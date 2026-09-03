package replication

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

const testToken = "node-token-not-the-api-token"

func testSchema(name string) vcollection.CollectionSchema {
	return vcollection.CollectionSchema{
		Name: name,
		Fields: []vcollection.VectorField{{
			Name:  "embedding",
			Type:  vcollection.VectorTypeDense,
			Dim:   4,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}
}

func testDocument(value float32) vcollection.Document {
	return vcollection.Document{
		Vectors:  map[string]interface{}{"embedding": []float32{value, 0, 0, 0}},
		Metadata: map[string]interface{}{"value": value},
	}
}

// openStore opens a durable store under dir/name/collections.
func openStore(t *testing.T, dir, name string) *vcollection.DurableStore {
	t.Helper()
	base := storePath(t, dir, name)
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("open %s store: %v", name, err)
	}
	return store
}

func storePath(t *testing.T, dir, name string) string {
	t.Helper()
	base := filepath.Join(dir, name, "collections")
	if err := os.MkdirAll(filepath.Dir(base), 0o755); err != nil {
		t.Fatal(err)
	}
	return base
}

// serveLeader puts the node surface on a real HTTP server and returns a
// follower pointed at it.
func serveLeader(t *testing.T, leader *vcollection.DurableStore) *Follower {
	t.Helper()
	handler, err := NewLeaderHandler(leader, LeaderConfig{Token: testToken, SpoolDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return &Follower{LeaderURL: srv.URL, Token: testToken}
}

// The whole point of the transport, over a real socket: a replica materializes
// from nothing, reaches the leader's exact state including leader-minted
// document IDs, and keeps reaching it as the leader writes.
//
// Document IDs are the sharp edge. They are assigned by the leader, so a
// replica that re-mints them serves different data under the same key while
// every count-based health check reports agreement.
func TestReplicaBootstrapsOverHTTPAndTailsItsLeader(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })

	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	var seeded []uint64
	for _, v := range []float32{1, 2, 3} {
		doc := testDocument(v)
		if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
			t.Fatal(err)
		}
		seeded = append(seeded, doc.ID)
	}
	// Checkpoint first: this is the case a snapshot bootstrap exists for. The
	// leader no longer retains the records a replica starting at LSN 0 needs,
	// so a journal-only follower could never catch up.
	if err := leader.Checkpoint(); err != nil {
		t.Fatal(err)
	}

	follower := serveLeader(t, leader)
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatalf("bootstrap replica: %v", err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	if !replica.IsReplica() {
		t.Fatal("bootstrapped store is not marked a replica; a local write would wedge it")
	}
	for i, id := range seeded {
		doc, ok := replica.Tenants().GetDocument("tenant-a", "docs", id)
		if !ok {
			t.Fatalf("replica missing document %d (leader ID for value %d)", id, i+1)
		}
		if got := doc.Metadata["value"]; got != float64(i+1) && got != float32(i+1) {
			t.Errorf("replica document %d value = %v, want %d", id, got, i+1)
		}
	}

	// Park the follower on the leader's tail BEFORE the record under test
	// exists, so this fails if a streamed record never arrives.
	followCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	failed := make(chan error, 1)
	go func() { failed <- follower.Follow(followCtx, replica) }()

	tailed := testDocument(4)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &tailed); err != nil {
		t.Fatal(err)
	}
	waitFor(t, followCtx, failed, func() bool {
		_, ok := replica.Tenants().GetDocument("tenant-a", "docs", tailed.ID)
		return ok
	}, "tailed document never reached the replica over HTTP")

	// A second write proves the stream stays open rather than delivering once.
	second := testDocument(5)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &second); err != nil {
		t.Fatal(err)
	}
	waitFor(t, followCtx, failed, func() bool {
		_, ok := replica.Tenants().GetDocument("tenant-a", "docs", second.ID)
		return ok
	}, "second tailed document never arrived; the stream delivered once and stopped")

	// A collection created after bootstrap must replicate too: it is a
	// different mutation type and it moves the tenant/collection counters.
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-b", testSchema("later")); err != nil {
		t.Fatal(err)
	}
	waitFor(t, followCtx, failed, func() bool {
		_, err := replica.Tenants().GetCollectionInfo("tenant-b", "later")
		return err == nil
	}, "collection created after bootstrap never replicated")
}

// waitFor polls until done() or the follow loop dies. Polling rather than
// signalling because the assertion is about the replica's observable state,
// which is what a reader of this replica would see.
func waitFor(t *testing.T, ctx context.Context, failed <-chan error, done func() bool, msg string) {
	t.Helper()
	for {
		if done() {
			return
		}
		select {
		case err := <-failed:
			t.Fatalf("%s: follow loop returned %v", msg, err)
		case <-ctx.Done():
			t.Fatal(msg)
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// A restarted replica must resume from its own durable cursor and never
// re-apply a record: ApplyReplicated rejects a re-delivered LSN, so a follower
// that resumed from the wrong place would fail loudly instead of silently
// duplicating -- this proves it resumes from the right place.
func TestRestartedReplicaResumesWithoutReapplying(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	follower := serveLeader(t, leader)
	base := storePath(t, dir, "replica")

	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatal(err)
	}
	first := testDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &first); err != nil {
		t.Fatal(err)
	}
	runCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	failed := make(chan error, 1)
	go func() { failed <- follower.Follow(runCtx, replica) }()
	waitFor(t, runCtx, failed, func() bool {
		_, ok := replica.Tenants().GetDocument("tenant-a", "docs", first.ID)
		return ok
	}, "first document never replicated")
	cancel()
	<-failed
	cursorBefore := replica.ReplicaCursor().LSN
	if err := replica.Close(); err != nil {
		t.Fatalf("close replica: %v", err)
	}

	// The leader writes while the replica is down.
	offline := testDocument(2)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &offline); err != nil {
		t.Fatal(err)
	}

	// Reopen: the path is occupied now, so this must resume, not re-bootstrap.
	reopened, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatalf("reopen replica: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	if got := reopened.ReplicaCursor().LSN; got != cursorBefore {
		t.Fatalf("reopened cursor LSN = %d, want %d; it did not resume where it stopped", got, cursorBefore)
	}
	resumeCtx, cancelResume := context.WithTimeout(ctx, 30*time.Second)
	defer cancelResume()
	resumeFailed := make(chan error, 1)
	go func() { resumeFailed <- follower.Follow(resumeCtx, reopened) }()
	waitFor(t, resumeCtx, resumeFailed, func() bool {
		_, ok := reopened.Tenants().GetDocument("tenant-a", "docs", offline.ID)
		return ok
	}, "record written while the replica was down never arrived after restart")
}

// A replica too far behind the leader's retained journal must be told to
// resync, not left to apply a hole. The leader can only discover this after
// the 200, so the reason travels as a control frame.
func TestFollowerIsToldToResyncWhenTheLeaderDiscardedItsRecords(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	follower := serveLeader(t, leader)
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	// The leader moves on and checkpoints, which removes the records the
	// replica still needs.
	for _, v := range []float32{1, 2, 3} {
		doc := testDocument(v)
		if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
			t.Fatal(err)
		}
	}
	if err := leader.Checkpoint(); err != nil {
		t.Fatal(err)
	}

	followCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	err = follower.Follow(followCtx, replica)
	if !errors.Is(err, ErrResyncRequired) {
		t.Fatalf("Follow = %v, want ErrResyncRequired", err)
	}
}

// The node surface authorizes on its own credential. The snapshot route
// exports every tenant in one request, so an unauthenticated caller -- and a
// caller holding some other token -- must get nothing.
func TestNodeSurfaceRefusesEveryCallerWithoutTheNodeToken(t *testing.T) {
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	handler, err := NewLeaderHandler(leader, LeaderConfig{Token: testToken, SpoolDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	for _, route := range []string{"status", "snapshot", "journal"} {
		for name, auth := range map[string]string{
			"no header":    "",
			"wrong token":  "Bearer some-tenant-api-token",
			"empty bearer": "Bearer ",
		} {
			req, err := http.NewRequest(http.MethodGet, srv.URL+PathPrefix+route, nil)
			if err != nil {
				t.Fatal(err)
			}
			if auth != "" {
				req.Header.Set("Authorization", auth)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusUnauthorized {
				t.Errorf("%s %s = %d, want 401", route, name, resp.StatusCode)
			}
		}
	}
}

// A leader with no node token configured must not build a handler at all.
// Defaulting to "off" in the caller is one forgotten branch away from serving
// every tenant's state to whoever can reach the port.
func TestLeaderHandlerRefusesToBuildWithoutANodeToken(t *testing.T) {
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := NewLeaderHandler(leader, LeaderConfig{}); err == nil {
		t.Fatal("NewLeaderHandler succeeded with no token")
	}
}

// A follower pointed at a leader it has not been syncing with must refuse
// before applying a record, not interleave two histories.
func TestFollowerRefusesAForeignLeader(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leaderA := openStore(t, dir, "leader-a")
	t.Cleanup(func() { _ = leaderA.Close() })
	leaderB := openStore(t, dir, "leader-b")
	t.Cleanup(func() { _ = leaderB.Close() })

	followerA := serveLeader(t, leaderA)
	base := storePath(t, dir, "replica")
	replica, err := followerA.Open(ctx, base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	followerB := serveLeader(t, leaderB)
	err = followerB.Follow(ctx, replica)
	if !errors.Is(err, vcollection.ErrJournalStoreMismatch) {
		t.Fatalf("Follow against a foreign leader = %v, want ErrJournalStoreMismatch", err)
	}
}

// Open must refuse a directory holding a store that is not this leader's
// replica. Bootstrapping would destroy it; adopting it is worse -- the leader's
// records would be appended onto an unrelated history, and every later LSN
// check would agree because the numbering lines up.
func TestOpenRefusesAStoreThatIsNotThisLeadersReplica(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	follower := serveLeader(t, leader)

	// A store that is NOT a replica of this leader, sitting at the target path.
	base := storePath(t, dir, "occupied")
	other := openStore(t, dir, "occupied")
	if _, err := other.Tenants().CreateCollection(ctx, "tenant-x", testSchema("keep")); err != nil {
		t.Fatal(err)
	}
	if err := other.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := follower.Open(ctx, base, base); !errors.Is(err, vcollection.ErrJournalStoreMismatch) {
		t.Fatalf("Open on a foreign store = %v, want ErrJournalStoreMismatch", err)
	}
	// The existing state must still be there.
	reopened, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("existing store no longer opens after a refused bootstrap: %v", err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	if _, err := reopened.Tenants().GetCollectionInfo("tenant-x", "keep"); err != nil {
		t.Fatalf("refused bootstrap destroyed the existing store's data: %v", err)
	}
}
