package replication

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
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
		Vectors:  map[string]vcollection.Vector{"embedding": vcollection.Vector{Dense: []float32{value, 0, 0, 0}}},
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

// serveLeader puts the node surface on a real HTTP server, scoped to a
// single tenant (the id is arbitrary -- NewLeaderHandler is gone, so every
// bare-route test now speaks to a store through the per-tenant leader), and
// returns a follower pointed at it.
func serveLeader(t *testing.T, leader *vcollection.DurableStore) *Follower {
	t.Helper()
	srv := serveTenantLeader(t, map[string]*vcollection.DurableStore{"leader": leader})
	return &Follower{LeaderURL: srv.URL, Token: testToken, Tenant: "leader"}
}

// serveLeaderWithEpoch is serveLeader with a controllable epoch, so a test
// can simulate a promotion happening on the leader mid-test by changing what
// epoch returns between calls.
func serveLeaderWithEpoch(t *testing.T, leader *vcollection.DurableStore, epoch func() (Epoch, error)) *Follower {
	t.Helper()
	handler, err := NewTenantLeaderHandler(
		LeaderConfig{Token: testToken, SpoolDir: t.TempDir(), Epoch: func(string) (Epoch, error) { return epoch() }},
		func() []string { return []string{"leader"} },
		func(id string) (Source, bool) {
			if id != "leader" {
				return nil, false
			}
			return leader, true
		})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return &Follower{LeaderURL: srv.URL, Token: testToken, Tenant: "leader"}
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
	go func() { failed <- follower.Follow(followCtx, replica, base) }()

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
	go func() { failed <- follower.Follow(runCtx, replica, base) }()
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
	go func() { resumeFailed <- follower.Follow(resumeCtx, reopened, base) }()
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
	err = follower.Follow(followCtx, replica, base)
	if !errors.Is(err, ErrResyncRequired) {
		t.Fatalf("Follow = %v, want ErrResyncRequired", err)
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
	err = followerB.Follow(ctx, replica, base)
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

// Replica-ness has to outlive the process that established it.
//
// DurableStore.replica is in-memory by design, so a directory synced here and
// opened later by `deepdata serve` would come up as an ordinary store and
// accept writes -- forking the two histories at the same LSN. The marker the
// follower leaves is the only thing standing between a read replica and that
// split brain, so it must be there the moment Open returns, name the right
// leader, and still be there after a resume.
func TestASyncedDirectoryDeclaresItsLeaderOnDisk(t *testing.T) {
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
		t.Fatalf("bootstrap replica: %v", err)
	}

	leaderID := leader.Metadata().StoreID
	id, isReplica, err := ReplicaLeaderID(base)
	if err != nil {
		t.Fatalf("read replica marker: %v", err)
	}
	if !isReplica {
		t.Fatal("a bootstrapped replica directory does not say so on disk; serving it would accept writes")
	}
	if id != leaderID {
		t.Fatalf("marker names leader %x, store follows %x", id, leaderID)
	}
	if err := replica.Close(); err != nil {
		t.Fatal(err)
	}

	// Resuming must not lose it either: the second run takes Open's other path.
	resumed, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatalf("resume replica: %v", err)
	}
	t.Cleanup(func() { _ = resumed.Close() })
	if _, isReplica, err = ReplicaLeaderID(base); err != nil || !isReplica {
		t.Fatalf("resume dropped the replica marker: isReplica=%v err=%v", isReplica, err)
	}

	// A directory nobody replicated must not claim a leader; that is the
	// difference between "serve read-only" and "serve".
	if _, isReplica, err := ReplicaLeaderID(storePath(t, dir, "unrelated")); err != nil || isReplica {
		t.Fatalf("a plain directory reported itself a replica: isReplica=%v err=%v", isReplica, err)
	}
}

// The marker must not look like store state.
//
// Every durable artifact is basePath+"."+suffix, and both BootstrapReplica's
// emptiness scan and storeArtifactsExist read any such name as "a store lives
// here". A dotted marker would make a directory holding only a stale marker
// look occupied, sending a fresh sync down the resume path -- where it would
// open an empty store under a new random ID and fail with a store mismatch
// instead of bootstrapping.
func TestTheReplicaMarkerIsNotMistakenForStoreState(t *testing.T) {
	base := storePath(t, t.TempDir(), "replica")
	if err := MarkReplica(base, [16]byte{1}); err != nil {
		t.Fatal(err)
	}
	occupied, err := storeArtifactsExist(base)
	if err != nil {
		t.Fatal(err)
	}
	if occupied {
		t.Fatal("the replica marker counts as a durable store artifact; a re-sync would take the resume path over an empty directory")
	}
}

// Seed is the half of Open a StoreSet-owned store needs before it is ever
// opened for real: bootstrap from the leader, mark the directory, then let go
// of the lock so a plain open can follow -- exactly what a serve process does
// next.
func TestSeedBootstrapsAnEmptyDirAndLeavesItUnlocked(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}

	follower := serveLeader(t, leader)
	base := storePath(t, dir, "replica")

	bootstrapped, err := follower.Seed(ctx, base, base)
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	if !bootstrapped {
		t.Fatal("Seed on an empty directory reported no bootstrap")
	}
	leaderID := leader.Metadata().StoreID
	if id, isReplica, err := ReplicaLeaderID(base); err != nil || !isReplica || id != leaderID {
		t.Fatalf("marker after Seed: id=%x isReplica=%v err=%v, want %x", id, isReplica, err, leaderID)
	}

	// A directory Seed already occupied does nothing the second time.
	bootstrapped, err = follower.Seed(ctx, base, base)
	if err != nil {
		t.Fatalf("second seed: %v", err)
	}
	if bootstrapped {
		t.Fatal("Seed on an occupied directory reported a bootstrap")
	}

	// The lock must be free: Seed released it via Abort rather than holding it.
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("open seeded store: %v", err)
	}
	if got := store.Metadata().StoreID; got != leaderID {
		t.Fatalf("seeded StoreID = %x, want %x", got, leaderID)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

// Bind is the guard an occupied Open resumes through, exposed so a
// StoreSet-owned store can be bound the same way. It must mark a genuine
// replica of the leader, and refuse -- without touching -- a store that
// belongs to someone else.
func TestBindMarksAnOpenStoreAndRefusesAForeignOne(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	follower := serveLeader(t, leader)
	base := storePath(t, dir, "replica")
	if _, err := follower.Seed(ctx, base, base); err != nil {
		t.Fatal(err)
	}
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })

	if err := follower.Bind(ctx, store, base); err != nil {
		t.Fatalf("bind: %v", err)
	}
	if !store.IsReplica() {
		t.Fatal("Bind did not mark the store a replica")
	}
	if _, isReplica, err := ReplicaLeaderID(base); err != nil || !isReplica {
		t.Fatalf("Bind did not leave the on-disk marker: isReplica=%v err=%v", isReplica, err)
	}

	// A store minted under a different StoreID must be refused, and left
	// exactly as handed in: Bind does not own the store, so it must not abort
	// it on failure.
	otherBase := storePath(t, dir, "occupied")
	other, err := vcollection.OpenDurableStore(otherBase, otherBase)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = other.Close() })
	if err := follower.Bind(ctx, other, otherBase); !errors.Is(err, vcollection.ErrJournalStoreMismatch) {
		t.Fatalf("Bind on a foreign store = %v, want ErrJournalStoreMismatch", err)
	}
	if _, err := other.Tenants().CreateCollection(ctx, "tenant-x", testSchema("still-usable")); err != nil {
		t.Fatalf("store refused by Bind must still take a local write: %v", err)
	}
}

// A directory can end up with store artifacts but no epoch sidecar -- e.g. a
// crash between Seed's bootstrap and its WriteEpoch call. Bind is the guard
// every occupied Open resumes through, so it has to re-align the sidecar
// itself instead of leaving the directory permanently unfenced; MarkReplica
// already self-heals the same way on every Bind.
func TestBindWritesTheEpochSidecarEvenWhenMissing(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	want := Epoch{Number: 2, StartLSN: 7}
	follower := serveLeaderWithEpoch(t, leader, func() (Epoch, error) { return want, nil })
	base := storePath(t, dir, "replica")
	if _, err := follower.Seed(ctx, base, base); err != nil {
		t.Fatal(err)
	}

	// Simulate the crash: artifacts exist, sidecar does not.
	if err := os.Remove(epochPath(base)); err != nil {
		t.Fatal(err)
	}
	if got, err := ReadEpoch(base); err != nil || got != (Epoch{}) {
		t.Fatalf("ReadEpoch after removing the sidecar = %+v, %v; want zero value, nil error", got, err)
	}

	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if err := follower.Bind(ctx, store, base); err != nil {
		t.Fatalf("bind: %v", err)
	}
	if got, err := ReadEpoch(base); err != nil || got != want {
		t.Fatalf("ReadEpoch after Bind = %+v, %v; want %+v, nil", got, err, want)
	}
}

// OnPreamble is how a standby learns the leader's LatestLSN for a lag report
// without a second round trip: Follow already reads the preamble to check the
// StoreID, so this only has to hand the caller what it already parsed.
func TestFollowReportsThePreamble(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	doc := testDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
		t.Fatal(err)
	}
	wantLSN := leader.Metadata().AppliedLSN
	wantStoreID := leader.Metadata().StoreID

	follower := serveLeader(t, leader)
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	var mu sync.Mutex
	var got Preamble
	follower.OnPreamble = func(p Preamble) {
		mu.Lock()
		got = p
		mu.Unlock()
	}

	runCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	failed := make(chan error, 1)
	go func() { failed <- follower.Follow(runCtx, replica, base) }()
	waitFor(t, runCtx, failed, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return got.StoreID != ([16]byte{})
	}, "OnPreamble never fired")
	cancel()
	<-failed

	if got.StoreID != wantStoreID {
		t.Fatalf("OnPreamble StoreID = %x, want %x", got.StoreID, wantStoreID)
	}
	if got.LatestLSN != wantLSN {
		t.Fatalf("OnPreamble LatestLSN = %d, want %d", got.LatestLSN, wantLSN)
	}
}

// A leader restart must not stall the stream.
//
// The recorded two-node failure was a leader whose record order lived in
// memory. It restarted at 1, the follower asked for everything after 200, the
// leader answered that its latest was 1 -- so the follower saw nothing to
// fetch, applied nothing, and logged nothing, while every write taken after the
// restart stayed on the leader alone. Order is now derived from the durable
// journal, and this is what holds it there.
//
// The leader keeps one address across the restart, because that is what a
// restarted leader does: the follower reconnects to the same URL and has to be
// told the truth by a process that just rebuilt its state from disk.
func TestWritesAfterALeaderRestartStillReachTheReplica(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leaderBase := storePath(t, dir, "leader")

	leader, err := vcollection.OpenDurableStore(leaderBase, leaderBase)
	if err != nil {
		t.Fatalf("open leader: %v", err)
	}
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	handler := mustLeaderHandler(t, leader)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		h := handler
		mu.Unlock()
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	follower := &Follower{LeaderURL: srv.URL, Token: testToken, Tenant: "leader"}

	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatalf("bootstrap replica: %v", err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	followCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	fatal := make(chan error, 1)
	// Mirror the `deepdata replicate` loop rather than calling Follow once: a
	// dropped stream is retried from the replica's own cursor, and the restart
	// under test drops it. A single Follow would end at the restart, before the
	// assertion this test exists for.
	go func() {
		for followCtx.Err() == nil {
			// A resync demand is terminal for an operator -- the directory has
			// to be discarded -- so it is terminal here too. A leader restart
			// must never provoke one.
			if err := follower.Follow(followCtx, replica, base); errors.Is(err, ErrResyncRequired) {
				fatal <- err
				return
			}
			select {
			case <-followCtx.Done():
			case <-time.After(20 * time.Millisecond):
			}
		}
	}()

	before := testDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &before); err != nil {
		t.Fatal(err)
	}
	waitFor(t, followCtx, fatal, func() bool {
		_, ok := replica.Tenants().GetDocument("tenant-a", "docs", before.ID)
		return ok
	}, "pre-restart document never reached the replica; the stream was not live before the restart")

	// The restart: the leader's process-local state goes away entirely and the
	// next one rebuilds from the same directory.
	if err := leader.Close(); err != nil {
		t.Fatalf("close leader: %v", err)
	}
	restarted, err := vcollection.OpenDurableStore(leaderBase, leaderBase)
	if err != nil {
		t.Fatalf("reopen leader after restart: %v", err)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	mu.Lock()
	handler = mustLeaderHandler(t, restarted)
	mu.Unlock()
	// A process that exits takes its established connections with it. Without
	// this the follower stays parked on a stream the old leader will never
	// write to again, which is a hang, not the reconnect under test.
	srv.CloseClientConnections()

	after := testDocument(2)
	if err := restarted.Tenants().AddDocument(ctx, "tenant-a", "docs", &after); err != nil {
		t.Fatal(err)
	}
	// Content, not presence. A leader that lost its place would re-mint IDs from
	// the start and collide with a document the replica already holds, so asking
	// only whether the ID resolves would be answered by the stale document and
	// this test would pass while the write was never delivered.
	waitFor(t, followCtx, fatal, func() bool {
		return contentOf(replica, after.ID) == contentOf(restarted, after.ID)
	}, "a write taken after the leader restarted never reached the replica")

	// The restart must not have cost the replica what it already had.
	if got := contentOf(replica, before.ID); got != contentOf(restarted, before.ID) {
		t.Errorf("pre-restart document %d did not survive the leader restart intact: replica has %s",
			before.ID, got)
	}
}

// Equal document counts are not agreement.
//
// The recorded two-node failure had both nodes reporting 201 active documents
// with different documents behind that number, which every count-based health
// check calls healthy. The mutation pair here is chosen to hold the count
// still: one delete and one insert leave DocCount exactly where it was, so a
// replica that applied neither still passes the count check and only a
// document-for-document comparison can fail.
func TestEqualDocumentCountsAreNotEnoughToCallTwoNodesInSync(t *testing.T) {
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

	follower := serveLeader(t, leader)
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatalf("bootstrap replica: %v", err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	followCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	failed := make(chan error, 1)
	go func() { failed <- follower.Follow(followCtx, replica, base) }()

	if err := leader.Tenants().DeleteDocument(ctx, "tenant-a", "docs", seeded[0]); err != nil {
		t.Fatal(err)
	}
	replacement := testDocument(4)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &replacement); err != nil {
		t.Fatal(err)
	}
	waitFor(t, followCtx, failed, func() bool {
		_, ok := replica.Tenants().GetDocument("tenant-a", "docs", replacement.ID)
		return ok
	}, "the replacement document never reached the replica")

	// The weak check, asserted on purpose: it is what a count-based health probe
	// sees, and it has to agree here or the comparison below is measuring
	// something other than divergence behind an equal count.
	leaderInfo, err := leader.Tenants().GetCollectionInfo("tenant-a", "docs")
	if err != nil {
		t.Fatal(err)
	}
	replicaInfo, err := replica.Tenants().GetCollectionInfo("tenant-a", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if leaderInfo.DocCount != replicaInfo.DocCount {
		t.Fatalf("counts already disagree (leader %d, replica %d); this test is about divergence they cannot see",
			leaderInfo.DocCount, replicaInfo.DocCount)
	}

	// The check that actually holds: same key, same document.
	for _, id := range []uint64{seeded[1], seeded[2], replacement.ID} {
		if want, got := contentOf(leader, id), contentOf(replica, id); want != got {
			t.Errorf("document %d differs behind an equal count: leader %s, replica %s", id, want, got)
		}
	}
	if got := contentOf(replica, seeded[0]); got != absentDocument {
		t.Errorf("document %d was deleted on the leader but is still served by the replica: %s", seeded[0], got)
	}
}

const absentDocument = "<absent>"

// contentOf renders the parts of a document a reader would receive, so a
// comparison fails on a difference in served data and not on bookkeeping a
// caller never sees. Map printing is key-sorted, so the rendering is stable.
func contentOf(store *vcollection.DurableStore, id uint64) string {
	doc, ok := store.Tenants().GetDocument("tenant-a", "docs", id)
	if !ok {
		return absentDocument
	}
	return fmt.Sprintf("metadata=%v vectors=%v", doc.Metadata, doc.Vectors)
}

func mustLeaderHandler(t *testing.T, store *vcollection.DurableStore) http.Handler {
	t.Helper()
	handler, err := NewTenantLeaderHandler(LeaderConfig{Token: testToken, SpoolDir: t.TempDir()},
		func() []string { return []string{"leader"} },
		func(id string) (Source, bool) {
			if id != "leader" {
				return nil, false
			}
			return store, true
		})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

// serveTenantLeader puts the per-tenant node surface over stores (tenant ID
// -> its own store, each opened by openStore so it is a real DurableStore on
// disk) on a real HTTP server.
func serveTenantLeader(t *testing.T, stores map[string]*vcollection.DurableStore) *httptest.Server {
	t.Helper()
	tenants := func() []string {
		ids := make([]string, 0, len(stores))
		for id := range stores {
			ids = append(ids, id)
		}
		return ids
	}
	source := func(id string) (Source, bool) {
		store, ok := stores[id]
		return store, ok
	}
	handler, err := NewTenantLeaderHandler(LeaderConfig{Token: testToken, SpoolDir: t.TempDir()}, tenants, source)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

// A per-tenant leader lists exactly the tenants it was given, sorted, so a
// follower can discover what to replicate without an out-of-band directory.
func TestTenantLeaderListsTenantsSorted(t *testing.T) {
	dir := t.TempDir()
	acme := openStore(t, dir, "acme")
	t.Cleanup(func() { _ = acme.Close() })
	globex := openStore(t, dir, "globex")
	t.Cleanup(func() { _ = globex.Close() })
	srv := serveTenantLeader(t, map[string]*vcollection.DurableStore{"globex": globex, "acme": acme})

	follower := &Follower{LeaderURL: srv.URL, Token: testToken}
	ids, err := follower.Tenants(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != "acme" || ids[1] != "globex" {
		t.Fatalf("tenants = %v, want [acme globex]", ids)
	}
}

// A follower scoped to one tenant bootstraps and tails only that tenant's
// store. Each tenant behind a per-tenant leader is a completely separate
// DurableStore, so this is really testing that the leader's {tenant} routing
// and the follower's URL builder agree on which one "acme" means -- a
// routing bug that resolved every ID to the same store would leak globex's
// write into this replica.
func TestTenantLeaderFollowerFollowsOnlyItsTenant(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	acme := openStore(t, dir, "acme")
	t.Cleanup(func() { _ = acme.Close() })
	globex := openStore(t, dir, "globex")
	t.Cleanup(func() { _ = globex.Close() })
	if _, err := acme.Tenants().CreateCollection(ctx, "acme", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if _, err := globex.Tenants().CreateCollection(ctx, "globex", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	srv := serveTenantLeader(t, map[string]*vcollection.DurableStore{"acme": acme, "globex": globex})

	follower := &Follower{LeaderURL: srv.URL, Token: testToken, Tenant: "acme"}
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatalf("open replica: %v", err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	followCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	failed := make(chan error, 1)
	go func() { failed <- follower.Follow(followCtx, replica, base) }()

	acmeDoc := testDocument(1)
	if err := acme.Tenants().AddDocument(ctx, "acme", "docs", &acmeDoc); err != nil {
		t.Fatal(err)
	}
	globexDoc := testDocument(2)
	if err := globex.Tenants().AddDocument(ctx, "globex", "docs", &globexDoc); err != nil {
		t.Fatal(err)
	}
	waitFor(t, followCtx, failed, func() bool {
		_, ok := replica.Tenants().GetDocument("acme", "docs", acmeDoc.ID)
		return ok
	}, "acme's document never reached its replica")

	if _, ok := replica.Tenants().GetDocument("globex", "docs", globexDoc.ID); ok {
		t.Fatal("globex's document reached a replica scoped to acme")
	}
	if _, err := replica.Tenants().GetCollectionInfo("globex", "docs"); err == nil {
		t.Fatal("globex's collection reached a replica scoped to acme")
	}
}

// A tenant ID unknown to source(), or one that fails ValidTenantID, gets a
// 404 rather than falling through to some other tenant's store.
func TestTenantLeaderUnknownTenantIs404(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	acme := openStore(t, dir, "acme")
	t.Cleanup(func() { _ = acme.Close() })
	srv := serveTenantLeader(t, map[string]*vcollection.DurableStore{"acme": acme})

	for _, id := range []string{"globex", "bad.id"} {
		follower := &Follower{LeaderURL: srv.URL, Token: testToken, Tenant: id}
		if _, err := follower.Status(ctx); err == nil {
			t.Errorf("tenant %q: Status succeeded, want 404", id)
		}
	}
}

// The bare pre-multitenancy routes stay registered on a per-tenant leader so
// a misdirected caller gets an explanation instead of a generic 404, but they
// must never serve any tenant's data.
func TestTenantLeaderBareRouteHintsAtTenants(t *testing.T) {
	dir := t.TempDir()
	acme := openStore(t, dir, "acme")
	t.Cleanup(func() { _ = acme.Close() })
	srv := serveTenantLeader(t, map[string]*vcollection.DurableStore{"acme": acme})

	req, err := http.NewRequest(http.MethodGet, srv.URL+PathPrefix+"status", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+testToken)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("bare status route = %d, want 404", resp.StatusCode)
	}
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["error"] != "this leader replicates per tenant" {
		t.Errorf("error = %q", body["error"])
	}
	if body["hint"] == "" {
		t.Error("hint is empty")
	}
}

// The tenant list is gated the same way every other node route is: a caller
// without the node token gets nothing, not an inventory of tenant IDs.
func TestTenantLeaderListRequiresToken(t *testing.T) {
	dir := t.TempDir()
	acme := openStore(t, dir, "acme")
	t.Cleanup(func() { _ = acme.Close() })
	srv := serveTenantLeader(t, map[string]*vcollection.DurableStore{"acme": acme})

	resp, err := http.Get(srv.URL + PathPrefix + "tenants")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("tenants list without token = %d, want 401", resp.StatusCode)
	}
}

// A Follower with no Tenant set talks to the bare routes, which a per-tenant
// leader answers with a hint rather than any tenant's data -- so pointing an
// un-scoped follower at one is a hard error, not a silent no-op.
func TestFollowerWithoutTenantErrorsAgainstAPerTenantLeader(t *testing.T) {
	dir := t.TempDir()
	acme := openStore(t, dir, "acme")
	t.Cleanup(func() { _ = acme.Close() })
	srv := serveTenantLeader(t, map[string]*vcollection.DurableStore{"acme": acme})

	follower := &Follower{LeaderURL: srv.URL, Token: testToken}
	if _, err := follower.Status(context.Background()); err == nil {
		t.Fatal("Status succeeded with no Tenant set against a per-tenant leader")
	}
}

// The sidecar itself: absent reads back as zero, a write survives a read, and
// a corrupt file is an error rather than a guessed zero -- the same stance
// ReplicaLeaderID takes on marker.go, and for the same reason: the caller's
// next move is to decide whether to honor a leader's stream.
func TestEpochSidecarRoundTrip(t *testing.T) {
	base := filepath.Join(t.TempDir(), "store")

	if got, err := ReadEpoch(base); err != nil || got != (Epoch{}) {
		t.Fatalf("ReadEpoch on an absent sidecar = %+v, %v; want zero value, nil error", got, err)
	}

	want := Epoch{Number: 3, StartLSN: 41}
	if err := WriteEpoch(base, want); err != nil {
		t.Fatal(err)
	}
	if got, err := ReadEpoch(base); err != nil || got != want {
		t.Fatalf("ReadEpoch after WriteEpoch = %+v, %v; want %+v, nil", got, err, want)
	}

	if err := os.WriteFile(epochPath(base), []byte("not an epoch"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadEpoch(base); err == nil {
		t.Fatal("ReadEpoch on a corrupt sidecar succeeded; a demoted leader could pass as epoch 0")
	}
}

// A leader whose epoch is older than the replica's own was demoted: a
// promotion elsewhere already moved the replica ahead of it, and following it
// further would fork the tenant's history at the point the promotion started.
func TestFollowerRefusesAStaleLeader(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}

	follower := serveLeaderWithEpoch(t, leader, func() (Epoch, error) { return Epoch{}, nil }) // leader stuck at epoch 0
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = replica.Close() })

	// This replica already adopted epoch 1 from elsewhere; this leader never
	// advanced past 0.
	if err := WriteEpoch(base, Epoch{Number: 1}); err != nil {
		t.Fatal(err)
	}

	after := testDocument(9)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &after); err != nil {
		t.Fatal(err)
	}

	followCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := follower.Follow(followCtx, replica, base); !errors.Is(err, ErrStaleLeader) {
		t.Fatalf("Follow against a stale leader = %v, want ErrStaleLeader", err)
	}
	if _, ok := replica.Tenants().GetDocument("tenant-a", "docs", after.ID); ok {
		t.Fatal("a stale leader's record was applied")
	}
}

// A leader whose epoch just increased is the current one after a promotion.
// A replica that has not synced past the LSN the promotion started at has
// nothing at risk of forking, so it adopts the new epoch and keeps going.
func TestFollowerAdoptsAHigherEpochWhenNotPastItsStart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	first := testDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &first); err != nil {
		t.Fatal(err)
	}

	epoch := Epoch{} // matches what bootstrap will seed the replica's sidecar with
	follower := serveLeaderWithEpoch(t, leader, func() (Epoch, error) { return epoch, nil })
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = replica.Close() })
	startLSN := replica.ReplicaCursor().LSN

	// Promotion: the leader's epoch advances, starting exactly where this
	// replica already is -- nothing of the new history is missing to it.
	epoch = Epoch{Number: 1, StartLSN: startLSN}
	promoted := testDocument(2)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &promoted); err != nil {
		t.Fatal(err)
	}

	followCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	failed := make(chan error, 1)
	go func() { failed <- follower.Follow(followCtx, replica, base) }()
	waitFor(t, followCtx, failed, func() bool {
		_, ok := replica.Tenants().GetDocument("tenant-a", "docs", promoted.ID)
		return ok
	}, "record under the new epoch never reached the replica")
	cancel()
	<-failed

	if got, err := ReadEpoch(base); err != nil || got != epoch {
		t.Fatalf("replica's epoch sidecar = %+v, %v; want %+v, nil", got, err, epoch)
	}
}

// A replica that already synced past the LSN a promotion started at has
// records from a history the new epoch does not extend: adopting the new
// epoch here would silently fork, so it must resync from scratch instead.
func TestFollowerPastThePromotionPointMustResync(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	leader := openStore(t, dir, "leader")
	t.Cleanup(func() { _ = leader.Close() })
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", testSchema("docs")); err != nil {
		t.Fatal(err)
	}
	doc := testDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
		t.Fatal(err)
	}

	epoch := Epoch{}
	follower := serveLeaderWithEpoch(t, leader, func() (Epoch, error) { return epoch, nil })
	base := storePath(t, dir, "replica")
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = replica.Close() })
	pastLSN := replica.ReplicaCursor().LSN
	if pastLSN == 0 {
		t.Fatal("test needs a replica that has already applied at least one record")
	}

	// Promotion started one LSN before this replica's own cursor: it has a
	// record the new epoch's leader never had a chance to include.
	epoch = Epoch{Number: 1, StartLSN: pastLSN - 1}

	followCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := follower.Follow(followCtx, replica, base); !errors.Is(err, ErrResyncRequired) {
		t.Fatalf("Follow past the promotion point = %v, want ErrResyncRequired", err)
	}
}
