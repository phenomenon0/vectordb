package collection

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// openReplicaTestPair opens a leader and an empty replica bound to it.
func openReplicaTestPair(t *testing.T) (*DurableStore, *DurableStore) {
	t.Helper()
	dir := t.TempDir()
	open := func(name string) *DurableStore {
		base := filepath.Join(dir, name, "collections")
		if err := os.MkdirAll(filepath.Dir(base), 0o755); err != nil {
			t.Fatal(err)
		}
		store, err := OpenDurableStore(base, base)
		if err != nil {
			t.Fatalf("open %s store: %v", name, err)
		}
		return store
	}
	leader, replica := open("leader"), open("replica")
	t.Cleanup(func() { _ = replica.Close() })
	t.Cleanup(func() { _ = leader.Close() })

	status, err := leader.JournalStatus()
	if err != nil {
		t.Fatal(err)
	}
	if err := replica.MakeReplica(status.StoreID); err != nil {
		t.Fatalf("make replica: %v", err)
	}
	return leader, replica
}

// pumpReplica drains everything the leader has committed into the replica and
// returns the payloads it applied, keyed by LSN. Payloads are copied because
// StreamJournal hands out a view of its own read buffer.
func pumpReplica(t *testing.T, leader, replica *DurableStore) map[uint64][]byte {
	t.Helper()
	applied := map[uint64][]byte{}
	if _, err := leader.StreamJournal(replica.ReplicaCursor(), func(record JournalRecord) error {
		if err := replica.ApplyReplicated(context.Background(), record); err != nil {
			return err
		}
		applied[record.LSN] = append([]byte(nil), record.Payload...)
		return nil
	}); err != nil {
		t.Fatalf("pump replica: %v", err)
	}
	return applied
}

// A replica must reach the leader's exact state — same collection, same
// document IDs, same contents — from the journal alone, and must keep reaching
// it as the leader writes. Document IDs are the sharp edge: they are minted by
// the leader, so a replica that re-mints them serves different data under the
// same key while every count-based health check reports agreement.
func TestReplicaCatchesUpAndTailsItsLeader(t *testing.T) {
	ctx := context.Background()
	leader, replica := openReplicaTestPair(t)

	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	var leaderIDs []uint64
	for _, v := range []float32{1, 2, 3} {
		doc := durableTestDocument(v)
		if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
			t.Fatal(err)
		}
		leaderIDs = append(leaderIDs, doc.ID)
	}

	if applied := pumpReplica(t, leader, replica); len(applied) != 4 {
		t.Fatalf("applied %d records, want 4 (1 create + 3 inserts)", len(applied))
	}
	if _, err := replica.Tenants().GetCollectionInfo("tenant-a", "docs"); err != nil {
		t.Fatalf("replica missing the replicated collection: %v", err)
	}
	for i, id := range leaderIDs {
		doc, ok := durableTestStoredDocument(t, replica, "tenant-a", "docs", id)
		if !ok {
			t.Fatalf("replica missing document %d (leader ID for value %d)", id, i+1)
		}
		if got := doc.Metadata["value"]; got != float64(i+1) && got != float32(i+1) {
			t.Fatalf("replica document %d value = %v, want %d", id, got, i+1)
		}
	}
	if got, want := replica.ReplicaCursor().LSN, uint64(4); got != want {
		t.Fatalf("replica cursor LSN = %d, want %d", got, want)
	}

	// Live tail: the follower is parked on the leader's tail before the record
	// under test exists, so this fails if the streamed record never arrives.
	followCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	arrived := make(chan uint64, 8)
	go func() {
		_ = leader.FollowJournal(followCtx, replica.ReplicaCursor(), func(record JournalRecord) error {
			if err := replica.ApplyReplicated(followCtx, record); err != nil {
				return err
			}
			arrived <- record.LSN
			return nil
		})
	}()
	time.Sleep(100 * time.Millisecond)

	tailed := durableTestDocument(4)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &tailed); err != nil {
		t.Fatal(err)
	}
	select {
	case lsn := <-arrived:
		if lsn != 5 {
			t.Fatalf("tailed LSN = %d, want 5", lsn)
		}
	case <-followCtx.Done():
		t.Fatal("tailed record never reached the replica")
	}
	if _, ok := durableTestStoredDocument(t, replica, "tenant-a", "docs", tailed.ID); !ok {
		t.Fatalf("replica missing tailed document %d", tailed.ID)
	}
}

// Every local write path must be refused. A replica that accepts one forks the
// history at that LSN: the legacy cluster spike accepted writes on its replica
// and both nodes then reported identical document counts with different data.
func TestReplicaRefusesEveryLocalWrite(t *testing.T) {
	ctx := context.Background()
	leader, replica := openReplicaTestPair(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	seed := durableTestDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &seed); err != nil {
		t.Fatal(err)
	}
	applied := pumpReplica(t, leader, replica)
	before := replica.ReplicaCursor().LSN

	// The inverse mistake is worse: feeding a replicated record into a leader
	// renumbers its own history under someone else's LSNs.
	if err := leader.ApplyReplicated(ctx, JournalRecord{LSN: 3, Payload: applied[2]}); !errors.Is(err, ErrReplicaNotConfigured) {
		t.Fatalf("ApplyReplicated on a leader: error = %v, want ErrReplicaNotConfigured", err)
	}

	upsert := durableTestDocument(9)
	upsert.ID = seed.ID
	batch := []Document{durableTestDocument(7)}
	cases := []struct {
		name string
		call func() error
	}{
		{"create collection", func() error {
			_, err := replica.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("other"))
			return err
		}},
		{"add document", func() error {
			doc := durableTestDocument(5)
			return replica.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc)
		}},
		{"batch add documents", func() error {
			return replica.Tenants().BatchAddDocuments(ctx, "tenant-a", "docs", batch)
		}},
		{"upsert document", func() error {
			return replica.Tenants().UpsertDocument(ctx, "tenant-a", "docs", &upsert)
		}},
		{"delete document", func() error {
			return replica.Tenants().DeleteDocument(ctx, "tenant-a", "docs", seed.ID)
		}},
		{"delete collection", func() error {
			return replica.Tenants().DeleteCollection(ctx, "tenant-a", "docs")
		}},
	}
	for _, tc := range cases {
		if err := tc.call(); !errors.Is(err, ErrReplicaReadOnly) {
			t.Fatalf("%s on replica: error = %v, want ErrReplicaReadOnly", tc.name, err)
		}
	}
	if got := replica.ReplicaCursor().LSN; got != before {
		t.Fatalf("refused writes advanced the replica cursor: %d -> %d", before, got)
	}
	if doc, ok := durableTestStoredDocument(t, replica, "tenant-a", "docs", seed.ID); !ok {
		t.Fatal("refused writes removed the replicated document")
	} else if doc.Metadata["value"] == float64(9) {
		t.Fatal("refused upsert still mutated the replicated document")
	}
}

// The replica's cursor is its own durable AppliedLSN, so a restart must resume
// exactly where it stopped: not one record early (double apply) and not one
// late (silent hole). A leader replaying its stream after a reconnect is the
// normal case, not an error case.
func TestReplicaResumesFromItsDurableCursorAfterRestart(t *testing.T) {
	ctx := context.Background()
	leader, replica := openReplicaTestPair(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	first := durableTestDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &first); err != nil {
		t.Fatal(err)
	}
	applied := pumpReplica(t, leader, replica)

	base := replica.basePath
	leaderID := replica.leaderID
	if err := replica.Close(); err != nil {
		t.Fatalf("close replica: %v", err)
	}
	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen replica: %v", err)
	}
	defer reopened.Close()
	if err := reopened.MakeReplica(leaderID); err != nil {
		t.Fatal(err)
	}
	if got, want := reopened.ReplicaCursor().LSN, uint64(2); got != want {
		t.Fatalf("cursor after restart = %d, want %d", got, want)
	}

	// A record already applied must be refused, not applied twice.
	if err := reopened.ApplyReplicated(ctx, JournalRecord{LSN: 2, Payload: applied[2]}); !errors.Is(err, ErrReplicaOutOfOrder) {
		t.Fatalf("re-delivered LSN 2: error = %v, want ErrReplicaOutOfOrder", err)
	}
	// So must a record that skips one.
	if err := reopened.ApplyReplicated(ctx, JournalRecord{LSN: 4, Payload: applied[2]}); !errors.Is(err, ErrReplicaOutOfOrder) {
		t.Fatalf("skipped to LSN 4: error = %v, want ErrReplicaOutOfOrder", err)
	}

	second := durableTestDocument(2)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &second); err != nil {
		t.Fatal(err)
	}
	if got := pumpReplica(t, leader, reopened); len(got) != 1 {
		t.Fatalf("resumed pump applied %d records, want 1", len(got))
	}
	if _, ok := durableTestStoredDocument(t, reopened, "tenant-a", "docs", second.ID); !ok {
		t.Fatalf("restarted replica missing document %d", second.ID)
	}
	if got, want := reopened.ReplicaCursor().LSN, uint64(3); got != want {
		t.Fatalf("cursor after resume = %d, want %d", got, want)
	}
}

// openBootstrapTestLeader opens a standalone leader plus the temp dir its
// replicas can be placed beside.
func openBootstrapTestLeader(t *testing.T) (*DurableStore, string) {
	t.Helper()
	dir := t.TempDir()
	base := filepath.Join(dir, "leader", "collections")
	if err := os.MkdirAll(filepath.Dir(base), 0o755); err != nil {
		t.Fatal(err)
	}
	leader, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("open leader store: %v", err)
	}
	t.Cleanup(func() { _ = leader.Close() })
	return leader, dir
}

// leaderSnapshotBytes captures one generation of the leader exactly as a
// bootstrap would ship it.
func leaderSnapshotBytes(t *testing.T, leader *DurableStore) ([]byte, JournalPosition) {
	t.Helper()
	var buf bytes.Buffer
	pos, err := leader.WriteSnapshot(&buf)
	if err != nil {
		t.Fatalf("write leader snapshot: %v", err)
	}
	return buf.Bytes(), pos
}

// bootstrapTestReplica ships the leader's current generation into a fresh
// path and opens the replica on it.
func bootstrapTestReplica(t *testing.T, leader *DurableStore, base string) (*DurableStore, JournalPosition) {
	t.Helper()
	stream, pos := leaderSnapshotBytes(t, leader)
	if err := os.MkdirAll(filepath.Dir(base), 0o755); err != nil {
		t.Fatal(err)
	}
	replica, err := BootstrapReplica(base, base, bytes.NewReader(stream), pos.StoreID)
	if err != nil {
		t.Fatalf("bootstrap replica: %v", err)
	}
	t.Cleanup(func() { _ = replica.Close() })
	return replica, pos
}

// This is the reason the phase exists. A leader that has ever checkpointed has
// deleted the journal records a brand new follower would need, so a replica
// that can only replay history can never join a leader that has run for a day —
// it either stalls or, worse, starts from an empty state that no count-based
// health check distinguishes from a healthy one. The premise is asserted first:
// if the cold cursor did NOT report a gap, this whole mechanism would be
// unnecessary and the rest of the test would prove nothing.
func TestBootstrapReplicaStartsAfterLeaderCheckpoint(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	var leaderIDs []uint64
	addDoc := func(v float32) {
		t.Helper()
		doc := durableTestDocument(v)
		if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
			t.Fatal(err)
		}
		leaderIDs = append(leaderIDs, doc.ID)
	}
	for _, v := range []float32{1, 2, 3} {
		addDoc(v)
	}
	if err := leader.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	for _, v := range []float32{4, 5} {
		addDoc(v)
	}
	if _, err := leader.StreamJournal(JournalCursor{}, nil); !errors.Is(err, ErrJournalGap) {
		t.Fatalf("cold cursor on a checkpointed leader: error = %v, want ErrJournalGap", err)
	}

	replica, pos := bootstrapTestReplica(t, leader, filepath.Join(dir, "replica", "collections"))
	if got, want := pos.LatestLSN, uint64(6); got != want {
		t.Fatalf("bootstrap position LSN = %d, want %d (1 create + 5 inserts)", got, want)
	}
	if !replica.IsReplica() {
		t.Fatal("bootstrapped store is not read-only; a local write would take the leader's next LSN")
	}
	for i, id := range leaderIDs {
		doc, ok := durableTestStoredDocument(t, replica, "tenant-a", "docs", id)
		if !ok {
			t.Fatalf("bootstrapped replica missing document %d (leader ID for value %d)", id, i+1)
		}
		if got := doc.Metadata["value"]; got != float64(i+1) && got != float32(i+1) {
			t.Fatalf("bootstrapped document %d value = %v, want %d", id, got, i+1)
		}
	}

	// The load-bearing half: the replica's journal counter must have been
	// seeded from the snapshot header, so the leader's very next record is
	// accepted with no replay of anything the snapshot already holds.
	addDoc(6)
	applied := pumpReplica(t, leader, replica)
	if len(applied) != 1 {
		t.Fatalf("pump after bootstrap applied %d records, want exactly 1", len(applied))
	}
	if _, ok := applied[7]; !ok {
		t.Fatalf("pump applied %v, want the single record at LSN 7", applied)
	}
	if _, ok := durableTestStoredDocument(t, replica, "tenant-a", "docs", leaderIDs[5]); !ok {
		t.Fatalf("replica missing the post-bootstrap document %d", leaderIDs[5])
	}
	if got, want := replica.ReplicaCursor().LSN, uint64(7); got != want {
		t.Fatalf("replica cursor after resume = %d, want %d", got, want)
	}
}

// A bootstrap must reproduce the leader's WHOLE store, not just its root
// manager, and the LSN it reports must be the one actually baked into the bytes
// it wrote. A position that disagrees with the state is the hidden-divergence
// failure: the replica reports a healthy cursor while missing a tenant, and
// every count check on the root manager still passes.
func TestBootstrappedReplicaCarriesEveryTenantAtItsReportedLSN(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	want := map[string][]uint64{}
	for _, tenant := range []string{"tenant-a", "tenant-b"} {
		if _, err := leader.Tenants().CreateCollection(ctx, tenant, durableTestSchema("docs")); err != nil {
			t.Fatal(err)
		}
		for _, v := range []float32{1, 2} {
			doc := durableTestDocument(v)
			if err := leader.Tenants().AddDocument(ctx, tenant, "docs", &doc); err != nil {
				t.Fatal(err)
			}
			want[tenant] = append(want[tenant], doc.ID)
		}
	}

	replica, pos := bootstrapTestReplica(t, leader, filepath.Join(dir, "replica", "collections"))
	if got := replica.Metadata().AppliedLSN; got != pos.LatestLSN {
		t.Fatalf("replica AppliedLSN = %d, but WriteSnapshot reported %d", got, pos.LatestLSN)
	}
	if got := replica.ReplicaCursor().LSN; got != pos.LatestLSN {
		t.Fatalf("replica cursor = %d, but WriteSnapshot reported %d", got, pos.LatestLSN)
	}
	if applied := pumpReplica(t, leader, replica); len(applied) != 0 {
		t.Fatalf("pump applied %d records against a fully caught-up bootstrap, want 0", len(applied))
	}
	for tenant, ids := range want {
		for i, id := range ids {
			doc, ok := durableTestStoredDocument(t, replica, tenant, "docs", id)
			if !ok {
				t.Fatalf("replica missing %s document %d", tenant, id)
			}
			if got := doc.Metadata["value"]; got != float64(i+1) && got != float32(i+1) {
				t.Fatalf("replica %s document %d value = %v, want %d", tenant, id, got, i+1)
			}
			vec, ok := doc.Vectors["embedding"].([]float32)
			if !ok || len(vec) != 4 || vec[0] != float32(i+1) {
				t.Fatalf("replica %s document %d vector = %v, want [%d 0 0 0]", tenant, id, doc.Vectors["embedding"], i+1)
			}
		}
	}
}

// Bootstrap writes over basePath+".snapshot". Pointed at a path that already
// holds a store — a typo, a reused data dir, a re-run of a provisioning script —
// an unguarded copy destroys that store's newest generation with no way back.
// The byte-level assertion is the load-bearing one: an error-only check would
// still pass if the guard ran AFTER the copy had already landed.
func TestBootstrapReplicaRefusesNonEmptyBasePath(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	seed := durableTestDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &seed); err != nil {
		t.Fatal(err)
	}
	stream, pos := leaderSnapshotBytes(t, leader)

	victimBase := filepath.Join(dir, "victim", "collections")
	if err := os.MkdirAll(filepath.Dir(victimBase), 0o755); err != nil {
		t.Fatal(err)
	}
	victim, err := OpenDurableStore(victimBase, victimBase)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := victim.Tenants().CreateCollection(ctx, "tenant-victim", durableTestSchema("keep")); err != nil {
		t.Fatal(err)
	}
	kept := durableTestDocument(42)
	if err := victim.Tenants().AddDocument(ctx, "tenant-victim", "keep", &kept); err != nil {
		t.Fatal(err)
	}
	if err := victim.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(collectionSnapshotPath(victimBase))
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(before)

	store, err := BootstrapReplica(victimBase, victimBase, bytes.NewReader(stream), pos.StoreID)
	if err == nil {
		_ = store.Abort()
		t.Fatal("bootstrap overwrote an existing store instead of refusing it")
	}
	after, err := os.ReadFile(collectionSnapshotPath(victimBase))
	if err != nil {
		t.Fatal(err)
	}
	if sha256.Sum256(after) != digest {
		t.Fatal("refused bootstrap still rewrote the existing store's snapshot")
	}
	reopened, err := OpenDurableStore(victimBase, victimBase)
	if err != nil {
		t.Fatalf("existing store no longer opens after a refused bootstrap: %v", err)
	}
	defer reopened.Close()
	if _, ok := durableTestStoredDocument(t, reopened, "tenant-victim", "keep", kept.ID); !ok {
		t.Fatalf("existing store lost document %d to a refused bootstrap", kept.ID)
	}
}

// The snapshot is the only evidence of who the leader is, and MakeReplica
// cannot check it. If the copy never landed where the caller thinks it did,
// OpenDurableStore mints a fresh random StoreID and the store replicates
// happily under a foreign identity — the divergence surfaces much later as a
// gap or a missing collection mid-replay. The store-ID comparison turns that
// into a startup refusal, and the refusal must not leak the lifetime lock or
// the path can never be opened again without restarting the process.
func TestBootstrapReplicaRejectsForeignLeaderID(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	stream, pos := leaderSnapshotBytes(t, leader)
	foreign := pos.StoreID
	foreign[0]++

	base := filepath.Join(dir, "replica", "collections")
	if err := os.MkdirAll(filepath.Dir(base), 0o755); err != nil {
		t.Fatal(err)
	}
	store, err := BootstrapReplica(base, base, bytes.NewReader(stream), foreign)
	if !errors.Is(err, ErrJournalStoreMismatch) {
		if store != nil {
			_ = store.Abort()
		}
		t.Fatalf("bootstrap with a foreign leader ID: error = %v, want ErrJournalStoreMismatch", err)
	}
	if store != nil {
		_ = store.Abort()
		t.Fatal("rejected bootstrap still returned a usable store")
	}
	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("rejected bootstrap leaked the store lock: %v", err)
	}
	_ = reopened.Abort()
}

// A truncated transfer must never open. The snapshot's trailing SHA-256 is the
// only thing standing between a half-delivered stream and a replica that
// silently serves partial data while reporting the leader's LSN — exactly the
// hidden divergence a count-based health check cannot see. This test also pins
// the decision to route through OpenDurableStore rather than a bespoke loader:
// any hand-rolled decode that skipped the checksum would pass everything else.
func TestBootstrapReplicaRejectsTruncatedSnapshot(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	doc := durableTestDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
		t.Fatal(err)
	}
	stream, pos := leaderSnapshotBytes(t, leader)

	base := filepath.Join(dir, "replica", "collections")
	if err := os.MkdirAll(filepath.Dir(base), 0o755); err != nil {
		t.Fatal(err)
	}
	store, err := BootstrapReplica(base, base, bytes.NewReader(stream[:len(stream)-1]), pos.StoreID)
	if err == nil {
		_ = store.Abort()
		t.Fatal("bootstrap accepted a truncated snapshot")
	}
	if store != nil {
		_ = store.Abort()
		t.Fatal("failed bootstrap still returned a store")
	}
	// A marker without a loadable snapshot is permanently unopenable, so the
	// failure must not have minted one.
	if _, err := os.Stat(collectionMarkerPath(base)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed bootstrap wrote an initialization marker: %v", err)
	}
}

// A bootstrapped replica is a normal durable store afterwards: it must survive
// restart and resume from its own AppliedLSN. This is what proves the decision
// to let the open path mint .initialized instead of writing one during
// bootstrap — a marker authored with any other store ID makes every future open
// die on "initialization marker store ID does not match snapshot", with no
// repair path in code, and that failure only appears on the SECOND open.
func TestBootstrappedReplicaResumesAfterRestart(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	replica, pos := bootstrapTestReplica(t, leader, filepath.Join(dir, "replica", "collections"))

	first := durableTestDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &first); err != nil {
		t.Fatal(err)
	}
	if applied := pumpReplica(t, leader, replica); len(applied) != 1 {
		t.Fatalf("pump applied %d records, want 1", len(applied))
	}
	base := replica.basePath
	cursorBefore := replica.ReplicaCursor()
	if err := replica.Close(); err != nil {
		t.Fatalf("close bootstrapped replica: %v", err)
	}

	markerID, err := readCollectionInitializationMarker(base)
	if err != nil {
		t.Fatalf("read replica initialization marker: %v", err)
	}
	if markerID != pos.StoreID {
		t.Fatalf("replica marker store ID = %x, want the leader's %x", markerID, pos.StoreID)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen bootstrapped replica: %v", err)
	}
	defer reopened.Close()
	if err := reopened.MakeReplica(pos.StoreID); err != nil {
		t.Fatal(err)
	}
	if got := reopened.ReplicaCursor(); got != cursorBefore {
		t.Fatalf("cursor across restart = %+v, want %+v", got, cursorBefore)
	}

	second := durableTestDocument(2)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &second); err != nil {
		t.Fatal(err)
	}
	if applied := pumpReplica(t, leader, reopened); len(applied) != 1 {
		t.Fatalf("resumed pump applied %d records, want 1", len(applied))
	}
	if _, ok := durableTestStoredDocument(t, reopened, "tenant-a", "docs", second.ID); !ok {
		t.Fatalf("restarted replica missing document %d", second.ID)
	}
	if _, ok := durableTestStoredDocument(t, reopened, "tenant-a", "docs", first.ID); !ok {
		t.Fatalf("restart lost the pre-restart document %d", first.ID)
	}
}

// A checkpoint on an otherwise idle leader deletes every journal artifact while
// its in-memory LSN stays put. A follower behind that boundary then reads zero
// records and, without an explicit check, is told the read succeeded: it parks
// on the notifier and serves stale data forever with nothing logged anywhere.
// That is the exact silent stall the legacy cluster shipped, and it is the one
// gap the per-record successor check inside StreamJournal cannot see, because
// it only runs when a record is actually delivered. The follower must be told
// to re-bootstrap instead.
func TestStreamJournalReportsGapWhenCheckpointLeftNoRecords(t *testing.T) {
	ctx := context.Background()
	leader, _ := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	doc := durableTestDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
		t.Fatal(err)
	}
	if err := leader.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	// Premise: no records survive, and no write follows to reveal the hole.
	for _, suffix := range []string{".journal", ".journal.frozen"} {
		if _, err := os.Stat(leader.basePath + suffix); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("checkpoint left %s behind, test premise is void: %v", suffix, err)
		}
	}

	delivered := 0
	pos, err := leader.StreamJournal(JournalCursor{LSN: 1}, func(JournalRecord) error {
		delivered++
		return nil
	})
	if !errors.Is(err, ErrJournalGap) {
		t.Fatalf("stale cursor on an idle checkpointed leader: error = %v, want ErrJournalGap", err)
	}
	if delivered != 0 {
		t.Fatalf("delivered %d records across a gap; a follower must get none", delivered)
	}
	if pos.LatestLSN != 2 {
		t.Fatalf("reported leader LSN = %d, want 2", pos.LatestLSN)
	}
	// A caught-up follower must still see success, or every idle tail becomes
	// a spurious re-bootstrap.
	if _, err := leader.StreamJournal(JournalCursor{LSN: 2}, nil); err != nil {
		t.Fatalf("caught-up cursor on the same leader: error = %v, want nil", err)
	}
}

// A node holds many shards side by side, so the emptiness scan has to tell
// "this store" from "a store whose name starts the same way". Matching a bare
// prefix conflates them: bootstrapping shard1 is refused because shard10 exists
// beside it, and the operator sees a corruption-shaped error for a healthy
// directory. Deleting the "." from the prefix in BootstrapReplica makes this
// fail.
func TestBootstrapReplicaIgnoresSiblingStoresSharingAPrefix(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	node := filepath.Join(dir, "node")
	if err := os.MkdirAll(node, 0o755); err != nil {
		t.Fatal(err)
	}
	// An unrelated shard whose name extends the bootstrap target's name.
	sibling, err := OpenDurableStore(filepath.Join(node, "shard10"), filepath.Join(node, "shard10"))
	if err != nil {
		t.Fatal(err)
	}
	if err := sibling.Close(); err != nil {
		t.Fatal(err)
	}

	replica, _ := bootstrapTestReplica(t, leader, filepath.Join(node, "shard1"))
	if !replica.IsReplica() {
		t.Fatal("bootstrapped store beside a prefix-sharing sibling is not a replica")
	}
	// The sibling must still be intact: the scan is a refusal, never a cleanup.
	if _, err := os.Stat(collectionSnapshotPath(filepath.Join(node, "shard10"))); err != nil {
		t.Fatalf("sibling shard10 snapshot disturbed by bootstrapping shard1: %v", err)
	}
}

// WriteSnapshot reports an LSN that a follower then trusts as its starting
// point. A closed store's manager and tenant maps are torn down, so snapshotting
// one would hand back a header LSN over near-empty contents -- the replica opens
// "successfully", reports the leader's LSN, and holds none of its data. That is
// hidden divergence: every count-based health check passes. Deleting the
// stateErrorLocked call in WriteSnapshot makes this fail.
func TestWriteSnapshotRefusesAClosedStore(t *testing.T) {
	ctx := context.Background()
	leader, _ := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	doc := durableTestDocument(1)
	if err := leader.Tenants().AddDocument(ctx, "tenant-a", "docs", &doc); err != nil {
		t.Fatal(err)
	}
	if err := leader.Close(); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	pos, err := leader.WriteSnapshot(&buf)
	if err == nil {
		t.Fatalf("WriteSnapshot on a closed store succeeded, reporting LSN %d over %d bytes", pos.LatestLSN, buf.Len())
	}
	if buf.Len() != 0 {
		t.Errorf("refused WriteSnapshot still wrote %d bytes to the sink", buf.Len())
	}
}

// BootstrapReplica is a trust boundary: its arguments come from whatever
// transport carries the snapshot, and each of these degenerate inputs fails far
// away from its cause if it is allowed through. A zero leader ID is the one that
// matters most -- it is what an unmarshalled-but-never-populated peer identity
// looks like, and it would bind the replica to a leader that cannot exist while
// the store still reports IsReplica.
func TestBootstrapReplicaRejectsDegenerateArguments(t *testing.T) {
	ctx := context.Background()
	leader, dir := openBootstrapTestLeader(t)
	if _, err := leader.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	stream, pos := leaderSnapshotBytes(t, leader)
	base := filepath.Join(dir, "replica", "collections")
	if err := os.MkdirAll(filepath.Dir(base), 0o755); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name     string
		base     string
		reader   io.Reader
		leaderID [16]byte
	}{
		{"zero leader ID", base, bytes.NewReader(stream), [16]byte{}},
		{"empty base path", "", bytes.NewReader(stream), pos.StoreID},
		{"nil reader", base, nil, pos.StoreID},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store, err := BootstrapReplica(tc.base, tc.base, tc.reader, tc.leaderID)
			if err == nil {
				_ = store.Abort()
				t.Fatal("bootstrap accepted a degenerate argument")
			}
			// Refused before touching the filesystem: a rejected bootstrap must
			// not leave a half-written snapshot that a retry then reads as a
			// non-empty target and refuses forever.
			if _, statErr := os.Stat(collectionSnapshotPath(base)); !os.IsNotExist(statErr) {
				t.Fatalf("refused bootstrap left a snapshot behind: %v", statErr)
			}
		})
	}
}
