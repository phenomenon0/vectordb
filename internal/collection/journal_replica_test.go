package collection

import (
	"context"
	"errors"
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
