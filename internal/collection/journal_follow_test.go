package collection

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// collectFollowedLSNs drains a follower until it has seen want records or the
// context expires, returning the LSNs and payload lengths it observed.
func collectFollowedLSNs(t *testing.T, store *DurableStore, cursor JournalCursor) []uint64 {
	t.Helper()
	var seen []uint64
	if _, err := store.StreamJournal(cursor, func(record JournalRecord) error {
		if len(record.Payload) == 0 {
			t.Fatalf("LSN %d streamed an empty payload", record.LSN)
		}
		seen = append(seen, record.LSN)
		return nil
	}); err != nil {
		t.Fatalf("stream journal: %v", err)
	}
	return seen
}

func mutateFollowTestStore(t *testing.T, store *DurableStore, tenant string, values ...float32) {
	t.Helper()
	ctx := context.Background()
	tenants := store.Tenants()
	for _, v := range values {
		doc := durableTestDocument(v)
		if err := tenants.AddDocument(ctx, tenant, "docs", &doc); err != nil {
			t.Fatal(err)
		}
	}
}

// A follower that starts from zero must see every committed mutation exactly
// once, in LSN order, with no holes. Ordering is the whole contract: applying
// a delete before its insert corrupts the replica silently.
func TestStreamJournalDeliversEveryRecordInOrder(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	mutateFollowTestStore(t, store, "tenant-a", 1, 2, 3)

	got := collectFollowedLSNs(t, store, JournalCursor{})
	want := []uint64{1, 2, 3, 4}
	if len(got) != len(want) {
		t.Fatalf("streamed LSNs = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("streamed LSNs = %v, want %v", got, want)
		}
	}

	// Resuming from a cursor must deliver the remainder and nothing already seen.
	if rest := collectFollowedLSNs(t, store, JournalCursor{LSN: 2}); len(rest) != 2 || rest[0] != 3 || rest[1] != 4 {
		t.Fatalf("resumed LSNs = %v, want [3 4]", rest)
	}
	// A cursor at the head must deliver nothing rather than replaying.
	if head := collectFollowedLSNs(t, store, JournalCursor{LSN: 4}); len(head) != 0 {
		t.Fatalf("cursor at head streamed %v, want nothing", head)
	}
}

// Follow mode exists so a replica sees writes as they happen, not only what
// was already on disk when it connected. A follower that only ever drains the
// backlog is a snapshot copier, not a replica.
//
// The follower is driven to the tail and parked there before any of the
// records under test are written, so this fails if the append-side wakeup is
// ever dropped — with only a backlog drain it would block forever.
func TestFollowJournalDeliversRecordsAppendedAfterTheFollowerStarted(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// LSN 1 is the entire backlog; the follower consumes it and then has
	// nothing left to read.
	if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	followCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	var (
		mu       sync.Mutex
		seen     []uint64
		atTail   = make(chan struct{})
		tailOnce sync.Once
		done     = make(chan error, 1)
	)
	go func() {
		done <- store.FollowJournal(followCtx, JournalCursor{}, func(record JournalRecord) error {
			mu.Lock()
			seen = append(seen, record.LSN)
			n := len(seen)
			mu.Unlock()
			if record.LSN == 1 {
				tailOnce.Do(func() { close(atTail) })
			}
			if n == 4 {
				cancel()
			}
			return nil
		})
	}()

	select {
	case <-atTail:
	case err := <-done:
		t.Fatalf("follower exited before draining the backlog: %v", err)
	}
	// The backlog is drained; give the follower time to park on the tail so the
	// records below can only reach it through the append-side wakeup.
	time.Sleep(100 * time.Millisecond)
	mutateFollowTestStore(t, store, "tenant-a", 1, 2, 3)

	if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("follow journal: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 4 {
		t.Fatalf("follower saw LSNs %v, want the 4 committed records", seen)
	}
	for i, lsn := range seen {
		if lsn != uint64(i+1) {
			t.Fatalf("follower saw LSNs %v, want [1 2 3 4]", seen)
		}
	}
}

// The regression the multi-node spike surfaced on the retired engine: its WAL
// sequence lived in memory, so a leader restart reset it to 1 while the
// follower's cursor sat at 200. The follower asked for records after 200,
// received an empty page, logged nothing, and never replicated again.
//
// The collection journal recovers its LSN from durable state, so the cursor
// stays meaningful across a restart. This test fails if that ever regresses.
func TestStreamJournalLSNSurvivesLeaderRestart(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	mutateFollowTestStore(t, store, "tenant-a", 1, 2)
	before, err := store.JournalStatus()
	if err != nil {
		t.Fatal(err)
	}
	if before.LatestLSN != 3 {
		t.Fatalf("pre-restart LSN = %d, want 3", before.LatestLSN)
	}
	// Abort, not Close: no checkpoint, so recovery runs the journal replay path
	// a crashed leader would take.
	abandonDurableStoreForTest(t, store)

	restarted, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("restart leader: %v", err)
	}
	defer restarted.Close()

	after, err := restarted.JournalStatus()
	if err != nil {
		t.Fatal(err)
	}
	if after.StoreID != before.StoreID {
		t.Fatalf("store UUID changed across restart: %x -> %x", before.StoreID, after.StoreID)
	}
	if after.LatestLSN < before.LatestLSN {
		t.Fatalf("LSN went backwards across restart: %d -> %d", before.LatestLSN, after.LatestLSN)
	}

	// A follower that was caught up before the restart must resume cleanly and
	// receive the post-restart write — the exact step the legacy engine dropped.
	mutateFollowTestStore(t, restarted, "tenant-a", 3)
	got := collectFollowedLSNs(t, restarted, JournalCursor{StoreID: before.StoreID, LSN: before.LatestLSN})
	if len(got) != 1 || got[0] != before.LatestLSN+1 {
		t.Fatalf("post-restart stream = %v, want [%d]", got, before.LatestLSN+1)
	}
}

// A cursor issued by another store must be refused. Without this a replica
// repointed at the wrong leader, or at one restored from a backup, would
// interleave foreign mutations into its own state and report success.
func TestStreamJournalRejectsForeignCursor(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	foreign := JournalCursor{StoreID: [16]byte{0xFF}, LSN: 0}
	if _, err := store.StreamJournal(foreign, nil); !errors.Is(err, ErrJournalStoreMismatch) {
		t.Fatalf("foreign cursor error = %v, want ErrJournalStoreMismatch", err)
	}
}

// A checkpoint removes journal artifacts it has covered. A follower whose
// cursor predates that boundary can no longer be caught up incrementally and
// must be told to resnapshot rather than handed a partial range that would
// leave a hole in its state.
func TestStreamJournalReportsGapAfterCheckpointRemovesRecords(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	mutateFollowTestStore(t, store, "tenant-a", 1, 2)
	if err := store.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(base + ".journal"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("checkpoint left a journal behind, test premise is void: %v", err)
	}
	// New writes after the checkpoint start a fresh artifact at LSN 4.
	mutateFollowTestStore(t, store, "tenant-a", 3)

	stale := JournalCursor{LSN: 1}
	delivered := 0
	if _, err := store.StreamJournal(stale, func(JournalRecord) error {
		delivered++
		return nil
	}); !errors.Is(err, ErrJournalGap) {
		t.Fatalf("stale cursor error = %v, want ErrJournalGap", err)
	}
	if delivered != 0 {
		t.Fatalf("delivered %d records across a gap; a follower must get none", delivered)
	}
}
