package index

import (
	"context"
	"testing"
)

func mustVec(dim int, n float32) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(i) + n
	}
	return v
}

// TestHNSWBatchAddRollbackCancel verifies that a batch whose parallel graph
// insertion fails (cancelled context) is rolled back atomically: counts are
// unchanged, resurrected tombstones are restored, and every ID in the failed
// batch is re-addable as if the batch had never run.
func TestHNSWBatchAddRollbackPartial(t *testing.T) {
	dim := 8
	idx, err := NewHNSWIndex(dim, nil)
	if err != nil {
		t.Fatalf("NewHNSWIndex: %v", err)
	}
	hnswIdx := idx.(*HNSWIndex)

	ctx := context.Background()

	// Seed one vector, then tombstone it so its ID can be resurrected.
	if err := idx.Add(ctx, 1, mustVec(dim, 1)); err != nil {
		t.Fatalf("seed Add: %v", err)
	}
	if err := idx.Delete(ctx, 1); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// A batch large enough to take the parallel graph-insertion path, with a
	// cancelled context so the insertion fails mid-flight.
	vectors := map[uint64][]float32{1: mustVec(dim, 100)}
	for id := uint64(2); id <= 101; id++ {
		vectors[id] = mustVec(dim, float32(id))
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	if err := hnswIdx.BatchAdd(cancelled, vectors); err == nil {
		t.Fatal("BatchAdd under a cancelled context: expected error, got nil")
	}

	stats := idx.Stats()
	if stats.Count != 1 {
		t.Errorf("stats.Count = %d after failed batch, want 1", stats.Count)
	}
	if stats.Deleted != 1 {
		t.Errorf("stats.Deleted = %d after failed batch, want 1 (tombstone restored)", stats.Deleted)
	}

	// The tombstoned ID must still be resurrectable rather than "already exists".
	if err := idx.Add(ctx, 1, mustVec(dim, 1)); err != nil {
		t.Fatalf("re-add tombstoned ID 1 after failed batch: %v", err)
	}

	// Every new ID from the failed batch must be addable with no orphan state.
	if err := idx.Add(ctx, 50, mustVec(dim, 50)); err != nil {
		t.Fatalf("re-add new ID 50 after failed batch: %v", err)
	}

	if got := idx.Stats().Count; got != 2 {
		t.Errorf("stats.Count = %d after rollback + re-adds, want 2", got)
	}
}

// TestHNSWBatchAddDuplicateLeavesNoMutation verifies that a duplicate-ID error
// inside a batch does not partially mutate the index.
func TestHNSWBatchAddDuplicateLeavesNoMutation(t *testing.T) {
	dim := 8
	ctx := context.Background()
	idx, err := NewHNSWIndex(dim, nil)
	if err != nil {
		t.Fatalf("NewHNSWIndex: %v", err)
	}
	hnswIdx := idx.(*HNSWIndex)

	if err := idx.Add(ctx, 7, mustVec(dim, 7)); err != nil {
		t.Fatalf("seed Add: %v", err)
	}

	// Duplicate of the live ID 7 plus a brand-new ID in the same batch.
	vectors := map[uint64][]float32{
		7:  mustVec(dim, 7),
		42: mustVec(dim, 42),
	}
	if err := hnswIdx.BatchAdd(ctx, vectors); err == nil {
		t.Fatal("BatchAdd with a duplicate live ID: expected error, got nil")
	}

	if got := idx.Stats().Count; got != 1 {
		t.Errorf("stats.Count = %d after duplicate-rejected batch, want 1", got)
	}
	if err := idx.Add(ctx, 42, mustVec(dim, 42)); err != nil {
		t.Fatalf("re-add new ID 42 after rejected batch: %v", err)
	}
	if got := idx.Stats().Count; got != 2 {
		t.Errorf("stats.Count = %d after re-add, want 2", got)
	}
}
