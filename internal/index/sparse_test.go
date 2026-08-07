package index

import (
	"context"
	"testing"
)

// TestSparseIndexOverwriteRecomputesNorm verifies that re-adding an existing ID
// (an overwrite) recomputes the stored cosine norm from the new vector. The old
// implementation folded norm accumulation into the posting loop, so an
// overwrite with an all-zero vector left the previous norm in the map.
func TestSparseIndexOverwriteRecomputesNorm(t *testing.T) {
	ctx := context.Background()
	idx, err := NewSparseIndex(4, map[string]interface{}{"metric": "cosine"})
	if err != nil {
		t.Fatalf("NewSparseIndex: %v", err)
	}
	s := idx.(*SparseIndex)

	if err := idx.Add(ctx, 1, []float32{1, 0, 0, 0}); err != nil {
		t.Fatalf("initial Add: %v", err)
	}

	// Overwrite with an all-zero vector: the empty sparse vector must yield a
	// zero norm, not the previously stored norm of 1.
	if err := idx.Add(ctx, 1, []float32{0, 0, 0, 0}); err != nil {
		t.Fatalf("all-zero overwrite: %v", err)
	}
	s.mu.RLock()
	got := s.norms[1]
	s.mu.RUnlock()
	if got != 0 {
		t.Errorf("norm after all-zero overwrite = %v, want 0 (stale value survived)", got)
	}

	// Overwrite again with a 3-4-5 right triangle: norm must be 5, not 0 or 1.
	if err := idx.Add(ctx, 1, []float32{0, 3, 0, 4}); err != nil {
		t.Fatalf("norm overwrite: %v", err)
	}
	s.mu.RLock()
	got = s.norms[1]
	s.mu.RUnlock()
	if got != 5 {
		t.Errorf("norm after 3/4 overwrite = %v, want 5", got)
	}

	if got := idx.Stats().Count; got != 1 {
		t.Errorf("Stats().Count = %d after overwrites, want 1 (no count inflation)", got)
	}
}

// TestSparseIndexOverwriteUpdatesPostings verifies an overwrite replaces the
// old inverted-list membership rather than retaining stale postings for terms
// that are no longer present.
func TestSparseIndexOverwriteUpdatesPostings(t *testing.T) {
	ctx := context.Background()
	idx, err := NewSparseIndex(4, map[string]interface{}{"metric": "dot"})
	if err != nil {
		t.Fatalf("NewSparseIndex: %v", err)
	}
	s := idx.(*SparseIndex)

	if err := idx.Add(ctx, 1, []float32{1, 1, 1, 1}); err != nil {
		t.Fatalf("initial Add: %v", err)
	}
	if err := idx.Add(ctx, 1, []float32{0, 0, 0, 2}); err != nil {
		t.Fatalf("overwrite Add: %v", err)
	}

	s.mu.RLock()
	for term, ids := range s.inverted {
		if term != 3 {
			t.Errorf("term %d still has postings after overwrite dropped it", term)
		}
		if len(ids) != 1 || ids[0] != 1 {
			t.Errorf("term %d postings = %v, want [1]", term, ids)
		}
	}
	s.mu.RUnlock()
}
