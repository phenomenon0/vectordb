package index

import (
	"context"
	"testing"
)

// TestFLATDuplicateIDNoCountInflation verifies that re-adding an existing ID
// replaces the vector without inflating the index count, and that a tombstoned
// ID can be resurrected without counting it twice.
func TestFLATDuplicateIDNoCountInflation(t *testing.T) {
	ctx := context.Background()
	idx, err := NewFLATIndex(4, nil)
	if err != nil {
		t.Fatalf("NewFLATIndex: %v", err)
	}
	flat := idx.(*FLATIndex)

	with := func(vals ...float32) []float32 { return append([]float32{}, vals...) }

	if err := idx.Add(ctx, 1, with(1, 0, 0, 0)); err != nil {
		t.Fatalf("first Add: %v", err)
	}
	if got := flat.Stats().Count; got != 1 {
		t.Fatalf("Count after first Add = %d, want 1", got)
	}

	// Overwrite the same ID: count must not grow.
	if err := idx.Add(ctx, 1, with(0, 1, 0, 0)); err != nil {
		t.Fatalf("duplicate Add: %v", err)
	}
	if got := flat.Stats().Count; got != 1 {
		t.Fatalf("Count after duplicate Add = %d, want 1 (inflation)", got)
	}

	// Tombstone then resurrect: count must stay 1.
	if err := idx.Delete(ctx, 1); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if err := idx.Add(ctx, 1, with(0, 0, 1, 0)); err != nil {
		t.Fatalf("resurrect Add: %v", err)
	}
	if got := flat.Stats().Count; got != 1 {
		t.Fatalf("Count after tombstone resurrection = %d, want 1", got)
	}

	// A genuinely new ID counts exactly once.
	if err := idx.Add(ctx, 2, with(0, 0, 0, 1)); err != nil {
		t.Fatalf("new Add: %v", err)
	}
	if got := flat.Stats().Count; got != 2 {
		t.Fatalf("Count after new Add = %d, want 2", got)
	}
}
