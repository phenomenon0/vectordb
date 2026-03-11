package index

import (
	"testing"
)

// testGraphStoreContract runs the common contract tests against any GraphStore implementation.
func testGraphStoreContract(t *testing.T, gs GraphStore) {
	t.Helper()

	// Empty store
	if gs.Len() != 0 {
		t.Fatalf("expected empty store, got len=%d", gs.Len())
	}
	if gs.HasNode(1) {
		t.Fatal("HasNode should return false for empty store")
	}
	if n := gs.GetNeighbors(1); n != nil {
		t.Fatalf("GetNeighbors on missing node should return nil, got %v", n)
	}

	// SetNeighbors + GetNeighbors
	gs.SetNeighbors(1, []uint64{2, 3, 4})
	gs.SetNeighbors(2, []uint64{1, 5})
	gs.SetNeighbors(3, []uint64{1})

	if gs.Len() != 3 {
		t.Fatalf("expected len=3, got %d", gs.Len())
	}
	if !gs.HasNode(1) {
		t.Fatal("HasNode(1) should be true")
	}

	neighbors := gs.GetNeighbors(1)
	if len(neighbors) != 3 || neighbors[0] != 2 || neighbors[1] != 3 || neighbors[2] != 4 {
		t.Fatalf("unexpected neighbors for node 1: %v", neighbors)
	}

	// Overwrite
	gs.SetNeighbors(1, []uint64{10, 20})
	neighbors = gs.GetNeighbors(1)
	if len(neighbors) != 2 || neighbors[0] != 10 || neighbors[1] != 20 {
		t.Fatalf("overwrite failed, got %v", neighbors)
	}

	// DeleteNode
	gs.DeleteNode(2)
	if gs.HasNode(2) {
		t.Fatal("node 2 should be deleted")
	}
	if gs.Len() != 2 {
		t.Fatalf("expected len=2 after delete, got %d", gs.Len())
	}

	// Range
	visited := make(map[uint64]bool)
	gs.Range(func(id uint64, _ []uint64) bool {
		visited[id] = true
		return true
	})
	if !visited[1] || !visited[3] || visited[2] {
		t.Fatalf("Range unexpected: %v", visited)
	}

	// Range early exit
	count := 0
	gs.Range(func(_ uint64, _ []uint64) bool {
		count++
		return false // stop after first
	})
	if count != 1 {
		t.Fatalf("Range early exit: expected 1 iteration, got %d", count)
	}

	// Snapshot: new nodes added after snapshot should not be visible
	gs.SetNeighbors(100, []uint64{200, 300})
	snap := gs.Snapshot()
	gs.SetNeighbors(999, []uint64{1}) // add new node after snapshot

	if snap.HasNode(999) {
		t.Fatal("snapshot should not see node 999 added after snapshot")
	}
	// Snapshot should see node 100
	snapNeighbors := snap.GetNeighbors(100)
	if snapNeighbors == nil {
		t.Fatal("snapshot should see node 100")
	}

	// Clone
	gs.SetNeighbors(50, []uint64{51, 52})
	cloned := gs.Clone()
	gs.SetNeighbors(50, []uint64{99})

	if len(cloned[50]) != 2 || cloned[50][0] != 51 {
		t.Fatalf("Clone should be independent, got %v", cloned[50])
	}

	// ReplaceAll
	gs.ReplaceAll(map[uint64][]uint64{
		10: {11, 12},
		20: {21},
	})
	if gs.Len() != 2 {
		t.Fatalf("ReplaceAll: expected len=2, got %d", gs.Len())
	}
	if !gs.HasNode(10) || !gs.HasNode(20) {
		t.Fatal("ReplaceAll: missing expected nodes")
	}
	if gs.HasNode(1) || gs.HasNode(3) {
		t.Fatal("ReplaceAll: old nodes should be gone")
	}
}

func TestMemoryGraphStore(t *testing.T) {
	testGraphStoreContract(t, NewMemoryGraphStore())
}

func TestMemoryGraphStoreFrom(t *testing.T) {
	initial := map[uint64][]uint64{
		1: {2, 3},
		2: {1},
	}
	gs := NewMemoryGraphStoreFrom(initial)
	if gs.Len() != 2 {
		t.Fatalf("expected len=2, got %d", gs.Len())
	}
	n := gs.GetNeighbors(1)
	if len(n) != 2 {
		t.Fatalf("expected 2 neighbors for node 1, got %d", len(n))
	}
}

func TestFirstNodeID(t *testing.T) {
	gs := NewMemoryGraphStore()
	gs.SetNeighbors(5, []uint64{6})
	gs.SetNeighbors(10, []uint64{11})

	deleted := map[uint64]bool{5: true}
	id, found := FirstNodeID(gs, deleted)
	if !found {
		t.Fatal("expected to find a node")
	}
	if id == 5 {
		t.Fatal("should skip deleted node 5")
	}
	if id != 10 {
		t.Fatalf("expected node 10, got %d", id)
	}

	// All deleted
	deleted[10] = true
	_, found = FirstNodeID(gs, deleted)
	if found {
		t.Fatal("expected no node found when all are deleted")
	}

	// Nil deleted map
	id, found = FirstNodeID(gs, nil)
	if !found {
		t.Fatal("expected to find a node with nil deleted map")
	}
}
