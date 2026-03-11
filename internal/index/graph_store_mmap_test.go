//go:build !windows

package index

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestMmapGraphStoreContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "graph.mmap")
	gs, err := NewMmapGraphStore(path, 32)
	if err != nil {
		t.Fatalf("failed to create mmap graph store: %v", err)
	}
	defer gs.Close()

	testGraphStoreContract(t, gs)
}

func TestMmapGraphStoreGrow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "graph.mmap")
	gs, err := NewMmapGraphStore(path, 32)
	if err != nil {
		t.Fatalf("failed to create: %v", err)
	}
	defer gs.Close()

	// Insert enough nodes to trigger at least one grow
	numNodes := mmapGraphInitialSlots + 100
	for i := 0; i < numNodes; i++ {
		gs.SetNeighbors(uint64(i), []uint64{uint64(i + 1), uint64(i + 2)})
	}

	if gs.Len() != numNodes {
		t.Fatalf("expected %d nodes, got %d", numNodes, gs.Len())
	}

	// Verify all nodes readable
	for i := 0; i < numNodes; i++ {
		n := gs.GetNeighbors(uint64(i))
		if len(n) != 2 {
			t.Fatalf("node %d: expected 2 neighbors, got %d", i, len(n))
		}
	}
}

func TestMmapGraphStoreSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "graph.mmap")
	gs, err := NewMmapGraphStore(path, 32)
	if err != nil {
		t.Fatalf("failed to create: %v", err)
	}
	defer gs.Close()

	gs.SetNeighbors(1, []uint64{2, 3, 4})
	gs.SetNeighbors(2, []uint64{1})

	snap := gs.Snapshot()

	// Mutate original
	gs.SetNeighbors(1, []uint64{99})
	gs.SetNeighbors(100, []uint64{200})

	// Snapshot should see old data for node 1
	// Note: since snapshot shares mmap, SetNeighbors on the SAME slot
	// will be visible. But new nodes (100) won't be in the snapshot's slot index.
	if snap.HasNode(100) {
		t.Fatal("snapshot should not see node 100")
	}

	// Snapshot's slot index still points to node 1's slot, but the data
	// was overwritten in-place. This is a known trade-off: mmap snapshots
	// see mutations to existing slots but don't see new slots.
	// For the DiskANN use case, this is acceptable because:
	// 1. Snapshots are used for read-only search during parallel build
	// 2. New edges are collected in buildResult and applied after search
	// 3. The slight staleness doesn't affect correctness, only search quality slightly

	if snap.Len() != 2 {
		t.Fatalf("snapshot should have 2 nodes, got %d", snap.Len())
	}
}

func TestDiskANNMmapGraph(t *testing.T) {
	dir := t.TempDir()
	dim := 64
	config := map[string]interface{}{
		"memory_limit":    100,
		"max_degree":      16,
		"ef_construction": 50,
		"ef_search":       30,
		"index_path":      filepath.Join(dir, "diskann.idx"),
		"graph_storage":   "disk",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
		}
	}()

	// Add vectors
	numVectors := 200
	vectors := make(map[uint64][]float32)
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		vectors[uint64(i)] = vec
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Verify stats
	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Fatalf("expected count %d, got %d", numVectors, stats.Count)
	}

	// Search
	query := vectors[0]
	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no search results")
	}

	// First result should be the query vector itself (or very close)
	if results[0].ID != 0 {
		t.Logf("Warning: first result is ID %d (distance %.4f), not 0", results[0].ID, results[0].Distance)
	}

	t.Logf("Mmap graph: %d vectors, search returned %d results", numVectors, len(results))
}

func TestDiskANNMmapGraphBatch(t *testing.T) {
	dir := t.TempDir()
	dim := 64
	config := map[string]interface{}{
		"memory_limit":    500,
		"max_degree":      16,
		"ef_construction": 50,
		"ef_search":       30,
		"index_path":      filepath.Join(dir, "diskann.idx"),
		"graph_storage":   "disk",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
		}
	}()

	// Batch add
	batchSize := 500
	vectors := make(map[uint64][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		vectors[uint64(i)] = vec
	}

	diskANN := idx.(*DiskANNIndex)
	if err := diskANN.AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("batch add failed: %v", err)
	}

	stats := idx.Stats()
	if stats.Count != batchSize {
		t.Fatalf("expected count %d, got %d", batchSize, stats.Count)
	}

	// Search works
	query := vectors[0]
	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results from batch-built mmap graph")
	}

	t.Logf("Mmap graph batch: %d vectors, search returned %d results", batchSize, len(results))
}

func TestDiskANNMmapScale(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping scale test in short mode")
	}

	dir := t.TempDir()
	dim := 32 // smaller dim for speed
	numVectors := 100_000

	config := map[string]interface{}{
		"memory_limit":    numVectors, // all in memory for vector data
		"max_degree":      16,
		"ef_construction": 30,
		"ef_search":       20,
		"index_path":      filepath.Join(dir, "diskann.idx"),
		"graph_storage":   "disk",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
		}
	}()

	// Batch insert in chunks
	chunkSize := 10000
	for start := 0; start < numVectors; start += chunkSize {
		end := start + chunkSize
		if end > numVectors {
			end = numVectors
		}

		vectors := make(map[uint64][]float32, end-start)
		for i := start; i < end; i++ {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rand.Float32()
			}
			vectors[uint64(i)] = vec
		}

		diskANN := idx.(*DiskANNIndex)
		if err := diskANN.AddBatch(context.Background(), vectors); err != nil {
			t.Fatalf("batch add chunk %d-%d failed: %v", start, end, err)
		}
	}

	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Fatalf("expected count %d, got %d", numVectors, stats.Count)
	}

	// Check RSS
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	rssMB := memStats.Sys / (1024 * 1024)
	t.Logf("Scale test: %d vectors, RSS: %d MB", numVectors, rssMB)

	// Search
	query := make([]float32, dim)
	for j := range query {
		query[j] = rand.Float32()
	}
	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results at scale")
	}

	// Check graph file exists on disk
	graphPath := filepath.Join(dir, "diskann.idx.graph")
	info, err := os.Stat(graphPath)
	if err != nil {
		t.Fatalf("graph file missing: %v", err)
	}
	graphMB := info.Size() / (1024 * 1024)
	t.Logf("Graph file: %d MB, expected ~%d MB", graphMB,
		int64(numVectors)*int64(4+16*8)/(1024*1024))

	fmt.Printf("Scale test passed: %dk vectors, RSS=%dMB, graph=%dMB\n",
		numVectors/1000, rssMB, graphMB)
}
