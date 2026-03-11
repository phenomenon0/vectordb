package index

import (
	"context"
	"math/rand"
	"path/filepath"
	"testing"
)

func TestParallelCompaction(t *testing.T) {
	dim := 64
	numVectors := 1000
	deleteRatio := 0.2 // delete 20%

	indexPath := filepath.Join(t.TempDir(), "test_parallel_compact.idx")
	config := map[string]interface{}{
		"memory_limit":    numVectors,
		"max_degree":      16,
		"ef_construction": 50,
		"ef_search":       30,
		"index_path":      indexPath,
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.(*DiskANNIndex).Close()

	// Add vectors
	vectors := make(map[uint64][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		vectors[uint64(i)] = vec
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("add %d: %v", i, err)
		}
	}

	// Delete some vectors
	numToDelete := int(float64(numVectors) * deleteRatio)
	for i := 0; i < numToDelete; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	// Compact (uses parallel reads now)
	diskANN := idx.(*DiskANNIndex)
	stats, err := diskANN.Compact(context.Background())
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	expectedAfter := numVectors - numToDelete
	if stats.VectorsAfter != expectedAfter {
		t.Fatalf("expected %d vectors after compaction, got %d", expectedAfter, stats.VectorsAfter)
	}
	if stats.VectorsRemoved != numToDelete {
		t.Fatalf("expected %d removed, got %d", numToDelete, stats.VectorsRemoved)
	}

	// Verify search still works on non-deleted vectors
	query := vectors[uint64(numToDelete)] // first non-deleted
	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("search after compaction failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no search results after compaction")
	}

	// Verify deleted vectors are gone
	for i := 0; i < numToDelete; i++ {
		for _, r := range results {
			if r.ID == uint64(i) {
				t.Fatalf("deleted vector %d appeared in results", i)
			}
		}
	}

	t.Logf("Parallel compaction: %d -> %d vectors, reclaimed %d bytes in %v",
		stats.VectorsBefore, stats.VectorsAfter, stats.SpaceReclaimed, stats.Duration)
}

func TestParallelCompactionLarger(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large compaction test")
	}

	dim := 32
	numVectors := 10000
	deleteRatio := 0.3

	indexPath := filepath.Join(t.TempDir(), "test_parallel_compact_large.idx")
	config := map[string]interface{}{
		"memory_limit":    numVectors,
		"max_degree":      16,
		"ef_construction": 30,
		"ef_search":       20,
		"index_path":      indexPath,
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer idx.(*DiskANNIndex).Close()

	// Batch add
	diskANN := idx.(*DiskANNIndex)
	vectors := make(map[uint64][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		vectors[uint64(i)] = vec
	}
	if err := diskANN.AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("batch add: %v", err)
	}

	// Delete
	numToDelete := int(float64(numVectors) * deleteRatio)
	for i := 0; i < numToDelete; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	// Compact
	stats, err := diskANN.Compact(context.Background())
	if err != nil {
		t.Fatalf("compact: %v", err)
	}

	expected := numVectors - numToDelete
	if stats.VectorsAfter != expected {
		t.Fatalf("expected %d, got %d", expected, stats.VectorsAfter)
	}

	t.Logf("Large parallel compaction: %d -> %d, reclaimed %d bytes in %v",
		stats.VectorsBefore, stats.VectorsAfter, stats.SpaceReclaimed, stats.Duration)
}
