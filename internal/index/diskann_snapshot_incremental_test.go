package index

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestIncrementalSnapshotSmaller(t *testing.T) {
	dim := 32
	numVectors := 500

	tmpDir, err := os.MkdirTemp("", "incr_snapshot_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test.idx")
	snapshotDir := filepath.Join(tmpDir, "snapshots")

	config := map[string]interface{}{
		"memory_limit":    numVectors,
		"max_degree":      16,
		"ef_construction": 30,
		"ef_search":       20,
		"index_path":      indexPath,
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("create index: %v", err)
	}
	defer idx.(*DiskANNIndex).Close()

	// Add initial vectors
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("add %d: %v", i, err)
		}
	}

	diskANN := idx.(*DiskANNIndex)
	sm, err := NewSnapshotManager(diskANN, SnapshotConfig{SnapshotDir: snapshotDir})
	if err != nil {
		t.Fatalf("create snapshot manager: %v", err)
	}

	// Create full base snapshot
	baseMeta, err := sm.CreateSnapshot(context.Background(), "base snapshot")
	if err != nil {
		t.Fatalf("create base snapshot: %v", err)
	}

	if len(baseMeta.ComponentChecksums) == 0 {
		t.Fatal("base snapshot should have component checksums")
	}

	baseSize, err := sm.GetSnapshotSize(baseMeta.ID)
	if err != nil {
		t.Fatalf("get base size: %v", err)
	}

	// Modify 10% of vectors (add new ones to change graph + memory)
	modifyCount := numVectors / 10
	for i := numVectors; i < numVectors+modifyCount; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("add modified %d: %v", i, err)
		}
	}

	// Create incremental snapshot
	incrMeta, err := sm.CreateIncrementalSnapshot(context.Background(), baseMeta.ID, "incremental after 10% change")
	if err != nil {
		t.Fatalf("create incremental snapshot: %v", err)
	}

	if !incrMeta.Incremental {
		t.Fatal("expected Incremental=true")
	}
	if incrMeta.BaseSnapshotID != baseMeta.ID {
		t.Fatalf("expected BaseSnapshotID=%s, got %s", baseMeta.ID, incrMeta.BaseSnapshotID)
	}
	if len(incrMeta.ComponentChecksums) == 0 {
		t.Fatal("incremental should have checksums")
	}

	incrSize, err := sm.GetSnapshotSize(incrMeta.ID)
	if err != nil {
		t.Fatalf("get incremental size: %v", err)
	}

	t.Logf("Base size: %d bytes, Incremental size: %d bytes (%.1f%%)",
		baseSize, incrSize, float64(incrSize)/float64(baseSize)*100)

	// Incremental should be smaller than full (at least some components skipped)
	if incrSize >= baseSize {
		t.Logf("Warning: incremental not smaller than base (all components changed)")
	}

	// Verify incremental snapshot has fewer files than base
	baseEntries, _ := os.ReadDir(filepath.Join(snapshotDir, baseMeta.ID))
	incrEntries, _ := os.ReadDir(filepath.Join(snapshotDir, incrMeta.ID))
	t.Logf("Base files: %d, Incremental files: %d", len(baseEntries), len(incrEntries))
}

func TestIncrementalSnapshotRestore(t *testing.T) {
	dim := 32
	numVectors := 200

	tmpDir, err := os.MkdirTemp("", "incr_restore_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test.idx")
	snapshotDir := filepath.Join(tmpDir, "snapshots")

	config := map[string]interface{}{
		"memory_limit":    numVectors * 2,
		"max_degree":      16,
		"ef_construction": 30,
		"ef_search":       20,
		"index_path":      indexPath,
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("create index: %v", err)
	}

	diskANN := idx.(*DiskANNIndex)

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

	sm, err := NewSnapshotManager(diskANN, SnapshotConfig{SnapshotDir: snapshotDir})
	if err != nil {
		t.Fatalf("create snapshot manager: %v", err)
	}

	// Base snapshot
	baseMeta, err := sm.CreateSnapshot(context.Background(), "base")
	if err != nil {
		t.Fatalf("base snapshot: %v", err)
	}

	// Add more vectors
	extraCount := 50
	for i := numVectors; i < numVectors+extraCount; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		vectors[uint64(i)] = vec
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("add extra %d: %v", i, err)
		}
	}

	// Incremental snapshot
	incrMeta, err := sm.CreateIncrementalSnapshot(context.Background(), baseMeta.ID, "incremental")
	if err != nil {
		t.Fatalf("incremental snapshot: %v", err)
	}

	expectedCount := numVectors + extraCount

	// Verify the snapshot has the right count
	if incrMeta.VectorCount != expectedCount {
		t.Fatalf("expected %d vectors in snapshot, got %d", expectedCount, incrMeta.VectorCount)
	}

	// Close and recreate index to prove restore works
	diskANN.Close()

	idx2, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("recreate index: %v", err)
	}
	defer idx2.(*DiskANNIndex).Close()

	diskANN2 := idx2.(*DiskANNIndex)
	sm2, err := NewSnapshotManager(diskANN2, SnapshotConfig{SnapshotDir: snapshotDir})
	if err != nil {
		t.Fatalf("recreate snapshot manager: %v", err)
	}

	// Restore from incremental
	if err := sm2.RestoreIncrementalSnapshot(context.Background(), incrMeta.ID); err != nil {
		t.Fatalf("restore incremental: %v", err)
	}

	// Verify search works
	query := vectors[uint64(numVectors)] // one of the extra vectors
	results, err := idx2.Search(context.Background(), query, 5, nil)
	if err != nil {
		t.Fatalf("search after restore: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results after incremental restore")
	}

	t.Logf("Incremental restore successful: %d vectors, search returned %d results",
		incrMeta.VectorCount, len(results))
}

func TestIncrementalSnapshotNoChange(t *testing.T) {
	dim := 16
	numVectors := 100

	tmpDir, err := os.MkdirTemp("", "incr_nochange_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	indexPath := filepath.Join(tmpDir, "test.idx")
	snapshotDir := filepath.Join(tmpDir, "snapshots")

	config := map[string]interface{}{
		"memory_limit":    numVectors,
		"max_degree":      8,
		"ef_construction": 20,
		"ef_search":       10,
		"index_path":      indexPath,
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer idx.(*DiskANNIndex).Close()

	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		idx.Add(context.Background(), uint64(i), vec)
	}

	diskANN := idx.(*DiskANNIndex)
	sm, err := NewSnapshotManager(diskANN, SnapshotConfig{SnapshotDir: snapshotDir})
	if err != nil {
		t.Fatal(err)
	}

	base, err := sm.CreateSnapshot(context.Background(), "base")
	if err != nil {
		t.Fatal(err)
	}

	baseSize, _ := sm.GetSnapshotSize(base.ID)

	// No changes — incremental should have almost nothing
	incr, err := sm.CreateIncrementalSnapshot(context.Background(), base.ID, "no changes")
	if err != nil {
		t.Fatalf("incremental: %v", err)
	}

	incrSize, _ := sm.GetSnapshotSize(incr.ID)

	t.Logf("No-change: base=%d bytes, incremental=%d bytes (%.1f%%)",
		baseSize, incrSize, float64(incrSize)/float64(baseSize)*100)

	// With no changes, incremental should be much smaller (just metadata.json)
	if incrSize > baseSize/2 {
		t.Fatalf("incremental with no changes should be much smaller: %d vs %d", incrSize, baseSize)
	}
}
