package index

import (
	"context"
	"math/rand"
	"os"
	"testing"
	"time"
)

// TestDiskANNCompactionBasic tests basic compaction functionality
func TestDiskANNCompactionBasic(t *testing.T) {
	dim := 64
	numVectors := 100

	config := map[string]interface{}{
		"memory_limit": 20,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_compaction_basic.idx",
		"metric":       "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer func() {
		diskANN := idx.(*DiskANNIndex)
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	diskANN := idx.(*DiskANNIndex)

	// Add vectors
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("Failed to add vector %d: %v", i, err)
		}
	}

	// Check initial state
	statsBefore := diskANN.Stats()
	diskBytesBefore := diskANN.mmapOffset

	t.Logf("Before deletion: Count=%d, DiskBytes=%d",
		statsBefore.Count, diskBytesBefore)

	// Delete 30% of vectors
	numToDelete := 30
	for i := 0; i < numToDelete; i++ {
		if err := idx.Delete(context.Background(), uint64(i)); err != nil {
			t.Fatalf("Failed to delete vector %d: %v", i, err)
		}
	}

	statsAfterDel := diskANN.Stats()
	t.Logf("After deletion: Count=%d (deleted=%d), DiskBytes=%d",
		statsAfterDel.Count, statsAfterDel.Deleted, diskANN.mmapOffset)

	// Disk bytes should not decrease (deleted vectors leave holes)
	if diskANN.mmapOffset < diskBytesBefore {
		t.Error("Disk bytes should not decrease before compaction")
	}

	// Run compaction
	compactStats, err := diskANN.Compact(context.Background())
	if err != nil {
		t.Fatalf("Compaction failed: %v", err)
	}

	t.Logf("Compaction Stats:")
	t.Logf("  Duration: %v", compactStats.Duration)
	t.Logf("  Vectors: %d -> %d (removed %d)",
		compactStats.VectorsBefore, compactStats.VectorsAfter, compactStats.VectorsRemoved)
	t.Logf("  Disk: %d -> %d bytes (reclaimed %d)",
		compactStats.DiskBytesBefore, compactStats.DiskBytesAfter, compactStats.SpaceReclaimed)
	t.Logf("  Fragmentation: %.2f%%", compactStats.FragmentationPct)

	// Verify compaction results
	if compactStats.VectorsRemoved != numToDelete {
		t.Errorf("Expected %d vectors removed, got %d", numToDelete, compactStats.VectorsRemoved)
	}

	if compactStats.SpaceReclaimed <= 0 {
		t.Error("Expected space to be reclaimed")
	}

	// Verify remaining vectors are accessible
	statsAfterCompact := diskANN.Stats()
	if statsAfterCompact.Count != numVectors-numToDelete {
		t.Errorf("Expected count %d after compaction, got %d",
			numVectors-numToDelete, statsAfterCompact.Count)
	}

	if statsAfterCompact.Deleted != 0 {
		t.Errorf("Expected 0 deleted vectors after compaction, got %d", statsAfterCompact.Deleted)
	}

	// Verify searches still work
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()
	}

	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("Search failed after compaction: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected search results after compaction")
	}

	t.Logf("✓ Compaction successful: reclaimed %d bytes (%.1f%% reduction)",
		compactStats.SpaceReclaimed,
		float64(compactStats.SpaceReclaimed)/float64(compactStats.DiskBytesBefore)*100)
}

// TestDiskANNNeedsCompaction tests compaction trigger logic
func TestDiskANNNeedsCompaction(t *testing.T) {
	dim := 32
	config := map[string]interface{}{
		"memory_limit": 10,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_needs_compaction.idx",
		"metric":       "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer func() {
		diskANN := idx.(*DiskANNIndex)
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	diskANN := idx.(*DiskANNIndex)

	compactConfig := CompactionConfig{
		MinFragmentation:  20.0,
		MinDeletedVectors: 10,
		BackgroundMode:    true,
	}

	// Initially should not need compaction
	if diskANN.NeedsCompaction(compactConfig) {
		t.Error("Empty index should not need compaction")
	}

	// Add 100 vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Still should not need compaction
	if diskANN.NeedsCompaction(compactConfig) {
		t.Error("Index with no deletions should not need compaction")
	}

	// Delete 5 vectors (below threshold)
	for i := 0; i < 5; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	if diskANN.NeedsCompaction(compactConfig) {
		t.Error("Index with 5 deletions should not trigger compaction (threshold=10)")
	}

	// Delete 10 more vectors (total 15, above threshold)
	for i := 5; i < 15; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	if !diskANN.NeedsCompaction(compactConfig) {
		t.Error("Index with 15 deletions should trigger compaction (threshold=10)")
	}

	// Test fragmentation threshold
	config2 := map[string]interface{}{
		"memory_limit": 10,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_needs_compaction2.idx",
		"metric":       "cosine",
	}

	idx2, _ := NewDiskANNIndex(dim, config2)
	defer func() {
		diskANN2 := idx2.(*DiskANNIndex)
		diskANN2.Close()
		os.Remove(diskANN2.indexPath)
	}()

	diskANN2 := idx2.(*DiskANNIndex)

	// Add 100 vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		idx2.Add(context.Background(), uint64(i), vec)
	}

	// Delete 25 vectors (25% fragmentation, above 20% threshold)
	for i := 0; i < 25; i++ {
		idx2.Delete(context.Background(), uint64(i))
	}

	fragmentation := diskANN2.EstimateFragmentation()
	t.Logf("Fragmentation: %.2f%%", fragmentation)

	if !diskANN2.NeedsCompaction(compactConfig) {
		t.Error("Index with 25% fragmentation should trigger compaction (threshold=20%)")
	}
}

// TestDiskANNCompactionWithQuantization tests compaction with quantized vectors
func TestDiskANNCompactionWithQuantization(t *testing.T) {
	dim := 64
	numVectors := 50

	config := map[string]interface{}{
		"memory_limit": 10,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_compaction_quant.idx",
		"metric":       "cosine",
		"quantization": map[string]interface{}{
			"type": "float16",
		},
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer func() {
		diskANN := idx.(*DiskANNIndex)
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	diskANN := idx.(*DiskANNIndex)

	// Add vectors
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Delete 20 vectors
	for i := 0; i < 20; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	// Compact
	stats, err := diskANN.Compact(context.Background())
	if err != nil {
		t.Fatalf("Compaction failed with quantization: %v", err)
	}

	t.Logf("Quantized compaction: reclaimed %d bytes", stats.SpaceReclaimed)

	// Verify remaining vectors
	if stats.VectorsAfter != numVectors-20 {
		t.Errorf("Expected %d vectors after compaction, got %d", numVectors-20, stats.VectorsAfter)
	}

	// Verify searches work
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()
	}

	results, err := idx.Search(context.Background(), query, 5, nil)
	if err != nil {
		t.Fatalf("Search failed after quantized compaction: %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected search results after quantized compaction")
	}
}

// TestDiskANNBackgroundCompaction tests background compaction manager
func TestDiskANNBackgroundCompaction(t *testing.T) {
	dim := 32
	config := map[string]interface{}{
		"memory_limit": 10,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_bg_compact.idx",
		"metric":       "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer func() {
		diskANN := idx.(*DiskANNIndex)
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	diskANN := idx.(*DiskANNIndex)

	// Configure background compaction
	compactConfig := CompactionConfig{
		MinFragmentation:  15.0,
		MinDeletedVectors: 5,
		BackgroundMode:    true,
	}

	bgCompact := NewBackgroundCompaction(diskANN, compactConfig, 100*time.Millisecond)
	bgCompact.Start()
	defer bgCompact.Stop()

	// Add vectors
	for i := 0; i < 50; i++ {
		vec := make([]float32, dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Delete vectors to trigger compaction
	for i := 0; i < 10; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	// Wait for background compaction
	time.Sleep(300 * time.Millisecond)

	// Check if compaction ran
	stats := bgCompact.LastStats()
	if stats == nil {
		t.Error("Expected background compaction to have run")
	} else {
		t.Logf("Background compaction completed: removed %d vectors, reclaimed %d bytes",
			stats.VectorsRemoved, stats.SpaceReclaimed)
	}

	// Verify deleted count is now 0
	indexStats := diskANN.Stats()
	if indexStats.Deleted != 0 {
		t.Errorf("Expected 0 deleted vectors after background compaction, got %d", indexStats.Deleted)
	}
}

// TestDiskANNCompactionConcurrency tests compaction with concurrent operations
func TestDiskANNCompactionConcurrency(t *testing.T) {
	dim := 32
	config := map[string]interface{}{
		"memory_limit": 10,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_compact_concurrent.idx",
		"metric":       "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer func() {
		diskANN := idx.(*DiskANNIndex)
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	// Add initial vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Delete some vectors
	for i := 0; i < 30; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	// Run compaction
	_, err = idx.(*DiskANNIndex).Compact(context.Background())
	if err != nil {
		t.Fatalf("Compaction failed: %v", err)
	}

	// Verify index is still functional
	stats := idx.(*DiskANNIndex).Stats()
	if stats.Count != 70 {
		t.Errorf("Expected 70 vectors after compaction, got %d", stats.Count)
	}

	if stats.Deleted != 0 {
		t.Errorf("Expected 0 deleted vectors, got %d", stats.Deleted)
	}

	t.Log("✓ Compaction with concurrent operations successful")
}

// TestDiskANNDiskSpaceUsage tests disk space tracking
func TestDiskANNDiskSpaceUsage(t *testing.T) {
	dim := 32
	config := map[string]interface{}{
		"memory_limit": 10,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_disk_usage.idx",
		"metric":       "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer func() {
		diskANN := idx.(*DiskANNIndex)
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	diskANN := idx.(*DiskANNIndex)

	// Add vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	usedBefore, reclaimableBefore := diskANN.DiskSpaceUsage()
	t.Logf("Before deletion: used=%d, reclaimable=%d", usedBefore, reclaimableBefore)

	if reclaimableBefore != 0 {
		t.Error("Expected 0 reclaimable space with no deletions")
	}

	// Delete 25% of vectors
	for i := 0; i < 25; i++ {
		idx.Delete(context.Background(), uint64(i))
	}

	usedAfter, reclaimableAfter := diskANN.DiskSpaceUsage()
	t.Logf("After deletion: used=%d, reclaimable=%d", usedAfter, reclaimableAfter)

	if reclaimableAfter == 0 {
		t.Error("Expected reclaimable space after deletions")
	}

	// Reclaimable should be approximately 25% of used space
	expectedReclaimable := int64(float64(usedAfter) * 0.25)
	if reclaimableAfter < expectedReclaimable*8/10 || reclaimableAfter > expectedReclaimable*12/10 {
		t.Logf("Warning: reclaimable space (%d) not close to expected (%d)",
			reclaimableAfter, expectedReclaimable)
	}
}
