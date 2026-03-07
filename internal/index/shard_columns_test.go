package index

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestColumnarExport_HNSW(t *testing.T) {
	idx, err := NewHNSWIndex(4, map[string]interface{}{
		"m": 8, "ef_construction": 100, "metric": "cosine",
	})
	if err != nil {
		t.Fatalf("create HNSW: %v", err)
	}

	// Add vectors
	ctx := context.Background()
	for i := uint64(1); i <= 20; i++ {
		vec := []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3, float32(i) * 0.4}
		if err := idx.Add(ctx, i, vec); err != nil {
			t.Fatalf("add %d: %v", i, err)
		}
	}
	// Delete some
	idx.Delete(ctx, 5)
	idx.Delete(ctx, 15)

	dir := t.TempDir()
	path := filepath.Join(dir, "test_hnsw.shard")

	// Export columnar
	if err := ExportToShardColumnar(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	// Verify file exists and is non-trivial
	fi, _ := os.Stat(path)
	if fi.Size() < 100 {
		t.Fatalf("shard file too small: %d bytes", fi.Size())
	}

	// Open column reader
	cr, err := OpenColumnReader(path)
	if err != nil {
		t.Fatalf("open column reader: %v", err)
	}
	defer cr.Close()

	// Check meta
	meta := cr.Meta()
	if meta.IndexType != "HNSW" {
		t.Errorf("expected HNSW, got %s", meta.IndexType)
	}
	if meta.Dim != 4 {
		t.Errorf("expected dim 4, got %d", meta.Dim)
	}
	if meta.Version != 2 {
		t.Errorf("expected version 2, got %d", meta.Version)
	}
	if !cr.IsColumnar() {
		t.Error("expected columnar layout")
	}

	// Read config only (no vectors loaded)
	cfg, err := cr.ReadConfig()
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if m, ok := cfg["m"].(float64); !ok || int(m) != 8 {
		t.Errorf("expected m=8, got %v", cfg["m"])
	}

	// Read vectors only
	vectors, err := cr.ReadVectors()
	if err != nil {
		t.Fatalf("read vectors: %v", err)
	}
	if len(vectors) != 20 {
		t.Errorf("expected 20 vectors, got %d", len(vectors))
	}

	// Verify vector precision
	for _, v := range vectors {
		if v.ID == 1 {
			expected := []float32{0.1, 0.2, 0.3, 0.4}
			for j, val := range v.Vector {
				if math.Abs(float64(val-expected[j])) > 1e-6 {
					t.Errorf("vector[%d] precision loss: got %f, want %f", j, val, expected[j])
				}
			}
		}
	}

	// Read deleted only
	deleted, err := cr.ReadDeleted()
	if err != nil {
		t.Fatalf("read deleted: %v", err)
	}
	if len(deleted) != 2 {
		t.Errorf("expected 2 deleted, got %d", len(deleted))
	}
	deletedSet := map[uint64]bool{}
	for _, id := range deleted {
		deletedSet[id] = true
	}
	if !deletedSet[5] || !deletedSet[15] {
		t.Errorf("expected deleted IDs 5 and 15, got %v", deleted)
	}

	// Check columns listing
	cols := cr.Columns()
	if len(cols) < 3 {
		t.Errorf("expected at least 3 columns, got %v", cols)
	}

	// Check entry sizes
	vecSize := cr.EntrySize(ColVectors)
	if vecSize <= 0 {
		t.Errorf("expected positive vector column size, got %d", vecSize)
	}
	t.Logf("vector column disk size: %d bytes (20 vectors, dim=4)", vecSize)
}

func TestColumnarImport_HNSW(t *testing.T) {
	// Create and export
	idx, _ := NewHNSWIndex(4, map[string]interface{}{
		"m": 8, "ef_construction": 100, "metric": "cosine",
	})
	ctx := context.Background()
	for i := uint64(1); i <= 10; i++ {
		vec := []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3, float32(i) * 0.4}
		idx.Add(ctx, i, vec)
	}
	idx.Delete(ctx, 3)

	dir := t.TempDir()
	path := filepath.Join(dir, "import_test.shard")
	if err := ExportToShardColumnar(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	// Import from columnar shard
	factory := func(indexType string, dim int) (Index, error) {
		return NewHNSWIndex(dim, map[string]interface{}{
			"m": 8, "ef_construction": 100, "metric": "cosine",
		})
	}
	imported, err := ImportFromShardColumnar(path, factory)
	if err != nil {
		t.Fatalf("import: %v", err)
	}

	stats := imported.Stats()
	if stats.Count != 10 {
		t.Errorf("expected count 10, got %d", stats.Count)
	}

	// Search should work on the imported index
	results, err := imported.Search(ctx, []float32{0.1, 0.2, 0.3, 0.4}, 5, nil)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Error("expected search results after columnar import")
	}
}

func TestColumnarExport_FLAT(t *testing.T) {
	idx, _ := NewFLATIndex(3, map[string]interface{}{"metric": "cosine"})
	ctx := context.Background()
	idx.Add(ctx, 1, []float32{1, 0, 0})
	idx.Add(ctx, 2, []float32{0, 1, 0})
	idx.Add(ctx, 3, []float32{0, 0, 1})

	dir := t.TempDir()
	path := filepath.Join(dir, "flat.shard")
	if err := ExportToShardColumnar(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	cr, err := OpenColumnReader(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer cr.Close()

	if cr.Meta().IndexType != "FLAT" {
		t.Errorf("expected FLAT, got %s", cr.Meta().IndexType)
	}

	vectors, err := cr.ReadVectors()
	if err != nil {
		t.Fatalf("read vectors: %v", err)
	}
	if len(vectors) != 3 {
		t.Errorf("expected 3 vectors, got %d", len(vectors))
	}

	cfg, err := cr.ReadConfig()
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if cfg["metric"] != "cosine" {
		t.Errorf("expected metric=cosine, got %v", cfg["metric"])
	}
}

func TestColumnarExport_IVF(t *testing.T) {
	idx, _ := NewIVFIndex(3, map[string]interface{}{
		"nlist": 2, "nprobe": 1,
	})
	ctx := context.Background()
	for i := uint64(1); i <= 10; i++ {
		idx.Add(ctx, i, []float32{float32(i), float32(i) * 2, float32(i) * 3})
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "ivf.shard")
	if err := ExportToShardColumnar(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	cr, err := OpenColumnReader(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer cr.Close()

	if cr.Meta().IndexType != "IVF" {
		t.Errorf("expected IVF, got %s", cr.Meta().IndexType)
	}
	if cr.Meta().Count != 10 {
		t.Errorf("expected count 10, got %d", cr.Meta().Count)
	}
}

func TestColumnReader_SelectiveLoading(t *testing.T) {
	// Create a shard with vectors
	idx, _ := NewHNSWIndex(8, map[string]interface{}{
		"m": 16, "ef_construction": 200, "metric": "cosine",
	})
	ctx := context.Background()
	for i := uint64(1); i <= 100; i++ {
		vec := make([]float32, 8)
		for j := range vec {
			vec[j] = float32(i) * 0.01 * float32(j+1)
		}
		idx.Add(ctx, i, vec)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "selective.shard")
	ExportToShardColumnar(idx, path)

	// Open and read ONLY metadata — vectors never touched
	cr, _ := OpenColumnReader(path)
	defer cr.Close()

	meta := cr.Meta()
	if meta.Count != 100 {
		t.Errorf("expected 100, got %d", meta.Count)
	}
	if meta.Dim != 8 {
		t.Errorf("expected dim 8, got %d", meta.Dim)
	}

	// Verify HasColumn
	if !cr.HasColumn(ColVectors) {
		t.Error("expected vectors column")
	}
	if !cr.HasColumn(ColConfig) {
		t.Error("expected config column")
	}
	if cr.HasColumn(ColDeleted) {
		t.Error("should not have deleted column (no deletions)")
	}
	if cr.HasColumn(ColGraph) {
		t.Error("should not have graph column (not implemented yet)")
	}

	// Entry size check — vectors for 100 * (8 + 8*4) = 4000 bytes uncompressed
	// With zstd compression should be smaller
	vecSize := cr.EntrySize(ColVectors)
	t.Logf("vector column: %d bytes on disk (100 vectors, dim=8, 4000 bytes raw)", vecSize)
	if vecSize <= 0 {
		t.Error("expected positive vector size")
	}
	if vecSize > 4000 {
		t.Error("compression should reduce size below raw")
	}
}

func TestColumnReader_MonolithicFallback(t *testing.T) {
	// Export using monolithic (v1) format
	idx, _ := NewHNSWIndex(3, map[string]interface{}{
		"m": 8, "ef_construction": 100, "metric": "cosine",
	})
	ctx := context.Background()
	idx.Add(ctx, 1, []float32{1, 0, 0})

	dir := t.TempDir()
	path := filepath.Join(dir, "mono.shard")
	if err := ExportToShard(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	// ColumnReader should still open and read meta from monolithic shard
	cr, err := OpenColumnReader(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer cr.Close()

	if cr.IsColumnar() {
		t.Error("monolithic shard should not be columnar")
	}
	if cr.Meta().IndexType != "HNSW" {
		t.Errorf("expected HNSW, got %s", cr.Meta().IndexType)
	}
}

func TestColumnarExport_NoDeletedColumn(t *testing.T) {
	// Export with no deletions — should not create deleted column
	idx, _ := NewFLATIndex(2, map[string]interface{}{"metric": "cosine"})
	ctx := context.Background()
	idx.Add(ctx, 1, []float32{1, 0})

	dir := t.TempDir()
	path := filepath.Join(dir, "nodelete.shard")
	ExportToShardColumnar(idx, path)

	cr, _ := OpenColumnReader(path)
	defer cr.Close()

	deleted, err := cr.ReadDeleted()
	if err != nil {
		t.Fatalf("read deleted: %v", err)
	}
	if len(deleted) != 0 {
		t.Errorf("expected no deleted IDs, got %d", len(deleted))
	}
}
