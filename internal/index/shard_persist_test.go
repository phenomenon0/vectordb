package index

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	shard "github.com/Neumenon/shard/go/shard"
)

// indexFactory creates a fresh index by type for import.
func indexFactory(indexType string, dim int) (Index, error) {
	switch indexType {
	case "HNSW":
		return NewHNSWIndex(dim, map[string]interface{}{"m": 16, "ef_construction": 200, "metric": "cosine"})
	case "IVF":
		return NewIVFIndex(dim, map[string]interface{}{"nlist": 4, "nprobe": 2, "metric": "cosine"})
	case "FLAT":
		return NewFLATIndex(dim, map[string]interface{}{"metric": "cosine"})
	case "Sparse":
		return NewSparseIndex(dim, map[string]interface{}{"metric": "cosine"})
	default:
		return nil, fmt.Errorf("unknown index type: %s", indexType)
	}
}

func TestShardPersistHNSW(t *testing.T) {
	idx, err := NewHNSWIndex(8, map[string]interface{}{"m": 16, "ef_construction": 200, "metric": "cosine"})
	if err != nil {
		t.Fatalf("create index: %v", err)
	}
	ctx := context.Background()

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 50; i++ {
		vec := make([]float32, 8)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		if err := idx.Add(ctx, uint64(i+1), vec); err != nil {
			t.Fatalf("add vector %d: %v", i, err)
		}
	}

	path := filepath.Join(t.TempDir(), "hnsw.shard")

	if err := ExportToShard(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	// Verify file exists and has SHRD magic
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	magic := make([]byte, 4)
	f.Read(magic)
	f.Close()
	if string(magic) != "SHRD" {
		t.Fatalf("expected SHRD magic, got %q", magic)
	}

	// Import back
	restored, err := ImportFromShard(path, indexFactory)
	if err != nil {
		t.Fatalf("import: %v", err)
	}

	origStats := idx.Stats()
	restStats := restored.Stats()
	if restStats.Name != origStats.Name {
		t.Errorf("name mismatch: %s vs %s", restStats.Name, origStats.Name)
	}
	if restStats.Count != origStats.Count {
		t.Errorf("count mismatch: %d vs %d", restStats.Count, origStats.Count)
	}

	// Search should return valid results (HNSW rebuilds graph so exact order may differ)
	query := make([]float32, 8)
	for j := range query {
		query[j] = rng.Float32()
	}
	restResults, err := restored.Search(ctx, query, 5, DefaultSearchParams{})
	if err != nil {
		t.Fatalf("search restored: %v", err)
	}
	if len(restResults) != 5 {
		t.Errorf("expected 5 results, got %d", len(restResults))
	}
}

func TestShardPersistFLAT(t *testing.T) {
	idx, err := NewFLATIndex(4, map[string]interface{}{"metric": "cosine"})
	if err != nil {
		t.Fatalf("create index: %v", err)
	}
	ctx := context.Background()

	vectors := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}
	for i, v := range vectors {
		idx.Add(ctx, uint64(i+1), v)
	}

	path := filepath.Join(t.TempDir(), "flat.shard")

	if err := ExportToShard(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	restored, err := ImportFromShard(path, indexFactory)
	if err != nil {
		t.Fatalf("import: %v", err)
	}

	if restored.Stats().Count != 3 {
		t.Errorf("expected 3 vectors, got %d", restored.Stats().Count)
	}
}

func TestShardPersistIVF(t *testing.T) {
	idx, err := NewIVFIndex(4, map[string]interface{}{"nlist": 2, "nprobe": 1, "metric": "cosine"})
	if err != nil {
		t.Fatalf("create index: %v", err)
	}
	ctx := context.Background()

	rng := rand.New(rand.NewSource(99))
	for i := 0; i < 20; i++ {
		vec := make([]float32, 4)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		idx.Add(ctx, uint64(i+1), vec)
	}

	path := filepath.Join(t.TempDir(), "ivf.shard")

	if err := ExportToShard(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	restored, err := ImportFromShard(path, indexFactory)
	if err != nil {
		t.Fatalf("import: %v", err)
	}

	if restored.Stats().Count != 20 {
		t.Errorf("expected 20 vectors, got %d", restored.Stats().Count)
	}
}

func TestShardPersistSparse(t *testing.T) {
	idx, err := NewSparseIndex(5, map[string]interface{}{"metric": "cosine"})
	if err != nil {
		t.Fatalf("create index: %v", err)
	}
	ctx := context.Background()

	if err := idx.Add(ctx, 1, []float32{1, 0, 0, 2, 0}); err != nil {
		t.Fatalf("add 1: %v", err)
	}
	if err := idx.Add(ctx, 2, []float32{0, 3, 0, 0, 1}); err != nil {
		t.Fatalf("add 2: %v", err)
	}

	path := filepath.Join(t.TempDir(), "sparse.shard")

	if err := ExportToShard(idx, path); err != nil {
		t.Fatalf("export: %v", err)
	}

	restored, err := ImportFromShard(path, indexFactory)
	if err != nil {
		t.Fatalf("import: %v", err)
	}

	if restored.Stats().Count != 2 {
		t.Errorf("expected 2 vectors, got %d", restored.Stats().Count)
	}
}

func TestShardReadMeta(t *testing.T) {
	idx, _ := NewFLATIndex(16, map[string]interface{}{"metric": "cosine"})
	ctx := context.Background()

	for i := 0; i < 10; i++ {
		vec := make([]float32, 16)
		vec[i%16] = 1
		idx.Add(ctx, uint64(i+1), vec)
	}

	path := filepath.Join(t.TempDir(), "meta.shard")
	ExportToShard(idx, path)

	meta, err := ReadShardMeta(path)
	if err != nil {
		t.Fatalf("read meta: %v", err)
	}
	if meta.IndexType != "FLAT" {
		t.Errorf("expected FLAT, got %s", meta.IndexType)
	}
	if meta.Dim != 16 {
		t.Errorf("expected dim 16, got %d", meta.Dim)
	}
	if meta.Count != 10 {
		t.Errorf("expected count 10, got %d", meta.Count)
	}
	if meta.CreatedAt == "" {
		t.Error("expected non-empty created_at")
	}
}

func TestShardIntegrity(t *testing.T) {
	idx, _ := NewFLATIndex(4, map[string]interface{}{"metric": "cosine"})
	ctx := context.Background()
	idx.Add(ctx, 1, []float32{1, 2, 3, 4})

	path := filepath.Join(t.TempDir(), "corrupt.shard")
	ExportToShard(idx, path)

	// Corrupt the file by flipping bytes in the data section
	data, _ := os.ReadFile(path)
	if len(data) > 200 {
		data[len(data)-10] ^= 0xFF
		data[len(data)-20] ^= 0xFF
		os.WriteFile(path, data, 0o644)
	}

	// Import should fail with checksum mismatch
	_, err := ImportFromShard(path, indexFactory)
	if err == nil {
		t.Error("expected error from corrupted shard")
	}
}

func TestShardCompression(t *testing.T) {
	idx, _ := NewHNSWIndex(64, map[string]interface{}{"m": 16, "ef_construction": 200, "metric": "cosine"})
	ctx := context.Background()

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 200; i++ {
		vec := make([]float32, 64)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		idx.Add(ctx, uint64(i+1), vec)
	}

	dir := t.TempDir()
	shardPath := filepath.Join(dir, "compressed.shard")
	ExportToShard(idx, shardPath)

	// Verify compression happened by checking shard entry flags
	r, err := shard.OpenShardV2(shardPath)
	if err != nil {
		t.Fatalf("open shard: %v", err)
	}
	defer r.Close()

	dataIdx := r.Lookup("index_data")
	if dataIdx < 0 {
		t.Fatal("missing index_data entry")
	}
	entry := r.GetEntryInfo(dataIdx)
	if !entry.IsCompressed() {
		t.Error("expected index_data to be compressed")
	}
	if entry.DiskSize >= entry.OrigSize {
		t.Errorf("compressed size %d should be smaller than original %d", entry.DiskSize, entry.OrigSize)
	}

	t.Logf("compression ratio: %.1f%% (disk=%d, orig=%d)",
		float64(entry.DiskSize)/float64(entry.OrigSize)*100,
		entry.DiskSize, entry.OrigSize)
}

func TestShardMissingMeta(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nometa.shard")
	w, _ := shard.NewShardV2Writer(path, ShardRoleVectorIndex)
	w.WriteEntry("something", []byte("data"))
	w.Close()

	_, err := ImportFromShard(path, indexFactory)
	if err == nil {
		t.Error("expected error for missing meta")
	}
}

func TestShardMissingData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nodata.shard")
	w, _ := shard.NewShardV2Writer(path, ShardRoleVectorIndex)
	meta := `{"index_type":"FLAT","version":1,"dim":4,"count":0,"created_at":"2026-01-01T00:00:00Z"}`
	w.WriteEntryTyped("meta", []byte(meta), shard.ContentTypeJSON)
	w.Close()

	_, err := ImportFromShard(path, indexFactory)
	if err == nil {
		t.Error("expected error for missing index_data")
	}
}
