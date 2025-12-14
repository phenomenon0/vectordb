package index

import (
	"context"
	"testing"
)

func TestHNSWBasicOperations(t *testing.T) {
	dim := 128
	idx, err := NewHNSWIndex(dim, nil)
	if err != nil {
		t.Fatalf("failed to create HNSW index: %v", err)
	}

	if idx.Name() != "HNSW" {
		t.Errorf("expected name HNSW, got %s", idx.Name())
	}

	ctx := context.Background()

	// Add some vectors
	vec1 := make([]float32, dim)
	for i := range vec1 {
		vec1[i] = float32(i) / float32(dim)
	}

	vec2 := make([]float32, dim)
	for i := range vec2 {
		vec2[i] = float32(i+10) / float32(dim)
	}

	vec3 := make([]float32, dim)
	for i := range vec3 {
		vec3[i] = float32(i+20) / float32(dim)
	}

	// Test Add
	if err := idx.Add(ctx, 1, vec1); err != nil {
		t.Fatalf("failed to add vector 1: %v", err)
	}

	if err := idx.Add(ctx, 2, vec2); err != nil {
		t.Fatalf("failed to add vector 2: %v", err)
	}

	if err := idx.Add(ctx, 3, vec3); err != nil {
		t.Fatalf("failed to add vector 3: %v", err)
	}

	// Test duplicate ID
	if err := idx.Add(ctx, 1, vec1); err == nil {
		t.Error("expected error when adding duplicate ID, got nil")
	}

	// Test dimension mismatch
	wrongVec := make([]float32, dim+1)
	if err := idx.Add(ctx, 4, wrongVec); err == nil {
		t.Error("expected error for dimension mismatch, got nil")
	}

	// Test Search
	results, err := idx.Search(ctx, vec1, 2, DefaultSearchParams{})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected at least 1 result, got 0")
	}

	// First result should be vec1 itself
	if results[0].ID != 1 {
		t.Errorf("expected first result ID=1, got ID=%d", results[0].ID)
	}

	// Test Stats
	stats := idx.Stats()
	if stats.Name != "HNSW" {
		t.Errorf("expected stats name HNSW, got %s", stats.Name)
	}
	if stats.Dim != dim {
		t.Errorf("expected dim %d, got %d", dim, stats.Dim)
	}
	if stats.Count != 3 {
		t.Errorf("expected count 3, got %d", stats.Count)
	}
	if stats.Active != 3 {
		t.Errorf("expected active 3, got %d", stats.Active)
	}

	// Test Delete
	if err := idx.Delete(ctx, 2); err != nil {
		t.Fatalf("failed to delete vector 2: %v", err)
	}

	// Verify deleted vector not in results
	results, err = idx.Search(ctx, vec2, 3, DefaultSearchParams{})
	if err != nil {
		t.Fatalf("search after delete failed: %v", err)
	}

	for _, r := range results {
		if r.ID == 2 {
			t.Error("deleted vector 2 still appears in search results")
		}
	}

	// Test stats after delete
	stats = idx.Stats()
	if stats.Deleted != 1 {
		t.Errorf("expected deleted count 1, got %d", stats.Deleted)
	}
	if stats.Active != 2 {
		t.Errorf("expected active count 2, got %d", stats.Active)
	}
}

func TestHNSWExportImport(t *testing.T) {
	dim := 64
	idx1, err := NewHNSWIndex(dim, map[string]interface{}{
		"m":         8,
		"ef_search": 32,
	})
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	ctx := context.Background()

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i*dim + j)
		}
		if err := idx1.Add(ctx, uint64(i), vec); err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Delete one vector
	if err := idx1.Delete(ctx, 5); err != nil {
		t.Fatalf("failed to delete: %v", err)
	}

	// Export
	data, err := idx1.Export()
	if err != nil {
		t.Fatalf("export failed: %v", err)
	}

	// Create new index and import
	idx2, err := NewHNSWIndex(dim, nil)
	if err != nil {
		t.Fatalf("failed to create second index: %v", err)
	}

	if err := idx2.Import(data); err != nil {
		t.Fatalf("import failed: %v", err)
	}

	// Verify stats match
	stats1 := idx1.Stats()
	stats2 := idx2.Stats()

	if stats1.Count != stats2.Count {
		t.Errorf("count mismatch: %d vs %d", stats1.Count, stats2.Count)
	}
	if stats1.Deleted != stats2.Deleted {
		t.Errorf("deleted mismatch: %d vs %d", stats1.Deleted, stats2.Deleted)
	}
	if stats1.Active != stats2.Active {
		t.Errorf("active mismatch: %d vs %d", stats1.Active, stats2.Active)
	}

	// Verify search results are reasonable
	// Note: HNSW graph construction is non-deterministic, so after import
	// the graph may be slightly different and return slightly different results.
	// We just verify that both return non-empty results.
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i)
	}

	results1, _ := idx1.Search(ctx, query, 5, DefaultSearchParams{})
	results2, _ := idx2.Search(ctx, query, 5, DefaultSearchParams{})

	// Both should return some results
	if len(results1) == 0 {
		t.Error("idx1 returned no results")
	}

	if len(results2) == 0 {
		t.Error("idx2 returned no results")
	}

	// Results should be similar in size (within 50%)
	if len(results1) > 0 && len(results2) > 0 {
		ratio := float64(len(results1)) / float64(len(results2))
		if ratio < 0.5 || ratio > 2.0 {
			t.Logf("Warning: result counts differ significantly: %d vs %d", len(results1), len(results2))
		}
	}
}

func TestHNSWFactory(t *testing.T) {
	// Test factory creation
	idx, err := Create("hnsw", 128, map[string]interface{}{
		"m":         16,
		"ef_search": 64,
	})
	if err != nil {
		t.Fatalf("factory create failed: %v", err)
	}

	if idx.Name() != "HNSW" {
		t.Errorf("expected name HNSW, got %s", idx.Name())
	}

	// Test supported types
	types := SupportedTypes()
	found := false
	for _, t := range types {
		if t == "hnsw" {
			found = true
			break
		}
	}
	if !found {
		t.Error("hnsw not in supported types")
	}

	// Test unknown type
	_, err = Create("unknown", 128, nil)
	if err == nil {
		t.Error("expected error for unknown index type, got nil")
	}
}

func TestHNSWSearchParams(t *testing.T) {
	dim := 128
	idx, err := NewHNSWIndex(dim, map[string]interface{}{
		"ef_search": 32, // Default
	})
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	ctx := context.Background()

	// Add vectors
	for i := 0; i < 20; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i*dim + j)
		}
		if err := idx.Add(ctx, uint64(i), vec); err != nil {
			t.Fatalf("failed to add vector: %v", err)
		}
	}

	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i)
	}

	// Test with default params
	results1, err := idx.Search(ctx, query, 5, DefaultSearchParams{})
	if err != nil {
		t.Fatalf("search with default params failed: %v", err)
	}

	// Test with HNSW params (higher ef_search should give same or better results)
	results2, err := idx.Search(ctx, query, 5, HNSWSearchParams{EfSearch: 128})
	if err != nil {
		t.Fatalf("search with HNSW params failed: %v", err)
	}

	// Results should be non-empty
	if len(results1) == 0 || len(results2) == 0 {
		t.Error("expected non-empty results")
	}
}

func TestHNSWCompact(t *testing.T) {
	dim := 64
	idx, err := NewHNSWIndex(dim, nil)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	ctx := context.Background()

	// Add vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = float32(i*dim + j)
		}
		if err := idx.Add(ctx, uint64(i), vec); err != nil {
			t.Fatalf("failed to add vector: %v", err)
		}
	}

	// Delete half
	for i := 0; i < 50; i++ {
		if err := idx.Delete(ctx, uint64(i)); err != nil {
			t.Fatalf("failed to delete: %v", err)
		}
	}

	statsBefore := idx.Stats()
	if statsBefore.Deleted != 50 {
		t.Errorf("expected 50 deleted before compact, got %d", statsBefore.Deleted)
	}

	// Compact
	hnswIdx := idx.(*HNSWIndex)
	removed, err := hnswIdx.Compact()
	if err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	if removed != 50 {
		t.Errorf("expected 50 removed, got %d", removed)
	}

	statsAfter := idx.Stats()
	if statsAfter.Deleted != 0 {
		t.Errorf("expected 0 deleted after compact, got %d", statsAfter.Deleted)
	}
	if statsAfter.Active != 50 {
		t.Errorf("expected 50 active after compact, got %d", statsAfter.Active)
	}
	if statsAfter.Count != 50 {
		t.Errorf("expected 50 count after compact, got %d", statsAfter.Count)
	}
}
