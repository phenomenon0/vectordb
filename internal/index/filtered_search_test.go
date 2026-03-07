package index

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/phenomenon0/vectordb/internal/filter"
)

// TestHNSWFilteredSearch tests filtered search with HNSW index
func TestHNSWFilteredSearch(t *testing.T) {
	dim := 32
	idx, err := NewHNSWIndex(dim, map[string]interface{}{
		"m":         16,
		"ef_search": 64,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW index: %v", err)
	}

	hnswIdx := idx.(*HNSWIndex)

	// Add test vectors with metadata
	ctx := context.Background()
	testData := []struct {
		id       uint64
		category string
		price    float64
		inStock  bool
		tags     []string
	}{
		{1, "electronics", 999.0, true, []string{"phone", "5g", "premium"}},
		{2, "electronics", 599.0, true, []string{"phone", "budget"}},
		{3, "clothing", 49.99, true, []string{"shirt", "cotton"}},
		{4, "electronics", 1299.0, false, []string{"laptop", "premium"}},
		{5, "clothing", 79.99, true, []string{"jeans", "denim"}},
		{6, "electronics", 399.0, true, []string{"tablet", "budget"}},
	}

	for _, data := range testData {
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = rand.Float32()
		}

		if err := hnswIdx.Add(ctx, data.id, vec); err != nil {
			t.Fatalf("Failed to add vector %d: %v", data.id, err)
		}

		metadata := map[string]interface{}{
			"category": data.category,
			"price":    data.price,
			"in_stock": data.inStock,
			"tags":     data.tags,
		}
		if err := hnswIdx.SetMetadata(data.id, metadata); err != nil {
			t.Fatalf("Failed to set metadata for vector %d: %v", data.id, err)
		}
	}

	// Test 1: Filter by category
	t.Run("FilterByCategory", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		params := HNSWSearchParams{
			EfSearch: 64,
			Filter:   filter.Eq("category", "electronics"),
		}

		results, err := hnswIdx.Search(ctx, query, 10, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should be electronics
		for _, r := range results {
			if r.Metadata["category"] != "electronics" {
				t.Errorf("Result ID %d has category %v, expected electronics", r.ID, r.Metadata["category"])
			}
		}

		t.Logf("Found %d electronics items", len(results))
	})

	// Test 2: Filter by price range
	t.Run("FilterByPriceRange", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		params := HNSWSearchParams{
			EfSearch: 64,
			Filter: filter.And(
				filter.Gte("price", 500.0),
				filter.Lt("price", 1000.0),
			),
		}

		results, err := hnswIdx.Search(ctx, query, 10, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should have price in [500, 1000)
		for _, r := range results {
			price := r.Metadata["price"].(float64)
			if price < 500.0 || price >= 1000.0 {
				t.Errorf("Result ID %d has price %.2f, expected [500, 1000)", r.ID, price)
			}
		}

		t.Logf("Found %d items in price range [500, 1000)", len(results))
	})

	// Test 3: Complex filter with AND/OR
	t.Run("ComplexFilter", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		// (category == "electronics" AND price < 1000) OR (category == "clothing" AND in_stock == true)
		params := HNSWSearchParams{
			EfSearch: 64,
			Filter: filter.Or(
				filter.And(
					filter.Eq("category", "electronics"),
					filter.Lt("price", 1000.0),
				),
				filter.And(
					filter.Eq("category", "clothing"),
					filter.Eq("in_stock", true),
				),
			),
		}

		results, err := hnswIdx.Search(ctx, query, 10, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Verify results match the complex filter
		for _, r := range results {
			category := r.Metadata["category"].(string)
			price := r.Metadata["price"].(float64)
			inStock := r.Metadata["in_stock"].(bool)

			valid := (category == "electronics" && price < 1000.0) ||
				(category == "clothing" && inStock)

			if !valid {
				t.Errorf("Result ID %d doesn't match filter: category=%s, price=%.2f, in_stock=%v",
					r.ID, category, price, inStock)
			}
		}

		t.Logf("Found %d items matching complex filter", len(results))
	})

	// Test 4: Filter by array contains
	t.Run("FilterByArrayContains", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		params := HNSWSearchParams{
			EfSearch: 64,
			Filter:   filter.Contains("tags", "premium"),
		}

		results, err := hnswIdx.Search(ctx, query, 10, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should have "premium" in tags
		for _, r := range results {
			tags := r.Metadata["tags"].([]string)
			hasPremium := false
			for _, tag := range tags {
				if tag == "premium" {
					hasPremium = true
					break
				}
			}
			if !hasPremium {
				t.Errorf("Result ID %d doesn't have 'premium' tag: %v", r.ID, tags)
			}
		}

		t.Logf("Found %d premium items", len(results))
	})

	// Test 5: No filter (should return all results)
	t.Run("NoFilter", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		params := HNSWSearchParams{
			EfSearch: 64,
			Filter:   nil, // No filter
		}

		results, err := hnswIdx.Search(ctx, query, 6, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Should return all 6 vectors
		if len(results) != 6 {
			t.Errorf("Expected 6 results without filter, got %d", len(results))
		}

		t.Logf("Found %d items without filter", len(results))
	})
}

// TestIVFFilteredSearch tests filtered search with IVF index
func TestIVFFilteredSearch(t *testing.T) {
	dim := 32
	idx, err := NewIVFIndex(dim, map[string]interface{}{
		"nlist":  4,
		"nprobe": 2,
	})
	if err != nil {
		t.Fatalf("Failed to create IVF index: %v", err)
	}

	ivfIdx := idx.(*IVFIndex)

	// Add test vectors with metadata
	ctx := context.Background()
	for i := uint64(1); i <= 50; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}

		if err := ivfIdx.Add(ctx, i, vec); err != nil {
			t.Fatalf("Failed to add vector %d: %v", i, err)
		}

		metadata := map[string]interface{}{
			"id":    int(i),
			"group": fmt.Sprintf("group_%d", i%5),
			"value": float64(i * 10),
		}
		if err := ivfIdx.SetMetadata(i, metadata); err != nil {
			t.Fatalf("Failed to set metadata for vector %d: %v", i, err)
		}
	}

	// Test filtered search
	t.Run("FilterByGroup", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		params := IVFSearchParams{
			NProbe: 2,
			Filter: filter.Eq("group", "group_2"),
		}

		results, err := ivfIdx.Search(ctx, query, 20, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should be in group_2
		for _, r := range results {
			if r.Metadata["group"] != "group_2" {
				t.Errorf("Result ID %d has group %v, expected group_2", r.ID, r.Metadata["group"])
			}
		}

		t.Logf("IVF: Found %d items in group_2", len(results))
	})

	// Test range filter
	t.Run("FilterByValueRange", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		params := IVFSearchParams{
			NProbe: 2,
			Filter: filter.And(
				filter.Gte("value", 100.0),
				filter.Lt("value", 300.0),
			),
		}

		results, err := ivfIdx.Search(ctx, query, 30, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should have value in [100, 300)
		for _, r := range results {
			value := r.Metadata["value"].(float64)
			if value < 100.0 || value >= 300.0 {
				t.Errorf("Result ID %d has value %.2f, expected [100, 300)", r.ID, value)
			}
		}

		t.Logf("IVF: Found %d items with value in [100, 300)", len(results))
	})
}

// TestDiskANNFilteredSearch tests filtered search with DiskANN index
func TestDiskANNFilteredSearch(t *testing.T) {
	dim := 32
	idx, err := NewDiskANNIndex(dim, map[string]interface{}{
		"memory_limit":    20,
		"max_degree":      16,
		"ef_construction": 50,
		"ef_search":       30,
		"index_path":      "/tmp/diskann_filter_test.idx",
	})
	if err != nil {
		t.Fatalf("Failed to create DiskANN index: %v", err)
	}
	defer idx.(*DiskANNIndex).Close()

	diskannIdx := idx.(*DiskANNIndex)

	// Add test vectors with metadata
	ctx := context.Background()
	for i := uint64(1); i <= 30; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}

		if err := diskannIdx.Add(ctx, i, vec); err != nil {
			t.Fatalf("Failed to add vector %d: %v", i, err)
		}

		metadata := map[string]interface{}{
			"id":       int(i),
			"priority": i % 3, // 0, 1, or 2
			"score":    float64(i) * 1.5,
		}
		if err := diskannIdx.SetMetadata(i, metadata); err != nil {
			t.Fatalf("Failed to set metadata for vector %d: %v", i, err)
		}
	}

	// Test filtered search
	t.Run("FilterByPriority", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		// Use HNSW params for DiskANN (it will accept them)
		params := HNSWSearchParams{
			EfSearch: 30,
			Filter:   filter.Eq("priority", uint64(1)),
		}

		results, err := diskannIdx.Search(ctx, query, 15, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should have priority == 1
		for _, r := range results {
			priority := r.Metadata["priority"].(uint64)
			if priority != 1 {
				t.Errorf("Result ID %d has priority %d, expected 1", r.ID, priority)
			}
		}

		t.Logf("DiskANN: Found %d items with priority 1", len(results))
	})

	// Test range filter
	t.Run("FilterByScoreRange", func(t *testing.T) {
		query := make([]float32, dim)
		for i := range query {
			query[i] = rand.Float32()
		}

		params := HNSWSearchParams{
			EfSearch: 30,
			Filter: filter.And(
				filter.Gt("score", 15.0),
				filter.Lte("score", 45.0),
			),
		}

		results, err := diskannIdx.Search(ctx, query, 20, params)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should have score in (15, 45]
		for _, r := range results {
			score := r.Metadata["score"].(float64)
			if score <= 15.0 || score > 45.0 {
				t.Errorf("Result ID %d has score %.2f, expected (15, 45]", r.ID, score)
			}
		}

		t.Logf("DiskANN: Found %d items with score in (15, 45]", len(results))
	})
}

// BenchmarkFilteredSearch benchmarks filtered vs unfiltered search
func BenchmarkFilteredSearch(b *testing.B) {
	dim := 128
	idx, err := NewHNSWIndex(dim, map[string]interface{}{
		"m":         16,
		"ef_search": 64,
	})
	if err != nil {
		b.Fatalf("Failed to create HNSW index: %v", err)
	}

	hnswIdx := idx.(*HNSWIndex)
	ctx := context.Background()

	// Add 1000 vectors with metadata
	for i := uint64(1); i <= 1000; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}

		hnswIdx.Add(ctx, i, vec)
		metadata := map[string]interface{}{
			"category": fmt.Sprintf("cat_%d", i%10),
			"price":    float64(i),
		}
		hnswIdx.SetMetadata(i, metadata)
	}

	query := make([]float32, dim)
	for i := range query {
		query[i] = rand.Float32()
	}

	b.Run("Unfiltered", func(b *testing.B) {
		params := HNSWSearchParams{
			EfSearch: 64,
			Filter:   nil,
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hnswIdx.Search(ctx, query, 10, params)
		}
	})

	b.Run("SimpleFilter", func(b *testing.B) {
		params := HNSWSearchParams{
			EfSearch: 64,
			Filter:   filter.Eq("category", "cat_5"),
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hnswIdx.Search(ctx, query, 10, params)
		}
	})

	b.Run("ComplexFilter", func(b *testing.B) {
		params := HNSWSearchParams{
			EfSearch: 64,
			Filter: filter.And(
				filter.Eq("category", "cat_5"),
				filter.Gte("price", 500.0),
				filter.Lt("price", 600.0),
			),
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hnswIdx.Search(ctx, query, 10, params)
		}
	})
}
