package collection

import (
	"context"
	"testing"

	"github.com/phenomenon0/vectordb/internal/filter"
)

// TestCollectionFilteredSearch tests filtered search through the collection layer
func TestCollectionFilteredSearch(t *testing.T) {
	// Create a simple collection with one dense vector field
	schema := CollectionSchema{
		Name: "products",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  32,
				Index: IndexConfig{
					Type: IndexTypeHNSW,
					Params: map[string]interface{}{
						"m":         16,
						"ef_search": 64,
					},
				},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Add test documents with metadata
	ctx := context.Background()
	testDocs := []struct {
		id       uint64
		category string
		price    float64
		inStock  bool
	}{
		{1, "electronics", 999.0, true},
		{2, "electronics", 599.0, true},
		{3, "clothing", 49.99, true},
		{4, "electronics", 1299.0, false},
		{5, "clothing", 79.99, true},
	}

	for _, td := range testDocs {
		// Create a random vector
		vec := make([]float32, 32)
		for i := range vec {
			vec[i] = float32(td.id) / 10.0 // Simple deterministic vector
		}

		doc := Document{
			ID: td.id,
			Vectors: map[string]interface{}{
				"embedding": vec,
			},
			Metadata: map[string]interface{}{
				"category": td.category,
				"price":    td.price,
				"in_stock": td.inStock,
			},
		}

		if err := coll.Add(ctx, &doc); err != nil {
			t.Fatalf("Failed to add document %d: %v", td.id, err)
		}
	}

	// Test 1: Search with category filter
	t.Run("FilterByCategory", func(t *testing.T) {
		query := make([]float32, 32)
		for i := range query {
			query[i] = 0.1
		}

		// Create filter map (as would come from HTTP request)
		filterMap := map[string]interface{}{
			"category": map[string]interface{}{
				"$eq": "electronics",
			},
		}

		req := SearchRequest{
			CollectionName: "products",
			Queries: map[string]interface{}{
				"embedding": query,
			},
			TopK:    10,
			Filters: filterMap,
		}

		resp, err := coll.Search(ctx, req)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// All results should be electronics
		for _, doc := range resp.Documents {
			category, ok := doc.Metadata["category"]
			if !ok {
				t.Errorf("Document %d missing category metadata", doc.ID)
				continue
			}
			if category != "electronics" {
				t.Errorf("Document %d has category %v, expected electronics", doc.ID, category)
			}
		}

		t.Logf("Found %d electronics items", len(resp.Documents))
	})

	// Test 2: Search with price range filter
	t.Run("FilterByPriceRange", func(t *testing.T) {
		query := make([]float32, 32)
		for i := range query {
			query[i] = 0.2
		}

		// Create complex filter: price >= 100 AND price < 1000
		filterMap := map[string]interface{}{
			"$and": []interface{}{
				map[string]interface{}{
					"price": map[string]interface{}{
						"$gte": 100.0,
					},
				},
				map[string]interface{}{
					"price": map[string]interface{}{
						"$lt": 1000.0,
					},
				},
			},
		}

		req := SearchRequest{
			CollectionName: "products",
			Queries: map[string]interface{}{
				"embedding": query,
			},
			TopK:    10,
			Filters: filterMap,
		}

		resp, err := coll.Search(ctx, req)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Verify all results match price range
		for _, doc := range resp.Documents {
			price, ok := doc.Metadata["price"].(float64)
			if !ok {
				t.Errorf("Document %d missing or invalid price metadata", doc.ID)
				continue
			}
			if price < 100.0 || price >= 1000.0 {
				t.Errorf("Document %d has price %.2f, expected [100, 1000)", doc.ID, price)
			}
		}

		t.Logf("Found %d items in price range [100, 1000)", len(resp.Documents))
	})

	// Test 3: Search with complex OR filter
	t.Run("ComplexORFilter", func(t *testing.T) {
		query := make([]float32, 32)
		for i := range query {
			query[i] = 0.3
		}

		// (category == "electronics" AND price < 700) OR (category == "clothing" AND in_stock == true)
		filterMap := map[string]interface{}{
			"$or": []interface{}{
				map[string]interface{}{
					"$and": []interface{}{
						map[string]interface{}{
							"category": map[string]interface{}{
								"$eq": "electronics",
							},
						},
						map[string]interface{}{
							"price": map[string]interface{}{
								"$lt": 700.0,
							},
						},
					},
				},
				map[string]interface{}{
					"$and": []interface{}{
						map[string]interface{}{
							"category": map[string]interface{}{
								"$eq": "clothing",
							},
						},
						map[string]interface{}{
							"in_stock": map[string]interface{}{
								"$eq": true,
							},
						},
					},
				},
			},
		}

		req := SearchRequest{
			CollectionName: "products",
			Queries: map[string]interface{}{
				"embedding": query,
			},
			TopK:    10,
			Filters: filterMap,
		}

		resp, err := coll.Search(ctx, req)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Verify all results match the complex filter
		for _, doc := range resp.Documents {
			category, _ := doc.Metadata["category"].(string)
			price, _ := doc.Metadata["price"].(float64)
			inStock, _ := doc.Metadata["in_stock"].(bool)

			valid := (category == "electronics" && price < 700.0) ||
				(category == "clothing" && inStock)

			if !valid {
				t.Errorf("Document %d doesn't match filter: category=%s, price=%.2f, in_stock=%v",
					doc.ID, category, price, inStock)
			}
		}

		t.Logf("Found %d items matching complex OR filter", len(resp.Documents))
	})

	// Test 4: Search without filter (sanity check)
	t.Run("NoFilter", func(t *testing.T) {
		query := make([]float32, 32)
		for i := range query {
			query[i] = 0.1
		}

		req := SearchRequest{
			CollectionName: "products",
			Queries: map[string]interface{}{
				"embedding": query,
			},
			TopK:    10,
			Filters: nil, // No filter
		}

		resp, err := coll.Search(ctx, req)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Should return all 5 documents
		if len(resp.Documents) != 5 {
			t.Errorf("Expected 5 documents without filter, got %d", len(resp.Documents))
		}

		t.Logf("Found %d items without filter", len(resp.Documents))
	})
}

// TestFilterFromMap validates the filter.FromMap function
func TestFilterFromMap(t *testing.T) {
	testCases := []struct {
		name      string
		filterMap map[string]interface{}
		metadata  map[string]interface{}
		shouldMatch bool
	}{
		{
			name: "SimpleEquality",
			filterMap: map[string]interface{}{
				"category": "electronics",
			},
			metadata: map[string]interface{}{
				"category": "electronics",
			},
			shouldMatch: true,
		},
		{
			name: "GreaterThan",
			filterMap: map[string]interface{}{
				"price": map[string]interface{}{
					"$gt": 100.0,
				},
			},
			metadata: map[string]interface{}{
				"price": 500.0,
			},
			shouldMatch: true,
		},
		{
			name: "AND Operator",
			filterMap: map[string]interface{}{
				"$and": []interface{}{
					map[string]interface{}{
						"category": "electronics",
					},
					map[string]interface{}{
						"price": map[string]interface{}{
							"$lt": 1000.0,
						},
					},
				},
			},
			metadata: map[string]interface{}{
				"category": "electronics",
				"price":    799.0,
			},
			shouldMatch: true,
		},
		{
			name: "OR Operator",
			filterMap: map[string]interface{}{
				"$or": []interface{}{
					map[string]interface{}{
						"category": "electronics",
					},
					map[string]interface{}{
						"category": "clothing",
					},
				},
			},
			metadata: map[string]interface{}{
				"category": "clothing",
			},
			shouldMatch: true,
		},
		{
			name: "NOT Operator",
			filterMap: map[string]interface{}{
				"$not": map[string]interface{}{
					"category": "electronics",
				},
			},
			metadata: map[string]interface{}{
				"category": "clothing",
			},
			shouldMatch: true,
		},
		{
			name: "MismatchedValue",
			filterMap: map[string]interface{}{
				"category": "electronics",
			},
			metadata: map[string]interface{}{
				"category": "clothing",
			},
			shouldMatch: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			f, err := filter.FromMap(tc.filterMap)
			if err != nil {
				t.Fatalf("FromMap failed: %v", err)
			}

			if f == nil {
				t.Fatal("Filter should not be nil")
			}

			matches := f.Evaluate(tc.metadata)
			if matches != tc.shouldMatch {
				t.Errorf("Filter evaluation mismatch: got %v, expected %v", matches, tc.shouldMatch)
			}
		})
	}
}
