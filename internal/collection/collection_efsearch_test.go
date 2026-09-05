package collection

import "testing"

// TestNewCollectionUsesDefaultEfSearchVar encodes the intent that
// DefaultEfSearch is the knob for a new collection's ef_search, not the
// HNSW_EFSEARCH environment variable: setting the package var must take
// effect, and setting the environment variable must have no effect at all.
func TestNewCollectionUsesDefaultEfSearchVar(t *testing.T) {
	orig := DefaultEfSearch
	t.Cleanup(func() { DefaultEfSearch = orig })
	DefaultEfSearch = 17

	t.Setenv("HNSW_EFSEARCH", "999")

	coll, err := NewCollection(CollectionSchema{
		Name: "efsearch-test",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  4,
				Index: IndexConfig{
					Type: IndexTypeHNSW,
					Params: map[string]interface{}{
						"m":               16,
						"ef_construction": 200,
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	if coll.defaultEfSearch != 17 {
		t.Fatalf("defaultEfSearch = %d, want 17 (DefaultEfSearch); HNSW_EFSEARCH=999 must have no effect", coll.defaultEfSearch)
	}
}
