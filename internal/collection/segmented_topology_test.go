package collection

import (
	"testing"

	denseindex "github.com/phenomenon0/vectordb/internal/index"
)

func TestHNSWSegmentTopologyIsSchemaExplicit(t *testing.T) {
	tests := []struct {
		name      string
		params    map[string]interface{}
		segmented bool
	}{
		{name: "omitted remains one graph", params: nil},
		{name: "explicit one remains one graph", params: map[string]interface{}{"segments": 1}},
		{name: "explicit multiple opts in", params: map[string]interface{}{"segments": 2}, segmented: true},
		{name: "json decoded multiple opts in", params: map[string]interface{}{"segments": float64(2)}, segmented: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			coll, err := NewCollection(CollectionSchema{
				Name: "topology",
				Fields: []VectorField{{
					Name:  "embedding",
					Type:  VectorTypeDense,
					Dim:   4,
					Index: IndexConfig{Type: IndexTypeHNSW, Params: tc.params},
				}},
			})
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			defer coll.Close()

			_, gotSegmented := coll.indexes["embedding"].(*denseindex.SegmentedIndex)
			if gotSegmented != tc.segmented {
				t.Fatalf("segmented = %v, want %v (index %T)", gotSegmented, tc.segmented, coll.indexes["embedding"])
			}
		})
	}
}
