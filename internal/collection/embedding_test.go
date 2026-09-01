package collection

import (
	"fmt"
	"strings"
	"testing"
)

// TextToSparse is the client-reproducible half of "text in": a caller that
// hashes terms itself must land on the same vector the server builds for a
// bm25-bound field, so the function has to be deterministic and total.
func TestTextToSparseIsDeterministicAndAggregatesCollisions(t *testing.T) {
	a, err := TextToSparse("Durable vector storage for agents", 64)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TextToSparse("Durable vector storage for agents", 64)
	if err != nil {
		t.Fatal(err)
	}
	if a.Dim != 64 || len(a.Indices) == 0 {
		t.Fatalf("unexpected sparse shape: dim=%d nnz=%d", a.Dim, len(a.Indices))
	}
	if fmt.Sprint(a.Indices, a.Values) != fmt.Sprint(b.Indices, b.Values) {
		t.Fatalf("same text produced different sparse vectors:\n%v %v\n%v %v", a.Indices, a.Values, b.Indices, b.Values)
	}

	// Repeated terms accumulate weight instead of being dropped, and with dim
	// 1 every term collides on the single bucket.
	one, err := TextToSparse("alpha alpha beta", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(one.Indices) != 1 || one.Values[0] != 3 {
		t.Fatalf("dim 1 should aggregate all 3 terms into one bucket, got indices=%v values=%v", one.Indices, one.Values)
	}

	empty, err := TextToSparse("", 8)
	if err != nil {
		t.Fatalf("empty text must produce an empty vector, not an error: %v", err)
	}
	if len(empty.Indices) != 0 {
		t.Fatalf("empty text produced terms: %v", empty.Indices)
	}

	if _, err := TextToSparse("anything", 0); err == nil || !strings.Contains(err.Error(), "positive") {
		t.Fatalf("dim 0 must be rejected, got %v", err)
	}
}

// The embedding binding is journaled with the schema, so the type matrix is
// enforced at Validate: bm25 belongs to sparse fields only, and a sparse
// field cannot pretend to a model it does not have.
func TestVectorFieldEmbeddingBindingRules(t *testing.T) {
	dense := func(e *EmbeddingConfig) VectorField {
		return VectorField{Name: "text", Type: VectorTypeDense, Dim: 4, Index: IndexConfig{Type: IndexTypeFLAT}, Embedding: e}
	}
	sparse := func(e *EmbeddingConfig) VectorField {
		return VectorField{Name: "terms", Type: VectorTypeSparse, Dim: 1024, Index: IndexConfig{Type: IndexTypeInverted}, Embedding: e}
	}
	cases := []struct {
		name    string
		field   VectorField
		wantErr string
	}{
		{"dense unbound", dense(nil), ""},
		{"dense bound to an embedder", dense(&EmbeddingConfig{Provider: "ollama", Model: "nomic-embed-text"}), ""},
		{"dense bound without a model", dense(&EmbeddingConfig{Provider: "hash"}), ""},
		{"dense bm25", dense(&EmbeddingConfig{Provider: EmbeddingProviderBM25}), "sparse fields"},
		{"empty provider", dense(&EmbeddingConfig{}), "provider"},
		{"sparse bm25", sparse(&EmbeddingConfig{Provider: EmbeddingProviderBM25}), ""},
		{"sparse bm25 with a model", sparse(&EmbeddingConfig{Provider: EmbeddingProviderBM25, Model: "x"}), "without a model"},
		{"sparse dense embedder", sparse(&EmbeddingConfig{Provider: "ollama"}), "bm25"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.field.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}
