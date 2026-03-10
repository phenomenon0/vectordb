package collection

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

func boolPtr(v bool) *bool { return &v }

func TestNewCollection_DenseOnly(t *testing.T) {
	schema := CollectionSchema{
		Name: "test_dense",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  128,
				Index: IndexConfig{
					Type: IndexTypeHNSW,
					Params: map[string]interface{}{
						"m":               16,
						"ef_construction": 200,
					},
				},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	if coll.Name() != "test_dense" {
		t.Errorf("name mismatch: got %s, want test_dense", coll.Name())
	}

	if len(coll.indexes) != 1 {
		t.Errorf("expected 1 index, got %d", len(coll.indexes))
	}
}

func TestNewCollection_SparseOnly(t *testing.T) {
	schema := CollectionSchema{
		Name: "test_sparse",
		Fields: []VectorField{
			{
				Name: "keywords",
				Type: VectorTypeSparse,
				Dim:  10000,
				Index: IndexConfig{
					Type: IndexTypeInverted,
					Params: map[string]interface{}{
						"k1": 1.2,
						"b":  0.75,
					},
				},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	if len(coll.sparse) != 1 {
		t.Errorf("expected 1 sparse index, got %d", len(coll.sparse))
	}
}

func TestNewCollection_MultiVector(t *testing.T) {
	schema := CollectionSchema{
		Name: "test_multi",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  384,
				Index: IndexConfig{
					Type: IndexTypeHNSW,
				},
			},
			{
				Name: "keywords",
				Type: VectorTypeSparse,
				Dim:  10000,
				Index: IndexConfig{
					Type: IndexTypeInverted,
				},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	if len(coll.indexes) != 1 {
		t.Errorf("expected 1 dense index, got %d", len(coll.indexes))
	}

	if len(coll.sparse) != 1 {
		t.Errorf("expected 1 sparse index, got %d", len(coll.sparse))
	}
}

func TestCollection_AddDense(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Add document
	doc := Document{
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 2.0, 3.0, 4.0},
		},
		Metadata: map[string]interface{}{
			"title": "test doc",
		},
	}

	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	if coll.Count() != 1 {
		t.Errorf("expected 1 document, got %d", coll.Count())
	}
}

func TestCollection_AddSparse(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "keywords",
				Type:  VectorTypeSparse,
				Dim:   100,
				Index: IndexConfig{Type: IndexTypeInverted},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Create sparse vector
	sparseVec, err := sparse.NewSparseVector(
		[]uint32{0, 5, 10},
		[]float32{1.0, 2.0, 3.0},
		100,
	)
	if err != nil {
		t.Fatalf("failed to create sparse vector: %v", err)
	}

	doc := Document{
		Vectors: map[string]interface{}{
			"keywords": sparseVec,
		},
		Metadata: map[string]interface{}{
			"title": "test doc",
		},
	}

	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	if coll.Count() != 1 {
		t.Errorf("expected 1 document, got %d", coll.Count())
	}
}

func TestCollection_AddMultiVector(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
			{
				Name:  "keywords",
				Type:  VectorTypeSparse,
				Dim:   100,
				Index: IndexConfig{Type: IndexTypeInverted},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	sparseVec, _ := sparse.NewSparseVector(
		[]uint32{0, 5, 10},
		[]float32{1.0, 2.0, 3.0},
		100,
	)

	doc := Document{
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 2.0, 3.0, 4.0},
			"keywords":  sparseVec,
		},
		Metadata: map[string]interface{}{
			"title": "test doc",
		},
	}

	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	if coll.Count() != 1 {
		t.Errorf("expected 1 document, got %d", coll.Count())
	}
}

func TestCollection_SearchDense(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Add documents
	ctx := context.Background()
	docs := []Document{
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{1.0, 0.0, 0.0, 0.0},
			},
			Metadata: map[string]interface{}{"id": 1},
		},
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{0.0, 1.0, 0.0, 0.0},
			},
			Metadata: map[string]interface{}{"id": 2},
		},
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{0.0, 0.0, 1.0, 0.0},
			},
			Metadata: map[string]interface{}{"id": 3},
		},
	}

	for _, doc := range docs {
		if err := coll.Add(ctx, &doc); err != nil {
			t.Fatalf("failed to add document: %v", err)
		}
	}

	// Search
	req := SearchRequest{
		CollectionName: "test",
		Queries: map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
		TopK: 2,
	}

	resp, err := coll.Search(ctx, req)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(resp.Documents) != 2 {
		t.Errorf("expected 2 results, got %d", len(resp.Documents))
	}

	// First result should be doc1 (exact match)
	if id, ok := resp.Documents[0].Metadata["id"].(int); !ok || id != 1 {
		t.Errorf("expected first result to be doc1, got id=%v", resp.Documents[0].Metadata["id"])
	}
	if len(resp.Documents[0].Vectors) == 0 {
		t.Fatal("expected vectors to be included by default")
	}
}

func TestCollection_SearchSparse(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "keywords",
				Type:  VectorTypeSparse,
				Dim:   100,
				Index: IndexConfig{Type: IndexTypeInverted},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Add documents
	ctx := context.Background()
	vec1, _ := sparse.NewSparseVector([]uint32{0, 1, 2}, []float32{1.0, 2.0, 3.0}, 100)
	vec2, _ := sparse.NewSparseVector([]uint32{0, 5, 10}, []float32{2.0, 1.0, 1.0}, 100)

	docs := []Document{
		{
			Vectors: map[string]interface{}{
				"keywords": vec1,
			},
			Metadata: map[string]interface{}{"id": 1},
		},
		{
			Vectors: map[string]interface{}{
				"keywords": vec2,
			},
			Metadata: map[string]interface{}{"id": 2},
		},
	}

	for _, doc := range docs {
		if err := coll.Add(ctx, &doc); err != nil {
			t.Fatalf("failed to add document: %v", err)
		}
	}

	// Search
	query, _ := sparse.NewSparseVector([]uint32{0, 1}, []float32{1.0, 2.0}, 100)
	req := SearchRequest{
		CollectionName: "test",
		Queries: map[string]interface{}{
			"keywords": query,
		},
		TopK: 2,
	}

	resp, err := coll.Search(ctx, req)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(resp.Documents) == 0 {
		t.Error("expected results from sparse search")
	}
}

func TestCollection_SearchHybrid(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
			{
				Name:  "keywords",
				Type:  VectorTypeSparse,
				Dim:   100,
				Index: IndexConfig{Type: IndexTypeInverted},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Add documents
	ctx := context.Background()
	vec1, _ := sparse.NewSparseVector([]uint32{0, 1, 2}, []float32{1.0, 2.0, 3.0}, 100)
	vec2, _ := sparse.NewSparseVector([]uint32{0, 5, 10}, []float32{2.0, 1.0, 1.0}, 100)

	docs := []Document{
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{1.0, 0.0, 0.0, 0.0},
				"keywords":  vec1,
			},
			Metadata: map[string]interface{}{"id": 1},
		},
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{0.0, 1.0, 0.0, 0.0},
				"keywords":  vec2,
			},
			Metadata: map[string]interface{}{"id": 2},
		},
	}

	for _, doc := range docs {
		if err := coll.Add(ctx, &doc); err != nil {
			t.Fatalf("failed to add document: %v", err)
		}
	}

	// Hybrid search
	query, _ := sparse.NewSparseVector([]uint32{0, 1}, []float32{1.0, 2.0}, 100)
	req := SearchRequest{
		CollectionName: "test",
		Queries: map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
			"keywords":  query,
		},
		TopK: 2,
		HybridParams: &HybridSearchParams{
			Strategy: "rrf",
			Weights: map[string]float32{
				"dense":  0.7,
				"sparse": 0.3,
			},
			RRFConstant: 60.0,
		},
	}

	resp, err := coll.Search(ctx, req)
	if err != nil {
		t.Fatalf("hybrid search failed: %v", err)
	}

	if len(resp.Documents) == 0 {
		t.Error("expected results from hybrid search")
	}

	t.Logf("Hybrid search returned %d results, examined %d candidates",
		len(resp.Documents), resp.CandidatesExamined)
}

func TestCollection_SearchCanOmitVectors(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	doc := Document{
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
	}
	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	resp, err := coll.Search(ctx, SearchRequest{
		CollectionName: "test",
		Queries: map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
		TopK:           1,
		IncludeVectors: boolPtr(false),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if resp.Documents[0].Vectors != nil {
		t.Fatalf("expected vectors to be omitted when explicitly disabled")
	}
}

func TestCollectionAcceptsJSONDenseDocumentAndQuery(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	var doc Document
	if err := json.Unmarshal([]byte(`{
		"vectors": {"embedding": [1, 0, 0, 0]},
		"metadata": {"source": "json"}
	}`), &doc); err != nil {
		t.Fatalf("unmarshal document failed: %v", err)
	}

	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add JSON document: %v", err)
	}

	var req SearchRequest
	if err := json.Unmarshal([]byte(`{
		"collection_name": "test",
		"queries": {"embedding": [1, 0, 0, 0]},
		"top_k": 1
	}`), &req); err != nil {
		t.Fatalf("unmarshal search request failed: %v", err)
	}

	resp, err := coll.Search(ctx, req)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(resp.Documents) != 1 {
		t.Fatalf("expected one result, got %d", len(resp.Documents))
	}
}

func TestCollectionAcceptsJSONSparseDocumentAndQuery(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "keywords",
				Type:  VectorTypeSparse,
				Dim:   16,
				Index: IndexConfig{Type: IndexTypeInverted},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	var doc Document
	if err := json.Unmarshal([]byte(`{
		"vectors": {
			"keywords": {
				"indices": [0, 5],
				"values": [1.25, 2.5],
				"dim": 16
			}
		}
	}`), &doc); err != nil {
		t.Fatalf("unmarshal sparse document failed: %v", err)
	}

	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add JSON sparse document: %v", err)
	}

	var req SearchRequest
	if err := json.Unmarshal([]byte(`{
		"collection_name": "test",
		"queries": {
			"keywords": {
				"indices": [0],
				"values": [1.0],
				"dim": 16
			}
		},
		"top_k": 1
	}`), &req); err != nil {
		t.Fatalf("unmarshal sparse search request failed: %v", err)
	}

	resp, err := coll.Search(ctx, req)
	if err != nil {
		t.Fatalf("sparse search failed: %v", err)
	}
	if len(resp.Documents) != 1 {
		t.Fatalf("expected one sparse result, got %d", len(resp.Documents))
	}
}

func TestCollectionSchemaJSONAcceptsStringEnums(t *testing.T) {
	var schema CollectionSchema
	err := json.Unmarshal([]byte(`{
		"name": "docs",
		"fields": [
			{"name": "embedding", "type": "dense", "dim": 4, "index": {"type": "flat"}}
		]
	}`), &schema)
	if err != nil {
		t.Fatalf("unmarshal schema failed: %v", err)
	}

	if schema.Fields[0].Type != VectorTypeDense {
		t.Fatalf("expected dense vector type, got %v", schema.Fields[0].Type)
	}
	if schema.Fields[0].Index.Type != IndexTypeFLAT {
		t.Fatalf("expected flat index type, got %v", schema.Fields[0].Index.Type)
	}

	data, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("marshal schema failed: %v", err)
	}
	if !strings.Contains(string(data), `"type":"dense"`) || !strings.Contains(string(data), `"type":"flat"`) {
		t.Fatalf("expected string enum encoding, got %s", string(data))
	}
}

func TestCollection_BatchAdd(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Batch add
	docs := make([]Document, 10)
	for i := range docs {
		docs[i] = Document{
			Vectors: map[string]interface{}{
				"embedding": []float32{float32(i), 0.0, 0.0, 0.0},
			},
		}
	}

	ctx := context.Background()
	if err := coll.BatchAdd(ctx, docs); err != nil {
		t.Fatalf("batch add failed: %v", err)
	}

	if coll.Count() != 10 {
		t.Errorf("expected 10 documents, got %d", coll.Count())
	}
}

func TestCollection_Delete(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	// Add document
	doc := Document{
		ID: 123,
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 2.0, 3.0, 4.0},
		},
	}

	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	// Delete
	if err := coll.Delete(ctx, 123); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if coll.Count() != 0 {
		t.Errorf("expected 0 documents after delete, got %d", coll.Count())
	}

	// Verify document is gone
	if _, ok := coll.GetDocument(123); ok {
		t.Error("document should not exist after delete")
	}
}

func TestCollection_GetDocument(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}

	doc := Document{
		ID: 456,
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 2.0, 3.0, 4.0},
		},
		Metadata: map[string]interface{}{
			"title": "test",
		},
	}

	ctx := context.Background()
	if err := coll.Add(ctx, &doc); err != nil {
		t.Fatalf("failed to add document: %v", err)
	}

	// Get document
	retrieved, ok := coll.GetDocument(456)
	if !ok {
		t.Fatal("document not found")
	}

	if retrieved.ID != 456 {
		t.Errorf("id mismatch: got %d, want 456", retrieved.ID)
	}

	if title, ok := retrieved.Metadata["title"].(string); !ok || title != "test" {
		t.Errorf("metadata mismatch: got %v", retrieved.Metadata["title"])
	}
}

func TestCollection_ValidationErrors(t *testing.T) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   4,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, _ := NewCollection(schema)
	ctx := context.Background()

	// Missing vector field
	doc1 := Document{
		Vectors: map[string]interface{}{},
	}
	if err := coll.Add(ctx, &doc1); err == nil {
		t.Error("should error on missing vector field")
	}

	// Wrong vector type
	doc2 := Document{
		Vectors: map[string]interface{}{
			"embedding": "not a vector",
		},
	}
	if err := coll.Add(ctx, &doc2); err == nil {
		t.Error("should error on wrong vector type")
	}

	// Extra field
	doc3 := Document{
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 2.0, 3.0, 4.0},
			"extra":     []float32{1.0, 2.0},
		},
	}
	if err := coll.Add(ctx, &doc3); err == nil {
		t.Error("should error on extra field")
	}
}

func BenchmarkCollection_AddDense(b *testing.B) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   384,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	coll, _ := NewCollection(schema)
	ctx := context.Background()

	vec := make([]float32, 384)
	for i := range vec {
		vec[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc := Document{
			Vectors: map[string]interface{}{
				"embedding": vec,
			},
		}
		_ = coll.Add(ctx, &doc)
	}
}

func BenchmarkCollection_AddMultiVector(b *testing.B) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   384,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
			{
				Name:  "keywords",
				Type:  VectorTypeSparse,
				Dim:   10000,
				Index: IndexConfig{Type: IndexTypeInverted},
			},
		},
	}

	coll, _ := NewCollection(schema)
	ctx := context.Background()

	denseVec := make([]float32, 384)
	for i := range denseVec {
		denseVec[i] = float32(i)
	}

	indices := make([]uint32, 300)
	values := make([]float32, 300)
	for i := range indices {
		indices[i] = uint32(i * 33)
		values[i] = float32(i + 1)
	}
	sparseVec, _ := sparse.NewSparseVector(indices, values, 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc := Document{
			Vectors: map[string]interface{}{
				"embedding": denseVec,
				"keywords":  sparseVec,
			},
		}
		_ = coll.Add(ctx, &doc)
	}
}

func BenchmarkCollection_SearchHybrid(b *testing.B) {
	schema := CollectionSchema{
		Name: "test",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   384,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
			{
				Name:  "keywords",
				Type:  VectorTypeSparse,
				Dim:   10000,
				Index: IndexConfig{Type: IndexTypeInverted},
			},
		},
	}

	coll, _ := NewCollection(schema)
	ctx := context.Background()

	// Add 100 documents
	for i := 0; i < 100; i++ {
		denseVec := make([]float32, 384)
		for j := range denseVec {
			denseVec[j] = float32(i*j) / 100.0
		}

		indices := make([]uint32, 50)
		values := make([]float32, 50)
		for j := range indices {
			indices[j] = uint32((i * j) % 10000)
			values[j] = float32(j + 1)
		}
		sparseVec, _ := sparse.NewSparseVector(indices, values, 10000)

		doc := Document{
			Vectors: map[string]interface{}{
				"embedding": denseVec,
				"keywords":  sparseVec,
			},
		}
		_ = coll.Add(ctx, &doc)
	}

	// Prepare query
	queryDense := make([]float32, 384)
	for i := range queryDense {
		queryDense[i] = float32(i) / 100.0
	}
	querySparse, _ := sparse.NewSparseVector([]uint32{0, 10, 20}, []float32{1.0, 2.0, 3.0}, 10000)

	req := SearchRequest{
		CollectionName: "test",
		Queries: map[string]interface{}{
			"embedding": queryDense,
			"keywords":  querySparse,
		},
		TopK: 10,
		HybridParams: &HybridSearchParams{
			Strategy:    "rrf",
			RRFConstant: 60.0,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = coll.Search(ctx, req)
	}
}
