package collection

import (
	"context"
	"fmt"
	"testing"

	"agentscope/vectordb/sparse"
)

func TestNewCollectionManager(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")

	if cm == nil {
		t.Fatal("NewCollectionManager returned nil")
	}

	if cm.CollectionCount() != 0 {
		t.Errorf("expected 0 collections, got %d", cm.CollectionCount())
	}
}

func TestCollectionManager_CreateCollection(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	coll, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	if coll.Name() != "test" {
		t.Errorf("collection name mismatch: got %s, want test", coll.Name())
	}

	if cm.CollectionCount() != 1 {
		t.Errorf("expected 1 collection, got %d", cm.CollectionCount())
	}
}

func TestCollectionManager_CreateDuplicateCollection(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	// Create first collection
	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("first CreateCollection failed: %v", err)
	}

	// Try to create duplicate
	_, err = cm.CreateCollection(ctx, schema)
	if err == nil {
		t.Error("expected error creating duplicate collection")
	}
}

func TestCollectionManager_GetCollection(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	// Get collection
	coll, err := cm.GetCollection("test")
	if err != nil {
		t.Fatalf("GetCollection failed: %v", err)
	}

	if coll.Name() != "test" {
		t.Errorf("collection name mismatch: got %s, want test", coll.Name())
	}

	// Try to get non-existent collection
	_, err = cm.GetCollection("nonexistent")
	if err == nil {
		t.Error("expected error getting non-existent collection")
	}
}

func TestCollectionManager_DeleteCollection(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	// Delete collection
	err = cm.DeleteCollection(ctx, "test")
	if err != nil {
		t.Fatalf("DeleteCollection failed: %v", err)
	}

	if cm.CollectionCount() != 0 {
		t.Errorf("expected 0 collections after delete, got %d", cm.CollectionCount())
	}

	// Try to delete non-existent collection
	err = cm.DeleteCollection(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error deleting non-existent collection")
	}
}

func TestCollectionManager_ListCollections(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	// Create multiple collections
	for i := 1; i <= 3; i++ {
		schema := CollectionSchema{
			Name: fmt.Sprintf("coll%d", i),
			Fields: []VectorField{
				{
					Name:  "embedding",
					Type:  VectorTypeDense,
					Dim:   384,
					Index: IndexConfig{Type: IndexTypeFLAT},
				},
			},
		}
		_, err := cm.CreateCollection(ctx, schema)
		if err != nil {
			t.Fatalf("CreateCollection failed: %v", err)
		}
	}

	names := cm.ListCollections()
	if len(names) != 3 {
		t.Errorf("expected 3 collections, got %d", len(names))
	}

	// Check all names are present
	expectedNames := map[string]bool{
		"coll1": false,
		"coll2": false,
		"coll3": false,
	}

	for _, name := range names {
		if _, ok := expectedNames[name]; ok {
			expectedNames[name] = true
		}
	}

	for name, found := range expectedNames {
		if !found {
			t.Errorf("collection %s not in list", name)
		}
	}
}

func TestCollectionManager_HasCollection(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	if !cm.HasCollection("test") {
		t.Error("HasCollection returned false for existing collection")
	}

	if cm.HasCollection("nonexistent") {
		t.Error("HasCollection returned true for non-existent collection")
	}
}

func TestCollectionManager_GetCollectionInfo(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	schema := CollectionSchema{
		Name:        "test",
		Description: "Test collection",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   384,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
		Metadata: map[string]interface{}{
			"version": "1.0",
		},
	}

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	info, err := cm.GetCollectionInfo("test")
	if err != nil {
		t.Fatalf("GetCollectionInfo failed: %v", err)
	}

	if info.Name != "test" {
		t.Errorf("name mismatch: got %s, want test", info.Name)
	}

	if info.Description != "Test collection" {
		t.Errorf("description mismatch: got %s, want 'Test collection'", info.Description)
	}

	if len(info.Fields) != 1 {
		t.Errorf("expected 1 field, got %d", len(info.Fields))
	}

	if info.DocCount != 0 {
		t.Errorf("expected 0 documents, got %d", info.DocCount)
	}
}

func TestCollectionManager_AddDocument(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	doc := Document{
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 2.0, 3.0, 4.0},
		},
		Metadata: map[string]interface{}{
			"title": "test doc",
		},
	}

	err = cm.AddDocument(ctx, "test", doc)
	if err != nil {
		t.Fatalf("AddDocument failed: %v", err)
	}

	// Verify document was added
	info, _ := cm.GetCollectionInfo("test")
	if info.DocCount != 1 {
		t.Errorf("expected 1 document, got %d", info.DocCount)
	}
}

func TestCollectionManager_BatchAddDocuments(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	docs := make([]Document, 5)
	for i := range docs {
		docs[i] = Document{
			Vectors: map[string]interface{}{
				"embedding": []float32{float32(i), 0.0, 0.0, 0.0},
			},
		}
	}

	err = cm.BatchAddDocuments(ctx, "test", docs)
	if err != nil {
		t.Fatalf("BatchAddDocuments failed: %v", err)
	}

	// Verify documents were added
	info, _ := cm.GetCollectionInfo("test")
	if info.DocCount != 5 {
		t.Errorf("expected 5 documents, got %d", info.DocCount)
	}
}

func TestCollectionManager_SearchCollection(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	// Add documents
	docs := []Document{
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{1.0, 0.0, 0.0, 0.0},
			},
		},
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{0.0, 1.0, 0.0, 0.0},
			},
		},
	}

	err = cm.BatchAddDocuments(ctx, "test", docs)
	if err != nil {
		t.Fatalf("BatchAddDocuments failed: %v", err)
	}

	// Search
	req := SearchRequest{
		CollectionName: "test",
		Queries: map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
		TopK: 2,
	}

	resp, err := cm.SearchCollection(ctx, req)
	if err != nil {
		t.Fatalf("SearchCollection failed: %v", err)
	}

	if len(resp.Documents) == 0 {
		t.Error("expected search results")
	}
}

func TestCollectionManager_DeleteDocument(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	doc := Document{
		ID: 123,
		Vectors: map[string]interface{}{
			"embedding": []float32{1.0, 2.0, 3.0, 4.0},
		},
	}

	err = cm.AddDocument(ctx, "test", doc)
	if err != nil {
		t.Fatalf("AddDocument failed: %v", err)
	}

	// Delete document
	err = cm.DeleteDocument(ctx, "test", 123)
	if err != nil {
		t.Fatalf("DeleteDocument failed: %v", err)
	}

	// Verify document was deleted
	info, _ := cm.GetCollectionInfo("test")
	if info.DocCount != 0 {
		t.Errorf("expected 0 documents after delete, got %d", info.DocCount)
	}
}

func TestCollectionManager_GetDocument(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
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

	err = cm.AddDocument(ctx, "test", doc)
	if err != nil {
		t.Fatalf("AddDocument failed: %v", err)
	}

	// Get document
	retrieved, err := cm.GetDocument("test", 456)
	if err != nil {
		t.Fatalf("GetDocument failed: %v", err)
	}

	if retrieved.ID != 456 {
		t.Errorf("ID mismatch: got %d, want 456", retrieved.ID)
	}

	if title, ok := retrieved.Metadata["title"].(string); !ok || title != "test" {
		t.Errorf("metadata mismatch: got %v", retrieved.Metadata["title"])
	}
}

func TestCollectionManager_GetStats(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	// Create multiple collections with different doc counts
	for i := 1; i <= 3; i++ {
		schema := CollectionSchema{
			Name: fmt.Sprintf("coll%d", i),
			Fields: []VectorField{
				{
					Name:  "embedding",
					Type:  VectorTypeDense,
					Dim:   4,
					Index: IndexConfig{Type: IndexTypeFLAT},
				},
			},
		}

		_, err := cm.CreateCollection(ctx, schema)
		if err != nil {
			t.Fatalf("CreateCollection failed: %v", err)
		}

		// Add i documents to collection i
		for j := 0; j < i; j++ {
			doc := Document{
				Vectors: map[string]interface{}{
					"embedding": []float32{float32(j), 0.0, 0.0, 0.0},
				},
			}
			cm.AddDocument(ctx, fmt.Sprintf("coll%d", i), doc)
		}
	}

	stats := cm.GetStats()

	if stats.CollectionCount != 3 {
		t.Errorf("expected 3 collections, got %d", stats.CollectionCount)
	}

	// Total: 1 + 2 + 3 = 6 documents
	if stats.TotalDocuments != 6 {
		t.Errorf("expected 6 total documents, got %d", stats.TotalDocuments)
	}

	// Check individual collection stats
	for i := 1; i <= 3; i++ {
		name := fmt.Sprintf("coll%d", i)
		collStats, ok := stats.Collections[name]
		if !ok {
			t.Errorf("stats missing for collection %s", name)
			continue
		}

		if collStats.DocCount != i {
			t.Errorf("collection %s: expected %d docs, got %d", name, i, collStats.DocCount)
		}
	}
}

func TestCollectionManager_RenameCollection(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	schema := CollectionSchema{
		Name: "oldname",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   384,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	// Rename
	err = cm.RenameCollection("oldname", "newname")
	if err != nil {
		t.Fatalf("RenameCollection failed: %v", err)
	}

	// Check old name doesn't exist
	if cm.HasCollection("oldname") {
		t.Error("old collection name still exists")
	}

	// Check new name exists
	if !cm.HasCollection("newname") {
		t.Error("new collection name doesn't exist")
	}

	// Check schema was updated
	coll, _ := cm.GetCollection("newname")
	if coll.Name() != "newname" {
		t.Errorf("schema name not updated: got %s, want newname", coll.Name())
	}
}

func TestCollectionManager_DropAllCollections(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	// Create multiple collections
	for i := 1; i <= 3; i++ {
		schema := CollectionSchema{
			Name: fmt.Sprintf("coll%d", i),
			Fields: []VectorField{
				{
					Name:  "embedding",
					Type:  VectorTypeDense,
					Dim:   384,
					Index: IndexConfig{Type: IndexTypeFLAT},
				},
			},
		}
		_, err := cm.CreateCollection(ctx, schema)
		if err != nil {
			t.Fatalf("CreateCollection failed: %v", err)
		}
	}

	// Drop all
	err := cm.DropAllCollections(ctx)
	if err != nil {
		t.Fatalf("DropAllCollections failed: %v", err)
	}

	if cm.CollectionCount() != 0 {
		t.Errorf("expected 0 collections after drop all, got %d", cm.CollectionCount())
	}
}

func TestCollectionManager_HybridSearch(t *testing.T) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	// Create hybrid collection
	schema := CollectionSchema{
		Name: "hybrid",
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

	_, err := cm.CreateCollection(ctx, schema)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	// Add hybrid documents
	vec1, _ := sparse.NewSparseVector([]uint32{0, 1, 2}, []float32{1.0, 2.0, 3.0}, 100)
	vec2, _ := sparse.NewSparseVector([]uint32{0, 5, 10}, []float32{2.0, 1.0, 1.0}, 100)

	docs := []Document{
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{1.0, 0.0, 0.0, 0.0},
				"keywords":  vec1,
			},
		},
		{
			Vectors: map[string]interface{}{
				"embedding": []float32{0.0, 1.0, 0.0, 0.0},
				"keywords":  vec2,
			},
		},
	}

	err = cm.BatchAddDocuments(ctx, "hybrid", docs)
	if err != nil {
		t.Fatalf("BatchAddDocuments failed: %v", err)
	}

	// Hybrid search
	query, _ := sparse.NewSparseVector([]uint32{0, 1}, []float32{1.0, 2.0}, 100)
	req := SearchRequest{
		CollectionName: "hybrid",
		Queries: map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
			"keywords":  query,
		},
		TopK: 2,
		HybridParams: &HybridSearchParams{
			Strategy:    "rrf",
			RRFConstant: 60.0,
		},
	}

	resp, err := cm.SearchCollection(ctx, req)
	if err != nil {
		t.Fatalf("hybrid search failed: %v", err)
	}

	if len(resp.Documents) == 0 {
		t.Error("expected hybrid search results")
	}

	t.Logf("Hybrid search returned %d results", len(resp.Documents))
}

func BenchmarkCollectionManager_AddDocument(b *testing.B) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	schema := CollectionSchema{
		Name: "bench",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   384,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	cm.CreateCollection(ctx, schema)

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
		_ = cm.AddDocument(ctx, "bench", doc)
	}
}

func BenchmarkCollectionManager_SearchCollection(b *testing.B) {
	cm := NewCollectionManager("/tmp/vectordb")
	ctx := context.Background()

	schema := CollectionSchema{
		Name: "bench",
		Fields: []VectorField{
			{
				Name:  "embedding",
				Type:  VectorTypeDense,
				Dim:   384,
				Index: IndexConfig{Type: IndexTypeFLAT},
			},
		},
	}

	cm.CreateCollection(ctx, schema)

	// Add documents
	for i := 0; i < 100; i++ {
		vec := make([]float32, 384)
		for j := range vec {
			vec[j] = float32(i*j) / 100.0
		}

		doc := Document{
			Vectors: map[string]interface{}{
				"embedding": vec,
			},
		}
		cm.AddDocument(ctx, "bench", doc)
	}

	// Prepare query
	query := make([]float32, 384)
	for i := range query {
		query[i] = float32(i) / 100.0
	}

	req := SearchRequest{
		CollectionName: "bench",
		Queries: map[string]interface{}{
			"embedding": query,
		},
		TopK: 10,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cm.SearchCollection(ctx, req)
	}
}
