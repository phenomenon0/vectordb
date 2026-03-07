package main

import (
	"os"
	"testing"
)

func TestVectorStoreIndexIntegration(t *testing.T) {
	// Create temporary directory for test
	tmpDir := t.TempDir()
	indexPath := tmpDir + "/test-index.gob"

	// Create a new vector store
	dim := 128
	store := NewVectorStore(100, dim)

	// Test Add with index abstraction
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(i) / float32(dim)
	}

	id, err := store.Add(vec, "test document", "", nil, "default", "test-tenant")
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	if id == "" {
		t.Error("Add returned empty ID")
	}

	// Test SearchANN with index abstraction
	results := store.SearchANN(vec, 1)
	if len(results) == 0 {
		t.Error("SearchANN returned no results")
	}

	// Test persistence
	if err := store.Save(indexPath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		t.Error("Index file was not created")
	}

	// Test loading
	loadedStore, loaded := loadOrInitStore(indexPath, 100, dim)
	if !loaded {
		t.Error("Failed to load index")
	}

	// Verify loaded data
	if loadedStore.Count != store.Count {
		t.Errorf("Count mismatch after load: expected %d, got %d", store.Count, loadedStore.Count)
	}

	// Test that loaded store has indexes
	if len(loadedStore.indexes) == 0 {
		t.Error("Loaded store has no indexes")
	}

	if _, ok := loadedStore.indexes["default"]; !ok {
		t.Error("Loaded store missing default index")
	}

	// Test search on loaded store
	results2 := loadedStore.SearchANN(vec, 1)
	if len(results2) == 0 {
		t.Error("SearchANN on loaded store returned no results")
	}
}
