package collection

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// CollectionManager manages multiple vector collections.
//
// It provides:
//   - Collection lifecycle: create, get, delete, list
//   - Thread-safe operations
//   - Metadata persistence (future)
type CollectionManager struct {
	collections map[string]*Collection
	mu          sync.RWMutex

	// Storage path for persistence (future use)
	storagePath string
}

// NewCollectionManager creates a new collection manager.
func NewCollectionManager(storagePath string) *CollectionManager {
	return &CollectionManager{
		collections: make(map[string]*Collection),
		storagePath: storagePath,
	}
}

// CreateCollection creates a new collection with the given schema.
//
// Returns an error if a collection with the same name already exists.
func (cm *CollectionManager) CreateCollection(ctx context.Context, schema CollectionSchema) (*Collection, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Check if collection already exists
	if _, exists := cm.collections[schema.Name]; exists {
		return nil, fmt.Errorf("collection %s already exists", schema.Name)
	}

	// Create collection
	coll, err := NewCollection(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}

	cm.collections[schema.Name] = coll
	return coll, nil
}

// GetCollection retrieves a collection by name.
//
// Returns nil if the collection does not exist.
func (cm *CollectionManager) GetCollection(name string) (*Collection, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	coll, exists := cm.collections[name]
	if !exists {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	return coll, nil
}

// DeleteCollection deletes a collection by name.
//
// Returns an error if the collection does not exist.
func (cm *CollectionManager) DeleteCollection(ctx context.Context, name string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	coll, exists := cm.collections[name]
	if !exists {
		return fmt.Errorf("collection %s not found", name)
	}

	// Cleanup collection resources (indexes, documents)
	coll.Close()

	delete(cm.collections, name)
	return nil
}

// ListCollections returns a list of all collection names.
func (cm *CollectionManager) ListCollections() []string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	names := make([]string, 0, len(cm.collections))
	for name := range cm.collections {
		names = append(names, name)
	}
	return names
}

// HasCollection checks if a collection exists.
func (cm *CollectionManager) HasCollection(name string) bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	_, exists := cm.collections[name]
	return exists
}

// CollectionCount returns the number of collections.
func (cm *CollectionManager) CollectionCount() int {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return len(cm.collections)
}

// GetCollectionInfo returns schema information for a collection.
func (cm *CollectionManager) GetCollectionInfo(name string) (*CollectionInfo, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	coll, exists := cm.collections[name]
	if !exists {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	schema := coll.Schema()
	return &CollectionInfo{
		Name:        schema.Name,
		Fields:      schema.Fields,
		Description: schema.Description,
		Metadata:    schema.Metadata,
		DocCount:    coll.Count(),
	}, nil
}

// ListCollectionInfos returns detailed information about all collections.
func (cm *CollectionManager) ListCollectionInfos() []CollectionInfo {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	infos := make([]CollectionInfo, 0, len(cm.collections))
	for _, coll := range cm.collections {
		schema := coll.Schema()
		infos = append(infos, CollectionInfo{
			Name:        schema.Name,
			Fields:      schema.Fields,
			Description: schema.Description,
			Metadata:    schema.Metadata,
			DocCount:    coll.Count(),
		})
	}
	return infos
}

// CollectionInfo contains metadata about a collection.
type CollectionInfo struct {
	Name        string
	Fields      []VectorField
	Description string
	Metadata    map[string]interface{}
	DocCount    int
}

// AddDocument adds a document to a collection.
//
// Convenience method that gets the collection and adds the document.
// Takes *Document so that server-assigned IDs are visible to the caller.
func (cm *CollectionManager) AddDocument(ctx context.Context, collectionName string, doc *Document) error {
	coll, err := cm.GetCollection(collectionName)
	if err != nil {
		return err
	}

	return coll.Add(ctx, doc)
}

// BatchAddDocuments adds multiple documents to a collection.
func (cm *CollectionManager) BatchAddDocuments(ctx context.Context, collectionName string, docs []Document) error {
	coll, err := cm.GetCollection(collectionName)
	if err != nil {
		return err
	}

	return coll.BatchAdd(ctx, docs)
}

// BulkAddDense inserts raw dense vectors into a single field of a collection.
func (cm *CollectionManager) BulkAddDense(ctx context.Context, collectionName, fieldName string, ids []uint64, vectors [][]float32) error {
	coll, err := cm.GetCollection(collectionName)
	if err != nil {
		return err
	}

	return coll.BulkAddDense(ctx, fieldName, ids, vectors)
}

// SearchCollection performs a search on a collection.
func (cm *CollectionManager) SearchCollection(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	coll, err := cm.GetCollection(req.CollectionName)
	if err != nil {
		return nil, err
	}

	return coll.Search(ctx, req)
}

// Recommend performs a recommendation search on a collection.
func (cm *CollectionManager) Recommend(ctx context.Context, req RecommendRequest) (*SearchResponse, error) {
	coll, err := cm.GetCollection(req.CollectionName)
	if err != nil {
		return nil, err
	}
	return coll.Recommend(ctx, req)
}

// Discover performs a context-based discovery search on a collection.
func (cm *CollectionManager) Discover(ctx context.Context, req DiscoverRequest) (*SearchResponse, error) {
	coll, err := cm.GetCollection(req.CollectionName)
	if err != nil {
		return nil, err
	}
	return coll.Discover(ctx, req)
}

// DeleteDocument deletes a document from a collection.
func (cm *CollectionManager) DeleteDocument(ctx context.Context, collectionName string, docID uint64) error {
	coll, err := cm.GetCollection(collectionName)
	if err != nil {
		return err
	}

	return coll.Delete(ctx, docID)
}

// GetDocument retrieves a document from a collection.
func (cm *CollectionManager) GetDocument(collectionName string, docID uint64) (*Document, error) {
	coll, err := cm.GetCollection(collectionName)
	if err != nil {
		return nil, err
	}

	doc, ok := coll.GetDocument(docID)
	if !ok {
		return nil, fmt.Errorf("document %d not found in collection %s", docID, collectionName)
	}

	return doc, nil
}

// UpdateCollectionMetadata updates the metadata for a collection.
func (cm *CollectionManager) UpdateCollectionMetadata(name string, metadata map[string]interface{}) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	coll, exists := cm.collections[name]
	if !exists {
		return fmt.Errorf("collection %s not found", name)
	}

	// Update schema metadata (note: this modifies the schema in place)
	schema := coll.Schema()
	if schema.Metadata == nil {
		schema.Metadata = make(map[string]interface{})
	}
	for k, v := range metadata {
		schema.Metadata[k] = v
	}

	return nil
}

// Stats returns statistics about all collections.
type ManagerStats struct {
	CollectionCount int
	TotalDocuments  int
	Collections     map[string]CollectionStats
}

// CollectionStats contains statistics for a single collection.
type CollectionStats struct {
	Name       string
	DocCount   int
	FieldCount int
}

// GetStats returns statistics about all collections.
func (cm *CollectionManager) GetStats() ManagerStats {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	stats := ManagerStats{
		CollectionCount: len(cm.collections),
		Collections:     make(map[string]CollectionStats),
	}

	totalDocs := 0
	for name, coll := range cm.collections {
		docCount := coll.Count()
		totalDocs += docCount

		schema := coll.Schema()
		stats.Collections[name] = CollectionStats{
			Name:       name,
			DocCount:   docCount,
			FieldCount: len(schema.Fields),
		}
	}

	stats.TotalDocuments = totalDocs
	return stats
}

// RenameCollection renames a collection.
func (cm *CollectionManager) RenameCollection(oldName, newName string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Check old collection exists
	coll, exists := cm.collections[oldName]
	if !exists {
		return fmt.Errorf("collection %s not found", oldName)
	}

	// Check new name is not taken
	if _, exists := cm.collections[newName]; exists {
		return fmt.Errorf("collection %s already exists", newName)
	}

	// Update schema name directly in the collection
	coll.schema.Name = newName

	// Rename in map
	delete(cm.collections, oldName)
	cm.collections[newName] = coll

	return nil
}

// DropAllCollections deletes all collections.
//
// WARNING: This is a destructive operation and cannot be undone.
func (cm *CollectionManager) DropAllCollections(ctx context.Context) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Cleanup each collection's resources
	for _, coll := range cm.collections {
		coll.Close()
	}

	cm.collections = make(map[string]*Collection)
	return nil
}

// ValidateCollection validates a collection schema without creating it.
func (cm *CollectionManager) ValidateCollection(schema CollectionSchema) error {
	return schema.Validate()
}

// persistedCollection holds the serialized state of a single collection.
type persistedCollection struct {
	Schema  CollectionSchema          `json:"schema"`
	Indexes map[string]json.RawMessage `json:"indexes"`
	NextID  uint64                     `json:"next_id"`
	Docs    map[string]*Document       `json:"docs,omitempty"` // string keys for JSON compat
}

// persistedManagerState holds the serialized state of all collections.
type persistedManagerState struct {
	Collections map[string]*persistedCollection `json:"collections"`
}

// Save serializes all collections to a JSON file.
func (cm *CollectionManager) Save(path string) error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	state := persistedManagerState{
		Collections: make(map[string]*persistedCollection, len(cm.collections)),
	}

	for name, coll := range cm.collections {
		indexes, err := coll.ExportIndexes()
		if err != nil {
			return fmt.Errorf("export collection %s: %w", name, err)
		}

		rawIndexes := make(map[string]json.RawMessage, len(indexes))
		for field, data := range indexes {
			rawIndexes[field] = json.RawMessage(data)
		}

		docs := coll.ExportDocuments()
		docMap := make(map[string]*Document, len(docs))
		for id, doc := range docs {
			docMap[fmt.Sprintf("%d", id)] = doc
		}

		state.Collections[name] = &persistedCollection{
			Schema:  coll.Schema(),
			Indexes: rawIndexes,
			NextID:  coll.GetNextID(),
			Docs:    docMap,
		}
	}

	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}

	return os.WriteFile(path, data, 0644)
}

// Load deserializes collections from a JSON file.
func (cm *CollectionManager) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No saved state, nothing to load
		}
		return fmt.Errorf("read state file: %w", err)
	}

	var state persistedManagerState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("unmarshal state: %w", err)
	}

	cm.mu.Lock()
	defer cm.mu.Unlock()

	for name, pc := range state.Collections {
		coll, err := NewCollection(pc.Schema)
		if err != nil {
			return fmt.Errorf("recreate collection %s: %w", name, err)
		}

		// Restore indexes
		indexes := make(map[string][]byte, len(pc.Indexes))
		for field, raw := range pc.Indexes {
			indexes[field] = []byte(raw)
		}
		if err := coll.ImportIndexes(indexes); err != nil {
			return fmt.Errorf("import indexes for %s: %w", name, err)
		}

		// Restore documents
		if pc.Docs != nil {
			docs := make(map[uint64]*Document, len(pc.Docs))
			for idStr, doc := range pc.Docs {
				var id uint64
				fmt.Sscanf(idStr, "%d", &id)
				doc.ID = id
				docs[id] = doc
			}
			coll.ImportDocuments(docs)
		}

		coll.SetNextID(pc.NextID)
		cm.collections[name] = coll
	}

	return nil
}
