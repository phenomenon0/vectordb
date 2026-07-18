package collection

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
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

	// durableReadOnly is immutable after a DurableStore is opened. The legacy
	// V2 manager remains available for reads and snapshot preservation, but its
	// mutation methods cannot bypass the canonical tenant journal.
	durableReadOnly bool

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
	if cm.isDurableReadOnly() {
		return nil, ErrCanonicalMutationRequired
	}
	return cm.createCollectionDirect(ctx, schema)
}

func (cm *CollectionManager) createCollectionDirect(ctx context.Context, schema CollectionSchema) (*Collection, error) {
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
	if cm.durableReadOnly {
		coll.setDurableReadOnly()
	}
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
	if cm.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
	return cm.deleteCollectionDirect(ctx, name)
}

func (cm *CollectionManager) deleteCollectionDirect(ctx context.Context, name string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	coll, exists := cm.collections[name]
	if !exists {
		return fmt.Errorf("collection %s not found", name)
	}

	// Cleanup collection resources (indexes, documents)
	coll.closeDirect()

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
	if cm.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
	coll, err := cm.GetCollection(collectionName)
	if err != nil {
		return err
	}

	return coll.Add(ctx, doc)
}

// BatchAddDocuments adds multiple documents to a collection.
func (cm *CollectionManager) BatchAddDocuments(ctx context.Context, collectionName string, docs []Document) error {
	if cm.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
	coll, err := cm.GetCollection(collectionName)
	if err != nil {
		return err
	}

	return coll.BatchAdd(ctx, docs)
}

// BulkAddDense inserts raw dense vectors into a single field of a collection.
func (cm *CollectionManager) BulkAddDense(ctx context.Context, collectionName, fieldName string, ids []uint64, vectors [][]float32) error {
	if cm.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
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
	if cm.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
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
	if cm.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
	cm.mu.Lock()
	defer cm.mu.Unlock()

	coll, exists := cm.collections[name]
	if !exists {
		return fmt.Errorf("collection %s not found", name)
	}

	coll.UpdateMetadata(metadata)
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
	if cm.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
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

	// Update schema name under the collection's own lock to prevent
	// races with concurrent readers (e.g., Search reading coll.schema.Name).
	coll.mu.Lock()
	coll.schema.Name = newName
	coll.mu.Unlock()

	// Rename in map
	delete(cm.collections, oldName)
	cm.collections[newName] = coll

	return nil
}

// DropAllCollections deletes all collections.
//
// WARNING: This is a destructive operation and cannot be undone.
func (cm *CollectionManager) DropAllCollections(ctx context.Context) error {
	if cm.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Cleanup each collection's resources
	for _, coll := range cm.collections {
		coll.closeDirect()
	}

	cm.collections = make(map[string]*Collection)
	return nil
}

func (cm *CollectionManager) setDurableReadOnly() {
	cm.mu.Lock()
	cm.durableReadOnly = true
	for _, coll := range cm.collections {
		coll.setDurableReadOnly()
	}
	cm.mu.Unlock()
}

func (cm *CollectionManager) isDurableReadOnly() bool {
	cm.mu.RLock()
	readOnly := cm.durableReadOnly
	cm.mu.RUnlock()
	return readOnly
}

// ValidateCollection validates a collection schema without creating it.
func (cm *CollectionManager) ValidateCollection(schema CollectionSchema) error {
	return schema.Validate()
}

// persistedCollection holds the serialized state of a single collection.
type persistedCollection struct {
	Schema        CollectionSchema           `json:"schema"`
	Indexes       map[string]json.RawMessage `json:"indexes"`
	SparseIndexes map[string]json.RawMessage `json:"sparse_indexes,omitempty"`
	NextID        uint64                     `json:"next_id"`
	Docs          map[string]*Document       `json:"docs,omitempty"` // string keys for JSON compat
}

// persistedManagerState holds the serialized state of all collections.
type persistedManagerState struct {
	Collections map[string]*persistedCollection `json:"collections"`
}

func (cm *CollectionManager) marshalState() ([]byte, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	state := persistedManagerState{
		Collections: make(map[string]*persistedCollection, len(cm.collections)),
	}

	for name, coll := range cm.collections {
		persisted, err := coll.capturePersistedState()
		if err != nil {
			return nil, fmt.Errorf("export collection %s: %w", name, err)
		}
		state.Collections[name] = persisted
	}

	data, err := json.Marshal(state)
	if err != nil {
		return nil, fmt.Errorf("marshal state: %w", err)
	}
	return data, nil
}

// Save serializes all collections using a same-directory, fsynced atomic
// replacement. The in-memory codec is also used by the unified V2/V3 snapshot.
func (cm *CollectionManager) Save(path string) error {
	data, err := cm.marshalState()
	if err != nil {
		return err
	}
	if err := writeCollectionFileAtomic(path, data, 0o600); err != nil {
		return fmt.Errorf("write state: %w", err)
	}
	return nil
}

func decodeCollectionManagerState(data []byte, storagePath string) (*CollectionManager, error) {
	var state persistedManagerState
	if err := decodeCollectionJSON(data, &state); err != nil {
		return nil, fmt.Errorf("unmarshal state: %w", err)
	}
	if state.Collections == nil {
		return nil, fmt.Errorf("unmarshal state: collections map is missing or null")
	}

	loadedCollections := make(map[string]*Collection, len(state.Collections))
	closeLoaded := func() {
		for _, coll := range loadedCollections {
			coll.Close()
		}
	}

	for name, pc := range state.Collections {
		if pc == nil {
			closeLoaded()
			return nil, fmt.Errorf("collection %s has null state", name)
		}
		if pc.Schema.Name != name {
			closeLoaded()
			return nil, fmt.Errorf("collection key %q does not match schema name %q", name, pc.Schema.Name)
		}
		coll, err := NewCollection(pc.Schema)
		if err != nil {
			closeLoaded()
			return nil, fmt.Errorf("recreate collection %s: %w", name, err)
		}

		// Require an exact persisted index set. Missing fields would otherwise
		// silently produce an empty index while the document map looks populated.
		denseFields := make(map[string]struct{})
		sparseFields := make(map[string]struct{})
		for _, field := range pc.Schema.Fields {
			if field.Type == VectorTypeDense {
				denseFields[field.Name] = struct{}{}
			} else if field.Type == VectorTypeSparse {
				sparseFields[field.Name] = struct{}{}
			}
		}
		if err := validatePersistedIndexSet(name, "dense", denseFields, pc.Indexes); err != nil {
			coll.Close()
			closeLoaded()
			return nil, err
		}
		if err := validatePersistedIndexSet(name, "sparse", sparseFields, pc.SparseIndexes); err != nil {
			coll.Close()
			closeLoaded()
			return nil, err
		}

		indexes := make(map[string][]byte, len(pc.Indexes))
		for field, raw := range pc.Indexes {
			indexes[field] = []byte(raw)
		}
		if err := coll.ImportIndexes(indexes); err != nil {
			coll.Close()
			closeLoaded()
			return nil, fmt.Errorf("import indexes for %s: %w", name, err)
		}

		// Restore sparse indexes
		if pc.SparseIndexes != nil {
			sparseIndexes := make(map[string][]byte, len(pc.SparseIndexes))
			for field, raw := range pc.SparseIndexes {
				sparseIndexes[field] = []byte(raw)
			}
			if err := coll.ImportSparseIndexes(sparseIndexes); err != nil {
				coll.Close()
				closeLoaded()
				return nil, fmt.Errorf("import sparse indexes for %s: %w", name, err)
			}
		}

		// Restore documents
		maxDocumentID := uint64(0)
		if pc.Docs != nil {
			docs := make(map[uint64]*Document, len(pc.Docs))
			for idStr, doc := range pc.Docs {
				id, err := strconv.ParseUint(idStr, 10, 64)
				if err != nil {
					coll.Close()
					closeLoaded()
					return nil, fmt.Errorf("invalid document ID %q in collection %s: %w", idStr, name, err)
				}
				if doc == nil {
					coll.Close()
					closeLoaded()
					return nil, fmt.Errorf("invalid document %q in collection %s", idStr, name)
				}
				if doc.ID != 0 && doc.ID != id {
					coll.Close()
					closeLoaded()
					return nil, fmt.Errorf("document key %d does not match embedded ID %d in collection %s", id, doc.ID, name)
				}
				doc.ID = id
				if err := validatePersistedDocument(doc, &pc.Schema); err != nil {
					coll.Close()
					closeLoaded()
					return nil, fmt.Errorf("invalid document %d in collection %s: %w", id, name, err)
				}
				docs[id] = doc
				if id > maxDocumentID {
					maxDocumentID = id
				}
			}
			coll.ImportDocuments(docs)
		}

		nextID := pc.NextID
		if nextID == 0 {
			nextID = 1
		}
		if nextID <= maxDocumentID {
			nextID = maxDocumentID + 1
		}
		coll.SetNextID(nextID)
		loadedCollections[name] = coll
	}

	return &CollectionManager{
		collections: loadedCollections,
		storagePath: storagePath,
	}, nil
}

// validatePersistedDocument accepts field-partial records produced by the
// supported BulkAddDense API while still rejecting unknown or malformed vector
// fields. Index snapshots remain authoritative for fields not present in the
// lightweight document record.
func validatePersistedDocument(doc *Document, schema *CollectionSchema) error {
	for fieldName, vector := range doc.Vectors {
		field := schema.GetField(fieldName)
		if field == nil {
			return fmt.Errorf("unknown vector field %s", fieldName)
		}
		switch field.Type {
		case VectorTypeDense:
			dense, err := coerceDenseVector(vector)
			if err != nil {
				return fmt.Errorf("field %s: %w", fieldName, err)
			}
			if len(dense) != field.Dim {
				return fmt.Errorf("field %s dimension mismatch: got %d, want %d", fieldName, len(dense), field.Dim)
			}
		case VectorTypeSparse:
			sparseVector, err := coerceSparseVector(vector)
			if err != nil {
				return fmt.Errorf("field %s: %w", fieldName, err)
			}
			if sparseVector.Dim != field.Dim {
				return fmt.Errorf("field %s dimension mismatch: got %d, want %d", fieldName, sparseVector.Dim, field.Dim)
			}
		default:
			return fmt.Errorf("unsupported vector type %s for field %s", field.Type, fieldName)
		}
	}
	return nil
}

func validatePersistedIndexSet(collectionName, kind string, expected map[string]struct{}, actual map[string]json.RawMessage) error {
	if len(expected) != len(actual) {
		return fmt.Errorf("collection %s has incomplete %s index set: got %d fields, want %d", collectionName, kind, len(actual), len(expected))
	}
	for name := range expected {
		raw, ok := actual[name]
		if !ok || len(raw) == 0 || string(raw) == "null" {
			return fmt.Errorf("collection %s is missing %s index %s", collectionName, kind, name)
		}
	}
	for name := range actual {
		if _, ok := expected[name]; !ok {
			return fmt.Errorf("collection %s has unexpected %s index %s", collectionName, kind, name)
		}
	}
	return nil
}

func (cm *CollectionManager) replaceState(loaded *CollectionManager) {
	cm.mu.Lock()
	oldCollections := cm.collections
	cm.collections = loaded.collections
	cm.mu.Unlock()

	for _, coll := range oldCollections {
		coll.closeDirect()
	}
}

// Load deserializes collections transactionally. Missing legacy files remain a
// no-op for compatibility; the unified store layer distinguishes a fresh store
// from a disappeared initialized snapshot.
func (cm *CollectionManager) Load(path string) error {
	if cm.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read state file: %w", err)
	}

	loaded, err := decodeCollectionManagerState(data, cm.storagePath)
	if err != nil {
		return err
	}
	cm.replaceState(loaded)
	return nil
}

func (c *Collection) capturePersistedState() (*persistedCollection, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	rawIndexes := make(map[string]json.RawMessage, len(c.indexes))
	for field, idx := range c.indexes {
		data, err := idx.Export()
		if err != nil {
			return nil, fmt.Errorf("export index %s: %w", field, err)
		}
		rawIndexes[field] = append(json.RawMessage(nil), data...)
	}

	var rawSparseIndexes map[string]json.RawMessage
	if len(c.sparse) > 0 {
		rawSparseIndexes = make(map[string]json.RawMessage, len(c.sparse))
		for field, idx := range c.sparse {
			data, err := idx.Export()
			if err != nil {
				return nil, fmt.Errorf("export sparse index %s: %w", field, err)
			}
			rawSparseIndexes[field] = append(json.RawMessage(nil), data...)
		}
	}

	schemaData, err := json.Marshal(c.schema)
	if err != nil {
		return nil, fmt.Errorf("marshal schema: %w", err)
	}
	var schema CollectionSchema
	if err := json.Unmarshal(schemaData, &schema); err != nil {
		return nil, fmt.Errorf("clone schema: %w", err)
	}

	docMap := make(map[string]*Document, len(c.documents))
	for id, doc := range c.documents {
		if doc == nil {
			return nil, fmt.Errorf("document %d is nil", id)
		}
		docData, err := json.Marshal(doc)
		if err != nil {
			return nil, fmt.Errorf("marshal document %d: %w", id, err)
		}
		var clone Document
		if err := json.Unmarshal(docData, &clone); err != nil {
			return nil, fmt.Errorf("clone document %d: %w", id, err)
		}
		docMap[strconv.FormatUint(id, 10)] = &clone
	}

	return &persistedCollection{
		Schema:        schema,
		Indexes:       rawIndexes,
		SparseIndexes: rawSparseIndexes,
		NextID:        c.nextID,
		Docs:          docMap,
	}, nil
}
