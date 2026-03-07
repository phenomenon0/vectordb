package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/phenomenon0/vectordb/internal/index"
)

// CollectionConfig defines the configuration for a collection
type CollectionConfig struct {
	Name      string                 `json:"name"`
	IndexType string                 `json:"index_type"` // "hnsw", "ivf", "flat", etc.
	Dimension int                    `json:"dimension"`
	Config    map[string]interface{} `json:"config,omitempty"` // Index-specific config
}

// CollectionInfo contains collection metadata and statistics
type CollectionInfo struct {
	Name       string                 `json:"name"`
	IndexType  string                 `json:"index_type"`
	Dimension  int                    `json:"dimension"`
	VectorCount int                   `json:"vector_count"`
	IndexStats index.IndexStats       `json:"index_stats"`
	Config     map[string]interface{} `json:"config,omitempty"`
}

// Collection manager methods on VectorStore

// CreateCollection creates a new collection with the specified index type and configuration
func (vs *VectorStore) CreateCollection(config CollectionConfig) error {
	vs.Lock()
	defer vs.Unlock()

	// Validate collection name
	if config.Name == "" {
		return fmt.Errorf("collection name cannot be empty")
	}

	// Check if collection already exists
	if _, exists := vs.indexes[config.Name]; exists {
		return fmt.Errorf("collection %q already exists", config.Name)
	}

	// Validate dimension matches VectorStore dimension
	if config.Dimension != vs.Dim {
		return fmt.Errorf("collection dimension %d does not match VectorStore dimension %d", config.Dimension, vs.Dim)
	}

	// Default to HNSW if no index type specified
	indexType := config.IndexType
	if indexType == "" {
		indexType = "hnsw"
	}

	// Create index using factory
	idx, err := index.Create(indexType, config.Dimension, config.Config)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	// Add to indexes map
	vs.indexes[config.Name] = idx

	return nil
}

// DeleteCollection removes a collection and its index
func (vs *VectorStore) DeleteCollection(name string) error {
	vs.Lock()
	defer vs.Unlock()

	// Cannot delete default collection
	if name == "default" {
		return fmt.Errorf("cannot delete default collection")
	}

	// Check if collection exists
	if _, exists := vs.indexes[name]; !exists {
		return fmt.Errorf("collection %q does not exist", name)
	}

	// Remove vectors belonging to this collection
	toDelete := make([]uint64, 0)
	for hid, coll := range vs.Coll {
		if coll == name {
			toDelete = append(toDelete, hid)
		}
	}

	// Mark vectors as deleted and clean up metadata
	for _, hid := range toDelete {
		vs.Deleted[hid] = true
		vs.ejectLex(hid)
		vs.ejectMeta(hid)
		delete(vs.Meta, hid)
		delete(vs.Coll, hid)
		delete(vs.TenantID, hid)
	}

	// Remove index
	delete(vs.indexes, name)

	return nil
}

// ListCollections returns information about all collections
func (vs *VectorStore) ListCollections() []CollectionInfo {
	vs.RLock()
	defer vs.RUnlock()

	collections := make([]CollectionInfo, 0, len(vs.indexes))

	for name, idx := range vs.indexes {
		if idx == nil {
			continue
		}

		stats := idx.Stats()

		// Count vectors in this collection
		vectorCount := 0
		for hid, coll := range vs.Coll {
			if coll == name && !vs.Deleted[hid] {
				vectorCount++
			}
		}

		info := CollectionInfo{
			Name:        name,
			IndexType:   stats.Name,
			Dimension:   stats.Dim,
			VectorCount: vectorCount,
			IndexStats:  stats,
			Config:      stats.Extra,
		}

		collections = append(collections, info)
	}

	return collections
}

// GetCollectionInfo returns detailed information about a specific collection
func (vs *VectorStore) GetCollectionInfo(name string) (*CollectionInfo, error) {
	vs.RLock()
	defer vs.RUnlock()

	idx, exists := vs.indexes[name]
	if !exists {
		return nil, fmt.Errorf("collection %q does not exist", name)
	}

	if idx == nil {
		return nil, fmt.Errorf("collection %q has nil index", name)
	}

	stats := idx.Stats()

	// Count vectors in this collection
	vectorCount := 0
	for hid, coll := range vs.Coll {
		if coll == name && !vs.Deleted[hid] {
			vectorCount++
		}
	}

	info := &CollectionInfo{
		Name:        name,
		IndexType:   stats.Name,
		Dimension:   stats.Dim,
		VectorCount: vectorCount,
		IndexStats:  stats,
		Config:      stats.Extra,
	}

	return info, nil
}

// CollectionExists checks if a collection exists
func (vs *VectorStore) CollectionExists(name string) bool {
	vs.RLock()
	defer vs.RUnlock()
	_, exists := vs.indexes[name]
	return exists
}

// GetCollectionIndex returns the index for a specific collection
func (vs *VectorStore) GetCollectionIndex(name string) (index.Index, error) {
	vs.RLock()
	defer vs.RUnlock()

	idx, exists := vs.indexes[name]
	if !exists {
		return nil, fmt.Errorf("collection %q does not exist", name)
	}

	return idx, nil
}

// UpdateCollectionConfig updates the configuration of an existing collection
// Note: This creates a new index and rebuilds it from existing vectors
func (vs *VectorStore) UpdateCollectionConfig(name string, config map[string]interface{}) error {
	vs.Lock()
	defer vs.Unlock()

	// Get existing index
	oldIdx, exists := vs.indexes[name]
	if !exists {
		return fmt.Errorf("collection %q does not exist", name)
	}

	if oldIdx == nil {
		return fmt.Errorf("collection %q has nil index", name)
	}

	// Get index type and dimension from old index
	stats := oldIdx.Stats()
	indexType := stats.Name
	dimension := stats.Dim

	// Create new index with updated config
	newIdx, err := index.Create(indexType, dimension, config)
	if err != nil {
		return fmt.Errorf("failed to create new index: %w", err)
	}

	// Rebuild index with existing vectors from this collection
	ctx := context.Background()
	for i, id := range vs.IDs {
		hid := hashID(id)

		// Skip if not in this collection or deleted
		if vs.Coll[hid] != name || vs.Deleted[hid] {
			continue
		}

		// Get vector
		vec := vs.Data[i*vs.Dim : (i+1)*vs.Dim]

		// Add to new index
		if err := newIdx.Add(ctx, hid, vec); err != nil {
			return fmt.Errorf("failed to rebuild index: %w", err)
		}
	}

	// Replace old index with new index
	vs.indexes[name] = newIdx

	return nil
}

// CollectionStats returns statistics for all collections
type CollectionStats struct {
	TotalCollections int                        `json:"total_collections"`
	TotalVectors     int                        `json:"total_vectors"`
	Collections      map[string]CollectionInfo  `json:"collections"`
}

// GetAllCollectionStats returns comprehensive statistics for all collections
func (vs *VectorStore) GetAllCollectionStats() CollectionStats {
	vs.RLock()
	defer vs.RUnlock()

	stats := CollectionStats{
		TotalCollections: len(vs.indexes),
		TotalVectors:     0,
		Collections:      make(map[string]CollectionInfo),
	}

	for name, idx := range vs.indexes {
		if idx == nil {
			continue
		}

		indexStats := idx.Stats()

		// Count vectors in this collection
		vectorCount := 0
		for hid, coll := range vs.Coll {
			if coll == name && !vs.Deleted[hid] {
				vectorCount++
			}
		}

		stats.TotalVectors += vectorCount

		info := CollectionInfo{
			Name:        name,
			IndexType:   indexStats.Name,
			Dimension:   indexStats.Dim,
			VectorCount: vectorCount,
			IndexStats:  indexStats,
			Config:      indexStats.Extra,
		}

		stats.Collections[name] = info
	}

	return stats
}

// MigrateVectorsToCollection moves vectors from one collection to another
func (vs *VectorStore) MigrateVectorsToCollection(fromCollection, toCollection string, vectorIDs []string) error {
	vs.Lock()
	defer vs.Unlock()

	// Validate collections exist
	fromIdx, fromExists := vs.indexes[fromCollection]
	toIdx, toExists := vs.indexes[toCollection]

	if !fromExists {
		return fmt.Errorf("source collection %q does not exist", fromCollection)
	}
	if !toExists {
		return fmt.Errorf("destination collection %q does not exist", toCollection)
	}

	if fromIdx == nil || toIdx == nil {
		return fmt.Errorf("one or both collections have nil indexes")
	}

	ctx := context.Background()

	for _, id := range vectorIDs {
		hid := hashID(id)

		// Check if vector exists and belongs to source collection
		if vs.Coll[hid] != fromCollection {
			continue
		}

		if vs.Deleted[hid] {
			continue
		}

		// Get vector index
		ix, ok := vs.idToIx[hid]
		if !ok {
			continue
		}

		// Get vector data
		vec := vs.Data[ix*vs.Dim : (ix+1)*vs.Dim]

		// Delete from source index
		if err := fromIdx.Delete(ctx, hid); err != nil {
			return fmt.Errorf("failed to delete from source: %w", err)
		}

		// Add to destination index
		if err := toIdx.Add(ctx, hid, vec); err != nil {
			return fmt.Errorf("failed to add to destination: %w", err)
		}

		// Update collection mapping
		vs.Coll[hid] = toCollection
	}

	return nil
}

// Ensure thread safety for collection operations
var collectionMutex sync.RWMutex

// Thread-safe collection operations wrapper (if needed for external access)
type CollectionManager struct {
	store *VectorStore
}

// NewCollectionManager creates a new collection manager
func NewCollectionManager(store *VectorStore) *CollectionManager {
	return &CollectionManager{store: store}
}

// All methods delegate to VectorStore methods (which already have locking)
func (cm *CollectionManager) Create(config CollectionConfig) error {
	return cm.store.CreateCollection(config)
}

func (cm *CollectionManager) Delete(name string) error {
	return cm.store.DeleteCollection(name)
}

func (cm *CollectionManager) List() []CollectionInfo {
	return cm.store.ListCollections()
}

func (cm *CollectionManager) Get(name string) (*CollectionInfo, error) {
	return cm.store.GetCollectionInfo(name)
}

func (cm *CollectionManager) Update(name string, config map[string]interface{}) error {
	return cm.store.UpdateCollectionConfig(name, config)
}

func (cm *CollectionManager) Stats() CollectionStats {
	return cm.store.GetAllCollectionStats()
}
