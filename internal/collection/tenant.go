package collection

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
)

// TenantManager provides collection-level multi-tenancy by namespacing
// collections per tenant. Each tenant has its own isolated set of collections:
// tenant "A" collection "docs" is completely separate from tenant "B" collection "docs".
//
// This is a lightweight namespace layer on top of CollectionManager — it does NOT
// handle auth/RBAC (that's the security package's job).
type TenantManager struct {
	// mu protects the tenants map. Individual CollectionManagers have their own locks.
	mu      sync.RWMutex
	tenants map[string]*CollectionManager

	// storagePath base for persistence (future: each tenant gets storagePath/tenantID/)
	storagePath string
}

// NewTenantManager creates a new TenantManager.
func NewTenantManager(storagePath string) *TenantManager {
	return &TenantManager{
		tenants:     make(map[string]*CollectionManager),
		storagePath: storagePath,
	}
}

// tenantStoragePath returns the storage path for a specific tenant.
func (tm *TenantManager) tenantStoragePath(tenantID string) string {
	if tm.storagePath == "" {
		return ""
	}
	return tm.storagePath + "/" + tenantID
}

// getOrCreateManager returns the CollectionManager for a tenant, creating it if needed.
func (tm *TenantManager) getOrCreateManager(tenantID string) *CollectionManager {
	// Fast path: read lock
	tm.mu.RLock()
	mgr, exists := tm.tenants[tenantID]
	tm.mu.RUnlock()
	if exists {
		return mgr
	}

	// Slow path: write lock, double-check
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if mgr, exists = tm.tenants[tenantID]; exists {
		return mgr
	}

	mgr = NewCollectionManager(tm.tenantStoragePath(tenantID))
	tm.tenants[tenantID] = mgr
	return mgr
}

// getManager returns the CollectionManager for a tenant, or nil if the tenant has no collections.
func (tm *TenantManager) getManager(tenantID string) *CollectionManager {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.tenants[tenantID]
}

// CreateCollection creates a new collection for a tenant.
// Returns an error if a collection with the same name already exists for this tenant.
func (tm *TenantManager) CreateCollection(ctx context.Context, tenantID string, schema CollectionSchema) (*Collection, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getOrCreateManager(tenantID)
	return mgr.CreateCollection(ctx, schema)
}

// GetCollection retrieves a collection belonging to a specific tenant.
// Returns an error if the tenant has no such collection — this prevents cross-tenant access.
func (tm *TenantManager) GetCollection(tenantID, collectionName string) (*Collection, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("collection %s not found for tenant %s", collectionName, tenantID)
	}
	return mgr.GetCollection(collectionName)
}

// ListCollections returns the names of all collections belonging to a tenant.
// Returns an empty slice if the tenant has no collections.
func (tm *TenantManager) ListCollections(tenantID string) []string {
	if tenantID == "" {
		return nil
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return []string{}
	}
	names := mgr.ListCollections()
	sort.Strings(names)
	return names
}

// ListCollectionInfos returns detailed info for all collections belonging to a tenant.
func (tm *TenantManager) ListCollectionInfos(tenantID string) []CollectionInfo {
	if tenantID == "" {
		return nil
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return []CollectionInfo{}
	}
	return mgr.ListCollectionInfos()
}

// DeleteCollection deletes a collection belonging to a specific tenant.
func (tm *TenantManager) DeleteCollection(ctx context.Context, tenantID, collectionName string) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("collection %s not found for tenant %s", collectionName, tenantID)
	}
	return mgr.DeleteCollection(ctx, collectionName)
}

// GetCollectionInfo returns schema information for a tenant's collection.
func (tm *TenantManager) GetCollectionInfo(tenantID, collectionName string) (*CollectionInfo, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("collection %s not found for tenant %s", collectionName, tenantID)
	}
	return mgr.GetCollectionInfo(collectionName)
}

// AddDocument adds a document to a tenant's collection.
// Takes *Document so that server-assigned IDs are visible to the caller.
func (tm *TenantManager) AddDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("collection %s not found for tenant %s", collectionName, tenantID)
	}
	return mgr.AddDocument(ctx, collectionName, doc)
}

// SearchCollection performs a search on a tenant's collection.
func (tm *TenantManager) SearchCollection(ctx context.Context, tenantID string, req SearchRequest) (*SearchResponse, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("collection %s not found for tenant %s", req.CollectionName, tenantID)
	}
	return mgr.SearchCollection(ctx, req)
}

// DeleteDocument deletes a document from a tenant's collection.
func (tm *TenantManager) DeleteDocument(ctx context.Context, tenantID, collectionName string, docID uint64) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("collection %s not found for tenant %s", collectionName, tenantID)
	}
	return mgr.DeleteDocument(ctx, collectionName, docID)
}

// ListTenants returns a sorted list of all tenant IDs that have collections.
func (tm *TenantManager) ListTenants() []string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	ids := make([]string, 0, len(tm.tenants))
	for id := range tm.tenants {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// TenantCount returns the number of tenants.
func (tm *TenantManager) TenantCount() int {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return len(tm.tenants)
}

// DropTenant removes all collections for a tenant.
func (tm *TenantManager) DropTenant(ctx context.Context, tenantID string) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()

	mgr, exists := tm.tenants[tenantID]
	if !exists {
		return fmt.Errorf("tenant %s not found", tenantID)
	}

	if err := mgr.DropAllCollections(ctx); err != nil {
		return fmt.Errorf("failed to drop tenant %s: %w", tenantID, err)
	}

	delete(tm.tenants, tenantID)
	return nil
}

// TenantStats contains statistics for a single tenant.
type TenantStats struct {
	TenantID        string                   `json:"tenant_id"`
	CollectionCount int                      `json:"collection_count"`
	TotalDocuments  int                      `json:"total_documents"`
	Collections     map[string]CollectionStats `json:"collections"`
}

// GetTenantStats returns statistics for a specific tenant.
func (tm *TenantManager) GetTenantStats(tenantID string) (*TenantStats, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("tenant %s not found", tenantID)
	}

	mgrStats := mgr.GetStats()
	return &TenantStats{
		TenantID:        tenantID,
		CollectionCount: mgrStats.CollectionCount,
		TotalDocuments:  mgrStats.TotalDocuments,
		Collections:     mgrStats.Collections,
	}, nil
}

// persistedTenantState holds all tenant manager states.
type persistedTenantState struct {
	Tenants map[string]json.RawMessage `json:"tenants"`
}

// Save serializes all tenant collection managers to a JSON file.
func (tm *TenantManager) Save(path string) error {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if len(tm.tenants) == 0 {
		return nil // Nothing to save
	}

	state := persistedTenantState{
		Tenants: make(map[string]json.RawMessage, len(tm.tenants)),
	}

	for tenantID, mgr := range tm.tenants {
		// Save each manager to a temporary path and read the bytes
		tmpPath := path + "." + tenantID + ".tmp"
		if err := mgr.Save(tmpPath); err != nil {
			return fmt.Errorf("save tenant %s: %w", tenantID, err)
		}
		data, err := os.ReadFile(tmpPath)
		if err != nil {
			return fmt.Errorf("read tenant %s state: %w", tenantID, err)
		}
		os.Remove(tmpPath)
		state.Tenants[tenantID] = json.RawMessage(data)
	}

	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal tenant state: %w", err)
	}

	return os.WriteFile(path, data, 0644)
}

// Load deserializes tenant collection managers from a JSON file.
func (tm *TenantManager) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read tenant state file: %w", err)
	}

	var state persistedTenantState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("unmarshal tenant state: %w", err)
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()

	for tenantID, raw := range state.Tenants {
		// Write tenant data to temp file and use manager Load
		tmpPath := path + "." + tenantID + ".tmp"
		if err := os.WriteFile(tmpPath, []byte(raw), 0644); err != nil {
			return fmt.Errorf("write tenant %s temp: %w", tenantID, err)
		}
		mgr := NewCollectionManager(tm.tenantStoragePath(tenantID))
		if err := mgr.Load(tmpPath); err != nil {
			os.Remove(tmpPath)
			return fmt.Errorf("load tenant %s: %w", tenantID, err)
		}
		os.Remove(tmpPath)
		tm.tenants[tenantID] = mgr
	}

	return nil
}
