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
	// mu protects the tenants and records maps. Individual CollectionManagers
	// have their own locks.
	mu      sync.RWMutex
	tenants map[string]*CollectionManager

	// records holds the administrator-visible tenant record (status, quota) for
	// tenants that have one. A tenant can own collections without a record
	// (legacy/implicit tenants) or hold a record with zero collections
	// (provisioned-but-empty).
	records map[string]TenantRecord

	// storagePath base for persistence (future: each tenant gets storagePath/tenantID/)
	storagePath string

	// durable is set once, before the manager is returned from OpenDurableStore.
	// Mutation methods delegate to it; reads keep their stable pointer.
	durable *DurableStore
}

// NewTenantManager creates a new TenantManager.
func NewTenantManager(storagePath string) *TenantManager {
	return &TenantManager{
		tenants:     make(map[string]*CollectionManager),
		records:     make(map[string]TenantRecord),
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
	if tm.durable != nil {
		mgr.setDurableReadOnly()
	}
	tm.tenants[tenantID] = mgr
	return mgr
}

// getManager returns the CollectionManager for a tenant, or nil if the tenant has no collections.
func (tm *TenantManager) getManager(tenantID string) *CollectionManager {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.tenants[tenantID]
}

// getTenantRecord returns the administrator record for a tenant, if any.
func (tm *TenantManager) getTenantRecord(tenantID string) (TenantRecord, bool) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	rec, ok := tm.records[tenantID]
	return rec, ok
}

// GetTenantRecord returns the raw stored record for a tenant (status and its
// own quota override, not the effective-quota/usage view GetTenantInfo
// returns), so an UpdateTenant caller can merge in only the fields it means
// to change.
func (tm *TenantManager) GetTenantRecord(tenantID string) (TenantRecord, bool) {
	return tm.getTenantRecord(tenantID)
}

// putTenantRecordDirect stores a tenant record, ensuring the tenant also owns
// a (possibly empty) CollectionManager so a record-only tenant is visible to
// every map keyed by tm.tenants (list, count, prune).
func (tm *TenantManager) putTenantRecordDirect(rec TenantRecord) {
	tm.getOrCreateManager(rec.TenantID)
	tm.mu.Lock()
	tm.records[rec.TenantID] = rec
	tm.mu.Unlock()
}

// deleteTenantDirect removes every collection owned by a tenant, then its
// record and manager. Used only behind DurableStore's global mutation lock.
func (tm *TenantManager) deleteTenantDirect(ctx context.Context, tenantID string) error {
	for _, name := range tm.listCollectionsDirect(tenantID) {
		if err := tm.deleteCollectionDirect(ctx, tenantID, name); err != nil {
			return err
		}
	}
	tm.mu.Lock()
	delete(tm.records, tenantID)
	delete(tm.tenants, tenantID)
	tm.mu.Unlock()
	return nil
}

// tenantActive reports whether a tenant holds a MaxTenants admission slot: it
// has a record, or its manager owns at least one collection.
func (tm *TenantManager) tenantActive(tenantID string) bool {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.tenantActiveLocked(tenantID)
}

// tenantActiveLocked is tenantActive for callers already holding tm.mu.
func (tm *TenantManager) tenantActiveLocked(tenantID string) bool {
	if _, ok := tm.records[tenantID]; ok {
		return true
	}
	manager := tm.tenants[tenantID]
	return manager != nil && manager.CollectionCount() > 0
}

// CreateCollection creates a new collection for a tenant.
// Returns an error if a collection with the same name already exists for this
// tenant. Durable stores return a nil Collection on success so the caller cannot
// retain a raw handle outside the store read/health barrier.
func (tm *TenantManager) CreateCollection(ctx context.Context, tenantID string, schema CollectionSchema) (*Collection, error) {
	if store := tm.durableStore(); store != nil {
		return nil, store.createCollection(ctx, tenantID, schema)
	}
	return tm.createCollectionDirect(ctx, tenantID, schema)
}

func (tm *TenantManager) createCollectionDirect(ctx context.Context, tenantID string, schema CollectionSchema) (*Collection, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getOrCreateManager(tenantID)
	return mgr.createCollectionDirect(ctx, schema)
}

// GetCollection retrieves a collection belonging to a specific tenant in the
// legacy in-memory API. Durable stores reject raw handles because later method
// calls would escape the store health/read barrier; use the checked value reads
// (GetCollectionInfo, SearchCollection, and ListCollectionInfosChecked) instead.
func (tm *TenantManager) GetCollection(tenantID, collectionName string) (*Collection, error) {
	if store := tm.durableStore(); store != nil {
		return store.getCollection(tenantID, collectionName)
	}
	return tm.getCollectionDirect(tenantID, collectionName)
}

func (tm *TenantManager) getCollectionDirect(tenantID, collectionName string) (*Collection, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.GetCollection(collectionName)
}

// ListCollections is the compatibility no-error form. Callers must
// use ListCollectionsChecked so a fault cannot be mistaken for an empty list.
func (tm *TenantManager) ListCollections(tenantID string) []string {
	names, _ := tm.ListCollectionsChecked(tenantID)
	return names
}

// ListCollectionsChecked returns sorted collection names under the durable
// store read/health barrier when persistence is attached.
func (tm *TenantManager) ListCollectionsChecked(tenantID string) ([]string, error) {
	if store := tm.durableStore(); store != nil {
		return store.listCollections(tenantID)
	}
	return tm.listCollectionsDirect(tenantID), nil
}

func (tm *TenantManager) listCollectionsDirect(tenantID string) []string {
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
	infos, _ := tm.ListCollectionInfosChecked(tenantID)
	return infos
}

// ListCollectionInfosChecked is the fail-closed canonical read used by server
// surfaces. The historical no-error method remains for compatibility only.
func (tm *TenantManager) ListCollectionInfosChecked(tenantID string) ([]CollectionInfo, error) {
	if store := tm.durableStore(); store != nil {
		return store.listCollectionInfos(tenantID)
	}
	return tm.listCollectionInfosDirect(tenantID), nil
}

func (tm *TenantManager) listCollectionInfosDirect(tenantID string) []CollectionInfo {
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
	if store := tm.durableStore(); store != nil {
		return store.deleteCollection(ctx, tenantID, collectionName)
	}
	return tm.deleteCollectionDirect(ctx, tenantID, collectionName)
}

func (tm *TenantManager) deleteCollectionDirect(ctx context.Context, tenantID, collectionName string) error {
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.deleteCollectionDirect(ctx, collectionName)
}

// GetCollectionInfo returns schema information for a tenant's collection.
func (tm *TenantManager) GetCollectionInfo(tenantID, collectionName string) (*CollectionInfo, error) {
	if store := tm.durableStore(); store != nil {
		return store.getCollectionInfo(tenantID, collectionName)
	}
	return tm.getCollectionInfoDirect(tenantID, collectionName)
}

func (tm *TenantManager) getCollectionInfoDirect(tenantID, collectionName string) (*CollectionInfo, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.GetCollectionInfo(collectionName)
}

// AddDocument adds a document to a tenant's collection.
// Takes *Document so that server-assigned IDs are visible to the caller.
func (tm *TenantManager) AddDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	if store := tm.durableStore(); store != nil {
		return store.addDocument(ctx, tenantID, collectionName, doc)
	}
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.AddDocument(ctx, collectionName, doc)
}

// BatchAddDocuments adds multiple documents to a tenant's collection. In a
// DurableStore all IDs are assigned and the complete batch is journaled before
// any index or document state is changed.
func (tm *TenantManager) BatchAddDocuments(ctx context.Context, tenantID, collectionName string, docs []Document) error {
	if store := tm.durableStore(); store != nil {
		return store.batchAddDocuments(ctx, tenantID, collectionName, docs)
	}
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.BatchAddDocuments(ctx, collectionName, docs)
}

// SearchCollection performs a search on a tenant's collection.
func (tm *TenantManager) SearchCollection(ctx context.Context, tenantID string, req SearchRequest) (*SearchResponse, error) {
	if store := tm.durableStore(); store != nil {
		return store.searchCollection(ctx, tenantID, req)
	}
	return tm.searchCollectionDirect(ctx, tenantID, req)
}

func (tm *TenantManager) searchCollectionDirect(ctx context.Context, tenantID string, req SearchRequest) (*SearchResponse, error) {
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, req.CollectionName, tenantID)
	}
	return mgr.SearchCollection(ctx, req)
}

// DeleteDocument deletes a document from a tenant's collection.
func (tm *TenantManager) DeleteDocument(ctx context.Context, tenantID, collectionName string, docID uint64) error {
	if store := tm.durableStore(); store != nil {
		return store.deleteDocument(ctx, tenantID, collectionName, docID)
	}
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.DeleteDocument(ctx, collectionName, docID)
}

// UpsertDocument inserts or replaces a caller-addressed document in a tenant's
// collection. In a DurableStore the upsert is journaled so a replace survives
// crash and replay; the caller's ID is preserved verbatim.
func (tm *TenantManager) UpsertDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	if store := tm.durableStore(); store != nil {
		return store.upsertDocument(ctx, tenantID, collectionName, doc)
	}
	if tenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.UpsertDocument(ctx, collectionName, doc)
}

// GetDocument returns a single document by caller-supplied ID. Shared-barrier
// durable stores serve the read under an RLock so it cannot observe a
// partially-applied mutation or a store fault.
func (tm *TenantManager) GetDocument(tenantID, collectionName string, docID uint64) (*Document, bool) {
	if store := tm.durableStore(); store != nil {
		return store.getDocument(tenantID, collectionName, docID)
	}
	if tenantID == "" {
		return nil, false
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, false
	}
	doc, err := mgr.GetDocument(collectionName, docID)
	if err != nil {
		return nil, false
	}
	return doc, true
}

// GetDocumentChecked is GetDocument's error-returning form; on a durable
// store a suspended tenant surfaces ErrTenantSuspended instead of a bare
// not-found. Use this over GetDocument when the caller needs to distinguish
// the two.
func (tm *TenantManager) GetDocumentChecked(tenantID, collectionName string, docID uint64) (*Document, error) {
	if store := tm.durableStore(); store != nil {
		return store.getDocumentChecked(tenantID, collectionName, docID)
	}
	if tenantID == "" {
		return nil, fmt.Errorf("tenant ID cannot be empty")
	}
	mgr := tm.getManager(tenantID)
	if mgr == nil {
		return nil, fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	return mgr.GetDocument(collectionName, docID)
}

// ListTenants is the compatibility no-error form. Callers must use
// ListTenantsChecked so a fault cannot be mistaken for an empty tenant set.
func (tm *TenantManager) ListTenants() []string {
	ids, _ := tm.ListTenantsChecked()
	return ids
}

// ListTenantsChecked returns tenant IDs under the durable store read/health
// barrier when persistence is attached.
func (tm *TenantManager) ListTenantsChecked() ([]string, error) {
	if store := tm.durableStore(); store != nil {
		return store.listTenants()
	}
	return tm.listTenantsDirect(), nil
}

func (tm *TenantManager) listTenantsDirect() []string {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	ids := make([]string, 0, len(tm.tenants))
	for id := range tm.tenants {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// TenantCount is the compatibility no-error form. Callers must use
// TenantCountChecked so a fault cannot be mistaken for an empty store.
func (tm *TenantManager) TenantCount() int {
	count, _ := tm.TenantCountChecked()
	return count
}

// TenantCountChecked returns the tenant count under the durable store
// read/health barrier when persistence is attached.
func (tm *TenantManager) TenantCountChecked() (int, error) {
	if store := tm.durableStore(); store != nil {
		return store.tenantCount()
	}
	return tm.tenantCountDirect(), nil
}

func (tm *TenantManager) tenantCountDirect() int {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return len(tm.tenants)
}

// resourceCounts returns active tenants and total collections. Empty tenant
// managers can remain after deleting their final collection, but do not consume
// a tenant admission slot.
func (tm *TenantManager) resourceCounts() (activeTenants, collections int) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	for tenantID, manager := range tm.tenants {
		collections += manager.CollectionCount()
		if tm.tenantActiveLocked(tenantID) {
			activeTenants++
		}
	}
	return activeTenants, collections
}

// pruneEmptyManager is used only behind DurableStore's global mutation lock,
// so no canonical create can retain the manager pointer while it is removed.
// A tenant with a record stays even with zero collections: it still holds a
// MaxTenants admission slot.
func (tm *TenantManager) pruneEmptyManager(tenantID string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	if tm.tenantActiveLocked(tenantID) {
		return
	}
	delete(tm.tenants, tenantID)
}

func (tm *TenantManager) pruneEmptyManagers() {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	for tenantID := range tm.tenants {
		if !tm.tenantActiveLocked(tenantID) {
			delete(tm.tenants, tenantID)
		}
	}
}

// CreateTenant provisions a new tenant record. Tenant records are a
// durability feature: this returns ErrUnsupportedDurableMutation without a
// durable store attached, same as DropTenant does in the other direction.
func (tm *TenantManager) CreateTenant(ctx context.Context, rec TenantRecord) error {
	if store := tm.durableStore(); store != nil {
		return store.createTenant(ctx, rec)
	}
	return ErrUnsupportedDurableMutation
}

// UpdateTenant upserts a tenant record, including status transitions such as
// suspend/reactivate.
func (tm *TenantManager) UpdateTenant(ctx context.Context, rec TenantRecord) error {
	if store := tm.durableStore(); store != nil {
		return store.updateTenant(ctx, rec)
	}
	return ErrUnsupportedDurableMutation
}

// DeleteTenant removes a tenant's record and every collection it owns.
func (tm *TenantManager) DeleteTenant(ctx context.Context, tenantID string) error {
	if store := tm.durableStore(); store != nil {
		return store.deleteTenant(ctx, tenantID)
	}
	return ErrUnsupportedDurableMutation
}

// GetTenantInfo returns one tenant's lifecycle status, effective quota and
// current usage. Quota admission is a durability feature: this returns
// ErrUnsupportedDurableMutation without a durable store attached.
func (tm *TenantManager) GetTenantInfo(tenantID string) (TenantInfo, error) {
	if store := tm.durableStore(); store != nil {
		return store.getTenantInfo(tenantID)
	}
	return TenantInfo{}, ErrUnsupportedDurableMutation
}

// ListTenantInfos returns every tenant's info, sorted by tenant ID.
func (tm *TenantManager) ListTenantInfos() ([]TenantInfo, error) {
	if store := tm.durableStore(); store != nil {
		return store.listTenantInfos()
	}
	return nil, ErrUnsupportedDurableMutation
}

// DropTenant removes all collections for a tenant.
func (tm *TenantManager) DropTenant(ctx context.Context, tenantID string) error {
	if tm.durableStore() != nil {
		return ErrUnsupportedDurableMutation
	}
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

func (tm *TenantManager) durableStore() *DurableStore {
	tm.mu.RLock()
	store := tm.durable
	tm.mu.RUnlock()
	return store
}

func (tm *TenantManager) attachDurableStore(store *DurableStore) {
	tm.mu.Lock()
	tm.durable = store
	for _, manager := range tm.tenants {
		manager.setDurableReadOnly()
	}
	tm.mu.Unlock()
}

func (tm *TenantManager) addPreparedDocumentsDirect(ctx context.Context, tenantID, collectionName string, docs []Document, nextID uint64) error {
	manager := tm.getManager(tenantID)
	if manager == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	coll, err := manager.GetCollection(collectionName)
	if err != nil {
		return err
	}
	return coll.addPreparedDocuments(ctx, docs, nextID)
}

// upsertPreparedDocumentsDirect applies a prepared canonical upsert mutation
// (already validated and ID-placed by prepareCanonicalUpsert) to the live
// collection. It is the durable-store apply seam for the upsert journal op.
func (tm *TenantManager) upsertPreparedDocumentsDirect(ctx context.Context, tenantID, collectionName string, docs []Document, nextID uint64) error {
	manager := tm.getManager(tenantID)
	if manager == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	coll, err := manager.GetCollection(collectionName)
	if err != nil {
		return err
	}
	coll.mu.Lock()
	defer coll.mu.Unlock()
	return coll.upsertPreparedLocked(ctx, docs, nextID)
}

func (tm *TenantManager) deleteDocumentDirect(ctx context.Context, tenantID, collectionName string, docID uint64) error {
	manager := tm.getManager(tenantID)
	if manager == nil {
		return fmt.Errorf("%w: %s for tenant %s", ErrCollectionNotFound, collectionName, tenantID)
	}
	coll, err := manager.GetCollection(collectionName)
	if err != nil {
		return err
	}
	return coll.deleteDocumentDirect(ctx, docID)
}

// TenantStats contains statistics for a single tenant.
type TenantStats struct {
	TenantID        string                     `json:"tenant_id"`
	CollectionCount int                        `json:"collection_count"`
	TotalDocuments  int                        `json:"total_documents"`
	Collections     map[string]CollectionStats `json:"collections"`
}

// GetTenantStats returns statistics for a specific tenant.
func (tm *TenantManager) GetTenantStats(tenantID string) (*TenantStats, error) {
	if store := tm.durableStore(); store != nil {
		return store.getTenantStats(tenantID)
	}
	return tm.getTenantStatsDirect(tenantID)
}

func (tm *TenantManager) getTenantStatsDirect(tenantID string) (*TenantStats, error) {
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

func (tm *TenantManager) marshalState() ([]byte, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	state := persistedTenantState{
		Tenants: make(map[string]json.RawMessage, len(tm.tenants)),
	}

	for tenantID, mgr := range tm.tenants {
		data, err := mgr.marshalState()
		if err != nil {
			return nil, fmt.Errorf("save tenant %s: %w", tenantID, err)
		}
		state.Tenants[tenantID] = json.RawMessage(data)
	}

	data, err := json.Marshal(state)
	if err != nil {
		return nil, fmt.Errorf("marshal tenant state: %w", err)
	}
	return data, nil
}

// Save always writes an explicit state, including the empty tenant set, using
// an atomic durable replacement. Tenant identifiers never become filenames.
func (tm *TenantManager) Save(path string) error {
	data, err := tm.marshalState()
	if err != nil {
		return err
	}
	if err := writeCollectionFileAtomic(path, data, 0o600); err != nil {
		return fmt.Errorf("write tenant state: %w", err)
	}
	return nil
}

func decodeTenantManagerState(data []byte, storagePath string) (*TenantManager, error) {
	var state persistedTenantState
	if err := decodeCollectionJSON(data, &state); err != nil {
		return nil, fmt.Errorf("unmarshal tenant state: %w", err)
	}
	if state.Tenants == nil {
		return nil, fmt.Errorf("unmarshal tenant state: tenants map is missing or null")
	}

	loaded := make(map[string]*CollectionManager, len(state.Tenants))
	closeLoaded := func() {
		for _, mgr := range loaded {
			mgr.closeAll()
		}
	}
	for tenantID, raw := range state.Tenants {
		if tenantID == "" {
			closeLoaded()
			return nil, fmt.Errorf("tenant ID cannot be empty")
		}
		if len(raw) == 0 || string(raw) == "null" {
			closeLoaded()
			return nil, fmt.Errorf("tenant %s has null state", tenantID)
		}
		tenantPath := ""
		if storagePath != "" {
			tenantPath = storagePath + "/" + tenantID
		}
		mgr, err := decodeCollectionManagerState(raw, tenantPath)
		if err != nil {
			closeLoaded()
			return nil, fmt.Errorf("load tenant %s: %w", tenantID, err)
		}
		loaded[tenantID] = mgr
	}

	return &TenantManager{
		tenants:     loaded,
		records:     make(map[string]TenantRecord),
		storagePath: storagePath,
	}, nil
}

func (cm *CollectionManager) closeAll() {
	cm.mu.Lock()
	collections := cm.collections
	cm.collections = make(map[string]*Collection)
	cm.mu.Unlock()
	for _, coll := range collections {
		coll.closeDirect()
	}
}

func (tm *TenantManager) replaceState(loaded *TenantManager) {
	tm.mu.Lock()
	oldTenants := tm.tenants
	tm.tenants = loaded.tenants
	tm.mu.Unlock()

	for _, mgr := range oldTenants {
		mgr.closeAll()
	}
}

func (tm *TenantManager) closeAll() {
	tm.mu.Lock()
	tenants := tm.tenants
	tm.tenants = make(map[string]*CollectionManager)
	tm.mu.Unlock()
	for _, mgr := range tenants {
		mgr.closeAll()
	}
}

// Load decodes every tenant before changing live state. Missing legacy files
// remain a no-op; an existing explicit empty state clears prior tenants.
func (tm *TenantManager) Load(path string) error {
	if tm.durableStore() != nil {
		return ErrUnsupportedDurableMutation
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read tenant state file: %w", err)
	}

	loaded, err := decodeTenantManagerState(data, tm.storagePath)
	if err != nil {
		return err
	}
	tm.replaceState(loaded)
	return nil
}
