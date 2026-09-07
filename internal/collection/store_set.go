package collection

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// ValidTenantID reports whether id is safe to use as a filename: 1-64 chars
// of [A-Za-z0-9_-]. A StoreSet names each tenant's files after its ID, so
// this is a trust-boundary check (path traversal, empty name) that must hold
// regardless of any transport-side validation. Mirrors cmd/deepdata's
// isValidTenantID.
func ValidTenantID(id string) bool {
	if len(id) == 0 || len(id) > 64 {
		return false
	}
	for _, c := range id {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

// ErrInvalidTenantID marks a tenant ID that fails ValidTenantID. It wraps the
// existing generic invalid-argument sentinel so apierror.FromEngine already
// maps it (400 / InvalidArgument) with no new table entry.
var ErrInvalidTenantID = fmt.Errorf("%w: invalid tenant id", ErrInvalidArgument)

// tenantArtifactSuffixes is the fixed set of file names one tenant's base
// path can own. Deliberately not a glob: a bare prefix also matches a
// sibling tenant whose ID extends this one ("acme" vs "acme2"), the same
// hazard BootstrapReplica's comment documents for the unified store.
var tenantArtifactSuffixes = []string{
	".journal", ".journal.frozen", ".usage.json", ".snapshot", ".initialized", ".lock", "-replica",
}

// StoreSet gives every tenant its own durable store rooted at dir/<tenantID>.
// A tenant's files never share a name with another tenant's: acme owns
// acme.lock, acme.snapshot, acme.initialized, acme.journal,
// acme.journal.frozen and acme.usage.json, exactly as one unified store owns
// index.gob.collections.* today.
type StoreSet struct {
	mu     sync.Mutex
	dir    string
	limits StoreLimits
	stores map[string]*DurableStore
	broken map[string]error

	// readOnly refuses to mint a tenant that has never been seen on this set.
	// A following node must not mint local tenants the leader may ship later
	// -- see SetReadOnly.
	readOnly bool

	// reservedCollections is in-flight CreateCollection admission against the
	// dir-wide MaxCollections cap: held while one create is running, so two
	// concurrent creates across different tenants can't both slip in at the
	// limit before either's store.collectionCount reflects the new one.
	reservedCollections int

	// empty is a TenantManager with no durable store attached and no tenant
	// ever created on it. Dispatching an unknown tenant ID to it reuses
	// TenantManager's own no-store-attached fallback methods, so a not-found
	// error or an empty list is byte-identical to what a bare TenantManager
	// already returns today.
	empty *TenantManager
}

// perTenantLimits derives one tenant's own store limits from the dir-wide
// config: exactly one tenant per store, the same collection/quota ceilings
// applied dir-wide.
func perTenantLimits(limits StoreLimits) StoreLimits {
	return StoreLimits{
		MaxTenants:           1,
		MaxCollections:       limits.MaxCollections,
		MaxTenantDocuments:   limits.MaxTenantDocuments,
		MaxTenantBytes:       limits.MaxTenantBytes,
		MaxTenantCollections: limits.MaxTenantCollections,
	}
}

// OpenStoreSet opens dir, eagerly opening every tenant store it already has
// (one <id>.initialized marker per tenant). A tenant whose ID is unsafe as a
// filename, or whose store fails to open, is recorded in broken and does not
// stop the rest of the directory from opening. A dir-wide MaxTenants already
// exceeded by existing tenants is not itself an open error -- it only blocks
// creating another tenant later.
func OpenStoreSet(dir string, limits StoreLimits) (*StoreSet, error) {
	if err := perTenantLimits(limits).validateRequired(); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("create tenant store directory: %w", err)
	}
	matches, err := filepath.Glob(filepath.Join(dir, "*.initialized"))
	if err != nil {
		return nil, fmt.Errorf("scan tenant store directory: %w", err)
	}
	set := &StoreSet{
		dir:    dir,
		limits: limits,
		stores: make(map[string]*DurableStore),
		broken: make(map[string]error),
		empty:  NewTenantManager(dir),
	}
	for _, marker := range matches {
		id := strings.TrimSuffix(filepath.Base(marker), ".initialized")
		if !ValidTenantID(id) {
			set.broken[id] = fmt.Errorf("%w: %q", ErrInvalidTenantID, id)
			continue
		}
		base := filepath.Join(dir, id)
		// ponytail: eager open is about 2 fds per tenant (lock + journal);
		// lazy open + LRU close when 2x tenants nears ulimit -n.
		store, err := OpenDurableStoreWithLimits(base, base, perTenantLimits(limits))
		if err != nil {
			if errors.Is(err, ErrCollectionStoreLocked) {
				// Another process holds this tenant (a serve or replicate on
				// the same dir): a deployment conflict, not a data fault, so
				// refuse the whole dir the way the single store did instead
				// of booting with every tenant faulted.
				for _, opened := range set.stores {
					_ = opened.Abort()
				}
				return nil, fmt.Errorf("tenant %q: %w", id, err)
			}
			set.broken[id] = err
			continue
		}
		set.stores[id] = store
	}
	return set, nil
}

// collectionCountSnapshot is a package-internal accessor StoreSet uses to sum
// collection counts across every tenant's own store for the dir-wide cap.
func (s *DurableStore) collectionCountSnapshot() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.collectionCount
}

func (s *StoreSet) faultedErr(id string, cause error) error {
	return fmt.Errorf("tenant %q: %w", id, fmt.Errorf("%w: %v", ErrDurableStoreFaulted, cause))
}

// lookup returns the open store for id, or nil with no error when id is
// unknown (the caller routes to s.empty), or nil with the wrapped fault
// sentinel when id failed to open at boot. Held only long enough to copy the
// pointer: never call into a store while holding mu.
func (s *StoreSet) lookup(id string) (*DurableStore, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if store, ok := s.stores[id]; ok {
		return store, nil
	}
	if cause, ok := s.broken[id]; ok {
		return nil, s.faultedErr(id, cause)
	}
	return nil, nil
}

// openOrCreate returns the open store for id, opening a fresh one-tenant
// DurableStore the first time id is seen. This and CreateTenant are the only
// two places a tenant store is minted for a brand new tenant.
func (s *StoreSet) openOrCreate(id string) (*DurableStore, error) {
	return s.openOrAdopt(id, nil)
}

// AdoptTenant returns the store for a tenant whose artifacts appeared on disk
// after boot -- a standby's follower seeding it mid-run. If id is already
// open, it is returned unchanged and bind is not called. Otherwise the store
// is opened exactly like openOrCreate's create path, bind runs on it with
// s.mu released (it may be a network round trip to the leader) and BEFORE the
// store is registered, and a bind failure aborts the store and registers
// nothing: no request can reach it, bound or not, until bind has said yes.
//
// bind is a caller-supplied hook, not a call into internal/replication,
// because this package must not import that one (it would be a cycle: that
// package already imports vcollection for DurableStore).
func (s *StoreSet) AdoptTenant(id string, bind func(*DurableStore) error) (*DurableStore, error) {
	return s.openOrAdopt(id, bind)
}

// openOrAdopt is the shared body behind openOrCreate and AdoptTenant: look up
// an already-open or already-broken store, enforce the dir-wide MaxTenants
// cap, and open a fresh one-tenant DurableStore. bind, when non-nil, runs on
// the freshly opened store before it is registered -- AdoptTenant's use for
// honoring a replica marker before any request can reach the store -- and a
// readOnly set refuses to mint a tenant nobody has bound yet (bind == nil is
// exactly the "mint a local tenant" case AdoptTenant is not).
func (s *StoreSet) openOrAdopt(id string, bind func(*DurableStore) error) (*DurableStore, error) {
	if !ValidTenantID(id) {
		return nil, fmt.Errorf("%w: %q", ErrInvalidTenantID, id)
	}
	s.mu.Lock()
	if store, ok := s.stores[id]; ok {
		s.mu.Unlock()
		return store, nil
	}
	if cause, ok := s.broken[id]; ok {
		s.mu.Unlock()
		return nil, s.faultedErr(id, cause)
	}
	if bind == nil && s.readOnly {
		s.mu.Unlock()
		return nil, ErrReplicaReadOnly
	}
	if s.limits.MaxTenants > 0 && len(s.stores)+len(s.broken) >= s.limits.MaxTenants {
		s.mu.Unlock()
		return nil, fmt.Errorf("%w: maximum is %d", ErrTenantLimitExceeded, s.limits.MaxTenants)
	}
	base := filepath.Join(s.dir, id)
	store, err := OpenDurableStoreWithLimits(base, base, perTenantLimits(s.limits))
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	if bind == nil {
		s.stores[id] = store
		s.mu.Unlock()
		return store, nil
	}
	// bind is typically a network round trip to the leader (Follower.Bind ->
	// Status()): release s.mu before calling it, or one tenant mid-seed
	// against a slow/unreachable leader stalls every other tenant on this
	// node -- lookup()'s "never call into a store while holding mu" rule
	// applies to this hook too.
	s.mu.Unlock()
	if err := bind(store); err != nil {
		return nil, errors.Join(err, store.Abort())
	}
	s.mu.Lock()
	s.stores[id] = store
	s.mu.Unlock()
	return store, nil
}

// SetReadOnly toggles whether openOrCreate may mint a brand new tenant store
// dir-wide. A following node must not mint local tenants the leader may ship
// later -- it only ever adopts tenants a bind vouches for. Existing tenants
// keep their own per-store replica flags; this does not touch them.
func (s *StoreSet) SetReadOnly(on bool) {
	s.mu.Lock()
	s.readOnly = on
	s.mu.Unlock()
}

// reserveCollection admits one more collection against the dir-wide
// MaxCollections cap, summing every open store's own collection count plus
// any other in-flight reservation.
// ponytail: sums on every call instead of a running total DeleteCollection
// would also have to maintain; a failed reservation can transiently refuse a
// concurrent create exactly at the cap, but never admits above it.
func (s *StoreSet) reserveCollection() (release func(), err error) {
	if s.limits.MaxCollections <= 0 {
		return func() {}, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	total := s.reservedCollections
	for _, store := range s.stores {
		total += store.collectionCountSnapshot()
	}
	if total >= s.limits.MaxCollections {
		return nil, fmt.Errorf("%w: maximum is %d", ErrCollectionLimitExceeded, s.limits.MaxCollections)
	}
	s.reservedCollections++
	return func() {
		s.mu.Lock()
		s.reservedCollections--
		s.mu.Unlock()
	}, nil
}

// CreateCollection opens or creates tenantID's store, admits against the
// dir-wide MaxCollections cap, then delegates. A failure after the store was
// freshly created leaves it open and registered -- it still holds a tenant
// slot, the same as a record-only tenant does after CreateTenant.
func (s *StoreSet) CreateCollection(ctx context.Context, tenantID string, schema CollectionSchema) (*Collection, error) {
	store, err := s.openOrCreate(tenantID)
	if err != nil {
		return nil, err
	}
	release, err := s.reserveCollection()
	if err != nil {
		return nil, err
	}
	defer release()
	return store.Tenants().CreateCollection(ctx, tenantID, schema)
}

func (s *StoreSet) DeleteCollection(ctx context.Context, tenantID, collectionName string) error {
	store, err := s.lookup(tenantID)
	if err != nil {
		return err
	}
	if store == nil {
		return s.empty.DeleteCollection(ctx, tenantID, collectionName)
	}
	return store.Tenants().DeleteCollection(ctx, tenantID, collectionName)
}

func (s *StoreSet) GetCollectionInfo(tenantID, collectionName string) (*CollectionInfo, error) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return nil, err
	}
	if store == nil {
		return s.empty.GetCollectionInfo(tenantID, collectionName)
	}
	return store.Tenants().GetCollectionInfo(tenantID, collectionName)
}

func (s *StoreSet) ListCollectionInfosChecked(tenantID string) ([]CollectionInfo, error) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return nil, err
	}
	if store == nil {
		return s.empty.ListCollectionInfosChecked(tenantID)
	}
	return store.Tenants().ListCollectionInfosChecked(tenantID)
}

func (s *StoreSet) ListCollectionsChecked(tenantID string) ([]string, error) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return nil, err
	}
	if store == nil {
		return s.empty.ListCollectionsChecked(tenantID)
	}
	return store.Tenants().ListCollectionsChecked(tenantID)
}

func (s *StoreSet) AddDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	store, err := s.lookup(tenantID)
	if err != nil {
		return err
	}
	if store == nil {
		return s.empty.AddDocument(ctx, tenantID, collectionName, doc)
	}
	return store.Tenants().AddDocument(ctx, tenantID, collectionName, doc)
}

func (s *StoreSet) BatchAddDocuments(ctx context.Context, tenantID, collectionName string, docs []Document) error {
	store, err := s.lookup(tenantID)
	if err != nil {
		return err
	}
	if store == nil {
		return s.empty.BatchAddDocuments(ctx, tenantID, collectionName, docs)
	}
	return store.Tenants().BatchAddDocuments(ctx, tenantID, collectionName, docs)
}

func (s *StoreSet) UpsertDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	store, err := s.lookup(tenantID)
	if err != nil {
		return err
	}
	if store == nil {
		return s.empty.UpsertDocument(ctx, tenantID, collectionName, doc)
	}
	return store.Tenants().UpsertDocument(ctx, tenantID, collectionName, doc)
}

func (s *StoreSet) DeleteDocument(ctx context.Context, tenantID, collectionName string, docID uint64) error {
	store, err := s.lookup(tenantID)
	if err != nil {
		return err
	}
	if store == nil {
		return s.empty.DeleteDocument(ctx, tenantID, collectionName, docID)
	}
	return store.Tenants().DeleteDocument(ctx, tenantID, collectionName, docID)
}

func (s *StoreSet) GetDocument(tenantID, collectionName string, docID uint64) (*Document, bool) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return nil, false
	}
	if store == nil {
		return s.empty.GetDocument(tenantID, collectionName, docID)
	}
	return store.Tenants().GetDocument(tenantID, collectionName, docID)
}

func (s *StoreSet) GetDocumentChecked(tenantID, collectionName string, docID uint64) (*Document, error) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return nil, err
	}
	if store == nil {
		return s.empty.GetDocumentChecked(tenantID, collectionName, docID)
	}
	return store.Tenants().GetDocumentChecked(tenantID, collectionName, docID)
}

func (s *StoreSet) SearchCollection(ctx context.Context, tenantID string, req SearchRequest) (*SearchResponse, error) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return nil, err
	}
	if store == nil {
		return s.empty.SearchCollection(ctx, tenantID, req)
	}
	return store.Tenants().SearchCollection(ctx, tenantID, req)
}

func (s *StoreSet) GetTenantStats(tenantID string) (*TenantStats, error) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return nil, err
	}
	if store == nil {
		return s.empty.GetTenantStats(tenantID)
	}
	return store.Tenants().GetTenantStats(tenantID)
}

func (s *StoreSet) GetTenantInfo(tenantID string) (TenantInfo, error) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return TenantInfo{}, err
	}
	if store == nil {
		return s.empty.GetTenantInfo(tenantID)
	}
	return store.Tenants().GetTenantInfo(tenantID)
}

// UpdateTenant upserts an existing tenant's record. Unlike a single unified
// store, PUT on a tenant ID the StoreSet has never seen does not create it --
// CreateCollection and CreateTenant are the only two places a tenant store
// comes into existence -- so an unknown ID gets s.empty's no-store fallback
// (ErrUnsupportedDurableMutation).
// ponytail: narrows the pre-StoreSet "PUT creates" contract; upgrade path is
// routing an unknown ID through openOrCreate here too, if a caller needs it.
func (s *StoreSet) UpdateTenant(ctx context.Context, rec TenantRecord) error {
	store, err := s.lookup(rec.TenantID)
	if err != nil {
		return err
	}
	if store == nil {
		return s.empty.UpdateTenant(ctx, rec)
	}
	return store.Tenants().UpdateTenant(ctx, rec)
}

func (s *StoreSet) CreateTenant(ctx context.Context, rec TenantRecord) error {
	store, err := s.openOrCreate(rec.TenantID)
	if err != nil {
		return err
	}
	return store.Tenants().CreateTenant(ctx, rec)
}

// DeleteTenant removes tenantID's entire store: the store IS the tenant's
// data, so there is no delete_tenant journal record to write -- once the
// files are gone there is nothing left to replay one into.
func (s *StoreSet) DeleteTenant(ctx context.Context, tenantID string) error {
	s.mu.Lock()
	store, ok := s.stores[tenantID]
	if ok {
		delete(s.stores, tenantID)
	}
	cause, broken := s.broken[tenantID]
	s.mu.Unlock()

	if !ok {
		if broken {
			return fmt.Errorf("tenant %q failed to open and was not deleted; its files may be the only copy, remove them by hand: %w", tenantID, cause)
		}
		return fmt.Errorf("%w: %s", ErrTenantNotFound, tenantID)
	}

	// Abort, never Checkpoint: a checkpoint would write a fresh snapshot for a
	// store that is about to be deleted anyway.
	errs := []error{store.Abort()}
	base := filepath.Join(s.dir, tenantID)
	for _, suffix := range tenantArtifactSuffixes {
		if err := os.Remove(base + suffix); err != nil && !errors.Is(err, os.ErrNotExist) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (s *StoreSet) GetTenantRecord(tenantID string) (TenantRecord, bool) {
	store, err := s.lookup(tenantID)
	if err != nil {
		return TenantRecord{}, false
	}
	if store == nil {
		return s.empty.GetTenantRecord(tenantID)
	}
	return store.Tenants().GetTenantRecord(tenantID)
}

// Store returns the open store for id, if any (not broken, not unknown).
func (s *StoreSet) Store(id string) (*DurableStore, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	store, ok := s.stores[id]
	return store, ok
}

// Base returns the file path prefix id's store owns, whether or not it is
// currently open.
func (s *StoreSet) Base(id string) string { return filepath.Join(s.dir, id) }

// Tenants lists every open tenant, sorted.
func (s *StoreSet) Tenants() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.stores))
	for id := range s.stores {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func (s *StoreSet) ListTenantsChecked() ([]string, error) { return s.Tenants(), nil }

func (s *StoreSet) TenantCountChecked() (int, error) { return len(s.Tenants()), nil }

// ListTenantInfos aggregates GetTenantInfo over every open tenant, sorted.
func (s *StoreSet) ListTenantInfos() ([]TenantInfo, error) {
	ids := s.Tenants()
	infos := make([]TenantInfo, 0, len(ids))
	var errs []error
	for _, id := range ids {
		store, ok := s.Store(id)
		if !ok {
			continue
		}
		info, err := store.Tenants().GetTenantInfo(id)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		infos = append(infos, info)
	}
	return infos, errors.Join(errs...)
}

// FaultedTenants returns a copy of every tenant ID that failed to open at
// boot, with the error that faulted it.
func (s *StoreSet) FaultedTenants() map[string]error {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[string]error, len(s.broken))
	for id, err := range s.broken {
		out[id] = err
	}
	return out
}

func (s *StoreSet) openStores() []*DurableStore {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*DurableStore, 0, len(s.stores))
	for _, store := range s.stores {
		out = append(out, store)
	}
	return out
}

// Err joins every open store's own health error. A per-tenant OPEN failure
// (see FaultedTenants) is a fault, not part of this set error.
func (s *StoreSet) Err() error {
	var errs []error
	for _, store := range s.openStores() {
		errs = append(errs, store.Err())
	}
	return errors.Join(errs...)
}

// UsageLoaded reports whether every open store loaded its usage sidecar.
func (s *StoreSet) UsageLoaded() bool {
	for _, store := range s.openStores() {
		if !store.UsageLoaded() {
			return false
		}
	}
	return true
}

// LegacyCollectionCount sums every open store's legacy V2 collection count.
func (s *StoreSet) LegacyCollectionCount() (int, error) {
	var total int
	var errs []error
	for _, store := range s.openStores() {
		n, err := store.LegacyCollectionCount()
		if err != nil {
			errs = append(errs, err)
			continue
		}
		total += n
	}
	return total, errors.Join(errs...)
}

// IsReplica reports whether every open store is a read replica, i.e. the
// whole node is read-only. A StoreSet can mix replica and normal tenants
// (docs/distributed-architecture.md's per-tenant replicate layout), so OR-ing
// across tenants here would make one replicated tenant declare every other,
// fully writable tenant read-only too. See ReplicaTenants for which tenants,
// if any, are replicas on a mixed set.
func (s *StoreSet) IsReplica() bool {
	stores := s.openStores()
	if len(stores) == 0 {
		return false
	}
	for _, store := range stores {
		if !store.IsReplica() {
			return false
		}
	}
	return true
}

// ReplicaTenants returns the sorted IDs of every open tenant whose store is a
// read replica, mirroring FaultedTenants: the per-tenant detail IsReplica's
// single bool can't carry on a mixed StoreSet.
func (s *StoreSet) ReplicaTenants() []string {
	s.mu.Lock()
	stores := make(map[string]*DurableStore, len(s.stores))
	for id, store := range s.stores {
		stores[id] = store
	}
	s.mu.Unlock()
	ids := make([]string, 0)
	for id, store := range stores {
		if store.IsReplica() {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

func (s *StoreSet) Checkpoint() error {
	var errs []error
	for _, store := range s.openStores() {
		errs = append(errs, store.Checkpoint())
	}
	return errors.Join(errs...)
}

// Close closes every open store, continuing past a failing one so the rest
// still release their locks.
func (s *StoreSet) Close() error {
	var errs []error
	for _, store := range s.openStores() {
		errs = append(errs, store.Close())
	}
	return errors.Join(errs...)
}

func (s *StoreSet) Abort() error {
	var errs []error
	for _, store := range s.openStores() {
		errs = append(errs, store.Abort())
	}
	return errors.Join(errs...)
}

// ExportTenantSnapshot writes a fresh, self-contained V2 snapshot for exactly
// one tenant of store -- its live collections plus its record, if any -- to
// dstBase, minting a new StoreID and AppliedLSN 0. OpenStoreSet can then open
// dstBase as that tenant's own store. dstBase's parent directory must already
// exist, the same contract OpenDurableStore has.
func ExportTenantSnapshot(store *DurableStore, tenantID, dstBase string) error {
	store.mu.RLock()
	defer store.mu.RUnlock()
	if err := store.stateErrorLocked(); err != nil {
		return err
	}
	manager := store.tenants.getManager(tenantID)
	if manager == nil {
		return fmt.Errorf("%w: %s", ErrTenantNotFound, tenantID)
	}
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		return err
	}
	exportTenants := NewTenantManager("")
	exportTenants.tenants[tenantID] = manager
	if rec, ok := store.tenants.getTenantRecord(tenantID); ok {
		exportTenants.records[tenantID] = rec
	}
	return saveUnifiedCollectionSnapshot(dstBase, NewCollectionManager(""), exportTenants, metadata)
}
