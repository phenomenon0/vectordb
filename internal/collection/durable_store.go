package collection

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/phenomenon0/vectordb/internal/logging"
)

const (
	durableCollectionMutationVersionV1 = uint16(1)
	durableCollectionMutationVersion   = uint16(2)

	mutationCreateCollection = "create_collection"
	mutationDeleteCollection = "delete_collection"
	mutationInsertDocument   = "insert_document"
	mutationBatchInsert      = "batch_insert_documents"
	mutationDeleteDocument   = "delete_document"
	mutationUpsertDocument   = "upsert_document"
	mutationCreateTenant     = "create_tenant"
	mutationUpdateTenant     = "update_tenant"
	mutationDeleteTenant     = "delete_tenant"
)

var (
	ErrDurableStoreClosed         = errors.New("durable collection store is closed")
	ErrDurableStoreFaulted        = errors.New("durable collection store is faulted")
	ErrDurableCollectionHandle    = errors.New("raw collection handles are unavailable for durable stores")
	ErrCanonicalMutationRequired  = errors.New("persistent collections must be mutated through TenantManager")
	ErrUnsupportedDurableMutation = errors.New("mutation is outside the release-candidate durable contract")
)

type durableMutationEnvelope struct {
	Version uint16          `json:"version"`
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

type durableCreateCollection struct {
	TenantID string           `json:"tenant_id"`
	Schema   CollectionSchema `json:"schema"`
}

type durableCollectionTarget struct {
	TenantID       string `json:"tenant_id"`
	CollectionName string `json:"collection_name"`
}

type durableInsertDocument struct {
	TenantID       string   `json:"tenant_id"`
	CollectionName string   `json:"collection_name"`
	Document       Document `json:"document"`
}

type durableBatchInsert struct {
	TenantID       string     `json:"tenant_id"`
	CollectionName string     `json:"collection_name"`
	Documents      []Document `json:"documents"`
}

type durableDeleteDocument struct {
	TenantID       string `json:"tenant_id"`
	CollectionName string `json:"collection_name"`
	DocumentID     uint64 `json:"document_id"`
}

type durableUpsertDocument struct {
	TenantID       string   `json:"tenant_id"`
	CollectionName string   `json:"collection_name"`
	Document       Document `json:"document"`
}

type durableTenantTarget struct {
	TenantID string `json:"tenant_id"`
}

type canonicalMutation struct {
	version        uint16
	typeName       string
	tenantID       string
	collectionName string
	schema         CollectionSchema
	documents      []Document
	documentID     uint64
	nextID         uint64
	record         TenantRecord
}

// DurableStore is the single Linux persistence boundary for the canonical,
// tenant-aware collection API. Its mutex deliberately spans WAL append, apply,
// snapshot capture, journal rotation, and coverage-checked cleanup.
type DurableStore struct {
	mu sync.RWMutex

	basePath string
	limits   StoreLimits
	// Resource counters are derived once after snapshot/WAL recovery and then
	// mutated only under mu alongside the corresponding journaled operation.
	activeTenants   int
	collectionCount int
	manager         *CollectionManager
	tenants         *TenantManager
	metadata        CollectionSnapshotMetadata
	journal         *collectionJournal
	lock            *collectionStoreLock

	fault  error
	closed bool

	// usageLoaded records whether the class B usage sidecar was read and
	// imported at open. It is written once during open and then only read.
	usageLoaded bool

	// apply is a per-store test seam. Production always points at
	// applyMutationDirect; an error after append permanently faults the store.
	apply func(context.Context, canonicalMutation) error

	// appended wakes journal followers. Zero value is usable and costs nothing
	// until a follower waits on it.
	appended journalNotifier

	// replica marks the store a read replica: local writes are refused and it
	// advances only through ApplyReplicated. Deliberately not persisted — which
	// leader a store follows is configuration, re-supplied by MakeReplica on
	// every open, so it never has to survive a snapshot format change.
	replica  bool
	leaderID [16]byte
}

// OpenDurableStore opens or initializes a durable unified collection store.
// Persistent startup is intentionally rejected by the build-tagged lock stub
// on non-Linux platforms for this release candidate.
func OpenDurableStore(basePath, storagePath string) (*DurableStore, error) {
	return openDurableStore(basePath, storagePath, StoreLimits{})
}

// OpenDurableStoreWithLimits opens a durable store with immutable admission
// limits for new collection creates. Existing acknowledged state is always
// replayed, even when it is already above a newly configured limit.
func OpenDurableStoreWithLimits(basePath, storagePath string, limits StoreLimits) (*DurableStore, error) {
	if err := limits.validateRequired(); err != nil {
		return nil, err
	}
	return openDurableStore(basePath, storagePath, limits)
}

func openDurableStore(basePath, storagePath string, limits StoreLimits) (*DurableStore, error) {
	if basePath == "" {
		return nil, errors.New("durable collection store base path cannot be empty")
	}
	basePath = filepath.Clean(basePath)
	lock, err := acquireCollectionStoreLock(basePath + ".lock")
	if err != nil {
		return nil, err
	}
	fail := func(manager *CollectionManager, tenants *TenantManager, cause error) (*DurableStore, error) {
		if manager != nil {
			manager.closeAll()
		}
		if tenants != nil {
			tenants.closeAll()
		}
		return nil, errors.Join(cause, lock.release())
	}

	manager, tenants, metadata, err := openUnifiedCollectionSnapshot(basePath, storagePath)
	if err != nil {
		return fail(nil, nil, err)
	}
	journal, replayPlan, err := openCollectionJournalValidated(
		basePath+".journal",
		basePath+".journal.frozen",
		metadata.StoreID,
		metadata.AppliedLSN,
		func(record collectionJournalRecord) error {
			if _, err := decodeDurableMutation(record.Payload); err != nil {
				return fmt.Errorf("decode collection mutation at LSN %d: %w", record.LSN, err)
			}
			return nil
		},
	)
	if err != nil {
		return fail(manager, tenants, fmt.Errorf("open durable collection journal: %w", err))
	}

	store := &DurableStore{
		basePath: basePath,
		limits:   limits,
		manager:  manager,
		tenants:  tenants,
		metadata: metadata,
		journal:  journal,
		lock:     lock,
	}
	store.apply = store.applyMutationDirect
	err = journal.streamReplay(replayPlan, func(record collectionJournalRecord) error {
		mutation, err := decodeDurableMutation(record.Payload)
		if err != nil {
			return fmt.Errorf("decode collection mutation at LSN %d: %w", record.LSN, err)
		}
		if err := store.prepareReplayMutation(&mutation); err != nil {
			return fmt.Errorf("validate collection mutation at LSN %d: %w", record.LSN, err)
		}
		if err := store.applyMutationDirect(context.Background(), mutation); err != nil {
			return fmt.Errorf("replay collection mutation at LSN %d: %w", record.LSN, err)
		}
		store.metadata.AppliedLSN = record.LSN
		return nil
	})
	if err != nil {
		return fail(manager, tenants, fmt.Errorf("stream durable collection journal replay: %w", err))
	}
	// Older snapshots could retain tenant managers after their final collection
	// was deleted. They carry no tenant data and must not grow the tenant map or
	// consume persistence on every subsequent checkpoint.
	store.tenants.pruneEmptyManagers()
	store.activeTenants, store.collectionCount = store.tenants.resourceCounts()
	// The usage sidecar is class B: it is restored after the canonical state
	// it annotates, and a failure here can never reach the caller.
	store.loadUsageSidecar()
	// Keep the fully validated journal after recovery. Snapshot serialization is
	// a separate bounded-memory track; invoking the current snapshot writer here
	// would reintroduce an unbounded startup allocation before callers can choose
	// when to checkpoint. Explicit Checkpoint and graceful Close still commit a
	// snapshot before coverage-checked cleanup.

	manager.setDurableReadOnly()
	tenants.attachDurableStore(store)
	return store, nil
}

func (s *DurableStore) Tenants() *TenantManager { return s.tenants }

// LegacyCollectionCount is a checked startup/migration inspection. Live
// request paths never receive the underlying V2 CollectionManager.
func (s *DurableStore) LegacyCollectionCount() (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.stateErrorLocked(); err != nil {
		return 0, err
	}
	return s.manager.CollectionCount(), nil
}

func (s *DurableStore) Metadata() CollectionSnapshotMetadata {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.metadata
}

func (s *DurableStore) Err() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.stateErrorLocked()
}

// UsageLoaded reports whether this store opened with the usage hints it was
// entitled to. It is false only when a sidecar existed and was discarded —
// the one case that also logs an error. A store with no sidecar at all had
// nothing to lose and reports true, so the signal never accuses a fresh
// deployment of losing data. Transports project it as status
// signals.usage.loaded.
func (s *DurableStore) UsageLoaded() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.usageLoaded
}

func (s *DurableStore) stateErrorLocked() error {
	if s.closed {
		return ErrDurableStoreClosed
	}
	if s.fault != nil {
		return fmt.Errorf("%w: %v", ErrDurableStoreFaulted, s.fault)
	}
	return nil
}

// tenantStateErrorLocked is stateErrorLocked plus a suspended-tenant check.
// Used only by data-plane methods (search/read/write); lifecycle, list, and
// stats methods stay on stateErrorLocked so a suspended tenant remains
// visible and administrable.
func (s *DurableStore) tenantStateErrorLocked(tenantID string) error {
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	if rec, ok := s.tenants.getTenantRecord(tenantID); ok && rec.Status == TenantStatusSuspended {
		return fmt.Errorf("%w: %s", ErrTenantSuspended, tenantID)
	}
	return nil
}

func (s *DurableStore) latchFaultLocked(err error) error {
	if s.fault == nil {
		s.fault = err
	}
	return fmt.Errorf("%w: %v", ErrDurableStoreFaulted, s.fault)
}

// Reads hold a shared store barrier for their full operation. A
// mutation/checkpoint has the exclusive side, so a read cannot pass a health
// check and then observe an append/apply fault or a partially applied batch.
func (s *DurableStore) getCollection(_, _ string) (*Collection, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.stateErrorLocked(); err != nil {
		return nil, err
	}
	return nil, ErrDurableCollectionHandle
}

func (s *DurableStore) getCollectionInfo(tenantID, collectionName string) (*CollectionInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return nil, err
	}
	return s.tenants.getCollectionInfoDirect(tenantID, collectionName)
}

func (s *DurableStore) listCollectionInfos(tenantID string) ([]CollectionInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return nil, err
	}
	return s.tenants.listCollectionInfosDirect(tenantID), nil
}

func (s *DurableStore) listCollections(tenantID string) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return nil, err
	}
	return s.tenants.listCollectionsDirect(tenantID), nil
}

func (s *DurableStore) listTenants() ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.stateErrorLocked(); err != nil {
		return nil, err
	}
	return s.tenants.listTenantsDirect(), nil
}

func (s *DurableStore) tenantCount() (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.stateErrorLocked(); err != nil {
		return 0, err
	}
	return s.tenants.tenantCountDirect(), nil
}

func (s *DurableStore) searchCollection(ctx context.Context, tenantID string, req SearchRequest) (*SearchResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return nil, err
	}
	return s.tenants.searchCollectionDirect(ctx, tenantID, req)
}

func (s *DurableStore) getTenantStats(tenantID string) (*TenantStats, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.stateErrorLocked(); err != nil {
		return nil, err
	}
	return s.tenants.getTenantStatsDirect(tenantID)
}

// getDocument serves a canonical single-document read under the store's shared
// barrier, so it cannot observe a partially applied mutation or a store fault.
func (s *DurableStore) getDocument(tenantID, collectionName string, docID uint64) (*Document, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.tenantStateErrorLocked(tenantID) != nil {
		return nil, false
	}
	coll, err := s.tenants.getCollectionDirect(tenantID, collectionName)
	if err != nil {
		return nil, false
	}
	return coll.GetDocument(docID)
}

// getDocumentChecked is getDocument's error-returning form so a suspended
// tenant's rejection is distinguishable from an ordinary not-found.
func (s *DurableStore) getDocumentChecked(tenantID, collectionName string, docID uint64) (*Document, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return nil, err
	}
	coll, err := s.tenants.getCollectionDirect(tenantID, collectionName)
	if err != nil {
		return nil, err
	}
	doc, ok := coll.GetDocument(docID)
	if !ok {
		return nil, fmt.Errorf("%w: %d in collection %s", ErrDocumentNotFound, docID, collectionName)
	}
	return doc, nil
}

func (s *DurableStore) createCollection(ctx context.Context, tenantID string, schema CollectionSchema) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return err
	}
	mutation := canonicalMutation{typeName: mutationCreateCollection, tenantID: tenantID, collectionName: schema.Name, schema: schema}
	if err := s.prepareCreateMutation(&mutation); err != nil {
		return err
	}
	if err := validateCanonicalSchemaResourceBounds(&mutation.schema); err != nil {
		return err
	}
	newTenant, err := s.checkCreateLimitsLocked(mutation.tenantID)
	if err != nil {
		return err
	}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return err
	}
	s.collectionCount++
	if newTenant {
		s.activeTenants++
	}
	return nil
}

func (s *DurableStore) checkCreateLimitsLocked(tenantID string) (bool, error) {
	if s.limits.MaxCollections > 0 && s.collectionCount >= s.limits.MaxCollections {
		return false, fmt.Errorf("%w: maximum is %d", ErrCollectionLimitExceeded, s.limits.MaxCollections)
	}
	newTenant := !s.tenants.tenantActive(tenantID)
	if newTenant {
		if err := s.checkMaxTenantsLocked(); err != nil {
			return false, err
		}
	}
	return newTenant, nil
}

// checkMaxTenantsLocked enforces the tenant admission limit for a caller that
// already determined the tenant is new. Shared by collection creation and
// tenant record creation so the two admission paths never drift apart.
func (s *DurableStore) checkMaxTenantsLocked() error {
	if s.limits.MaxTenants > 0 && s.activeTenants >= s.limits.MaxTenants {
		return fmt.Errorf("%w: maximum is %d", ErrTenantLimitExceeded, s.limits.MaxTenants)
	}
	return nil
}

func (s *DurableStore) addDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	if doc == nil {
		return errors.New("document cannot be nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return err
	}
	mutation, err := s.prepareDocumentsMutation(mutationInsertDocument, tenantID, collectionName, []Document{*doc})
	if err != nil {
		return err
	}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return err
	}
	doc.ID = mutation.documents[0].ID
	return nil
}

func (s *DurableStore) batchAddDocuments(ctx context.Context, tenantID, collectionName string, docs []Document) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return err
	}
	mutation, err := s.prepareDocumentsMutation(mutationBatchInsert, tenantID, collectionName, docs)
	if err != nil {
		return err
	}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return err
	}
	for i := range docs {
		docs[i].ID = mutation.documents[i].ID
	}
	return nil
}

func (s *DurableStore) upsertDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	if doc == nil {
		return errors.New("document cannot be nil")
	}
	if doc.ID == 0 {
		return errors.New("upsert requires a caller-supplied document ID")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return err
	}
	mutation, err := s.prepareUpsertMutation(tenantID, collectionName, *doc)
	if err != nil {
		return err
	}
	return s.appendApplyLocked(ctx, mutation)
}

func (s *DurableStore) prepareUpsertMutation(tenantID, collectionName string, doc Document) (canonicalMutation, error) {
	mutation := canonicalMutation{typeName: mutationUpsertDocument, tenantID: tenantID, collectionName: collectionName}
	if err := s.prepareCollectionTarget(mutation); err != nil {
		return mutation, err
	}
	coll, _ := s.tenants.getCollectionDirect(tenantID, collectionName)
	normalized, nextID, err := coll.prepareCanonicalUpsert([]Document{doc})
	if err != nil {
		return mutation, err
	}
	mutation.documents = normalized
	mutation.nextID = nextID
	return mutation, nil
}

func (s *DurableStore) deleteCollection(ctx context.Context, tenantID, collectionName string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return err
	}
	mutation := canonicalMutation{typeName: mutationDeleteCollection, tenantID: tenantID, collectionName: collectionName}
	if err := s.prepareCollectionTarget(mutation); err != nil {
		return err
	}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return err
	}
	s.collectionCount--
	// A tenant record keeps the admission slot even after its last collection
	// is gone, so only drop the count when nothing else holds the tenant active.
	if !s.tenants.tenantActive(tenantID) {
		s.activeTenants--
	}
	return nil
}

func (s *DurableStore) createTenant(ctx context.Context, rec TenantRecord) error {
	if err := rec.validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	if _, exists := s.tenants.getTenantRecord(rec.TenantID); exists {
		return fmt.Errorf("%w: %s", ErrTenantExists, rec.TenantID)
	}
	newTenant := !s.tenants.tenantActive(rec.TenantID)
	if newTenant {
		if err := s.checkMaxTenantsLocked(); err != nil {
			return err
		}
	}
	mutation := canonicalMutation{typeName: mutationCreateTenant, tenantID: rec.TenantID, record: rec}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return err
	}
	if newTenant {
		s.activeTenants++
	}
	return nil
}

// updateTenant upserts a tenant record: PUT on an unknown tenant creates it,
// so replay after a restart or from a replica needs only this one path.
func (s *DurableStore) updateTenant(ctx context.Context, rec TenantRecord) error {
	if err := rec.validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	newTenant := !s.tenants.tenantActive(rec.TenantID)
	if newTenant {
		if err := s.checkMaxTenantsLocked(); err != nil {
			return err
		}
	}
	mutation := canonicalMutation{typeName: mutationUpdateTenant, tenantID: rec.TenantID, record: rec}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return err
	}
	if newTenant {
		s.activeTenants++
	}
	return nil
}

func (s *DurableStore) deleteTenant(ctx context.Context, tenantID string) error {
	if tenantID == "" {
		return errors.New("tenant ID cannot be empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	if !s.tenants.tenantActive(tenantID) {
		return fmt.Errorf("%w: %s", ErrTenantNotFound, tenantID)
	}
	mutation := canonicalMutation{typeName: mutationDeleteTenant, tenantID: tenantID}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return err
	}
	// ponytail: recount instead of tracking how many collections the deleted
	// tenant owned; DeleteTenant is a rare admin action, so O(tenants) here is
	// cheaper than duplicating deleteCollectionDirect's per-collection math.
	s.activeTenants, s.collectionCount = s.tenants.resourceCounts()
	return nil
}

func (s *DurableStore) deleteDocument(ctx context.Context, tenantID, collectionName string, documentID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.tenantStateErrorLocked(tenantID); err != nil {
		return err
	}
	mutation := canonicalMutation{typeName: mutationDeleteDocument, tenantID: tenantID, collectionName: collectionName, documentID: documentID}
	if err := s.prepareDeleteDocument(mutation); err != nil {
		return err
	}
	return s.appendApplyLocked(ctx, mutation)
}

// ephemeralDocumentMutationLocked reports whether the mutation writes
// documents into a collection created with durability "ephemeral". Only the
// four document mutations qualify: create and delete-collection stay class A,
// so the collection's existence survives a restart while its documents do not.
// Caller holds mu.
func (s *DurableStore) ephemeralDocumentMutationLocked(m canonicalMutation) bool {
	switch m.typeName {
	case mutationInsertDocument, mutationBatchInsert, mutationUpsertDocument, mutationDeleteDocument:
	default:
		return false
	}
	coll, err := s.tenants.getCollectionDirect(m.tenantID, m.collectionName)
	if err != nil || coll == nil {
		return false
	}
	return coll.isEphemeral()
}

func (s *DurableStore) appendApplyLocked(ctx context.Context, mutation canonicalMutation) error {
	// Every locally originated write funnels through here, so this one guard
	// makes a read replica read-only on all six of them at once.
	if s.replica {
		return ErrReplicaReadOnly
	}
	// Durability class E (ADR 0009): an ephemeral collection's documents are
	// memory only, so the mutation is applied under the same store mutex but
	// is neither encoded nor appended, costs no fsync, and does not advance
	// AppliedLSN. Nothing durable was written, so a failed apply cannot split
	// journal and memory and must not latch a store-wide fault.
	if s.ephemeralDocumentMutationLocked(mutation) {
		if err := ctx.Err(); err != nil {
			return err
		}
		return s.apply(context.WithoutCancel(ctx), mutation)
	}
	payload, err := encodeDurableMutation(mutation)
	if err != nil {
		return err
	}
	return s.appendPayloadApplyLocked(ctx, payload, mutation)
}

// appendPayloadApplyLocked commits an already-encoded mutation: durable append
// first, then the mandatory in-memory apply. Shared with the replica applier,
// which appends the leader's exact payload bytes rather than re-encoding, so
// both journals carry identical records and a replica can be streamed from in
// turn.
func (s *DurableStore) appendPayloadApplyLocked(ctx context.Context, payload []byte, mutation canonicalMutation) error {
	// A request canceled while it was waiting for the store mutex has not
	// crossed the commit point and must not be appended. Once append begins,
	// however, cancellation can no longer be allowed to split durable and
	// in-memory state.
	if err := ctx.Err(); err != nil {
		return err
	}
	record, err := s.journal.append(payload)
	if err != nil {
		if journalFault := s.journal.writeFault(); journalFault != nil {
			return s.latchFaultLocked(fmt.Errorf("journal append failed: %w", journalFault))
		}
		return err
	}
	// Once append returns, the mutation is durably committed regardless of the
	// request lifetime. Applying with the caller's cancelable context would let a
	// disconnected client strand an acknowledged journal record between append
	// and in-memory state, faulting the entire store until restart. Preserve any
	// context values while detaching cancellation for this mandatory apply step.
	applyCtx := context.WithoutCancel(ctx)
	if err := s.apply(applyCtx, mutation); err != nil {
		return s.latchFaultLocked(fmt.Errorf("apply LSN %d after durable append: %w", record.LSN, err))
	}
	s.metadata.AppliedLSN = record.LSN
	// The record is durable and applied, so followers may read it now.
	s.appended.notify()
	return nil
}

func (s *DurableStore) prepareCreateMutation(m *canonicalMutation) error {
	return s.prepareCreateMutationWithValidator(m, validateCanonicalSchema)
}

func (s *DurableStore) prepareCreateMutationV1(m *canonicalMutation) error {
	return s.prepareCreateMutationWithValidator(m, validateDurableSchemaV1)
}

func (s *DurableStore) prepareCreateMutationWithValidator(
	m *canonicalMutation,
	validate func(*CollectionSchema) error,
) error {
	if m.tenantID == "" {
		return errors.New("tenant ID cannot be empty")
	}
	clone, err := cloneCanonicalSchema(m.schema)
	if err != nil {
		return err
	}
	if err := validate(&clone); err != nil {
		return err
	}
	if m.collectionName != "" && m.collectionName != clone.Name {
		return errors.New("collection name does not match schema")
	}
	m.collectionName = clone.Name
	m.schema = clone
	if manager := s.tenants.getManager(m.tenantID); manager != nil && manager.HasCollection(clone.Name) {
		return fmt.Errorf("%w: %s", ErrCollectionExists, clone.Name)
	}
	return nil
}

// validateDurableSchemaV1 freezes the admission rules used by mutation version
// 1. New canonical URL/index restrictions belong to v2 and must not make an
// already-acknowledged v1 create record unreplayable after an upgrade.
func validateDurableSchemaV1(schema *CollectionSchema) error {
	if err := schema.Validate(); err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	for _, field := range schema.Fields {
		switch field.Type {
		case VectorTypeDense, VectorTypeSparse:
			// The v1 admission set and the canonical one coincide because the
			// vocabulary itself is frozen (ADR 0001): IndexTypes is the whole
			// RC scope, so reading it here cannot widen or narrow v1.
			if err := validateFieldIndexType(field); err != nil {
				return err
			}
		default:
			return fmt.Errorf("field %s uses unsupported vector type %s", field.Name, field.Type)
		}
	}
	return nil
}

func validateCanonicalSchema(schema *CollectionSchema) error {
	if err := schema.Validate(); err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	if !IsValidCanonicalIdentifier(schema.Name) {
		return errors.New("collection name must be 1-64 alphanumeric/hyphen/underscore characters")
	}
	switch schema.Durability {
	case "", DurabilityDurable, DurabilityEphemeral:
	default:
		return fmt.Errorf("%w: durability must be %q or %q, got %q",
			ErrInvalidArgument, DurabilityDurable, DurabilityEphemeral, schema.Durability)
	}
	for _, field := range schema.Fields {
		switch field.Type {
		case VectorTypeDense, VectorTypeSparse:
			if err := validateFieldIndexType(field); err != nil {
				return err
			}
		default:
			return fmt.Errorf("field %s uses unsupported vector type %s", field.Name, field.Type)
		}
		if err := validateCanonicalIndexParams(field); err != nil {
			return err
		}
	}
	return nil
}

func validateCanonicalIndexParams(field VectorField) error {
	allowed := map[string]bool{}
	switch field.Index.Type {
	case IndexTypeHNSW:
		allowed = map[string]bool{
			"m": true, "ml": true, "ef_search": true,
			"ef_construction": true, "prenormalize": true,
			"segments": true,
		}
	case IndexTypeFLAT:
		allowed = map[string]bool{"metric": true}
	case IndexTypeInverted:
		allowed = map[string]bool{"k1": true, "b": true}
	}

	keys := make([]string, 0, len(field.Index.Params))
	for key := range field.Index.Params {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if !allowed[key] {
			return fmt.Errorf(
				"field %s index parameter %q is outside the canonical release contract",
				field.Name,
				key,
			)
		}
	}

	switch field.Index.Type {
	case IndexTypeHNSW:
		if err := validateCanonicalIntegerParam(field, "m", 2, 100); err != nil {
			return err
		}
		if err := validateCanonicalIntegerParam(field, "ef_search", 1, 1_000_000); err != nil {
			return err
		}
		if err := validateCanonicalIntegerParam(field, "ef_construction", 1, 1_000_000); err != nil {
			return err
		}
		if err := validateCanonicalIntegerParam(field, "segments", 1, 64); err != nil {
			return err
		}
		if value, ok, err := canonicalNumericParam(field, "ml"); err != nil {
			return err
		} else if ok && (value <= 0 || value > 10) {
			return fmt.Errorf("field %s index parameter %q must be in (0, 10]", field.Name, "ml")
		}
		if value, ok := field.Index.Params["prenormalize"]; ok {
			if _, valid := value.(bool); !valid {
				return fmt.Errorf("field %s index parameter %q must be a boolean", field.Name, "prenormalize")
			}
		}
	case IndexTypeFLAT:
		if value, ok := field.Index.Params["metric"]; ok {
			metric, valid := value.(string)
			if !valid || (metric != "cosine" && metric != "euclidean") {
				return fmt.Errorf("field %s index parameter %q must be cosine or euclidean", field.Name, "metric")
			}
		}
	case IndexTypeInverted:
		if value, ok, err := canonicalNumericParam(field, "k1"); err != nil {
			return err
		} else if ok && (value <= 0 || value > 100) {
			return fmt.Errorf("field %s index parameter %q must be in (0, 100]", field.Name, "k1")
		}
		if value, ok, err := canonicalNumericParam(field, "b"); err != nil {
			return err
		} else if ok && (value < 0 || value > 1) {
			return fmt.Errorf("field %s index parameter %q must be in [0, 1]", field.Name, "b")
		}
	}
	return nil
}

func canonicalNumericParam(field VectorField, key string) (float64, bool, error) {
	value, ok := field.Index.Params[key]
	if !ok {
		return 0, false, nil
	}
	number, valid := value.(float64)
	if !valid || math.IsNaN(number) || math.IsInf(number, 0) {
		return 0, false, fmt.Errorf("field %s index parameter %q must be a finite number", field.Name, key)
	}
	return number, true, nil
}

func validateCanonicalIntegerParam(field VectorField, key string, minimum, maximum int) error {
	value, ok, err := canonicalNumericParam(field, key)
	if err != nil || !ok {
		return err
	}
	if value != math.Trunc(value) || value < float64(minimum) || value > float64(maximum) {
		return fmt.Errorf(
			"field %s index parameter %q must be an integer in [%d, %d]",
			field.Name,
			key,
			minimum,
			maximum,
		)
	}
	return nil
}

func cloneCanonicalSchema(schema CollectionSchema) (CollectionSchema, error) {
	data, err := json.Marshal(schema)
	if err != nil {
		return CollectionSchema{}, fmt.Errorf("marshal schema: %w", err)
	}
	var clone CollectionSchema
	if err := decodeCollectionJSON(data, &clone); err != nil {
		return CollectionSchema{}, fmt.Errorf("clone schema: %w", err)
	}
	return clone, nil
}

func (s *DurableStore) prepareCollectionTarget(m canonicalMutation) error {
	if m.tenantID == "" {
		return errors.New("tenant ID cannot be empty")
	}
	if m.collectionName == "" {
		return errors.New("collection name cannot be empty")
	}
	_, err := s.tenants.getCollectionDirect(m.tenantID, m.collectionName)
	return err
}

func (s *DurableStore) prepareDocumentsMutation(typeName, tenantID, collectionName string, docs []Document) (canonicalMutation, error) {
	return s.prepareDocumentsMutationWithAdmission(typeName, tenantID, collectionName, docs, true)
}

func (s *DurableStore) prepareDocumentsMutationWithAdmission(
	typeName, tenantID, collectionName string,
	docs []Document,
	enforceCurrentAdmission bool,
) (canonicalMutation, error) {
	mutation := canonicalMutation{typeName: typeName, tenantID: tenantID, collectionName: collectionName}
	if len(docs) == 0 {
		return mutation, errors.New("documents cannot be empty")
	}
	if enforceCurrentAdmission && len(docs) > MaxBatchDocuments {
		return mutation, fmt.Errorf("document batch exceeds maximum of %d", MaxBatchDocuments)
	}
	if err := s.prepareCollectionTarget(mutation); err != nil {
		return mutation, err
	}
	coll, _ := s.tenants.getCollectionDirect(tenantID, collectionName)
	normalized, nextID, err := coll.prepareCanonicalDocuments(docs)
	if err != nil {
		return mutation, err
	}
	mutation.documents = normalized
	mutation.nextID = nextID
	return mutation, nil
}

func (s *DurableStore) prepareDeleteDocument(m canonicalMutation) error {
	if m.documentID == 0 {
		return errors.New("document ID cannot be zero")
	}
	if err := s.prepareCollectionTarget(m); err != nil {
		return err
	}
	coll, _ := s.tenants.getCollectionDirect(m.tenantID, m.collectionName)
	if _, ok := coll.GetDocument(m.documentID); !ok {
		return fmt.Errorf("%w: %d in collection %s", ErrDocumentNotFound, m.documentID, m.collectionName)
	}
	return nil
}

// normalizeReplayDocuments canonicalizes only documents freshly decoded from
// the durable journal. Those values are private to the recovery mutation, so
// replacing JSON-generic vector trees cannot alias a caller. Ordinary live
// preparation deliberately bypasses this method and continues to preserve
// caller-provided Go types while making its defensive deep copy.
func (s *DurableStore) normalizeReplayDocuments(m *canonicalMutation) error {
	if err := s.prepareCollectionTarget(*m); err != nil {
		return err
	}
	coll, _ := s.tenants.getCollectionDirect(m.tenantID, m.collectionName)
	coll.mu.RLock()
	defer coll.mu.RUnlock()
	for i := range m.documents {
		if err := normalizeDocumentVectorTypes(&m.documents[i], &coll.schema); err != nil {
			return fmt.Errorf("document %d vector normalization failed: %w", i, err)
		}
	}
	return nil
}

func (s *DurableStore) prepareReplayMutation(m *canonicalMutation) error {
	switch m.typeName {
	case mutationCreateCollection:
		if m.version == durableCollectionMutationVersionV1 {
			return s.prepareCreateMutationV1(m)
		}
		return s.prepareCreateMutation(m)
	case mutationDeleteCollection:
		return s.prepareCollectionTarget(*m)
	case mutationInsertDocument, mutationBatchInsert:
		if err := s.normalizeReplayDocuments(m); err != nil {
			return err
		}
		prepared, err := s.prepareDocumentsMutationWithAdmission(
			m.typeName,
			m.tenantID,
			m.collectionName,
			m.documents,
			m.version != durableCollectionMutationVersionV1,
		)
		if err != nil {
			return err
		}
		prepared.version = m.version
		*m = prepared
		return nil
	case mutationDeleteDocument:
		return s.prepareDeleteDocument(*m)
	case mutationUpsertDocument:
		if len(m.documents) != 1 {
			return errors.New("upsert mutation must contain exactly one document")
		}
		if err := s.normalizeReplayDocuments(m); err != nil {
			return err
		}
		prepared, err := s.prepareUpsertMutation(m.tenantID, m.collectionName, m.documents[0])
		if err != nil {
			return err
		}
		prepared.version = m.version
		*m = prepared
		return nil
	case mutationCreateTenant, mutationUpdateTenant, mutationDeleteTenant:
		// decodeDurableMutation already fully decoded and validated the
		// record (or, for delete, just needs the tenant id already set).
		return nil
	default:
		return fmt.Errorf("unknown mutation type %q", m.typeName)
	}
}

func (s *DurableStore) applyMutationDirect(ctx context.Context, m canonicalMutation) error {
	switch m.typeName {
	case mutationCreateCollection:
		_, err := s.tenants.createCollectionDirect(ctx, m.tenantID, m.schema)
		return err
	case mutationDeleteCollection:
		if err := s.tenants.deleteCollectionDirect(ctx, m.tenantID, m.collectionName); err != nil {
			return err
		}
		s.tenants.pruneEmptyManager(m.tenantID)
		return nil
	case mutationInsertDocument, mutationBatchInsert:
		return s.tenants.addPreparedDocumentsDirect(ctx, m.tenantID, m.collectionName, m.documents, m.nextID)
	case mutationUpsertDocument:
		return s.tenants.upsertPreparedDocumentsDirect(ctx, m.tenantID, m.collectionName, m.documents, m.nextID)
	case mutationDeleteDocument:
		return s.tenants.deleteDocumentDirect(ctx, m.tenantID, m.collectionName, m.documentID)
	case mutationCreateTenant:
		if _, exists := s.tenants.getTenantRecord(m.tenantID); exists {
			return fmt.Errorf("%w: %s", ErrTenantExists, m.tenantID)
		}
		s.tenants.putTenantRecordDirect(m.record)
		return nil
	case mutationUpdateTenant:
		s.tenants.putTenantRecordDirect(m.record)
		return nil
	case mutationDeleteTenant:
		if !s.tenants.tenantActive(m.tenantID) {
			return fmt.Errorf("%w: %s", ErrTenantNotFound, m.tenantID)
		}
		return s.tenants.deleteTenantDirect(ctx, m.tenantID)
	default:
		return fmt.Errorf("unknown mutation type %q", m.typeName)
	}
}

// Checkpoint commits all applied mutations to one unified snapshot, then and
// only then removes journal artifacts whose complete LSN range is covered.
func (s *DurableStore) Checkpoint() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	return s.commitSnapshotAndCleanupLocked(false)
}

func (s *DurableStore) commitSnapshotAndCleanupLocked(recovery bool) error {
	frozenExists, err := pathExists(s.journal.frozenPath)
	if err != nil {
		return fmt.Errorf("inspect frozen collection journal: %w", err)
	}
	if !frozenExists && !recovery {
		if err := s.journal.rotate(); err != nil {
			if journalFault := s.journal.writeFault(); journalFault != nil {
				return s.latchFaultLocked(fmt.Errorf("journal rotation failed: %w", journalFault))
			}
			return fmt.Errorf("rotate collection journal: %w", err)
		}
	}
	if err := saveUnifiedCollectionSnapshot(s.basePath, s.manager, s.tenants, s.metadata); err != nil {
		return err
	}
	if err := s.verifyJournalCoverageLocked(); err != nil {
		return s.latchFaultLocked(err)
	}
	if err := s.journal.cleanupAll(); err != nil {
		return fmt.Errorf("clean covered collection journals: %w", err)
	}
	s.writeUsageSidecarLocked()
	return nil
}

func usageSidecarPath(basePath string) string { return basePath + ".usage.json" }

// usageSidecarDocument is the whole store's class B usage state: one
// version for the file, then the tracker entries of every collection that
// has any, keyed by the tenant and collection they belong to.
type usageSidecarDocument struct {
	Version     int                      `json:"version"`
	Collections []usageSidecarCollection `json:"collections"`
}

type usageSidecarCollection struct {
	TenantID   string        `json:"tenant_id"`
	Collection string        `json:"collection"`
	Entries    []UsageRecord `json:"entries"`
}

// writeUsageSidecarLocked commits the usage sidecar beside the snapshot
// that was just written. Class B: every failure is logged and swallowed,
// because a ranking hint that cannot be persisted must not fail a
// checkpoint that already committed the canonical state. Caller holds mu.
//
// ponytail: whole-file rewrite of every tracked entry on every snapshot.
// Each collection is capped at usageEntryCap entries, so the cost is
// bounded but linear in tracked documents; past ~250k entries in a store,
// move the records into the snapshot format as their own framed section
// (snapshot.go:276) instead of growing a second full-file write.
func (s *DurableStore) writeUsageSidecarLocked() {
	doc := usageSidecarDocument{Version: usageDocumentVersion}
	for _, tenantID := range s.tenants.listTenantsDirect() {
		for _, name := range s.tenants.listCollectionsDirect(tenantID) {
			coll, err := s.tenants.getCollectionDirect(tenantID, name)
			if err != nil {
				continue
			}
			entries := coll.usage.Export().Entries
			if len(entries) == 0 {
				continue
			}
			doc.Collections = append(doc.Collections, usageSidecarCollection{
				TenantID:   tenantID,
				Collection: name,
				Entries:    entries,
			})
		}
	}
	path := usageSidecarPath(s.basePath)
	data, err := json.Marshal(doc)
	if err == nil {
		err = writeCollectionFileAtomic(path, data, 0o600)
	}
	if err != nil {
		logging.Default().Error("usage sidecar not written; ranking hints will be lost on restart",
			"path", path, "error", err)
	}
}

// loadUsageSidecar restores the class B usage state. An absent sidecar is
// not an error — every data directory written before the sidecar existed
// has none, and an empty tracker is exactly the right starting state. A
// sidecar that is present but unreadable, corrupt, or of an unknown
// version is logged and discarded whole: the collection stays up serving
// correct-by-similarity answers with no ranking hints, and no fault is
// latched.
func (s *DurableStore) loadUsageSidecar() {
	path := usageSidecarPath(s.basePath)
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		// Nothing was lost, so nothing is reported lost: a store with no
		// sidecar had no hints to discard.
		s.usageLoaded = true
		return
	}
	if err == nil {
		err = s.importUsageSidecar(data)
	}
	if err != nil {
		logging.Default().Error("usage sidecar discarded; collection stays up without ranking hints",
			"path", path, "error", err)
		return
	}
	s.usageLoaded = true
}

// importUsageSidecar validates the whole document before importing any of
// it, so one bad collection cannot leave the store half-restored.
// Collections named by the sidecar that no longer exist are skipped: a
// delete after the last checkpoint is ordinary, not corruption.
func (s *DurableStore) importUsageSidecar(data []byte) error {
	var doc usageSidecarDocument
	if err := decodeCollectionJSON(data, &doc); err != nil {
		return err
	}
	if doc.Version != usageDocumentVersion {
		return fmt.Errorf("unsupported usage sidecar version %d", doc.Version)
	}
	type restore struct {
		tracker *UsageTracker
		doc     UsageDocument
	}
	pending := make([]restore, 0, len(doc.Collections))
	for _, entry := range doc.Collections {
		tracked := UsageDocument{Version: usageDocumentVersion, Entries: entry.Entries}
		if err := validateUsageDocument(tracked); err != nil {
			return fmt.Errorf("tenant %s collection %s: %w", entry.TenantID, entry.Collection, err)
		}
		coll, err := s.tenants.getCollectionDirect(entry.TenantID, entry.Collection)
		if err != nil {
			continue
		}
		pending = append(pending, restore{tracker: coll.usage, doc: tracked})
	}
	for _, item := range pending {
		if err := item.tracker.Import(item.doc); err != nil {
			return err
		}
	}
	return nil
}

func (s *DurableStore) verifyJournalCoverageLocked() error {
	return s.journal.verifyCovered(s.metadata.AppliedLSN)
}

func (s *DurableStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	var checkpointErr error
	if s.fault == nil {
		checkpointErr = s.commitSnapshotAndCleanupLocked(false)
	} else {
		checkpointErr = fmt.Errorf("%w: %v", ErrDurableStoreFaulted, s.fault)
	}
	s.closed = true
	journalCloseErr := s.journal.closeWriter()
	s.manager.closeAll()
	s.tenants.closeAll()
	return errors.Join(checkpointErr, journalCloseErr, s.lock.release())
}

// Abort closes in-memory resources and releases the lifetime lock without
// creating a checkpoint or deleting journal evidence. It is reserved for
// startup refusal before any request can be served, and for crash-recovery
// tests. Normal graceful shutdown must use Close.
func (s *DurableStore) Abort() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	journalCloseErr := s.journal.closeWriter()
	s.manager.closeAll()
	s.tenants.closeAll()
	return errors.Join(journalCloseErr, s.lock.release())
}

func encodeDurableMutation(m canonicalMutation) ([]byte, error) {
	return encodeDurableMutationVersion(m, durableCollectionMutationVersion)
}

func encodeDurableMutationVersion(m canonicalMutation, version uint16) ([]byte, error) {
	if version != durableCollectionMutationVersionV1 && version != durableCollectionMutationVersion {
		return nil, fmt.Errorf("unsupported durable mutation version %d", version)
	}
	var payload any
	switch m.typeName {
	case mutationCreateCollection:
		payload = durableCreateCollection{TenantID: m.tenantID, Schema: m.schema}
	case mutationDeleteCollection:
		payload = durableCollectionTarget{TenantID: m.tenantID, CollectionName: m.collectionName}
	case mutationInsertDocument:
		if len(m.documents) != 1 {
			return nil, errors.New("insert mutation must contain exactly one document")
		}
		payload = durableInsertDocument{TenantID: m.tenantID, CollectionName: m.collectionName, Document: m.documents[0]}
	case mutationBatchInsert:
		payload = durableBatchInsert{TenantID: m.tenantID, CollectionName: m.collectionName, Documents: m.documents}
	case mutationUpsertDocument:
		if len(m.documents) != 1 {
			return nil, errors.New("upsert mutation must contain exactly one document")
		}
		payload = durableUpsertDocument{TenantID: m.tenantID, CollectionName: m.collectionName, Document: m.documents[0]}
	case mutationDeleteDocument:
		payload = durableDeleteDocument{TenantID: m.tenantID, CollectionName: m.collectionName, DocumentID: m.documentID}
	case mutationCreateTenant, mutationUpdateTenant:
		payload = m.record
	case mutationDeleteTenant:
		payload = durableTenantTarget{TenantID: m.tenantID}
	default:
		return nil, fmt.Errorf("unknown durable mutation type %q", m.typeName)
	}
	// Single traversal: the typed payload is marshaled inline instead of being
	// encoded to an intermediate buffer that a second pass copies into the
	// envelope. Field order matches durableMutationEnvelope (version, type,
	// payload), and the payload structs emit identical members nested or
	// standalone, so journal bytes stay byte-compatible with records written
	// by earlier encoders.
	return json.Marshal(struct {
		Version uint16 `json:"version"`
		Type    string `json:"type"`
		Payload any    `json:"payload"`
	}{
		Version: version,
		Type:    m.typeName,
		Payload: payload,
	})
}

func decodeDurableMutation(data []byte) (canonicalMutation, error) {
	var envelope durableMutationEnvelope
	if err := decodeCollectionJSON(data, &envelope); err != nil {
		return canonicalMutation{}, err
	}
	if envelope.Version != durableCollectionMutationVersionV1 && envelope.Version != durableCollectionMutationVersion {
		return canonicalMutation{}, fmt.Errorf("unsupported durable mutation version %d", envelope.Version)
	}
	if len(envelope.Payload) == 0 || string(envelope.Payload) == "null" {
		return canonicalMutation{}, errors.New("durable mutation payload is missing or null")
	}
	m := canonicalMutation{version: envelope.Version, typeName: envelope.Type}
	switch envelope.Type {
	case mutationCreateCollection:
		var payload durableCreateCollection
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName, m.schema = payload.TenantID, payload.Schema.Name, payload.Schema
		validate := validateCanonicalSchema
		if envelope.Version == durableCollectionMutationVersionV1 {
			validate = validateDurableSchemaV1
		}
		if err := validate(&m.schema); err != nil {
			return m, err
		}
	case mutationDeleteCollection:
		var payload durableCollectionTarget
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName = payload.TenantID, payload.CollectionName
	case mutationInsertDocument:
		var payload durableInsertDocument
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName, m.documents = payload.TenantID, payload.CollectionName, []Document{payload.Document}
	case mutationBatchInsert:
		var payload durableBatchInsert
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName, m.documents = payload.TenantID, payload.CollectionName, payload.Documents
	case mutationUpsertDocument:
		var payload durableUpsertDocument
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName, m.documents = payload.TenantID, payload.CollectionName, []Document{payload.Document}
	case mutationDeleteDocument:
		var payload durableDeleteDocument
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName, m.documentID = payload.TenantID, payload.CollectionName, payload.DocumentID
	case mutationCreateTenant, mutationUpdateTenant:
		var payload TenantRecord
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		if err := payload.validate(); err != nil {
			return m, err
		}
		m.tenantID, m.record = payload.TenantID, payload
	case mutationDeleteTenant:
		var payload durableTenantTarget
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID = payload.TenantID
	default:
		return m, fmt.Errorf("unknown durable mutation type %q", envelope.Type)
	}
	switch m.typeName {
	case mutationCreateTenant, mutationUpdateTenant, mutationDeleteTenant:
		if m.tenantID == "" {
			return m, errors.New("durable mutation tenant ID cannot be empty")
		}
	default:
		if m.tenantID == "" || m.collectionName == "" {
			return m, errors.New("durable mutation tenant and collection names cannot be empty")
		}
	}
	if (m.typeName == mutationInsertDocument || m.typeName == mutationBatchInsert || m.typeName == mutationUpsertDocument) && len(m.documents) == 0 {
		return m, errors.New("durable insert mutation has no documents")
	}
	if m.typeName == mutationInsertDocument || m.typeName == mutationBatchInsert || m.typeName == mutationUpsertDocument {
		for i := range m.documents {
			if m.documents[i].ID == 0 {
				return m, fmt.Errorf("durable insert mutation document %d has an unassigned ID", i)
			}
		}
	}
	if m.typeName == mutationDeleteDocument && m.documentID == 0 {
		return m, errors.New("durable delete mutation document ID cannot be zero")
	}
	return m, nil
}

func nextCanonicalID(candidate uint64, reserved map[uint64]struct{}, existing map[uint64]*Document) (uint64, uint64, error) {
	for {
		if candidate == 0 || candidate == math.MaxUint64 {
			return 0, 0, errors.New("document ID space exhausted")
		}
		if _, used := reserved[candidate]; !used {
			if _, used = existing[candidate]; !used {
				return candidate, candidate + 1, nil
			}
		}
		candidate++
	}
}
