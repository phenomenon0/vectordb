package collection

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sync"
)

const (
	durableCollectionMutationVersion = uint16(1)

	mutationCreateCollection = "create_collection"
	mutationDeleteCollection = "delete_collection"
	mutationInsertDocument   = "insert_document"
	mutationBatchInsert      = "batch_insert_documents"
	mutationDeleteDocument   = "delete_document"
)

var (
	ErrDurableStoreClosed         = errors.New("durable collection store is closed")
	ErrDurableStoreFaulted        = errors.New("durable collection store is faulted")
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

type canonicalMutation struct {
	typeName       string
	tenantID       string
	collectionName string
	schema         CollectionSchema
	documents      []Document
	documentID     uint64
	nextID         uint64
}

// DurableStore is the single Linux persistence boundary for the canonical,
// tenant-aware collection API. Its mutex deliberately spans WAL append, apply,
// snapshot capture, journal rotation, and coverage-checked cleanup.
type DurableStore struct {
	mu sync.Mutex

	basePath string
	manager  *CollectionManager
	tenants  *TenantManager
	metadata CollectionSnapshotMetadata
	journal  *collectionJournal
	lock     *collectionStoreLock

	fault  error
	closed bool

	// apply is a per-store test seam. Production always points at
	// applyMutationDirect; an error after append permanently faults the store.
	apply func(context.Context, canonicalMutation) error
}

// OpenDurableStore opens or initializes a durable unified collection store.
// Persistent startup is intentionally rejected by the build-tagged lock stub
// on non-Linux platforms for this release candidate.
func OpenDurableStore(basePath, storagePath string) (*DurableStore, error) {
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
	journal, records, err := openCollectionJournal(basePath+".journal", basePath+".journal.frozen", metadata.StoreID, metadata.AppliedLSN)
	if err != nil {
		return fail(manager, tenants, fmt.Errorf("open durable collection journal: %w", err))
	}

	// Decode every record before changing any loaded state. This makes unknown
	// versions, operations, fields, and trailing JSON a startup failure rather
	// than a partially replayed live store.
	mutations := make([]canonicalMutation, len(records))
	for i, record := range records {
		mutation, err := decodeDurableMutation(record.Payload)
		if err != nil {
			return fail(manager, tenants, fmt.Errorf("decode collection mutation at LSN %d: %w", record.LSN, err))
		}
		mutations[i] = mutation
	}

	store := &DurableStore{
		basePath: basePath,
		manager:  manager,
		tenants:  tenants,
		metadata: metadata,
		journal:  journal,
		lock:     lock,
	}
	store.apply = store.applyMutationDirect
	for i, mutation := range mutations {
		if err := store.prepareReplayMutation(&mutation); err != nil {
			return fail(manager, tenants, fmt.Errorf("validate collection mutation at LSN %d: %w", records[i].LSN, err))
		}
		if err := store.applyMutationDirect(context.Background(), mutation); err != nil {
			return fail(manager, tenants, fmt.Errorf("replay collection mutation at LSN %d: %w", records[i].LSN, err))
		}
		store.metadata.AppliedLSN = records[i].LSN
	}
	if len(records) > 0 {
		if err := store.commitSnapshotAndCleanupLocked(true); err != nil {
			return fail(manager, tenants, fmt.Errorf("checkpoint replayed collection mutations: %w", err))
		}
	}

	manager.setDurableReadOnly()
	tenants.attachDurableStore(store)
	return store, nil
}

func (s *DurableStore) Manager() *CollectionManager { return s.manager }
func (s *DurableStore) Tenants() *TenantManager     { return s.tenants }

func (s *DurableStore) Metadata() CollectionSnapshotMetadata {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.metadata
}

func (s *DurableStore) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stateErrorLocked()
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

func (s *DurableStore) latchFaultLocked(err error) error {
	if s.fault == nil {
		s.fault = err
	}
	return fmt.Errorf("%w: %v", ErrDurableStoreFaulted, s.fault)
}

func (s *DurableStore) createCollection(ctx context.Context, tenantID string, schema CollectionSchema) (*Collection, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return nil, err
	}
	mutation := canonicalMutation{typeName: mutationCreateCollection, tenantID: tenantID, collectionName: schema.Name, schema: schema}
	if err := s.prepareCreateMutation(&mutation); err != nil {
		return nil, err
	}
	if err := s.appendApplyLocked(ctx, mutation); err != nil {
		return nil, err
	}
	return s.tenants.GetCollection(tenantID, mutation.collectionName)
}

func (s *DurableStore) addDocument(ctx context.Context, tenantID, collectionName string, doc *Document) error {
	if doc == nil {
		return errors.New("document cannot be nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
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
	if err := s.stateErrorLocked(); err != nil {
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

func (s *DurableStore) deleteCollection(ctx context.Context, tenantID, collectionName string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	mutation := canonicalMutation{typeName: mutationDeleteCollection, tenantID: tenantID, collectionName: collectionName}
	if err := s.prepareCollectionTarget(mutation); err != nil {
		return err
	}
	return s.appendApplyLocked(ctx, mutation)
}

func (s *DurableStore) deleteDocument(ctx context.Context, tenantID, collectionName string, documentID uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.stateErrorLocked(); err != nil {
		return err
	}
	mutation := canonicalMutation{typeName: mutationDeleteDocument, tenantID: tenantID, collectionName: collectionName, documentID: documentID}
	if err := s.prepareDeleteDocument(mutation); err != nil {
		return err
	}
	return s.appendApplyLocked(ctx, mutation)
}

func (s *DurableStore) appendApplyLocked(ctx context.Context, mutation canonicalMutation) error {
	payload, err := encodeDurableMutation(mutation)
	if err != nil {
		return err
	}
	record, err := s.journal.append(payload)
	if err != nil {
		if journalFault := s.journal.writeFault(); journalFault != nil {
			return s.latchFaultLocked(fmt.Errorf("journal append failed: %w", journalFault))
		}
		return err
	}
	if err := s.apply(ctx, mutation); err != nil {
		return s.latchFaultLocked(fmt.Errorf("apply LSN %d after durable append: %w", record.LSN, err))
	}
	s.metadata.AppliedLSN = record.LSN
	return nil
}

func (s *DurableStore) prepareCreateMutation(m *canonicalMutation) error {
	if m.tenantID == "" {
		return errors.New("tenant ID cannot be empty")
	}
	clone, err := cloneCanonicalSchema(m.schema)
	if err != nil {
		return err
	}
	if err := validateCanonicalSchema(&clone); err != nil {
		return err
	}
	if m.collectionName != "" && m.collectionName != clone.Name {
		return errors.New("collection name does not match schema")
	}
	m.collectionName = clone.Name
	m.schema = clone
	if manager := s.tenants.getManager(m.tenantID); manager != nil && manager.HasCollection(clone.Name) {
		return fmt.Errorf("collection %s already exists", clone.Name)
	}
	return nil
}

func validateCanonicalSchema(schema *CollectionSchema) error {
	if err := schema.Validate(); err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	for _, field := range schema.Fields {
		switch field.Type {
		case VectorTypeDense:
			if field.Index.Type != IndexTypeHNSW && field.Index.Type != IndexTypeFLAT {
				return fmt.Errorf("field %s uses index %s; canonical release supports only HNSW or Flat dense indexes", field.Name, field.Index.Type)
			}
		case VectorTypeSparse:
			if field.Index.Type != IndexTypeInverted {
				return fmt.Errorf("field %s uses index %s; sparse fields require Inverted", field.Name, field.Index.Type)
			}
		default:
			return fmt.Errorf("field %s uses unsupported vector type %s", field.Name, field.Type)
		}
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
	_, err := s.tenants.GetCollection(m.tenantID, m.collectionName)
	return err
}

func (s *DurableStore) prepareDocumentsMutation(typeName, tenantID, collectionName string, docs []Document) (canonicalMutation, error) {
	mutation := canonicalMutation{typeName: typeName, tenantID: tenantID, collectionName: collectionName}
	if len(docs) == 0 {
		return mutation, errors.New("documents cannot be empty")
	}
	if err := s.prepareCollectionTarget(mutation); err != nil {
		return mutation, err
	}
	coll, _ := s.tenants.GetCollection(tenantID, collectionName)
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
	coll, _ := s.tenants.GetCollection(m.tenantID, m.collectionName)
	if _, ok := coll.GetDocument(m.documentID); !ok {
		return fmt.Errorf("document %d not found in collection %s", m.documentID, m.collectionName)
	}
	return nil
}

func (s *DurableStore) prepareReplayMutation(m *canonicalMutation) error {
	switch m.typeName {
	case mutationCreateCollection:
		return s.prepareCreateMutation(m)
	case mutationDeleteCollection:
		return s.prepareCollectionTarget(*m)
	case mutationInsertDocument, mutationBatchInsert:
		prepared, err := s.prepareDocumentsMutation(m.typeName, m.tenantID, m.collectionName, m.documents)
		if err != nil {
			return err
		}
		*m = prepared
		return nil
	case mutationDeleteDocument:
		return s.prepareDeleteDocument(*m)
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
		return s.tenants.deleteCollectionDirect(ctx, m.tenantID, m.collectionName)
	case mutationInsertDocument, mutationBatchInsert:
		return s.tenants.addPreparedDocumentsDirect(ctx, m.tenantID, m.collectionName, m.documents, m.nextID)
	case mutationDeleteDocument:
		return s.tenants.deleteDocumentDirect(ctx, m.tenantID, m.collectionName, m.documentID)
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
	return nil
}

func (s *DurableStore) verifyJournalCoverageLocked() error {
	for _, path := range []string{s.journal.frozenPath, s.journal.currentPath} {
		records, exists, err := readCollectionJournalFile(path, s.metadata.StoreID, s.journal.maxPayload)
		if err != nil {
			return fmt.Errorf("verify checkpoint coverage for %q: %w", path, err)
		}
		if !exists || len(records) == 0 {
			continue
		}
		for _, record := range records {
			if record.LSN > s.metadata.AppliedLSN {
				return fmt.Errorf("refusing journal cleanup: %q LSN %d exceeds snapshot LSN %d", path, record.LSN, s.metadata.AppliedLSN)
			}
		}
	}
	return nil
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
	s.manager.closeAll()
	s.tenants.closeAll()
	return errors.Join(checkpointErr, s.lock.release())
}

func encodeDurableMutation(m canonicalMutation) ([]byte, error) {
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
	case mutationDeleteDocument:
		payload = durableDeleteDocument{TenantID: m.tenantID, CollectionName: m.collectionName, DocumentID: m.documentID}
	default:
		return nil, fmt.Errorf("unknown durable mutation type %q", m.typeName)
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal durable mutation payload: %w", err)
	}
	return json.Marshal(durableMutationEnvelope{Version: durableCollectionMutationVersion, Type: m.typeName, Payload: payloadBytes})
}

func decodeDurableMutation(data []byte) (canonicalMutation, error) {
	var envelope durableMutationEnvelope
	if err := decodeCollectionJSON(data, &envelope); err != nil {
		return canonicalMutation{}, err
	}
	if envelope.Version != durableCollectionMutationVersion {
		return canonicalMutation{}, fmt.Errorf("unsupported durable mutation version %d", envelope.Version)
	}
	if len(envelope.Payload) == 0 || string(envelope.Payload) == "null" {
		return canonicalMutation{}, errors.New("durable mutation payload is missing or null")
	}
	m := canonicalMutation{typeName: envelope.Type}
	switch envelope.Type {
	case mutationCreateCollection:
		var payload durableCreateCollection
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName, m.schema = payload.TenantID, payload.Schema.Name, payload.Schema
		if err := validateCanonicalSchema(&m.schema); err != nil {
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
	case mutationDeleteDocument:
		var payload durableDeleteDocument
		if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
			return m, err
		}
		m.tenantID, m.collectionName, m.documentID = payload.TenantID, payload.CollectionName, payload.DocumentID
	default:
		return m, fmt.Errorf("unknown durable mutation type %q", envelope.Type)
	}
	if m.tenantID == "" || m.collectionName == "" {
		return m, errors.New("durable mutation tenant and collection names cannot be empty")
	}
	if (m.typeName == mutationInsertDocument || m.typeName == mutationBatchInsert) && len(m.documents) == 0 {
		return m, errors.New("durable insert mutation has no documents")
	}
	if m.typeName == mutationInsertDocument || m.typeName == mutationBatchInsert {
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
