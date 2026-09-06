package collection

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

const (
	collectionSnapshotMagic               = "deepdata-collections"
	collectionSnapshotV1Version           = uint32(1)
	collectionSnapshotVersion             = uint32(2)
	collectionMarkerMagic                 = "deepdata-collections-initialized"
	collectionMarkerVersion               = uint32(1)
	collectionSnapshotV2HeaderSize        = uint32(56)
	collectionSnapshotV2MaxFrameSize      = uint32(16 << 20)
	collectionSnapshotV1MaxReadableBytes  = int64(256 << 20)
	collectionLegacyStateMaxReadableBytes = int64(256 << 20)
	collectionMarkerMaxReadableBytes      = int64(4 << 10)
)

var collectionSnapshotV2Magic = [8]byte{'D', 'D', 'C', 'O', 'L', 'S', 'N', 'P'}

// ErrCollectionSnapshotNotFound means an initialized collection store lost its
// unified snapshot. Callers must fail closed instead of recreating empty state.
var ErrCollectionSnapshotNotFound = errors.New("initialized collection snapshot is missing")

// CollectionSnapshotMetadata ties the unified V2/V3 state to its journal.
// StoreID is stable across checkpoints; AppliedLSN is the journal high-water
// represented by the snapshot.
type CollectionSnapshotMetadata struct {
	StoreID    [16]byte
	AppliedLSN uint64
}

// NewCollectionSnapshotMetadata creates metadata for a newly initialized
// unified collection store.
func NewCollectionSnapshotMetadata() (CollectionSnapshotMetadata, error) {
	id, err := newCollectionStoreID()
	if err != nil {
		return CollectionSnapshotMetadata{}, err
	}
	return CollectionSnapshotMetadata{StoreID: id}, nil
}

type collectionSnapshotPayload struct {
	AppliedLSN uint64          `json:"applied_lsn"`
	Manager    json.RawMessage `json:"manager"`
	Tenants    json.RawMessage `json:"tenants"`
}

type collectionSnapshotEnvelope struct {
	Magic    string          `json:"magic"`
	Version  uint32          `json:"version"`
	StoreID  string          `json:"store_id"`
	Payload  json.RawMessage `json:"payload"`
	Checksum string          `json:"checksum"`
}

// Version 2 is a framed logical snapshot. It intentionally persists schemas
// and documents rather than monolithic index exports: the current Index
// interface can only return an entire []byte export, and several index
// implementations first clone all vectors before producing that byte slice.
// Rebuilding indexes from documents on load keeps snapshot memory bounded by
// one admitted document frame plus the live state and per-insert index scratch
// being reconstructed. json.Marshal still materializes that one frame before
// the 16 MiB frame limit is checked; durable admission already constrains one
// mutation to the same ceiling, while non-durable callers can fail after one
// larger document allocation. Approximate indexes such as HNSW and IVF may
// produce a different physical graph and result order after rebuild; durable
// semantics are the exact schema, IDs, vectors, metadata, and next-ID cursor,
// while recovery validation must separately gate query relevance and counts.
type collectionSnapshotV2Collection struct {
	Schema        CollectionSchema `json:"schema"`
	NextID        uint64           `json:"next_id"`
	DocumentCount uint64           `json:"document_count"`
}

type collectionSnapshotV2Tenant struct {
	TenantID        string `json:"tenant_id"`
	CollectionCount uint64 `json:"collection_count"`
	// Record is set only when the tenant has an administrator record. Older
	// binaries decode snapshots with DisallowUnknownFields, so a snapshot
	// written with a record cannot be opened by a binary predating this
	// field; there is no compatibility shim for that case.
	Record *TenantRecord `json:"record,omitempty"`
}

type collectionSnapshotV2Document struct {
	ID       uint64    `json:"id"`
	Document *Document `json:"document"`
}

type collectionInitializationMarker struct {
	Magic   string `json:"magic"`
	Version uint32 `json:"version"`
	StoreID string `json:"store_id"`
}

var (
	collectionSnapshotFileSync = func(f *os.File) error { return f.Sync() }
	collectionSnapshotDirSync  = syncCollectionSnapshotDirectory
	collectionSnapshotRename   = os.Rename
	// collectionSnapshotAfterV2Validation is a test seam used to prove that an
	// atomic path replacement between the validation and build passes is detected.
	collectionSnapshotAfterV2Validation func(string) error
	// A unified checkpoint is serialized from target validation through the
	// final rename. Unique temp names prevent collisions; this mutex makes the
	// existing-LSN regression check and replacement one same-process operation.
	collectionSnapshotCommitMu sync.Mutex
)

func decodeCollectionJSON(data []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON value")
		}
		return fmt.Errorf("trailing data: %w", err)
	}
	return nil
}

func syncCollectionSnapshotDirectory(path string) error {
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	if err := dir.Sync(); err != nil {
		_ = dir.Close()
		return err
	}
	return dir.Close()
}

// writeCollectionFileAtomic commits data through a unique same-directory temp
// file. If the final directory sync fails, the rename outcome is intentionally
// reported as indeterminate and is not rolled back.
func writeCollectionFileAtomic(path string, data []byte, mode os.FileMode) error {
	return writeCollectionFileAtomicStream(path, mode, func(w io.Writer) error {
		if _, err := io.Copy(w, bytes.NewReader(data)); err != nil {
			return fmt.Errorf("write temporary file: %w", err)
		}
		return nil
	})
}

// writeCollectionFileAtomicStream is the streaming equivalent of
// writeCollectionFileAtomic. The caller writes one logical generation to a
// unique same-directory inode; only a fully written, chmodded, fsynced file is
// renamed over the target.
func writeCollectionFileAtomicStream(path string, mode os.FileMode, write func(io.Writer) error) (retErr error) {
	dir := filepath.Dir(path)
	base := filepath.Base(path)
	f, err := os.CreateTemp(dir, "."+base+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary file: %w", err)
	}
	tmpPath := f.Name()
	committed := false
	defer func() {
		if !committed {
			_ = f.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	if err := f.Chmod(mode); err != nil {
		return fmt.Errorf("set temporary file mode: %w", err)
	}
	if err := write(f); err != nil {
		return err
	}
	if err := collectionSnapshotFileSync(f); err != nil {
		return fmt.Errorf("sync temporary file: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close temporary file: %w", err)
	}
	if err := collectionSnapshotRename(tmpPath, path); err != nil {
		return fmt.Errorf("rename temporary file: %w", err)
	}
	committed = true
	if err := collectionSnapshotDirSync(path); err != nil {
		return fmt.Errorf("sync parent directory after rename: %w", err)
	}
	return nil
}

func newCollectionStoreID() ([16]byte, error) {
	var id [16]byte
	if _, err := io.ReadFull(rand.Reader, id[:]); err != nil {
		return id, fmt.Errorf("generate collection store ID: %w", err)
	}
	return id, nil
}

func parseCollectionStoreID(encoded string) ([16]byte, error) {
	var id [16]byte
	if len(encoded) != hex.EncodedLen(len(id)) {
		return id, fmt.Errorf("store ID must be %d hexadecimal characters", hex.EncodedLen(len(id)))
	}
	decoded, err := hex.DecodeString(encoded)
	if err != nil {
		return id, fmt.Errorf("decode store ID: %w", err)
	}
	copy(id[:], decoded)
	var zero [16]byte
	if id == zero {
		return id, fmt.Errorf("store ID cannot be zero")
	}
	return id, nil
}

func collectionSnapshotChecksum(storeID [16]byte, payload []byte) string {
	h := sha256.New()
	_, _ = h.Write([]byte(collectionSnapshotMagic))
	var version [4]byte
	binary.BigEndian.PutUint32(version[:], collectionSnapshotV1Version)
	_, _ = h.Write(version[:])
	_, _ = h.Write(storeID[:])
	_, _ = h.Write(payload)
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func collectionSnapshotPath(basePath string) string { return basePath + ".snapshot" }
func collectionMarkerPath(basePath string) string   { return basePath + ".initialized" }

func saveUnifiedCollectionSnapshot(basePath string, manager *CollectionManager, tenants *TenantManager, metadata CollectionSnapshotMetadata) error {
	if basePath == "" {
		return fmt.Errorf("collection snapshot base path cannot be empty")
	}
	if manager == nil || tenants == nil {
		return fmt.Errorf("collection snapshot managers cannot be nil")
	}
	var zero [16]byte
	if metadata.StoreID == zero {
		return fmt.Errorf("collection snapshot store ID cannot be zero")
	}

	collectionSnapshotCommitMu.Lock()
	defer collectionSnapshotCommitMu.Unlock()

	if err := validateExistingCollectionSnapshotTarget(basePath, metadata); err != nil {
		return err
	}

	manager.mu.RLock()
	tenants.mu.RLock()
	err := writeCollectionFileAtomicStream(collectionSnapshotPath(basePath), 0o600, func(w io.Writer) error {
		return writeUnifiedCollectionSnapshotV2(w, manager, tenants, metadata)
	})
	tenants.mu.RUnlock()
	manager.mu.RUnlock()
	if err != nil {
		return fmt.Errorf("commit collection snapshot: %w", err)
	}
	if err := ensureCollectionInitializationMarker(basePath, metadata.StoreID); err != nil {
		return err
	}
	return nil
}

func writeUnifiedCollectionSnapshotV2(w io.Writer, manager *CollectionManager, tenants *TenantManager, metadata CollectionSnapshotMetadata) error {
	header := make([]byte, collectionSnapshotV2HeaderSize)
	copy(header[:8], collectionSnapshotV2Magic[:])
	binary.BigEndian.PutUint32(header[8:12], collectionSnapshotVersion)
	binary.BigEndian.PutUint32(header[12:16], collectionSnapshotV2HeaderSize)
	copy(header[16:32], metadata.StoreID[:])
	binary.BigEndian.PutUint64(header[32:40], metadata.AppliedLSN)
	binary.BigEndian.PutUint64(header[40:48], uint64(len(manager.collections)))
	binary.BigEndian.PutUint64(header[48:56], uint64(len(tenants.tenants)))

	h := sha256.New()
	body := io.MultiWriter(w, h)
	if err := writeCollectionSnapshotBytes(body, header); err != nil {
		return fmt.Errorf("write v2 header: %w", err)
	}
	if err := writeCollectionSnapshotV2ManagerLocked(body, manager); err != nil {
		return fmt.Errorf("write root manager: %w", err)
	}
	for tenantID, tenantManager := range tenants.tenants {
		if tenantID == "" {
			return fmt.Errorf("tenant ID cannot be empty")
		}
		if tenantManager == nil {
			return fmt.Errorf("tenant %s has nil manager", tenantID)
		}
		tenantManager.mu.RLock()
		err := func() error {
			descriptor := collectionSnapshotV2Tenant{
				TenantID:        tenantID,
				CollectionCount: uint64(len(tenantManager.collections)),
			}
			if rec, ok := tenants.records[tenantID]; ok {
				descriptor.Record = &rec
			}
			if err := writeCollectionSnapshotV2Frame(body, descriptor); err != nil {
				return fmt.Errorf("write tenant descriptor: %w", err)
			}
			return writeCollectionSnapshotV2ManagerLocked(body, tenantManager)
		}()
		tenantManager.mu.RUnlock()
		if err != nil {
			return fmt.Errorf("write tenant %s: %w", tenantID, err)
		}
	}
	if err := writeCollectionSnapshotBytes(w, h.Sum(nil)); err != nil {
		return fmt.Errorf("write v2 checksum: %w", err)
	}
	return nil
}

// writeCollectionSnapshotV2ManagerLocked writes a manager while its read lock
// is held. Each collection is captured under its own read lock and each
// document is encoded and released before the next one is visited.
func writeCollectionSnapshotV2ManagerLocked(w io.Writer, manager *CollectionManager) error {
	for name, coll := range manager.collections {
		if coll == nil {
			return fmt.Errorf("collection %s has nil state", name)
		}
		coll.mu.RLock()
		err := func() error {
			if coll.schema.Name != name {
				return fmt.Errorf("collection key %q does not match schema name %q", name, coll.schema.Name)
			}
			if err := validateCollectionSnapshotV2RepresentableLocked(coll); err != nil {
				return err
			}
			// Durability class E (ADR 0009): an ephemeral collection is
			// captured as its schema and nothing else, so the reload finds it
			// present and empty for its upstream to rebuild.
			documentCount := uint64(len(coll.documents))
			if coll.schema.Durability == DurabilityEphemeral {
				documentCount = 0
			}
			if err := writeCollectionSnapshotV2Frame(w, collectionSnapshotV2Collection{
				Schema:        coll.schema,
				NextID:        coll.nextID,
				DocumentCount: documentCount,
			}); err != nil {
				return fmt.Errorf("write descriptor: %w", err)
			}
			if documentCount == 0 {
				return nil
			}
			for id, doc := range coll.documents {
				if doc == nil {
					return fmt.Errorf("document %d is nil", id)
				}
				if doc.ID != 0 && doc.ID != id {
					return fmt.Errorf("document key %d does not match embedded ID %d", id, doc.ID)
				}
				if err := writeCollectionSnapshotV2Frame(w, collectionSnapshotV2Document{ID: id, Document: doc}); err != nil {
					return fmt.Errorf("write document %d: %w", id, err)
				}
			}
			return nil
		}()
		coll.mu.RUnlock()
		if err != nil {
			return fmt.Errorf("export collection %s: %w", name, err)
		}
	}
	return nil
}

// validateCollectionSnapshotV2RepresentableLocked checks the strongest
// bounded condition available through the Index interface: each field has the
// same number of retained document vectors as active index vectors. The API
// exposes neither point lookup nor a streaming iterator, so value-by-value
// equality cannot be proved without the same whole-index Export allocation v2
// avoids. The count check still prevents known legacy/imported state with
// index-only vectors from being silently rewritten into a document-only v2
// snapshot.
func validateCollectionSnapshotV2RepresentableLocked(coll *Collection) error {
	if err := coll.schema.Validate(); err != nil {
		return fmt.Errorf("invalid collection schema: %w", err)
	}
	// Recreate an empty collection from an isolated schema clone before any
	// bytes are committed. CollectionSchema.Validate intentionally does not
	// validate every concrete index parameter, while NewCollection does. The
	// clone is required because index constructors fill defaults into parameter
	// maps and must never mutate the live schema during a checkpoint.
	schemaClone, err := cloneCanonicalSchema(coll.schema)
	if err != nil {
		return fmt.Errorf("clone collection schema for reload validation: %w", err)
	}
	recreated, err := NewCollection(schemaClone)
	if err != nil {
		return fmt.Errorf("collection schema cannot be recreated: %w", err)
	}
	recreated.closeDirect()

	denseCounts := make(map[string]int)
	sparseCounts := make(map[string]int)
	var maxDocumentID uint64
	hasDocuments := false
	for id, doc := range coll.documents {
		if id == math.MaxUint64 {
			return fmt.Errorf("document ID %d cannot be represented by the next-ID cursor", id)
		}
		if doc == nil {
			return fmt.Errorf("document %d is nil", id)
		}
		if doc.ID != 0 && doc.ID != id {
			return fmt.Errorf("document key %d does not match embedded ID %d", id, doc.ID)
		}
		if err := validatePersistedDocument(doc, &coll.schema); err != nil {
			return fmt.Errorf("invalid document %d: %w", id, err)
		}
		for fieldName := range doc.Vectors {
			field := coll.schema.GetField(fieldName)
			if field == nil {
				return fmt.Errorf("document %d has unknown vector field %s", id, fieldName)
			}
			if field.Type == VectorTypeDense {
				denseCounts[fieldName]++
			} else if field.Type == VectorTypeSparse {
				sparseCounts[fieldName]++
			}
		}
		if !hasDocuments || id > maxDocumentID {
			maxDocumentID = id
		}
		hasDocuments = true
	}
	if coll.nextID == 0 {
		return fmt.Errorf("next document ID cannot be zero")
	}
	if hasDocuments && coll.nextID <= maxDocumentID {
		return fmt.Errorf("next document ID %d must be greater than maximum document ID %d", coll.nextID, maxDocumentID)
	}

	denseFields := 0
	sparseFields := 0
	for _, field := range coll.schema.Fields {
		switch field.Type {
		case VectorTypeDense:
			denseFields++
			idx, ok := coll.indexes[field.Name]
			if !ok {
				return fmt.Errorf("collection is missing dense index %s", field.Name)
			}
			if active := idx.Stats().Active; active != denseCounts[field.Name] {
				return fmt.Errorf(
					"dense index %s has %d active vectors but documents retain %d; v2 snapshot would lose state",
					field.Name,
					active,
					denseCounts[field.Name],
				)
			}
		case VectorTypeSparse:
			sparseFields++
			idx, ok := coll.sparse[field.Name]
			if !ok {
				return fmt.Errorf("collection is missing sparse index %s", field.Name)
			}
			if active := idx.Count(); active != sparseCounts[field.Name] {
				return fmt.Errorf(
					"sparse index %s has %d active vectors but documents retain %d; v2 snapshot would lose state",
					field.Name,
					active,
					sparseCounts[field.Name],
				)
			}
		}
	}
	if len(coll.indexes) != denseFields {
		return fmt.Errorf("collection has %d dense indexes for %d schema fields", len(coll.indexes), denseFields)
	}
	if len(coll.sparse) != sparseFields {
		return fmt.Errorf("collection has %d sparse indexes for %d schema fields", len(coll.sparse), sparseFields)
	}
	return nil
}

func writeCollectionSnapshotV2Frame(w io.Writer, value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encode frame: %w", err)
	}
	if len(payload) == 0 || uint64(len(payload)) > uint64(collectionSnapshotV2MaxFrameSize) {
		return fmt.Errorf("snapshot frame is %d bytes; maximum is %d", len(payload), collectionSnapshotV2MaxFrameSize)
	}
	var size [4]byte
	binary.BigEndian.PutUint32(size[:], uint32(len(payload)))
	if err := writeCollectionSnapshotBytes(w, size[:]); err != nil {
		return err
	}
	return writeCollectionSnapshotBytes(w, payload)
}

func writeCollectionSnapshotBytes(w io.Writer, data []byte) error {
	for len(data) > 0 {
		n, err := w.Write(data)
		if n < 0 || n > len(data) {
			return fmt.Errorf("invalid write count %d", n)
		}
		data = data[n:]
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func validateExistingCollectionSnapshotTarget(basePath string, metadata CollectionSnapshotMetadata) error {
	snapshotExists, err := pathExists(collectionSnapshotPath(basePath))
	if err != nil {
		return fmt.Errorf("inspect existing collection snapshot: %w", err)
	}
	markerExists, err := pathExists(collectionMarkerPath(basePath))
	if err != nil {
		return fmt.Errorf("inspect collection initialization marker: %w", err)
	}
	if markerExists {
		markerID, err := readCollectionInitializationMarker(basePath)
		if err != nil {
			return err
		}
		if markerID != metadata.StoreID {
			return fmt.Errorf("refusing to replace collection snapshot with a different store ID")
		}
		if !snapshotExists {
			return ErrCollectionSnapshotNotFound
		}
	}
	if snapshotExists {
		existingMetadata, err := inspectUnifiedCollectionSnapshot(basePath)
		if err != nil {
			return fmt.Errorf("refusing to replace invalid existing collection snapshot: %w", err)
		}
		if existingMetadata.StoreID != metadata.StoreID {
			return fmt.Errorf("refusing to replace collection snapshot with a different store ID")
		}
		if metadata.AppliedLSN < existingMetadata.AppliedLSN {
			return fmt.Errorf(
				"refusing to replace collection snapshot at applied LSN %d with regressed LSN %d",
				existingMetadata.AppliedLSN,
				metadata.AppliedLSN,
			)
		}
	}
	return nil
}

// SaveUnifiedCollectionSnapshot atomically persists both V2 and V3 managers in
// one checksummed generation.
func SaveUnifiedCollectionSnapshot(basePath string, manager *CollectionManager, tenants *TenantManager, metadata CollectionSnapshotMetadata) error {
	return saveUnifiedCollectionSnapshot(basePath, manager, tenants, metadata)
}

func loadUnifiedCollectionSnapshot(basePath, storagePath string) (*CollectionManager, *TenantManager, CollectionSnapshotMetadata, error) {
	path := collectionSnapshotPath(basePath)
	f, version, info, err := openCollectionSnapshotFile(path)
	if err != nil {
		return nil, nil, CollectionSnapshotMetadata{}, err
	}
	defer f.Close()

	if version == collectionSnapshotV1Version {
		data, err := readOpenCollectionFileBounded(
			f,
			info,
			collectionSnapshotV1MaxReadableBytes,
			"legacy v1 collection snapshot",
			"bounded compatibility loader",
		)
		if err != nil {
			return nil, nil, CollectionSnapshotMetadata{}, err
		}
		decoded, err := decodeUnifiedCollectionSnapshotV1(data)
		if err != nil {
			return nil, nil, CollectionSnapshotMetadata{}, err
		}
		if err := validateCollectionInitializationMarkerIfPresent(basePath, decoded.metadata.StoreID); err != nil {
			return nil, nil, CollectionSnapshotMetadata{}, err
		}
		if err := ensureCollectionSnapshotPathIdentity(path, info); err != nil {
			return nil, nil, CollectionSnapshotMetadata{}, err
		}
		manager, tenants, err := buildUnifiedCollectionSnapshotV1(decoded, storagePath)
		if err != nil {
			return nil, nil, CollectionSnapshotMetadata{}, err
		}
		if err := ensureCollectionSnapshotPathIdentity(path, info); err != nil {
			manager.closeAll()
			tenants.closeAll()
			return nil, nil, CollectionSnapshotMetadata{}, err
		}
		return manager, tenants, decoded.metadata, nil
	}

	validated, err := readUnifiedCollectionSnapshotV2File(f, info, "", false)
	if err != nil {
		return nil, nil, CollectionSnapshotMetadata{}, err
	}
	if err := validateCollectionInitializationMarkerIfPresent(basePath, validated.metadata.StoreID); err != nil {
		return nil, nil, CollectionSnapshotMetadata{}, err
	}
	if collectionSnapshotAfterV2Validation != nil {
		if err := collectionSnapshotAfterV2Validation(path); err != nil {
			return nil, nil, CollectionSnapshotMetadata{}, fmt.Errorf("after collection snapshot validation: %w", err)
		}
	}
	if err := ensureCollectionSnapshotPathIdentity(path, info); err != nil {
		return nil, nil, CollectionSnapshotMetadata{}, err
	}

	loaded, err := readUnifiedCollectionSnapshotV2File(f, info, storagePath, true)
	if err != nil {
		return nil, nil, CollectionSnapshotMetadata{}, err
	}
	if loaded.metadata != validated.metadata || loaded.digest != validated.digest {
		loaded.manager.closeAll()
		loaded.tenants.closeAll()
		return nil, nil, CollectionSnapshotMetadata{}, fmt.Errorf("collection snapshot changed between validation and load")
	}
	if err := ensureCollectionSnapshotPathIdentity(path, info); err != nil {
		loaded.manager.closeAll()
		loaded.tenants.closeAll()
		return nil, nil, CollectionSnapshotMetadata{}, err
	}
	return loaded.manager, loaded.tenants, loaded.metadata, nil
}

func inspectUnifiedCollectionSnapshot(basePath string) (CollectionSnapshotMetadata, error) {
	path := collectionSnapshotPath(basePath)
	f, version, info, err := openCollectionSnapshotFile(path)
	if err != nil {
		return CollectionSnapshotMetadata{}, err
	}
	defer f.Close()
	if version == collectionSnapshotVersion {
		validated, err := readUnifiedCollectionSnapshotV2File(f, info, "", false)
		if err != nil {
			return CollectionSnapshotMetadata{}, err
		}
		if err := ensureCollectionSnapshotPathIdentity(path, info); err != nil {
			return CollectionSnapshotMetadata{}, err
		}
		return validated.metadata, nil
	}
	data, err := readOpenCollectionFileBounded(
		f,
		info,
		collectionSnapshotV1MaxReadableBytes,
		"legacy v1 collection snapshot",
		"bounded compatibility loader",
	)
	if err != nil {
		return CollectionSnapshotMetadata{}, err
	}
	decoded, err := decodeUnifiedCollectionSnapshotV1(data)
	if err != nil {
		return CollectionSnapshotMetadata{}, err
	}
	if err := ensureCollectionSnapshotPathIdentity(path, info); err != nil {
		return CollectionSnapshotMetadata{}, err
	}
	return decoded.metadata, nil
}

func inspectCollectionSnapshotFile(path string) (uint32, os.FileInfo, error) {
	f, version, info, err := openCollectionSnapshotFile(path)
	if err != nil {
		return 0, nil, err
	}
	if err := f.Close(); err != nil {
		return 0, nil, fmt.Errorf("close collection snapshot after inspection: %w", err)
	}
	return version, info, nil
}

func openCollectionSnapshotFile(path string) (*os.File, uint32, os.FileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, nil, err
	}
	version, info, err := inspectOpenCollectionSnapshotFile(f)
	if err != nil {
		_ = f.Close()
		return nil, 0, nil, err
	}
	return f, version, info, nil
}

func inspectOpenCollectionSnapshotFile(f *os.File) (uint32, os.FileInfo, error) {
	info, err := f.Stat()
	if err != nil {
		return 0, nil, fmt.Errorf("stat collection snapshot: %w", err)
	}
	if !info.Mode().IsRegular() {
		return 0, nil, fmt.Errorf("collection snapshot is not a regular file")
	}
	if info.Mode().Perm() != 0o600 {
		return 0, nil, fmt.Errorf("collection snapshot has permissions %04o; expected 0600", info.Mode().Perm())
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return 0, nil, fmt.Errorf("seek collection snapshot: %w", err)
	}
	prefix := make([]byte, len(collectionSnapshotV2Magic))
	n, readErr := io.ReadFull(f, prefix)
	if readErr != nil && readErr != io.ErrUnexpectedEOF {
		return 0, nil, fmt.Errorf("read collection snapshot prefix: %w", readErr)
	}
	if n == len(prefix) && bytes.Equal(prefix, collectionSnapshotV2Magic[:]) {
		return collectionSnapshotVersion, info, nil
	}
	if n > 0 && prefix[0] == '{' {
		if info.Size() > collectionSnapshotV1MaxReadableBytes {
			return 0, nil, fmt.Errorf(
				"legacy v1 collection snapshot is %d bytes; bounded compatibility loader maximum is %d; migrate offline",
				info.Size(),
				collectionSnapshotV1MaxReadableBytes,
			)
		}
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return 0, nil, fmt.Errorf("rewind collection snapshot: %w", err)
		}
		return collectionSnapshotV1Version, info, nil
	}
	return 0, nil, fmt.Errorf("unknown collection snapshot format")
}

func readOpenCollectionFileBounded(
	f *os.File,
	info os.FileInfo,
	maximum int64,
	label string,
	limitDescription string,
) ([]byte, error) {
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", label)
	}
	if info.Size() < 0 {
		return nil, fmt.Errorf("%s has invalid negative size %d", label, info.Size())
	}
	if info.Size() > maximum {
		return nil, fmt.Errorf(
			"%s is %d bytes; %s maximum is %d; migrate offline",
			label,
			info.Size(),
			limitDescription,
			maximum,
		)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek %s: %w", label, err)
	}
	data := make([]byte, int(info.Size()))
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, fmt.Errorf("read %s: %w", label, err)
	}
	var extra [1]byte
	if n, err := f.Read(extra[:]); n != 0 || err != io.EOF {
		if err == nil {
			err = fmt.Errorf("unexpected trailing byte")
		}
		return nil, fmt.Errorf("%s changed while being read: %w", label, err)
	}
	if err := ensureOpenCollectionFileInfo(f, info, label); err != nil {
		return nil, err
	}
	return data, nil
}

func sameCollectionFileInfo(left, right os.FileInfo) bool {
	return left != nil && right != nil &&
		os.SameFile(left, right) &&
		left.Size() == right.Size() &&
		left.Mode() == right.Mode() &&
		left.ModTime().Equal(right.ModTime())
}

func ensureOpenCollectionFileInfo(f *os.File, expected os.FileInfo, label string) error {
	actual, err := f.Stat()
	if err != nil {
		return fmt.Errorf("stat %s after read: %w", label, err)
	}
	if !sameCollectionFileInfo(expected, actual) {
		return fmt.Errorf("%s changed while being read", label)
	}
	return nil
}

func ensureCollectionSnapshotPathIdentity(path string, expected os.FileInfo) error {
	actual, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat collection snapshot path after validation: %w", err)
	}
	if !sameCollectionFileInfo(expected, actual) {
		return fmt.Errorf("collection snapshot changed between validation and load")
	}
	return nil
}

func readLegacyCollectionStateFile(path, label string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", label, err)
	}
	data, err := readOpenCollectionFileBounded(
		f,
		info,
		collectionLegacyStateMaxReadableBytes,
		label,
		"bounded online migration",
	)
	if err != nil {
		return nil, err
	}
	actual, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat %s path after read: %w", label, err)
	}
	if !sameCollectionFileInfo(info, actual) {
		return nil, fmt.Errorf("%s changed while being read", label)
	}
	return data, nil
}

type decodedCollectionSnapshotV1 struct {
	metadata CollectionSnapshotMetadata
	manager  json.RawMessage
	tenants  json.RawMessage
}

func decodeUnifiedCollectionSnapshotV1(data []byte) (decodedCollectionSnapshotV1, error) {
	var decoded decodedCollectionSnapshotV1

	var envelope collectionSnapshotEnvelope
	if err := decodeCollectionJSON(data, &envelope); err != nil {
		return decoded, fmt.Errorf("decode collection snapshot envelope: %w", err)
	}
	if envelope.Magic != collectionSnapshotMagic {
		return decoded, fmt.Errorf("unknown collection snapshot magic %q", envelope.Magic)
	}
	if envelope.Version != collectionSnapshotV1Version {
		return decoded, fmt.Errorf("unsupported collection snapshot version %d", envelope.Version)
	}
	storeID, err := parseCollectionStoreID(envelope.StoreID)
	if err != nil {
		return decoded, fmt.Errorf("invalid collection snapshot store ID: %w", err)
	}
	if len(envelope.Payload) == 0 || bytes.Equal(envelope.Payload, []byte("null")) {
		return decoded, fmt.Errorf("collection snapshot payload is missing or null")
	}
	wantChecksum := collectionSnapshotChecksum(storeID, envelope.Payload)
	if subtle.ConstantTimeCompare([]byte(envelope.Checksum), []byte(wantChecksum)) != 1 {
		return decoded, fmt.Errorf("collection snapshot checksum mismatch")
	}

	var payload collectionSnapshotPayload
	if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
		return decoded, fmt.Errorf("decode collection snapshot payload: %w", err)
	}
	if len(payload.Manager) == 0 || bytes.Equal(payload.Manager, []byte("null")) {
		return decoded, fmt.Errorf("collection snapshot manager state is missing or null")
	}
	if len(payload.Tenants) == 0 || bytes.Equal(payload.Tenants, []byte("null")) {
		return decoded, fmt.Errorf("collection snapshot tenant state is missing or null")
	}
	decoded.metadata.StoreID = storeID
	decoded.metadata.AppliedLSN = payload.AppliedLSN
	decoded.manager = payload.Manager
	decoded.tenants = payload.Tenants
	return decoded, nil
}

func buildUnifiedCollectionSnapshotV1(decoded decodedCollectionSnapshotV1, storagePath string) (*CollectionManager, *TenantManager, error) {
	manager, err := decodeCollectionManagerState(decoded.manager, storagePath)
	if err != nil {
		return nil, nil, fmt.Errorf("decode collection manager: %w", err)
	}
	tenants, err := decodeTenantManagerState(decoded.tenants, storagePath)
	if err != nil {
		manager.closeAll()
		return nil, nil, fmt.Errorf("decode collection tenants: %w", err)
	}
	return manager, tenants, nil
}

func validateCollectionInitializationMarkerIfPresent(basePath string, storeID [16]byte) error {
	exists, err := pathExists(collectionMarkerPath(basePath))
	if err != nil {
		return fmt.Errorf("inspect collection initialization marker: %w", err)
	}
	if !exists {
		return nil
	}
	markerID, err := readCollectionInitializationMarker(basePath)
	if err != nil {
		return fmt.Errorf("read collection initialization marker: %w", err)
	}
	if markerID != storeID {
		return fmt.Errorf("collection initialization marker store ID does not match snapshot")
	}
	return nil
}

type collectionSnapshotV2ReadResult struct {
	manager  *CollectionManager
	tenants  *TenantManager
	metadata CollectionSnapshotMetadata
	digest   [sha256.Size]byte
}

// readUnifiedCollectionSnapshotV2 validates one framed generation without
// reading the file into memory. A normal open runs this once without building
// indexes, then a second time into detached managers. A malformed or changed
// file therefore never becomes live state.
func readUnifiedCollectionSnapshotV2File(f *os.File, info os.FileInfo, storagePath string, build bool) (collectionSnapshotV2ReadResult, error) {
	var result collectionSnapshotV2ReadResult
	if info.Size() < int64(collectionSnapshotV2HeaderSize)+sha256.Size {
		return result, fmt.Errorf("version 2 collection snapshot is too short")
	}
	if err := ensureOpenCollectionFileInfo(f, info, "collection snapshot"); err != nil {
		return result, err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return result, fmt.Errorf("rewind version 2 collection snapshot: %w", err)
	}
	bodySize := info.Size() - sha256.Size
	limited := &io.LimitedReader{R: f, N: bodySize}
	h := sha256.New()
	body := io.TeeReader(limited, h)
	header := make([]byte, collectionSnapshotV2HeaderSize)
	if _, err := io.ReadFull(body, header); err != nil {
		return result, fmt.Errorf("read version 2 collection snapshot header: %w", err)
	}
	metadata, rootCollectionCount, tenantCount, err := decodeCollectionSnapshotV2Header(header)
	if err != nil {
		return result, err
	}

	manager, err := readCollectionSnapshotV2Manager(body, rootCollectionCount, storagePath, build)
	if err != nil {
		return result, fmt.Errorf("decode root manager: %w", err)
	}
	var tenants *TenantManager
	if build {
		tenants = NewTenantManager(storagePath)
	}
	complete := false
	defer func() {
		if !complete && build {
			manager.closeAll()
			tenants.closeAll()
		}
	}()

	seenTenants := make(map[string]struct{})
	for i := uint64(0); i < tenantCount; i++ {
		frame, err := readCollectionSnapshotV2Frame(body)
		if err != nil {
			return result, fmt.Errorf("read tenant %d descriptor: %w", i, err)
		}
		var descriptor collectionSnapshotV2Tenant
		if err := decodeCollectionJSON(frame, &descriptor); err != nil {
			return result, fmt.Errorf("decode tenant %d descriptor: %w", i, err)
		}
		if descriptor.TenantID == "" {
			return result, fmt.Errorf("tenant ID cannot be empty")
		}
		if _, exists := seenTenants[descriptor.TenantID]; exists {
			return result, fmt.Errorf("duplicate tenant %q", descriptor.TenantID)
		}
		seenTenants[descriptor.TenantID] = struct{}{}
		tenantPath := ""
		if storagePath != "" {
			tenantPath = storagePath + "/" + descriptor.TenantID
		}
		tenantManager, err := readCollectionSnapshotV2Manager(body, descriptor.CollectionCount, tenantPath, build)
		if err != nil {
			return result, fmt.Errorf("decode tenant %s: %w", descriptor.TenantID, err)
		}
		if build {
			tenants.tenants[descriptor.TenantID] = tenantManager
			if descriptor.Record != nil {
				// The manager is already in tenants.tenants above, so
				// putTenantRecordDirect's getOrCreateManager call finds it
				// and does not create a second one.
				tenants.putTenantRecordDirect(*descriptor.Record)
			}
		}
	}

	if limited.N != 0 {
		var extra [1]byte
		n, err := body.Read(extra[:])
		if n != 0 {
			return result, fmt.Errorf("version 2 collection snapshot has unexpected trailing body bytes")
		}
		if err == io.EOF {
			return result, fmt.Errorf("version 2 collection snapshot was truncated while reading body")
		}
		if err != nil {
			return result, fmt.Errorf("finish version 2 collection snapshot body: %w", err)
		}
		return result, fmt.Errorf("version 2 collection snapshot body made no read progress")
	}

	wantChecksum := make([]byte, sha256.Size)
	if _, err := io.ReadFull(f, wantChecksum); err != nil {
		return result, fmt.Errorf("read version 2 collection snapshot checksum: %w", err)
	}
	digest := h.Sum(nil)
	if subtle.ConstantTimeCompare(wantChecksum, digest) != 1 {
		return result, fmt.Errorf("collection snapshot checksum mismatch")
	}
	var extra [1]byte
	if n, err := f.Read(extra[:]); n != 0 || err != io.EOF {
		if err == nil {
			err = fmt.Errorf("unexpected trailing byte")
		}
		return result, fmt.Errorf("collection snapshot changed or has trailing data: %w", err)
	}
	if err := ensureOpenCollectionFileInfo(f, info, "collection snapshot"); err != nil {
		return result, err
	}

	complete = true
	result.manager = manager
	result.tenants = tenants
	result.metadata = metadata
	copy(result.digest[:], digest)
	return result, nil
}

func decodeCollectionSnapshotV2Header(header []byte) (CollectionSnapshotMetadata, uint64, uint64, error) {
	var metadata CollectionSnapshotMetadata
	if len(header) != int(collectionSnapshotV2HeaderSize) {
		return metadata, 0, 0, fmt.Errorf("invalid version 2 collection snapshot header size %d", len(header))
	}
	if !bytes.Equal(header[:8], collectionSnapshotV2Magic[:]) {
		return metadata, 0, 0, fmt.Errorf("invalid version 2 collection snapshot magic")
	}
	if version := binary.BigEndian.Uint32(header[8:12]); version != collectionSnapshotVersion {
		return metadata, 0, 0, fmt.Errorf("unsupported collection snapshot version %d", version)
	}
	if headerSize := binary.BigEndian.Uint32(header[12:16]); headerSize != collectionSnapshotV2HeaderSize {
		return metadata, 0, 0, fmt.Errorf("unsupported version 2 collection snapshot header size %d", headerSize)
	}
	copy(metadata.StoreID[:], header[16:32])
	if metadata.StoreID == ([16]byte{}) {
		return CollectionSnapshotMetadata{}, 0, 0, fmt.Errorf("collection snapshot store ID cannot be zero")
	}
	metadata.AppliedLSN = binary.BigEndian.Uint64(header[32:40])
	return metadata, binary.BigEndian.Uint64(header[40:48]), binary.BigEndian.Uint64(header[48:56]), nil
}

func readCollectionSnapshotV2Manager(r io.Reader, collectionCount uint64, storagePath string, build bool) (*CollectionManager, error) {
	var manager *CollectionManager
	if build {
		manager = NewCollectionManager(storagePath)
	}
	complete := false
	defer func() {
		if build && !complete {
			manager.closeAll()
		}
	}()

	seenCollections := make(map[string]struct{})
	for i := uint64(0); i < collectionCount; i++ {
		frame, err := readCollectionSnapshotV2Frame(r)
		if err != nil {
			return nil, fmt.Errorf("read collection %d descriptor: %w", i, err)
		}
		var descriptor collectionSnapshotV2Collection
		if err := decodeCollectionJSON(frame, &descriptor); err != nil {
			return nil, fmt.Errorf("decode collection %d descriptor: %w", i, err)
		}
		name := descriptor.Schema.Name
		if name == "" {
			return nil, fmt.Errorf("collection %d has empty schema name", i)
		}
		if _, exists := seenCollections[name]; exists {
			return nil, fmt.Errorf("duplicate collection %q", name)
		}
		seenCollections[name] = struct{}{}
		if err := descriptor.Schema.Validate(); err != nil {
			return nil, fmt.Errorf("invalid schema for collection %s: %w", name, err)
		}

		var coll *Collection
		if build {
			coll, err = NewCollection(descriptor.Schema)
			if err != nil {
				return nil, fmt.Errorf("recreate collection %s: %w", name, err)
			}
		}

		var maxDocumentID uint64
		hasDocuments := false
		for documentIndex := uint64(0); documentIndex < descriptor.DocumentCount; documentIndex++ {
			frame, err := readCollectionSnapshotV2Frame(r)
			if err != nil {
				if build {
					coll.closeDirect()
				}
				return nil, fmt.Errorf("read document %d in collection %s: %w", documentIndex, name, err)
			}
			var record collectionSnapshotV2Document
			if err := decodeCollectionJSON(frame, &record); err != nil {
				if build {
					coll.closeDirect()
				}
				return nil, fmt.Errorf("decode document %d in collection %s: %w", documentIndex, name, err)
			}
			doc, err := normalizeCollectionSnapshotV2Document(record, &descriptor.Schema)
			if err != nil {
				if build {
					coll.closeDirect()
				}
				return nil, fmt.Errorf("invalid document %d in collection %s: %w", record.ID, name, err)
			}
			if record.ID == math.MaxUint64 {
				if build {
					coll.closeDirect()
				}
				return nil, fmt.Errorf("document ID %d in collection %s cannot be represented by the next-ID cursor", record.ID, name)
			}
			if build {
				if err := restoreCollectionSnapshotV2Document(coll, doc); err != nil {
					coll.closeDirect()
					return nil, fmt.Errorf("restore document %d in collection %s: %w", record.ID, name, err)
				}
			}
			if !hasDocuments || record.ID > maxDocumentID {
				maxDocumentID = record.ID
			}
			hasDocuments = true
		}

		if descriptor.NextID == 0 {
			if build {
				coll.closeDirect()
			}
			return nil, fmt.Errorf("collection %s next document ID cannot be zero", name)
		}
		if hasDocuments && descriptor.NextID <= maxDocumentID {
			if build {
				coll.closeDirect()
			}
			return nil, fmt.Errorf(
				"collection %s next document ID %d must be greater than maximum document ID %d",
				name,
				descriptor.NextID,
				maxDocumentID,
			)
		}
		if build {
			coll.nextID = descriptor.NextID
			manager.collections[name] = coll
		}
	}

	complete = true
	return manager, nil
}

func normalizeCollectionSnapshotV2Document(record collectionSnapshotV2Document, schema *CollectionSchema) (*Document, error) {
	if record.Document == nil {
		return nil, fmt.Errorf("document state is null")
	}
	doc := record.Document
	if doc.ID != 0 && doc.ID != record.ID {
		return nil, fmt.Errorf("document key %d does not match embedded ID %d", record.ID, doc.ID)
	}
	doc.ID = record.ID
	if err := normalizeDocumentVectorTypes(doc, schema); err != nil {
		return nil, err
	}
	return doc, nil
}

func restoreCollectionSnapshotV2Document(coll *Collection, doc *Document) error {
	if _, exists := coll.documents[doc.ID]; exists {
		return fmt.Errorf("duplicate document ID %d", doc.ID)
	}
	for fieldName, vector := range doc.Vectors {
		field := coll.schema.GetField(fieldName)
		if field == nil {
			return fmt.Errorf("unknown vector field %s", fieldName)
		}
		if err := coll.addToIndex(context.Background(), *field, doc.ID, vector); err != nil {
			return err
		}
		if len(doc.Metadata) > 0 {
			if err := coll.setIndexMetadata(*field, doc.ID, doc.Metadata); err != nil {
				return fmt.Errorf("metadata for field %s: %w", fieldName, err)
			}
		}
	}
	coll.documents[doc.ID] = doc
	return nil
}

func readCollectionSnapshotV2Frame(r io.Reader) ([]byte, error) {
	var sizeBytes [4]byte
	if _, err := io.ReadFull(r, sizeBytes[:]); err != nil {
		return nil, err
	}
	size := binary.BigEndian.Uint32(sizeBytes[:])
	if size == 0 || size > collectionSnapshotV2MaxFrameSize {
		return nil, fmt.Errorf("snapshot frame is %d bytes; maximum is %d", size, collectionSnapshotV2MaxFrameSize)
	}
	payload := make([]byte, int(size))
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func ensureCollectionInitializationMarker(basePath string, storeID [16]byte) error {
	path := collectionMarkerPath(basePath)
	markerID, err := readCollectionInitializationMarker(basePath)
	if err == nil {
		if markerID != storeID {
			return fmt.Errorf("collection initialization marker store ID does not match snapshot")
		}
		return nil
	}
	if !os.IsNotExist(err) {
		return fmt.Errorf("read collection initialization marker: %w", err)
	}

	markerBytes, err := json.Marshal(collectionInitializationMarker{
		Magic:   collectionMarkerMagic,
		Version: collectionMarkerVersion,
		StoreID: hex.EncodeToString(storeID[:]),
	})
	if err != nil {
		return fmt.Errorf("marshal collection initialization marker: %w", err)
	}
	if err := writeCollectionFileAtomic(path, markerBytes, 0o600); err != nil {
		return fmt.Errorf("commit collection initialization marker: %w", err)
	}
	return nil
}

func readCollectionInitializationMarker(basePath string) ([16]byte, error) {
	var zero [16]byte
	path := collectionMarkerPath(basePath)
	f, err := os.Open(path)
	if err != nil {
		return zero, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return zero, fmt.Errorf("stat collection initialization marker: %w", err)
	}
	if !info.Mode().IsRegular() {
		return zero, fmt.Errorf("collection initialization marker is not a regular file")
	}
	if info.Mode().Perm() != 0o600 {
		return zero, fmt.Errorf("collection initialization marker has permissions %04o; expected 0600", info.Mode().Perm())
	}
	if info.Size() > collectionMarkerMaxReadableBytes {
		return zero, fmt.Errorf("collection initialization marker is %d bytes; maximum is %d", info.Size(), collectionMarkerMaxReadableBytes)
	}
	data, err := io.ReadAll(io.LimitReader(f, collectionMarkerMaxReadableBytes+1))
	if err != nil {
		return zero, fmt.Errorf("read collection initialization marker: %w", err)
	}
	if int64(len(data)) > collectionMarkerMaxReadableBytes {
		return zero, fmt.Errorf("collection initialization marker exceeds %d bytes", collectionMarkerMaxReadableBytes)
	}
	var marker collectionInitializationMarker
	if err := decodeCollectionJSON(data, &marker); err != nil {
		return zero, fmt.Errorf("decode collection initialization marker: %w", err)
	}
	if marker.Magic != collectionMarkerMagic || marker.Version != collectionMarkerVersion {
		return zero, fmt.Errorf("invalid collection initialization marker")
	}
	markerID, err := parseCollectionStoreID(marker.StoreID)
	if err != nil {
		return zero, fmt.Errorf("invalid collection initialization marker store ID: %w", err)
	}
	return markerID, nil
}

func pathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// openUnifiedCollectionSnapshot loads a complete unified generation, or
// performs a one-time strict migration from the legacy manager/tenant files.
// It writes an initial empty snapshot before first service so later deletion is
// distinguishable from a genuinely fresh store.
func openUnifiedCollectionSnapshot(basePath, storagePath string) (*CollectionManager, *TenantManager, CollectionSnapshotMetadata, error) {
	var metadata CollectionSnapshotMetadata
	snapshotExists, err := pathExists(collectionSnapshotPath(basePath))
	if err != nil {
		return nil, nil, metadata, fmt.Errorf("inspect collection snapshot: %w", err)
	}
	markerExists, err := pathExists(collectionMarkerPath(basePath))
	if err != nil {
		return nil, nil, metadata, fmt.Errorf("inspect collection initialization marker: %w", err)
	}

	if snapshotExists {
		manager, tenants, loadedMetadata, err := loadUnifiedCollectionSnapshot(basePath, storagePath)
		if err != nil {
			return nil, nil, metadata, err
		}
		if err := ensureCollectionInitializationMarker(basePath, loadedMetadata.StoreID); err != nil {
			manager.closeAll()
			for _, tenantID := range tenants.ListTenants() {
				if mgr := tenants.getManager(tenantID); mgr != nil {
					mgr.closeAll()
				}
			}
			return nil, nil, metadata, err
		}
		return manager, tenants, loadedMetadata, nil
	}
	if markerExists {
		return nil, nil, metadata, ErrCollectionSnapshotNotFound
	}

	legacyManagerPath := basePath + ".manager"
	legacyTenantPath := basePath + ".tenants"
	managerExists, err := pathExists(legacyManagerPath)
	if err != nil {
		return nil, nil, metadata, fmt.Errorf("inspect legacy collection manager: %w", err)
	}
	tenantExists, err := pathExists(legacyTenantPath)
	if err != nil {
		return nil, nil, metadata, fmt.Errorf("inspect legacy tenant manager: %w", err)
	}
	if tenantExists && !managerExists {
		return nil, nil, metadata, fmt.Errorf("legacy tenant state exists without the required manager state")
	}

	manager := NewCollectionManager(storagePath)
	tenants := NewTenantManager(storagePath)
	if managerExists {
		managerData, err := readLegacyCollectionStateFile(legacyManagerPath, "legacy collection manager")
		if err != nil {
			return nil, nil, metadata, fmt.Errorf("read legacy collection manager: %w", err)
		}
		manager, err = decodeCollectionManagerState(managerData, storagePath)
		if err != nil {
			return nil, nil, metadata, fmt.Errorf("load legacy collection manager: %w", err)
		}
		if tenantExists {
			tenantData, err := readLegacyCollectionStateFile(legacyTenantPath, "legacy collection tenants")
			if err != nil {
				manager.closeAll()
				return nil, nil, metadata, fmt.Errorf("read legacy tenant manager: %w", err)
			}
			tenants, err = decodeTenantManagerState(tenantData, storagePath)
			if err != nil {
				manager.closeAll()
				return nil, nil, metadata, fmt.Errorf("load legacy tenant manager: %w", err)
			}
		}
	}

	storeID, err := newCollectionStoreID()
	if err != nil {
		manager.closeAll()
		return nil, nil, metadata, err
	}
	metadata.StoreID = storeID
	if err := saveUnifiedCollectionSnapshot(basePath, manager, tenants, metadata); err != nil {
		manager.closeAll()
		for _, tenantID := range tenants.ListTenants() {
			if mgr := tenants.getManager(tenantID); mgr != nil {
				mgr.closeAll()
			}
		}
		return nil, nil, CollectionSnapshotMetadata{}, fmt.Errorf("initialize unified collection snapshot: %w", err)
	}
	return manager, tenants, metadata, nil
}

// OpenUnifiedCollectionSnapshot validates and opens the unified snapshot,
// migrating legacy state exactly once when necessary.
func OpenUnifiedCollectionSnapshot(basePath, storagePath string) (*CollectionManager, *TenantManager, CollectionSnapshotMetadata, error) {
	return openUnifiedCollectionSnapshot(basePath, storagePath)
}
