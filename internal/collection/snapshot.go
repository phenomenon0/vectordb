package collection

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	collectionSnapshotMagic   = "deepdata-collections"
	collectionSnapshotVersion = uint32(1)
	collectionMarkerMagic     = "deepdata-collections-initialized"
)

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

type collectionInitializationMarker struct {
	Magic   string `json:"magic"`
	Version uint32 `json:"version"`
	StoreID string `json:"store_id"`
}

var (
	collectionSnapshotFileSync = func(f *os.File) error { return f.Sync() }
	collectionSnapshotDirSync  = syncCollectionSnapshotDirectory
	collectionSnapshotRename   = os.Rename
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
func writeCollectionFileAtomic(path string, data []byte, mode os.FileMode) (retErr error) {
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
	if n, err := f.Write(data); err != nil {
		return fmt.Errorf("write temporary file: %w", err)
	} else if n != len(data) {
		return fmt.Errorf("write temporary file: wrote %d of %d bytes", n, len(data))
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
	binary.BigEndian.PutUint32(version[:], collectionSnapshotVersion)
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
	var zero [16]byte
	if metadata.StoreID == zero {
		return fmt.Errorf("collection snapshot store ID cannot be zero")
	}
	if err := validateExistingCollectionSnapshotTarget(basePath, metadata.StoreID); err != nil {
		return err
	}

	managerState, err := manager.marshalState()
	if err != nil {
		return fmt.Errorf("capture manager state: %w", err)
	}
	tenantState, err := tenants.marshalState()
	if err != nil {
		return fmt.Errorf("capture tenant state: %w", err)
	}
	payloadBytes, err := json.Marshal(collectionSnapshotPayload{
		AppliedLSN: metadata.AppliedLSN,
		Manager:    managerState,
		Tenants:    tenantState,
	})
	if err != nil {
		return fmt.Errorf("marshal collection snapshot payload: %w", err)
	}

	envelopeBytes, err := json.Marshal(collectionSnapshotEnvelope{
		Magic:    collectionSnapshotMagic,
		Version:  collectionSnapshotVersion,
		StoreID:  hex.EncodeToString(metadata.StoreID[:]),
		Payload:  payloadBytes,
		Checksum: collectionSnapshotChecksum(metadata.StoreID, payloadBytes),
	})
	if err != nil {
		return fmt.Errorf("marshal collection snapshot envelope: %w", err)
	}
	if err := writeCollectionFileAtomic(collectionSnapshotPath(basePath), envelopeBytes, 0o600); err != nil {
		return fmt.Errorf("commit collection snapshot: %w", err)
	}
	if err := ensureCollectionInitializationMarker(basePath, metadata.StoreID); err != nil {
		return err
	}
	return nil
}

func validateExistingCollectionSnapshotTarget(basePath string, storeID [16]byte) error {
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
		if markerID != storeID {
			return fmt.Errorf("refusing to replace collection snapshot with a different store ID")
		}
		if !snapshotExists {
			return ErrCollectionSnapshotNotFound
		}
	}
	if snapshotExists {
		existingManager, existingTenants, existingMetadata, err := loadUnifiedCollectionSnapshot(basePath, "")
		if err != nil {
			return fmt.Errorf("refusing to replace invalid existing collection snapshot: %w", err)
		}
		existingManager.closeAll()
		existingTenants.closeAll()
		if existingMetadata.StoreID != storeID {
			return fmt.Errorf("refusing to replace collection snapshot with a different store ID")
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
	var metadata CollectionSnapshotMetadata
	data, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		return nil, nil, metadata, err
	}

	var envelope collectionSnapshotEnvelope
	if err := decodeCollectionJSON(data, &envelope); err != nil {
		return nil, nil, metadata, fmt.Errorf("decode collection snapshot envelope: %w", err)
	}
	if envelope.Magic != collectionSnapshotMagic {
		return nil, nil, metadata, fmt.Errorf("unknown collection snapshot magic %q", envelope.Magic)
	}
	if envelope.Version != collectionSnapshotVersion {
		return nil, nil, metadata, fmt.Errorf("unsupported collection snapshot version %d", envelope.Version)
	}
	storeID, err := parseCollectionStoreID(envelope.StoreID)
	if err != nil {
		return nil, nil, metadata, fmt.Errorf("invalid collection snapshot store ID: %w", err)
	}
	if len(envelope.Payload) == 0 || bytes.Equal(envelope.Payload, []byte("null")) {
		return nil, nil, metadata, fmt.Errorf("collection snapshot payload is missing or null")
	}
	wantChecksum := collectionSnapshotChecksum(storeID, envelope.Payload)
	if subtle.ConstantTimeCompare([]byte(envelope.Checksum), []byte(wantChecksum)) != 1 {
		return nil, nil, metadata, fmt.Errorf("collection snapshot checksum mismatch")
	}

	var payload collectionSnapshotPayload
	if err := decodeCollectionJSON(envelope.Payload, &payload); err != nil {
		return nil, nil, metadata, fmt.Errorf("decode collection snapshot payload: %w", err)
	}
	if len(payload.Manager) == 0 || bytes.Equal(payload.Manager, []byte("null")) {
		return nil, nil, metadata, fmt.Errorf("collection snapshot manager state is missing or null")
	}
	if len(payload.Tenants) == 0 || bytes.Equal(payload.Tenants, []byte("null")) {
		return nil, nil, metadata, fmt.Errorf("collection snapshot tenant state is missing or null")
	}

	manager, err := decodeCollectionManagerState(payload.Manager, storagePath)
	if err != nil {
		return nil, nil, metadata, fmt.Errorf("decode collection manager: %w", err)
	}
	tenants, err := decodeTenantManagerState(payload.Tenants, storagePath)
	if err != nil {
		manager.closeAll()
		return nil, nil, metadata, fmt.Errorf("decode collection tenants: %w", err)
	}
	metadata.StoreID = storeID
	metadata.AppliedLSN = payload.AppliedLSN
	return manager, tenants, metadata, nil
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
		Version: collectionSnapshotVersion,
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
	data, err := os.ReadFile(collectionMarkerPath(basePath))
	if err != nil {
		return zero, err
	}
	var marker collectionInitializationMarker
	if err := decodeCollectionJSON(data, &marker); err != nil {
		return zero, fmt.Errorf("decode collection initialization marker: %w", err)
	}
	if marker.Magic != collectionMarkerMagic || marker.Version != collectionSnapshotVersion {
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
		managerData, err := os.ReadFile(legacyManagerPath)
		if err != nil {
			return nil, nil, metadata, fmt.Errorf("read legacy collection manager: %w", err)
		}
		manager, err = decodeCollectionManagerState(managerData, storagePath)
		if err != nil {
			return nil, nil, metadata, fmt.Errorf("load legacy collection manager: %w", err)
		}
		if tenantExists {
			tenantData, err := os.ReadFile(legacyTenantPath)
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
