package collection

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	vectorindex "github.com/phenomenon0/vectordb/internal/index"
)

type snapshotExportRejectingIndex struct {
	vectorindex.Index
}

func (snapshotExportRejectingIndex) Export() ([]byte, error) {
	return nil, errors.New("index export must not be called by snapshot v2")
}

func writeLegacyV1UnifiedSnapshot(t *testing.T, basePath string, manager *CollectionManager, tenants *TenantManager, metadata CollectionSnapshotMetadata) {
	t.Helper()
	managerState, err := manager.marshalState()
	if err != nil {
		t.Fatalf("marshal legacy manager: %v", err)
	}
	tenantState, err := tenants.marshalState()
	if err != nil {
		t.Fatalf("marshal legacy tenants: %v", err)
	}
	payload, err := json.Marshal(collectionSnapshotPayload{
		AppliedLSN: metadata.AppliedLSN,
		Manager:    managerState,
		Tenants:    tenantState,
	})
	if err != nil {
		t.Fatalf("marshal legacy payload: %v", err)
	}
	envelope, err := json.Marshal(collectionSnapshotEnvelope{
		Magic:    collectionSnapshotMagic,
		Version:  collectionSnapshotV1Version,
		StoreID:  hex.EncodeToString(metadata.StoreID[:]),
		Payload:  payload,
		Checksum: collectionSnapshotChecksum(metadata.StoreID, payload),
	})
	if err != nil {
		t.Fatalf("marshal legacy envelope: %v", err)
	}
	if err := os.WriteFile(collectionSnapshotPath(basePath), envelope, 0o600); err != nil {
		t.Fatalf("write legacy snapshot: %v", err)
	}
}

func TestUnifiedCollectionSnapshotV2DoesNotExportIndexes(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := seedManagerForSnapshot(t, basePath, "docs")
	coll, err := manager.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	coll.mu.Lock()
	coll.indexes["embedding"] = snapshotExportRejectingIndex{Index: coll.indexes["embedding"]}
	coll.mu.Unlock()
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	metadata.AppliedLSN = 17

	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err != nil {
		t.Fatalf("save framed snapshot without index export: %v", err)
	}
	prefix := make([]byte, len(collectionSnapshotV2Magic))
	f, err := os.Open(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Read(prefix); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if string(prefix) != string(collectionSnapshotV2Magic[:]) {
		t.Fatalf("snapshot prefix = %q, want %q", prefix, collectionSnapshotV2Magic)
	}

	loaded, _, loadedMetadata, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("open framed snapshot: %v", err)
	}
	if loadedMetadata != metadata {
		t.Fatalf("metadata = %+v, want %+v", loadedMetadata, metadata)
	}
	loadedCollection, err := loaded.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	if loadedCollection.Count() != 1 {
		t.Fatalf("loaded document count = %d, want 1", loadedCollection.Count())
	}
	response, err := loaded.SearchCollection(context.Background(), SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"embedding": []float32{0, 0, 0, 0}},
		TopK:           1,
	})
	if err != nil {
		t.Fatalf("search rebuilt index: %v", err)
	}
	if len(response.Documents) != 1 {
		t.Fatalf("rebuilt index returned %d results, want 1", len(response.Documents))
	}
}

func TestUnifiedCollectionSnapshotV1RemainsReadableAndRewritesAsV2(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := seedManagerForSnapshot(t, basePath, "legacy")
	tenants := NewTenantManager(basePath)
	if _, err := tenants.CreateCollection(context.Background(), "tenant-a", testSchema("tenant-docs", 4)); err != nil {
		t.Fatal(err)
	}
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	metadata.AppliedLSN = 9
	writeLegacyV1UnifiedSnapshot(t, basePath, manager, tenants, metadata)

	loadedManager, loadedTenants, loadedMetadata, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("open legacy v1 snapshot: %v", err)
	}
	if loadedMetadata != metadata || !loadedManager.HasCollection("legacy") {
		t.Fatalf("legacy state was not preserved: metadata=%+v collections=%v", loadedMetadata, loadedManager.ListCollections())
	}
	if got := loadedTenants.ListTenants(); len(got) != 1 || got[0] != "tenant-a" {
		t.Fatalf("legacy tenants = %v", got)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, loadedManager, loadedTenants, loadedMetadata); err != nil {
		t.Fatalf("rewrite legacy snapshot as v2: %v", err)
	}
	version, _, err := inspectCollectionSnapshotFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	if version != collectionSnapshotVersion {
		t.Fatalf("rewritten version = %d, want %d", version, collectionSnapshotVersion)
	}
}

func TestUnifiedCollectionSnapshotV2RoundTripsFieldPartialDocuments(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := NewCollectionManager(basePath)
	schema := CollectionSchema{
		Name: "bulk",
		Fields: []VectorField{
			{Name: "first", Type: VectorTypeDense, Dim: 2, Index: IndexConfig{Type: IndexTypeFLAT}},
			{Name: "second", Type: VectorTypeDense, Dim: 2, Index: IndexConfig{Type: IndexTypeFLAT}},
		},
	}
	if _, err := manager.CreateCollection(context.Background(), schema); err != nil {
		t.Fatal(err)
	}
	ids := []uint64{0, 7}
	if err := manager.BulkAddDense(context.Background(), "bulk", "first", ids, [][]float32{{1, 0}, {0, 1}}); err != nil {
		t.Fatal(err)
	}
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}

	loaded, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("open field-partial v2 snapshot: %v", err)
	}
	for _, id := range ids {
		doc, err := loaded.GetDocument("bulk", id)
		if err != nil {
			t.Fatalf("get document %d: %v", id, err)
		}
		if doc.ID != id || len(doc.Vectors) != 1 || doc.Vectors["first"] == nil {
			t.Fatalf("partial document %d = %+v", id, doc)
		}
	}
	if err := loaded.BulkAddDense(context.Background(), "bulk", "second", ids, [][]float32{{0.5, 0.5}, {0.25, 0.75}}); err != nil {
		t.Fatalf("extend restored partial documents: %v", err)
	}
}

func TestUnifiedCollectionSnapshotV2RebuildsDenseAndSparseIndexes(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := NewCollectionManager(basePath)
	schema := CollectionSchema{
		Name: "hybrid",
		Fields: []VectorField{
			{Name: "embedding", Type: VectorTypeDense, Dim: 2, Index: IndexConfig{Type: IndexTypeFLAT}},
			{Name: "tokens", Type: VectorTypeSparse, Dim: 8, Index: IndexConfig{Type: IndexTypeInverted}},
		},
	}
	if _, err := manager.CreateCollection(context.Background(), schema); err != nil {
		t.Fatal(err)
	}
	doc := &Document{
		Vectors: map[string]interface{}{
			"embedding": []float32{1, 0},
			"tokens": map[string]interface{}{
				"indices": []uint32{2},
				"values":  []float32{1},
				"dim":     8,
			},
		},
		Metadata: map[string]interface{}{"title": "kept"},
	}
	if err := manager.AddDocument(context.Background(), "hybrid", doc); err != nil {
		t.Fatal(err)
	}
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}

	loaded, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatal(err)
	}
	for field, query := range map[string]interface{}{
		"embedding": []float32{1, 0},
		"tokens": map[string]interface{}{
			"indices": []uint32{2},
			"values":  []float32{1},
			"dim":     8,
		},
	} {
		response, err := loaded.SearchCollection(context.Background(), SearchRequest{
			CollectionName: "hybrid",
			Queries:        map[string]interface{}{field: query},
			TopK:           1,
		})
		if err != nil {
			t.Fatalf("search rebuilt %s index: %v", field, err)
		}
		if len(response.Documents) != 1 || response.Documents[0].Metadata["title"] != "kept" {
			t.Fatalf("rebuilt %s result = %+v", field, response.Documents)
		}
	}
}

func TestUnifiedCollectionSnapshotV2RejectsTrailingBodyBytes(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, seedManagerForSnapshot(t, basePath, "docs"), NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	bodyEnd := len(data) - 32
	corrupt := make([]byte, 0, len(data)+1)
	corrupt = append(corrupt, data[:bodyEnd]...)
	corrupt = append(corrupt, 0xff)
	corrupt = append(corrupt, data[bodyEnd:]...)
	if err := os.WriteFile(collectionSnapshotPath(basePath), corrupt, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "trailing body bytes") {
		t.Fatalf("expected trailing-body rejection, got %v", err)
	}
}

func TestUnifiedCollectionSnapshotV2WriteFailurePreservesPriorGeneration(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := seedManagerForSnapshot(t, basePath, "docs")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	coll, err := manager.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	coll.mu.Lock()
	for _, doc := range coll.documents {
		doc.Metadata["cannot_encode"] = make(chan struct{})
		break
	}
	coll.mu.Unlock()
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("expected encoding failure, got %v", err)
	}
	got, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("failed v2 save changed the committed generation")
	}
}

func TestUnifiedCollectionSnapshotV2RefusesIndexStateMissingFromDocuments(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := seedManagerForSnapshot(t, basePath, "docs")
	coll, err := manager.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	coll.mu.Lock()
	for _, doc := range coll.documents {
		delete(doc.Vectors, "embedding")
		break
	}
	coll.mu.Unlock()
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err == nil || !strings.Contains(err.Error(), "v2 snapshot would lose state") {
		t.Fatalf("expected non-representable state rejection, got %v", err)
	}
	if _, err := os.Stat(collectionSnapshotPath(basePath)); !os.IsNotExist(err) {
		t.Fatalf("rejected snapshot created a final generation: %v", err)
	}
}

func TestUnifiedCollectionSnapshotV1OversizeFailsBeforeAllocation(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	path := collectionSnapshotPath(basePath)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte{'{'}); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Truncate(collectionSnapshotV1MaxReadableBytes + 1); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "bounded compatibility loader maximum") {
		t.Fatalf("expected bounded v1 rejection, got %v", err)
	}
}

func TestUnifiedCollectionSnapshotV2RejectsPathSwapBetweenValidationAndBuild(t *testing.T) {
	dir := t.TempDir()
	basePath := filepath.Join(dir, "collections")
	replacementBasePath := filepath.Join(dir, "replacement")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	metadata.AppliedLSN = 11
	if err := SaveUnifiedCollectionSnapshot(basePath, seedManagerForSnapshot(t, basePath, "original"), NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(replacementBasePath, seedManagerForSnapshot(t, replacementBasePath, "replacement"), NewTenantManager(replacementBasePath), metadata); err != nil {
		t.Fatal(err)
	}

	originalHook := collectionSnapshotAfterV2Validation
	t.Cleanup(func() { collectionSnapshotAfterV2Validation = originalHook })
	hookCalled := false
	collectionSnapshotAfterV2Validation = func(path string) error {
		hookCalled = true
		return os.Rename(collectionSnapshotPath(replacementBasePath), path)
	}

	if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "changed between validation and load") {
		t.Fatalf("expected swapped-path rejection, got %v", err)
	}
	if !hookCalled {
		t.Fatal("validation/build swap hook was not called")
	}
}

func TestUnifiedCollectionSnapshotV2ChecksMarkerBeforeBuild(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, seedManagerForSnapshot(t, basePath, "docs"), NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	otherMetadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	marker, err := json.Marshal(collectionInitializationMarker{
		Magic:   collectionMarkerMagic,
		Version: collectionMarkerVersion,
		StoreID: hex.EncodeToString(otherMetadata.StoreID[:]),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(collectionMarkerPath(basePath), marker, 0o600); err != nil {
		t.Fatal(err)
	}

	originalHook := collectionSnapshotAfterV2Validation
	t.Cleanup(func() { collectionSnapshotAfterV2Validation = originalHook })
	hookCalled := false
	collectionSnapshotAfterV2Validation = func(string) error {
		hookCalled = true
		return nil
	}
	if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "marker store ID does not match") {
		t.Fatalf("expected marker mismatch, got %v", err)
	}
	if hookCalled {
		t.Fatal("snapshot build boundary was reached before marker mismatch rejection")
	}
}

func TestUnifiedCollectionSnapshotV2RejectsRegressedAppliedLSN(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := seedManagerForSnapshot(t, basePath, "docs")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	metadata.AppliedLSN = 10
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}

	regressed := metadata
	regressed.AppliedLSN = 9
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), regressed); err == nil || !strings.Contains(err.Error(), "regressed LSN") {
		t.Fatalf("expected applied-LSN regression rejection, got %v", err)
	}
	got, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("rejected applied-LSN regression changed the committed snapshot")
	}
}

func TestUnifiedCollectionSnapshotV2RejectsInvalidNextIDBeforeCommit(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := seedManagerForSnapshot(t, basePath, "docs")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	coll, err := manager.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	coll.SetNextID(1)
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err == nil || !strings.Contains(err.Error(), "must be greater than maximum") {
		t.Fatalf("expected next-ID rejection, got %v", err)
	}
	got, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("rejected next-ID state changed the committed snapshot")
	}
}

func TestUnifiedCollectionSnapshotV2RejectsMaximumDocumentIDBeforeCommit(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager := NewCollectionManager(basePath)
	schema := CollectionSchema{
		Name: "bulk",
		Fields: []VectorField{{
			Name:  "embedding",
			Type:  VectorTypeDense,
			Dim:   1,
			Index: IndexConfig{Type: IndexTypeFLAT},
		}},
	}
	if _, err := manager.CreateCollection(context.Background(), schema); err != nil {
		t.Fatal(err)
	}
	if err := manager.BulkAddDense(context.Background(), "bulk", "embedding", []uint64{math.MaxUint64}, [][]float32{{1}}); err != nil {
		t.Fatal(err)
	}
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err == nil || !strings.Contains(err.Error(), "cannot be represented") {
		t.Fatalf("expected maximum-ID rejection, got %v", err)
	}
	if _, err := os.Stat(collectionSnapshotPath(basePath)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("maximum-ID rejection committed a snapshot: %v", err)
	}
}

func TestUnifiedCollectionSnapshotV2RejectsReloadInvalidSchemaBeforeCommit(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	params := map[string]interface{}{"segments": float64(2)}
	manager := NewCollectionManager(basePath)
	schema := CollectionSchema{
		Name: "docs",
		Fields: []VectorField{{
			Name:  "embedding",
			Type:  VectorTypeDense,
			Dim:   2,
			Index: IndexConfig{Type: IndexTypeHNSW, Params: params},
		}},
	}
	if _, err := manager.CreateCollection(context.Background(), schema); err != nil {
		t.Fatal(err)
	}
	params["segments"] = float64(0)
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, manager, NewTenantManager(basePath), metadata); err == nil || !strings.Contains(err.Error(), "cannot be recreated") {
		t.Fatalf("expected reload-invalid schema rejection, got %v", err)
	}
	if _, err := os.Stat(collectionSnapshotPath(basePath)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("reload-invalid schema committed a snapshot: %v", err)
	}
}

func TestUnifiedCollectionSnapshotV2LoadRejectsRegressedNextID(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, seedManagerForSnapshot(t, basePath, "docs"), NewTenantManager(basePath), metadata); err != nil {
		t.Fatal(err)
	}
	rewriteFirstV2CollectionDescriptor(t, collectionSnapshotPath(basePath), func(descriptor *collectionSnapshotV2Collection) {
		descriptor.NextID = 1
	})
	if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "must be greater than maximum") {
		t.Fatalf("expected corrupt next-ID rejection, got %v", err)
	}
}

func TestLegacyCollectionStateOversizeRequiresOfflineMigration(t *testing.T) {
	writeOversize := func(t *testing.T, path string) {
		t.Helper()
		f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte{'{'}); err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
		if err := f.Truncate(collectionLegacyStateMaxReadableBytes + 1); err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("manager", func(t *testing.T) {
		basePath := filepath.Join(t.TempDir(), "collections")
		writeOversize(t, basePath+".manager")
		if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "bounded online migration maximum") {
			t.Fatalf("expected bounded manager migration rejection, got %v", err)
		}
	})
	t.Run("tenants", func(t *testing.T) {
		basePath := filepath.Join(t.TempDir(), "collections")
		if err := NewCollectionManager(basePath).Save(basePath + ".manager"); err != nil {
			t.Fatal(err)
		}
		writeOversize(t, basePath+".tenants")
		if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "bounded online migration maximum") {
			t.Fatalf("expected bounded tenant migration rejection, got %v", err)
		}
	})
}

func rewriteFirstV2CollectionDescriptor(t *testing.T, path string, mutate func(*collectionSnapshotV2Collection)) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	bodyEnd := len(data) - sha256.Size
	offset := int(collectionSnapshotV2HeaderSize)
	if bodyEnd < offset+4 {
		t.Fatalf("snapshot body is only %d bytes", bodyEnd)
	}
	frameSize := int(binary.BigEndian.Uint32(data[offset : offset+4]))
	frameStart := offset + 4
	frameEnd := frameStart + frameSize
	if frameSize <= 0 || frameEnd > bodyEnd {
		t.Fatalf("invalid first frame size %d", frameSize)
	}
	var descriptor collectionSnapshotV2Collection
	if err := decodeCollectionJSON(data[frameStart:frameEnd], &descriptor); err != nil {
		t.Fatal(err)
	}
	mutate(&descriptor)
	encoded, err := json.Marshal(descriptor)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) != frameSize {
		t.Fatalf("mutated descriptor changed frame size from %d to %d", frameSize, len(encoded))
	}
	copy(data[frameStart:frameEnd], encoded)
	digest := sha256.Sum256(data[:bodyEnd])
	copy(data[bodyEnd:], digest[:])
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
