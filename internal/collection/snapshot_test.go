package collection

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func seedManagerForSnapshot(t *testing.T, storagePath, collectionName string) *CollectionManager {
	t.Helper()
	manager := NewCollectionManager(storagePath)
	if _, err := manager.CreateCollection(context.Background(), testSchema(collectionName, 4)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	doc := testDoc(4)
	doc.Metadata["collection"] = collectionName
	if err := manager.AddDocument(context.Background(), collectionName, doc); err != nil {
		t.Fatalf("add document: %v", err)
	}
	return manager
}

func TestUnifiedCollectionSnapshotRoundTrip(t *testing.T) {
	dir := t.TempDir()
	basePath := filepath.Join(dir, "index.gob.collections")
	manager := seedManagerForSnapshot(t, basePath, "v2-docs")
	tenants := NewTenantManager(basePath)
	if _, err := tenants.CreateCollection(context.Background(), "tenant-a", testSchema("v3-docs", 4)); err != nil {
		t.Fatalf("create tenant collection: %v", err)
	}
	if err := tenants.AddDocument(context.Background(), "tenant-a", "v3-docs", testDoc(4)); err != nil {
		t.Fatalf("add tenant document: %v", err)
	}
	metadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	metadata.AppliedLSN = 42

	if err := SaveUnifiedCollectionSnapshot(basePath, manager, tenants, metadata); err != nil {
		t.Fatalf("save unified snapshot: %v", err)
	}
	for _, path := range []string{collectionSnapshotPath(basePath), collectionMarkerPath(basePath)} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat %s: %v", path, err)
		}
		if got := info.Mode().Perm(); got != 0o600 {
			t.Fatalf("mode for %s = %o, want 600", path, got)
		}
	}

	loadedManager, loadedTenants, loadedMetadata, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("open unified snapshot: %v", err)
	}
	if loadedMetadata != metadata {
		t.Fatalf("metadata = %+v, want %+v", loadedMetadata, metadata)
	}
	if loadedManager.CollectionCount() != 1 {
		t.Fatalf("loaded V2 collection count = %d, want 1", loadedManager.CollectionCount())
	}
	v2, err := loadedManager.GetCollection("v2-docs")
	if err != nil || v2.Count() != 1 {
		t.Fatalf("loaded V2 state: collection=%v err=%v count=%d", v2, err, func() int {
			if v2 == nil {
				return -1
			}
			return v2.Count()
		}())
	}
	if got := loadedTenants.ListTenants(); len(got) != 1 || got[0] != "tenant-a" {
		t.Fatalf("loaded tenants = %v, want [tenant-a]", got)
	}
	stats, err := loadedTenants.GetTenantStats("tenant-a")
	if err != nil || stats.TotalDocuments != 1 {
		t.Fatalf("loaded tenant stats = %+v, err=%v", stats, err)
	}
}

func TestUnifiedCollectionSnapshotRejectsChecksumCorruption(t *testing.T) {
	dir := t.TempDir()
	basePath := filepath.Join(dir, "collections")
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
	if len(data) < sha256.Size {
		t.Fatalf("snapshot is only %d bytes", len(data))
	}
	data[len(data)-1] ^= 0xff
	if err := os.WriteFile(collectionSnapshotPath(basePath), data, 0o600); err != nil {
		t.Fatal(err)
	}

	if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("expected checksum mismatch, got %v", err)
	}
}

func TestUnifiedCollectionSnapshotRefusesDifferentUnloadedStore(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	firstMetadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, seedManagerForSnapshot(t, basePath, "keep"), NewTenantManager(basePath), firstMetadata); err != nil {
		t.Fatal(err)
	}
	wantSnapshot, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}

	secondMetadata, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(basePath, NewCollectionManager(basePath), NewTenantManager(basePath), secondMetadata); err == nil || !strings.Contains(err.Error(), "different store ID") {
		t.Fatalf("expected store-ID replacement refusal, got %v", err)
	}
	gotSnapshot, err := os.ReadFile(collectionSnapshotPath(basePath))
	if err != nil {
		t.Fatal(err)
	}
	if string(gotSnapshot) != string(wantSnapshot) {
		t.Fatal("rejected save changed the existing snapshot")
	}
	manager, _, metadata, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("reopen original snapshot: %v", err)
	}
	if metadata.StoreID != firstMetadata.StoreID || !manager.HasCollection("keep") {
		t.Fatalf("original state was not preserved: metadata=%+v collections=%v", metadata, manager.ListCollections())
	}
}

func TestCollectionManagerRoundTripsFieldPartialBulkDocuments(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manager.json")
	manager := NewCollectionManager(dir)
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
	if err := manager.Save(path); err != nil {
		t.Fatal(err)
	}

	loaded := NewCollectionManager(dir)
	if err := loaded.Load(path); err != nil {
		t.Fatalf("load field-partial bulk snapshot: %v", err)
	}
	for _, id := range ids {
		doc, err := loaded.GetDocument("bulk", id)
		if err != nil {
			t.Fatalf("get bulk document %d: %v", id, err)
		}
		if doc.ID != id || len(doc.Vectors) != 1 || doc.Vectors["first"].IsZero() {
			t.Fatalf("bulk document %d = %+v", id, doc)
		}
	}

	if err := loaded.BulkAddDense(context.Background(), "bulk", "second", ids, [][]float32{{0.5, 0.5}, {0.25, 0.75}}); err != nil {
		t.Fatal(err)
	}
	for _, id := range ids {
		doc, err := loaded.GetDocument("bulk", id)
		if err != nil {
			t.Fatal(err)
		}
		if len(doc.Vectors) != 2 || doc.Vectors["first"].IsZero() || doc.Vectors["second"].IsZero() {
			t.Fatalf("second field import discarded document vectors for %d: %+v", id, doc.Vectors)
		}
	}
}

func TestInitializedCollectionSnapshotCannotDisappearSilently(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	manager, tenants, metadata, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("initialize store: %v", err)
	}
	if manager.CollectionCount() != 0 || tenants.TenantCount() != 0 {
		t.Fatal("fresh store was not empty")
	}
	if metadata.StoreID == ([16]byte{}) {
		t.Fatal("fresh store has zero store ID")
	}
	if err := os.Remove(collectionSnapshotPath(basePath)); err != nil {
		t.Fatal(err)
	}
	_, _, _, err = OpenUnifiedCollectionSnapshot(basePath, basePath)
	if !errors.Is(err, ErrCollectionSnapshotNotFound) {
		t.Fatalf("expected ErrCollectionSnapshotNotFound, got %v", err)
	}
}

func TestUnifiedCollectionSnapshotLegacyMigration(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	legacy := seedManagerForSnapshot(t, basePath, "legacy")
	if err := legacy.Save(basePath + ".manager"); err != nil {
		t.Fatalf("save legacy manager: %v", err)
	}

	manager, tenants, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath)
	if err != nil {
		t.Fatalf("migrate legacy state: %v", err)
	}
	if !manager.HasCollection("legacy") || tenants.TenantCount() != 0 {
		t.Fatalf("unexpected migrated state: collections=%v tenants=%v", manager.ListCollections(), tenants.ListTenants())
	}
	if _, err := os.Stat(collectionSnapshotPath(basePath)); err != nil {
		t.Fatalf("unified snapshot not written: %v", err)
	}
	if _, err := os.Stat(collectionMarkerPath(basePath)); err != nil {
		t.Fatalf("initialization marker not written: %v", err)
	}
}

func TestUnifiedCollectionSnapshotRejectsTenantOnlyLegacyState(t *testing.T) {
	basePath := filepath.Join(t.TempDir(), "collections")
	tenants := NewTenantManager(basePath)
	if _, err := tenants.CreateCollection(context.Background(), "tenant-a", testSchema("docs", 4)); err != nil {
		t.Fatal(err)
	}
	if err := tenants.Save(basePath + ".tenants"); err != nil {
		t.Fatal(err)
	}

	if _, _, _, err := OpenUnifiedCollectionSnapshot(basePath, basePath); err == nil || !strings.Contains(err.Error(), "without the required manager") {
		t.Fatalf("expected one-sided legacy error, got %v", err)
	}
	if _, err := os.Stat(collectionSnapshotPath(basePath)); !os.IsNotExist(err) {
		t.Fatalf("failed migration wrote unified snapshot: %v", err)
	}
}

func TestCollectionManagerLoadFailurePreservesLiveState(t *testing.T) {
	manager := seedManagerForSnapshot(t, "", "keep")
	path := filepath.Join(t.TempDir(), "manager.json")
	if err := os.WriteFile(path, []byte(`{"collections":{"bad":null}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := manager.Load(path); err == nil {
		t.Fatal("expected corrupt manager load to fail")
	}
	if !manager.HasCollection("keep") || manager.CollectionCount() != 1 {
		t.Fatalf("failed load mutated manager: %v", manager.ListCollections())
	}
}

func TestExplicitEmptyManagerStateReplacesLiveState(t *testing.T) {
	manager := seedManagerForSnapshot(t, "", "remove")
	path := filepath.Join(t.TempDir(), "manager.json")
	if err := os.WriteFile(path, []byte(`{"collections":{}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := manager.Load(path); err != nil {
		t.Fatalf("load explicit empty manager: %v", err)
	}
	if manager.CollectionCount() != 0 {
		t.Fatalf("collection count = %d, want 0", manager.CollectionCount())
	}
}

func TestTenantStateSaveAndLoadAreTransactional(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tenants.json")
	populated := NewTenantManager(dir)
	if _, err := populated.CreateCollection(context.Background(), "old", testSchema("docs", 4)); err != nil {
		t.Fatal(err)
	}
	if err := populated.Save(path); err != nil {
		t.Fatal(err)
	}

	empty := NewTenantManager(dir)
	if err := empty.Save(path); err != nil {
		t.Fatalf("overwrite with explicit empty state: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"tenants":{}}` {
		t.Fatalf("empty tenant encoding = %s", data)
	}
	if err := populated.Load(path); err != nil {
		t.Fatalf("load explicit empty tenants: %v", err)
	}
	if populated.TenantCount() != 0 {
		t.Fatalf("tenant count = %d, want 0", populated.TenantCount())
	}

	prior := NewTenantManager(dir)
	if _, err := prior.CreateCollection(context.Background(), "keep", testSchema("docs", 4)); err != nil {
		t.Fatal(err)
	}
	validManager, err := NewCollectionManager("").marshalState()
	if err != nil {
		t.Fatal(err)
	}
	mixed := persistedTenantState{Tenants: map[string]json.RawMessage{
		"would-add": validManager,
		"bad":       json.RawMessage(`{"collections":{"broken":null}}`),
	}}
	mixedBytes, err := json.Marshal(mixed)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, mixedBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := prior.Load(path); err == nil {
		t.Fatal("expected mixed tenant state to fail")
	}
	if got := prior.ListTenants(); len(got) != 1 || got[0] != "keep" {
		t.Fatalf("failed tenant load mutated live state: %v", got)
	}
}

func TestConcurrentCollectionManagerSavesUseUniqueTemps(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manager.json")
	left := seedManagerForSnapshot(t, "", "left")
	right := seedManagerForSnapshot(t, "", "right")

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for _, manager := range []*CollectionManager{left, right} {
		wg.Add(1)
		go func(manager *CollectionManager) {
			defer wg.Done()
			errs <- manager.Save(path)
		}(manager)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent save: %v", err)
		}
	}

	loaded := NewCollectionManager("")
	if err := loaded.Load(path); err != nil {
		t.Fatalf("load concurrent save result: %v", err)
	}
	if loaded.CollectionCount() != 1 {
		t.Fatalf("loaded collection count = %d, want 1", loaded.CollectionCount())
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("saved mode = %o, want 600", info.Mode().Perm())
	}
	temps, err := filepath.Glob(filepath.Join(dir, ".manager.json.tmp-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(temps) != 0 {
		t.Fatalf("temporary files leaked: %v", temps)
	}
}

func TestTenantIdentifierDoesNotBecomeTemporaryPath(t *testing.T) {
	dir := t.TempDir()
	tm := NewTenantManager(dir)
	if _, err := tm.CreateCollection(context.Background(), "../escape", testSchema("docs", 4)); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "tenants.json")
	if err := tm.Save(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(dir), "escape.tmp")); !os.IsNotExist(err) {
		t.Fatalf("tenant ID escaped persistence directory: %v", err)
	}
	loaded := NewTenantManager(dir)
	if err := loaded.Load(path); err != nil {
		t.Fatal(err)
	}
	if got := loaded.ListTenants(); len(got) != 1 || got[0] != "../escape" {
		t.Fatalf("tenant round trip = %v", got)
	}
}

func TestExistingStateRejectsMissingRequiredMaps(t *testing.T) {
	for name, data := range map[string]string{
		"manager-null":  `null`,
		"manager-empty": `{}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeCollectionManagerState([]byte(data), ""); err == nil {
				t.Fatal("expected manager decode to fail")
			}
		})
	}
	for name, data := range map[string]string{
		"tenant-null":  `null`,
		"tenant-empty": `{}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeTenantManagerState([]byte(data), ""); err == nil {
				t.Fatal("expected tenant decode to fail")
			}
		})
	}
}
