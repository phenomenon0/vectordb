package main

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/storage"
)

// TestLoadOrInitStoreExistingUnreadableSnapshotFailsClosed documents the
// required startup boundary: an existing but unreadable snapshot is evidence
// of prior state, not permission to initialize an empty database. The current
// boolean API has no error result, so loaded=false is the observable unsafe
// behavior this characterization test rejects.
func TestLoadOrInitStoreExistingUnreadableSnapshotFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.snapshot")
	corrupt := []byte("this is an existing but unreadable DeepData snapshot")
	if err := os.WriteFile(path, corrupt, 0o600); err != nil {
		t.Fatalf("write corrupt snapshot fixture: %v", err)
	}

	store, loaded, err := loadOrInitStore(path, 10, 3)
	if err == nil {
		t.Fatalf("existing unreadable snapshot returned no error (store=%v loaded=%t)", store, loaded)
	}
	if store != nil || loaded {
		t.Fatalf("corrupt snapshot returned usable state: store=%v loaded=%t error=%v", store, loaded, err)
	}
}

func TestLoadOrInitStoreInitializesOnlyWhenSnapshotIsMissing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.snapshot")
	store, loaded, err := loadOrInitStore(path, 10, 3)
	if err != nil {
		t.Fatalf("initialize missing snapshot: %v", err)
	}
	if loaded {
		t.Fatal("missing snapshot was reported as loaded")
	}
	if store == nil || store.walPath != path+".wal" {
		t.Fatalf("fresh store was not initialized correctly: %#v", store)
	}
}

func TestLoadOrInitStoreRejectsCurrentChecksumMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.snapshot")
	store := NewVectorStore(10, 3)
	if _, err := store.Add([]float32{1, 0, 0}, "doc", "id-1", nil, "default", "tenant-a"); err != nil {
		t.Fatalf("add fixture: %v", err)
	}
	if err := store.Save(path); err != nil {
		t.Fatalf("save fixture: %v", err)
	}

	payload, format, err := tryLoadPayload(path)
	if err != nil {
		t.Fatalf("load fixture payload: %v", err)
	}
	payload.Checksum = "tampered-checksum"
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("open fixture for tamper: %v", err)
	}
	if err := format.Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("write tampered fixture: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close tampered fixture: %v", err)
	}

	loadedStore, loaded, err := loadOrInitStore(path, 10, 3)
	if err == nil {
		t.Fatalf("checksum mismatch returned no error: store=%v loaded=%t", loadedStore, loaded)
	}
	if loadedStore != nil || loaded {
		t.Fatalf("checksum mismatch returned usable state: store=%v loaded=%t", loadedStore, loaded)
	}
}

func TestLoadOrInitStoreRejectsLogicalDataTamper(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.snapshot")
	store := NewVectorStore(10, 3)
	if _, err := store.Add([]float32{1, 0, 0}, "doc", "id-1", map[string]string{"kind": "fixture"}, "default", "tenant-a"); err != nil {
		t.Fatalf("add fixture: %v", err)
	}
	wantSeqs := append([]uint64(nil), store.Seqs...)
	if err := store.Save(path); err != nil {
		t.Fatalf("save fixture: %v", err)
	}

	payload, format, err := tryLoadPayload(path)
	if err != nil {
		t.Fatalf("load fixture payload: %v", err)
	}
	if !reflect.DeepEqual(payload.Seqs, wantSeqs) {
		t.Fatalf("sequence round trip: got %v, want %v", payload.Seqs, wantSeqs)
	}
	payload.Data[0] = 99
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("open fixture for tamper: %v", err)
	}
	if err := format.Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("write tampered fixture: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close tampered fixture: %v", err)
	}

	if loadedStore, loaded, err := loadOrInitStore(path, 10, 3); err == nil || loadedStore != nil || loaded {
		t.Fatalf("tampered vector state was accepted: store=%v loaded=%t error=%v", loadedStore, loaded, err)
	}
}

func TestChecksumIsCanonicalAcrossMapInsertionOrder(t *testing.T) {
	first := NewVectorStore(10, 3)
	second := NewVectorStore(10, 3)
	firstMeta := map[string]string{"a": "one", "b": "two"}
	secondMeta := make(map[string]string)
	secondMeta["b"] = "two"
	secondMeta["a"] = "one"
	for _, fixture := range []struct {
		store *VectorStore
		meta  map[string]string
	}{{first, firstMeta}, {second, secondMeta}} {
		if _, err := fixture.store.Add([]float32{1, 2, 3}, "canonical document", "id-1", fixture.meta, "default", "tenant-a"); err != nil {
			t.Fatalf("add fixture: %v", err)
		}
		hid := hashID("id-1")
		fixture.store.NumMeta[hid] = map[string]float64{"z": 1.5, "a": -2.25}
		fixture.store.TimeMeta[hid] = map[string]time.Time{
			"updated": time.Date(2026, time.July, 17, 12, 0, 0, 0, time.FixedZone("fixture", -5*60*60)),
		}
	}
	if got, want := first.computeChecksum(), second.computeChecksum(); got != want {
		t.Fatalf("checksum depends on map insertion order: got %q, want %q", got, want)
	}
}

func TestCompactPersistsSequencesAndAdvancesHighWaterMark(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact.snapshot")
	store := NewVectorStore(10, 3)
	for i, id := range []string{"id-1", "id-2", "id-3"} {
		if _, err := store.Add([]float32{float32(i + 1), 0, 0}, "doc", id, nil, "default", "tenant-a"); err != nil {
			t.Fatalf("add %s: %v", id, err)
		}
	}
	if err := store.Delete("id-2"); err != nil {
		t.Fatalf("delete fixture: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- store.Compact(path) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("compact: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("compact deadlocked while saving")
	}

	loaded, ok, err := loadOrInitStore(path, 10, 3)
	if err != nil || !ok {
		t.Fatalf("reload compacted snapshot: loaded=%t error=%v", ok, err)
	}
	if got, want := loaded.Seqs, []uint64{0, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("compacted sequences: got %v, want %v", got, want)
	}
	if _, err := loaded.Add([]float32{4, 0, 0}, "new doc", "id-4", nil, "default", "tenant-a"); err != nil {
		t.Fatalf("add after compaction: %v", err)
	}
	if got, want := loaded.Seqs[len(loaded.Seqs)-1], uint64(3); got != want {
		t.Fatalf("post-compaction sequence: got %d, want %d", got, want)
	}
}

func TestIndexTypeAndArtifactIntegrityRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "typed-index.snapshot")
	store := NewVectorStore(10, 3)
	for _, config := range []CollectionConfig{
		{Name: "flat-coll", IndexType: "flat", Dimension: 3},
		{Name: "ivf-coll", IndexType: "ivf", Dimension: 3, Config: map[string]interface{}{"nlist": 1, "nprobe": 1}},
	} {
		if err := store.CreateCollection(config); err != nil {
			t.Fatalf("create %s: %v", config.Name, err)
		}
	}
	fixtures := []struct {
		id         string
		collection string
		vector     []float32
	}{
		{"default-id", "default", []float32{1, 0, 0}},
		{"flat-id", "flat-coll", []float32{0, 1, 0}},
		{"ivf-id", "ivf-coll", []float32{0, 0, 1}},
	}
	for _, fixture := range fixtures {
		if _, err := store.Add(fixture.vector, "typed index fixture", fixture.id, nil, fixture.collection, "tenant-a"); err != nil {
			t.Fatalf("add %s: %v", fixture.id, err)
		}
	}
	if err := store.Save(path); err != nil {
		t.Fatalf("save typed indexes: %v", err)
	}

	loaded, ok, err := loadOrInitStore(path, 10, 3)
	if err != nil || !ok {
		t.Fatalf("load typed indexes: loaded=%t error=%v", ok, err)
	}
	for collection, want := range map[string]string{
		"default":   "HNSW",
		"flat-coll": "FLAT",
		"ivf-coll":  "IVF",
	} {
		idx := loaded.indexes[collection]
		if idx == nil || idx.Name() != want {
			t.Fatalf("index %q type: got %v, want %s", collection, idx, want)
		}
	}

	payload, format, err := tryLoadPayload(path)
	if err != nil {
		t.Fatalf("read typed snapshot: %v", err)
	}
	payload.Indexes["flat-coll"][0] ^= 0xff
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("open index tamper fixture: %v", err)
	}
	if err := format.Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("write index tamper fixture: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close index tamper fixture: %v", err)
	}
	if tampered, accepted, err := loadOrInitStore(path, 10, 3); err == nil || tampered != nil || accepted {
		t.Fatalf("tampered index artifact was accepted: store=%v loaded=%t error=%v", tampered, accepted, err)
	}
}

func TestTryLoadPayloadRejectsStructurallyInvalidCurrentSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "invalid.snapshot")
	payload := &storage.Payload{
		FormatVersion: storage.CurrentFormatVersion,
		Dim:           3,
		Count:         1,
		Next:          1,
		IDs:           []string{"id-1"},
		Checksum:      "present-but-structurally-invalid",
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create invalid fixture: %v", err)
	}
	if err := storage.Default().Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("save invalid fixture: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close invalid fixture: %v", err)
	}

	if _, _, err := tryLoadPayload(path); err == nil {
		t.Fatal("structurally invalid current snapshot was accepted")
	}
}

func TestLoadOrInitStoreMigratesRecognizedLegacyChecksum(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.gob")
	payload := &storage.Payload{
		FormatVersion: 0,
		Dim:           3,
		Data:          []float32{1, 0, 0},
		Docs:          []string{"legacy document"},
		IDs:           []string{"legacy-id"},
		Next:          1,
		Count:         1,
	}
	payload.Checksum = fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d-%d", payload.Count, payload.Next, len(payload.Docs))))
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create legacy fixture: %v", err)
	}
	if err := storage.Get("gob").Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("save legacy fixture: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close legacy fixture: %v", err)
	}

	store, loaded, err := loadOrInitStore(path, 10, 3)
	if err != nil {
		t.Fatalf("load legacy fixture: %v", err)
	}
	if !loaded || store == nil {
		t.Fatalf("legacy fixture was not loaded: store=%v loaded=%t", store, loaded)
	}
	if !store.validateChecksum() {
		t.Fatal("legacy checksum was not migrated to the current formula")
	}
	if tenant := store.TenantID[hashID("legacy-id")]; tenant != "default" {
		t.Fatalf("legacy tenant migration: got %q, want default", tenant)
	}
}

func TestLoadOrInitStoreRejectsUnrecognizedLegacyChecksum(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.gob")
	payload := &storage.Payload{
		FormatVersion: 0,
		Dim:           3,
		Data:          []float32{1, 0, 0},
		Docs:          []string{"legacy document"},
		IDs:           []string{"legacy-id"},
		Next:          1,
		Count:         1,
		Checksum:      "unrecognized-or-corrupt-checksum",
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create legacy fixture: %v", err)
	}
	if err := storage.Get("gob").Save(f, payload); err != nil {
		_ = f.Close()
		t.Fatalf("save legacy fixture: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close legacy fixture: %v", err)
	}

	if store, loaded, err := loadOrInitStore(path, 10, 3); err == nil || store != nil || loaded {
		t.Fatalf("unrecognized legacy checksum was accepted: store=%v loaded=%t error=%v", store, loaded, err)
	}
}
