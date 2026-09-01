package collection

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

func durableTestSchema(name string) CollectionSchema {
	return CollectionSchema{
		Name: name,
		Fields: []VectorField{{
			Name:  "embedding",
			Type:  VectorTypeDense,
			Dim:   4,
			Index: IndexConfig{Type: IndexTypeFLAT},
		}},
	}
}

func durableTestDocument(value float32) Document {
	return Document{
		Vectors: map[string]interface{}{
			"embedding": []float32{value, 0, 0, 0},
		},
		Metadata: map[string]interface{}{"value": value},
	}
}

func durableTestCollectionInfo(t *testing.T, store *DurableStore, tenantID, collectionName string) *CollectionInfo {
	t.Helper()
	info, err := store.Tenants().GetCollectionInfo(tenantID, collectionName)
	if err != nil {
		t.Fatal(err)
	}
	return info
}

func durableTestStoredDocument(t *testing.T, store *DurableStore, tenantID, collectionName string, documentID uint64) (*Document, bool) {
	t.Helper()
	store.mu.RLock()
	defer store.mu.RUnlock()
	if err := store.stateErrorLocked(); err != nil {
		t.Fatal(err)
	}
	coll, err := store.tenants.getCollectionDirect(tenantID, collectionName)
	if err != nil {
		t.Fatal(err)
	}
	return coll.GetDocument(documentID)
}

func abandonDurableStoreForTest(t *testing.T, store *DurableStore) {
	t.Helper()
	if err := store.Abort(); err != nil {
		t.Fatalf("abort durable store: %v", err)
	}
}

func TestDurableStoreSubprocessLockExclusion(t *testing.T) {
	if base := os.Getenv("DEEPDATA_LOCK_TEST_BASE"); base != "" {
		store, err := OpenDurableStore(base, base)
		if err == nil {
			_ = store.Close()
			t.Fatal("second process unexpectedly opened locked store")
		}
		if !strings.Contains(err.Error(), "already open") {
			t.Fatalf("second process error = %v, want already open", err)
		}
		return
	}

	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := store.Close(); err != nil {
			t.Errorf("close store: %v", err)
		}
	}()

	cmd := exec.Command(os.Args[0], "-test.run=^TestDurableStoreSubprocessLockExclusion$")
	cmd.Env = append(os.Environ(), "DEEPDATA_LOCK_TEST_BASE="+base)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("lock helper failed: %v\n%s", err, output)
	}
}

func TestDurableStoreRejectsSymlinkLockWithoutTouchingTarget(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux persistence contract")
	}
	dir := t.TempDir()
	base := filepath.Join(dir, "collections")
	target := filepath.Join(dir, "lock-target")
	if err := os.WriteFile(target, []byte("sentinel"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, base+".lock"); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenDurableStore(base, base); err == nil {
		t.Fatal("opened durable store through a symlink lock")
	}
	info, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0o644 {
		t.Fatalf("lock target permissions changed to %o", got)
	}
	data, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "sentinel" {
		t.Fatalf("lock target content changed: %q", data)
	}
}

func TestDurableStoreReplayRetainsJournalUntilExplicitCheckpoint(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	one := durableTestDocument(1)
	if err := tenants.AddDocument(ctx, "tenant-a", "docs", &one); err != nil {
		t.Fatal(err)
	}
	batch := []Document{durableTestDocument(2), durableTestDocument(3)}
	if err := tenants.BatchAddDocuments(ctx, "tenant-a", "docs", batch); err != nil {
		t.Fatal(err)
	}
	if one.ID != 1 || batch[0].ID != 2 || batch[1].ID != 3 {
		t.Fatalf("assigned IDs = %d, %d, %d", one.ID, batch[0].ID, batch[1].ID)
	}
	if err := tenants.DeleteDocument(ctx, "tenant-a", "docs", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := tenants.CreateCollection(ctx, "tenant-a", durableTestSchema("temporary")); err != nil {
		t.Fatal(err)
	}
	if err := tenants.DeleteCollection(ctx, "tenant-a", "temporary"); err != nil {
		t.Fatal(err)
	}
	wantLSN := store.Metadata().AppliedLSN
	if wantLSN != 6 {
		t.Fatalf("applied LSN = %d, want 6", wantLSN)
	}
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen after uncheckpointed mutations: %v", err)
	}
	defer reopened.Close()
	if got := reopened.Metadata().AppliedLSN; got != wantLSN {
		t.Fatalf("recovered LSN = %d, want %d", got, wantLSN)
	}
	info := durableTestCollectionInfo(t, reopened, "tenant-a", "docs")
	if got := info.DocCount; got != 2 {
		t.Fatalf("recovered document count = %d, want 2", got)
	}
	if _, ok := durableTestStoredDocument(t, reopened, "tenant-a", "docs", 2); ok {
		t.Fatal("deleted document reappeared after replay")
	}
	if _, err := reopened.Tenants().GetCollectionInfo("tenant-a", "temporary"); err == nil {
		t.Fatal("deleted collection reappeared after replay")
	}
	if _, err := os.Stat(base + ".journal"); err != nil {
		t.Fatalf("recovery did not retain validated current journal: %v", err)
	}
	if _, err := os.Stat(base + ".journal.frozen"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unexpected frozen journal after current-only recovery: %v", err)
	}
	if err := reopened.Checkpoint(); err != nil {
		t.Fatalf("explicit checkpoint after recovery: %v", err)
	}
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("explicit checkpoint did not clean covered journal %s: %v", path, err)
		}
	}
}

func TestDurableStoreUpsertReplacesAndReplays(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	// A new caller-supplied ID behaves like an insert.
	if err := tenants.UpsertDocument(ctx, "tenant-a", "docs", &Document{
		ID:       42,
		Vectors:  map[string]interface{}{"embedding": []float64{1, 0, 0, 0}},
		Metadata: map[string]interface{}{"source": "first"},
	}); err != nil {
		t.Fatal(err)
	}
	if got, ok := durableTestStoredDocument(t, store, "tenant-a", "docs", 42); !ok || got.Metadata["source"] != "first" {
		t.Fatalf("upsert insert not stored: ok=%v doc=%+v", ok, got)
	}

	// Replacing an existing live ID keeps exactly one storage entry.
	if err := tenants.UpsertDocument(ctx, "tenant-a", "docs", &Document{
		ID:       42,
		Vectors:  map[string]interface{}{"embedding": []float32{4, 0, 0, 0}},
		Metadata: map[string]interface{}{"source": "replaced"},
	}); err != nil {
		t.Fatal(err)
	}
	info := durableTestCollectionInfo(t, store, "tenant-a", "docs")
	if info.DocCount != 1 {
		t.Fatalf("doc count = %d after upsert-replace, want 1", info.DocCount)
	}
	wantLSN := store.Metadata().AppliedLSN
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen after upsert: %v", err)
	}
	defer reopened.Close()
	if got := reopened.Metadata().AppliedLSN; got != wantLSN {
		t.Fatalf("recovered LSN = %d, want %d", got, wantLSN)
	}
	info = durableTestCollectionInfo(t, reopened, "tenant-a", "docs")
	if info.DocCount != 1 {
		t.Fatalf("recovered doc count = %d after upsert replay, want 1", info.DocCount)
	}
	got, ok := durableTestStoredDocument(t, reopened, "tenant-a", "docs", 42)
	if !ok {
		t.Fatal("upserted document missing after replay")
	}
	if got.Metadata["source"] != "replaced" {
		t.Fatalf("replay applied first upsert instead of replacement: %+v", got.Metadata)
	}
	replayedVector, ok := got.Vectors["embedding"].([]float32)
	if !ok || len(replayedVector) != 4 || replayedVector[0] != 4 {
		t.Fatalf("upsert replay retained non-compact dense vector: %T %v", got.Vectors["embedding"], got.Vectors["embedding"])
	}
}

func TestDurableStoreUpsertContractAndReadNotFound(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "tenant-a", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	// Automatically-assigned (zero) IDs are refused: upsert is caller-addressed.
	zero := durableTestDocument(1)
	if err := tenants.UpsertDocument(ctx, "tenant-a", "docs", &zero); err == nil {
		t.Fatal("upsert with zero ID should fail")
	}

	// GetDocument is the canonical single-doc read.
	if doc, ok := tenants.GetDocument("tenant-a", "docs", 99); ok || doc != nil {
		t.Fatal("read of never-written ID must be not-found")
	}
	if err := tenants.UpsertDocument(ctx, "tenant-a", "docs", &Document{
		ID:       7,
		Vectors:  map[string]interface{}{"embedding": []float32{7, 0, 0, 0}},
		Metadata: map[string]interface{}{"value": float64(7)},
	}); err != nil {
		t.Fatal(err)
	}
	got, ok := tenants.GetDocument("tenant-a", "docs", 7)
	if !ok || got.ID != 7 {
		t.Fatalf("GetDocument after read-write = ok=%v id=%d", ok, got.ID)
	}
}

func TestDurableStoreRepairsPartialMutationTailAndDefersCheckpoint(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "tenant", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	first := durableTestDocument(1)
	first.ID = 1
	if err := store.Tenants().AddDocument(ctx, "tenant", "docs", &first); err != nil {
		t.Fatal(err)
	}
	metadata := store.Metadata()
	if metadata.AppliedLSN != 2 {
		t.Fatalf("applied LSN = %d, want 2", metadata.AppliedLSN)
	}

	second := durableTestDocument(2)
	second.ID = 2
	payload, err := encodeDurableMutation(canonicalMutation{
		typeName:       mutationInsertDocument,
		tenantID:       "tenant",
		collectionName: "docs",
		documents:      []Document{second},
	})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := encodeCollectionJournalFrame(metadata.StoreID, metadata.AppliedLSN+1, payload, collectionJournalMaxPayload)
	if err != nil {
		t.Fatal(err)
	}
	partialLength := int(collectionJournalHeaderSize) + len(payload)/2
	if partialLength >= len(frame) {
		t.Fatalf("partial fixture length %d is not below frame length %d", partialLength, len(frame))
	}
	abandonDurableStoreForTest(t, store)

	journal, err := os.OpenFile(base+".journal", os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := journal.Write(frame[:partialLength]); err != nil {
		_ = journal.Close()
		t.Fatal(err)
	}
	if err := journal.Sync(); err != nil {
		_ = journal.Close()
		t.Fatal(err)
	}
	if err := journal.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen with valid-prefix partial tail: %v", err)
	}
	if got := reopened.Metadata().AppliedLSN; got != 2 {
		t.Fatalf("recovered LSN = %d, want 2", got)
	}
	if got := durableTestCollectionInfo(t, reopened, "tenant", "docs").DocCount; got != 1 {
		t.Fatalf("recovered document count = %d, want 1", got)
	}
	if _, ok := durableTestStoredDocument(t, reopened, "tenant", "docs", 1); !ok {
		t.Fatal("complete acknowledged document was lost")
	}
	if _, ok := durableTestStoredDocument(t, reopened, "tenant", "docs", 2); ok {
		t.Fatal("partial unacknowledged document was applied")
	}
	if _, err := os.Stat(base + ".journal"); err != nil {
		t.Fatalf("partial-tail recovery did not retain repaired journal: %v", err)
	}
	if _, err := os.Stat(base + ".journal.frozen"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unexpected frozen journal after partial-tail recovery: %v", err)
	}
	if err := reopened.Checkpoint(); err != nil {
		t.Fatalf("explicit checkpoint after partial-tail recovery: %v", err)
	}
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("explicit checkpoint retained journal %s: %v", path, err)
		}
	}
	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}

	reopenedAgain, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("second reopen after partial-tail checkpoint: %v", err)
	}
	defer reopenedAgain.Close()
	if got := durableTestCollectionInfo(t, reopenedAgain, "tenant", "docs").DocCount; got != 1 {
		t.Fatalf("document count after second reopen = %d, want 1", got)
	}
}

func TestDurableStoreRestartRecoversFrozenAndCurrentAfterCheckpointFailureExactlyOnce(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "tenant", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	originalSync := collectionSnapshotFileSync
	collectionSnapshotFileSync = func(*os.File) error { return errors.New("injected snapshot sync failure") }
	checkpointErr := store.Checkpoint()
	collectionSnapshotFileSync = originalSync
	if checkpointErr == nil || !strings.Contains(checkpointErr.Error(), "injected snapshot sync failure") {
		t.Fatalf("checkpoint error = %v", checkpointErr)
	}
	if _, err := os.Stat(base + ".journal.frozen"); err != nil {
		t.Fatalf("failed checkpoint did not retain frozen journal: %v", err)
	}

	doc := durableTestDocument(1)
	doc.ID = 1
	if err := store.Tenants().AddDocument(ctx, "tenant", "docs", &doc); err != nil {
		t.Fatalf("append current journal after failed checkpoint: %v", err)
	}
	if _, err := os.Stat(base + ".journal"); err != nil {
		t.Fatalf("current journal missing after post-rotation mutation: %v", err)
	}
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("restart with frozen and current journals: %v", err)
	}
	if got := reopened.Metadata().AppliedLSN; got != 2 {
		t.Fatalf("recovered LSN = %d, want 2", got)
	}
	if got := durableTestCollectionInfo(t, reopened, "tenant", "docs").DocCount; got != 1 {
		t.Fatalf("recovered document count = %d, want 1", got)
	}
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("restart recovery did not retain validated journal %s: %v", path, err)
		}
	}
	if err := reopened.Checkpoint(); err != nil {
		t.Fatalf("explicit checkpoint after frozen/current recovery: %v", err)
	}
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("explicit checkpoint retained covered journal %s: %v", path, err)
		}
	}
	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}

	reopenedAgain, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("second reopen after frozen/current recovery: %v", err)
	}
	defer reopenedAgain.Close()
	if got := durableTestCollectionInfo(t, reopenedAgain, "tenant", "docs").DocCount; got != 1 {
		t.Fatalf("document count after second reopen = %d, want 1", got)
	}
}

func TestDurableStoreReplaysSupportedHNSWAndSparseState(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	schema := CollectionSchema{
		Name: "hybrid",
		Fields: []VectorField{
			{Name: "dense", Type: VectorTypeDense, Dim: 4, Index: IndexConfig{Type: IndexTypeHNSW}},
			{Name: "sparse", Type: VectorTypeSparse, Dim: 16, Index: IndexConfig{Type: IndexTypeInverted}},
		},
	}
	if _, err := store.Tenants().CreateCollection(ctx, "t", schema); err != nil {
		t.Fatal(err)
	}
	doc := Document{
		Vectors: map[string]interface{}{
			"dense":  []float32{1, 0, 0, 0},
			"sparse": map[string]interface{}{"indices": []uint32{1, 3}, "values": []float32{2, 1}, "dim": 16},
		},
		Metadata: map[string]interface{}{"kind": "hybrid"},
	}
	if err := store.Tenants().AddDocument(ctx, "t", "hybrid", &doc); err != nil {
		t.Fatal(err)
	}
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen HNSW+sparse store: %v", err)
	}
	defer reopened.Close()
	info := durableTestCollectionInfo(t, reopened, "t", "hybrid")
	if info.DocCount != 1 {
		t.Fatalf("hybrid collection after replay: count=%d", info.DocCount)
	}
	replayed, ok := durableTestStoredDocument(t, reopened, "t", "hybrid", doc.ID)
	if !ok {
		t.Fatal("hybrid document missing after replay")
	}
	dense, ok := replayed.Vectors["dense"].([]float32)
	if !ok || len(dense) != 4 || dense[0] != 1 {
		t.Fatalf("WAL replay retained non-compact dense vector: %T %v", replayed.Vectors["dense"], replayed.Vectors["dense"])
	}
	sparseVector, ok := replayed.Vectors["sparse"].(*sparse.SparseVector)
	if !ok || len(sparseVector.Indices) != 2 || sparseVector.Indices[0] != 1 || sparseVector.Indices[1] != 3 ||
		len(sparseVector.Values) != 2 || sparseVector.Values[0] != 2 || sparseVector.Values[1] != 1 {
		t.Fatalf("WAL replay retained non-compact or misaligned sparse vector: %T %+v", replayed.Vectors["sparse"], replayed.Vectors["sparse"])
	}
	response, err := reopened.Tenants().SearchCollection(ctx, "t", SearchRequest{
		CollectionName: "hybrid",
		Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
		TopK:           1,
	})
	if err != nil || len(response.Documents) != 1 || response.Documents[0].ID != doc.ID {
		t.Fatalf("HNSW search after replay: response=%+v err=%v", response, err)
	}
}

func TestDurableStorePreservesV2AndV3SnapshotState(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	v2 := NewCollectionManager(base)
	if _, err := v2.CreateCollection(ctx, durableTestSchema("legacy-v2")); err != nil {
		t.Fatal(err)
	}
	v2doc := durableTestDocument(7)
	if err := v2.AddDocument(ctx, "legacy-v2", &v2doc); err != nil {
		t.Fatal(err)
	}
	v3 := NewTenantManager(base)
	if _, err := v3.CreateCollection(ctx, "existing", durableTestSchema("existing-v3")); err != nil {
		t.Fatal(err)
	}
	meta, err := NewCollectionSnapshotMetadata()
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveUnifiedCollectionSnapshot(base, v2, v3, meta); err != nil {
		t.Fatal(err)
	}
	v2.closeAll()
	v3.closeAll()

	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	managerPtr, tenantPtr := store.manager, store.Tenants()
	if err := managerPtr.AddDocument(ctx, "legacy-v2", &Document{}); !errors.Is(err, ErrCanonicalMutationRequired) {
		t.Fatalf("V2 write error = %v, want canonical-only rejection", err)
	}
	v2coll, err := managerPtr.GetCollection("legacy-v2")
	if err != nil || v2coll.Count() != 1 {
		t.Fatalf("preserved V2 collection = %v count=%d err=%v", v2coll, v2coll.Count(), err)
	}
	if err := v2coll.Add(ctx, &Document{}); !errors.Is(err, ErrCanonicalMutationRequired) {
		t.Fatalf("direct collection write error = %v, want canonical-only rejection", err)
	}
	if _, err := tenantPtr.CreateCollection(ctx, "new", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if err := store.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	if store.manager != managerPtr || store.Tenants() != tenantPtr {
		t.Fatal("checkpoint replaced manager pointers")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	legacyCount, err := reopened.LegacyCollectionCount()
	if err != nil {
		t.Fatal(err)
	}
	if legacyCount != 1 {
		t.Fatalf("V2 collection count = %d, want 1", legacyCount)
	}
	if _, err := reopened.Tenants().GetCollectionInfo("existing", "existing-v3"); err != nil {
		t.Fatalf("existing V3 state lost: %v", err)
	}
	if _, err := reopened.Tenants().GetCollectionInfo("new", "docs"); err != nil {
		t.Fatalf("new V3 state lost: %v", err)
	}
}

func TestDurableStoreRejectsMalformedMutationBeforeReplay(t *testing.T) {
	cases := map[string]string{
		"unknown version": `{"version":3,"type":"delete_collection","payload":{"tenant_id":"t","collection_name":"c"}}`,
		"unknown type":    `{"version":1,"type":"future","payload":{}}`,
		"unknown field":   `{"version":1,"type":"delete_collection","payload":{"tenant_id":"t","collection_name":"c","extra":true}}`,
		"trailing JSON":   `{"version":1,"type":"delete_collection","payload":{"tenant_id":"t","collection_name":"c"}} true`,
		"unassigned ID":   `{"version":1,"type":"insert_document","payload":{"tenant_id":"t","collection_name":"c","document":{"id":0,"vectors":{"embedding":[1,0,0,0]}}}}`,
	}
	for name, payload := range cases {
		t.Run(name, func(t *testing.T) {
			base := filepath.Join(t.TempDir(), "collections")
			store, err := OpenDurableStore(base, base)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.journal.append([]byte(payload)); err != nil {
				t.Fatal(err)
			}
			abandonDurableStoreForTest(t, store)
			if reopened, err := OpenDurableStore(base, base); err == nil {
				_ = reopened.Close()
				t.Fatal("malformed mutation unexpectedly opened")
			}
		})
	}
}

func TestDurableMutationEncoderUsesCurrentVersion(t *testing.T) {
	payload, err := encodeDurableMutation(canonicalMutation{
		typeName:       mutationDeleteCollection,
		tenantID:       "tenant",
		collectionName: "docs",
	})
	if err != nil {
		t.Fatal(err)
	}
	var envelope durableMutationEnvelope
	if err := decodeCollectionJSON(payload, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Version != durableCollectionMutationVersion || envelope.Version == durableCollectionMutationVersionV1 {
		t.Fatalf("encoded mutation version = %d, want current v%d distinct from v1", envelope.Version, durableCollectionMutationVersion)
	}
}

func TestDurableStoreReplaysLegacyV1BatchAboveCurrentAdmissionLimit(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}

	// Both the path-like name and unknown Flat parameter were accepted by v1,
	// before the canonical v2 URL/index admission contract was narrowed.
	schema := durableTestSchema("legacy/name")
	schema.Fields[0].Index.Params = map[string]interface{}{"gpu": true}
	if _, err := store.Tenants().CreateCollection(context.Background(), "tenant", schema); err == nil {
		t.Fatal("current live admission unexpectedly accepted the legacy v1 schema")
	}
	createPayload, err := encodeDurableMutationVersion(canonicalMutation{
		typeName:       mutationCreateCollection,
		tenantID:       "tenant",
		collectionName: schema.Name,
		schema:         schema,
	}, durableCollectionMutationVersionV1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.journal.append(createPayload); err != nil {
		t.Fatal(err)
	}

	documents := make([]Document, MaxBatchDocuments+1)
	for i := range documents {
		documents[i] = durableTestDocument(float32(i + 1))
		documents[i].ID = uint64(i + 1)
	}
	batchPayload, err := encodeDurableMutationVersion(canonicalMutation{
		typeName:       mutationBatchInsert,
		tenantID:       "tenant",
		collectionName: schema.Name,
		documents:      documents,
	}, durableCollectionMutationVersionV1)
	if err != nil {
		t.Fatal(err)
	}
	if len(batchPayload) >= int(collectionJournalMaxPayload) {
		t.Fatalf("legacy compatibility fixture is %d bytes; must fit one v1 frame", len(batchPayload))
	}
	if _, err := store.journal.append(batchPayload); err != nil {
		t.Fatal(err)
	}
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen legacy v1 journal: %v", err)
	}
	info := durableTestCollectionInfo(t, reopened, "tenant", schema.Name)
	if got := info.DocCount; got != len(documents) {
		t.Fatalf("legacy v1 replay count = %d, want %d", got, len(documents))
	}
	if reopened.Metadata().AppliedLSN != 2 {
		t.Fatalf("legacy v1 applied LSN = %d, want 2", reopened.Metadata().AppliedLSN)
	}
	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}

	// Recovery checkpoints the legacy records. A second open proves they were
	// applied once, not retained and replayed on top of the checkpoint.
	reopenedAgain, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("second reopen after legacy checkpoint: %v", err)
	}
	defer reopenedAgain.Close()
	info = durableTestCollectionInfo(t, reopenedAgain, "tenant", schema.Name)
	if got := info.DocCount; got != len(documents) {
		t.Fatalf("legacy v1 count after checkpoint/reopen = %d, want %d", got, len(documents))
	}
}

func TestDurableStoreRejectsCorruptJournalFrame(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	abandonDurableStoreForTest(t, store)
	path := base + ".journal"
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-1] ^= 0xff
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if reopened, err := OpenDurableStore(base, base); err == nil {
		_ = reopened.Close()
		t.Fatal("corrupt journal unexpectedly opened")
	} else if !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("corrupt journal error = %v", err)
	}
}

func TestDurableStoreCanonicalIDsValidationAndDefensiveClone(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	explicit := durableTestDocument(40)
	explicit.ID = 40
	if err := tenants.AddDocument(ctx, "t", "docs", &explicit); err != nil {
		t.Fatal(err)
	}
	invalid := durableTestDocument(0)
	invalid.Vectors["embedding"] = []float32{1, 2}
	if err := tenants.AddDocument(ctx, "t", "docs", &invalid); err == nil {
		t.Fatal("dimension mismatch unexpectedly succeeded")
	}
	auto := durableTestDocument(41)
	auto.Metadata["nested"] = map[string]interface{}{"key": "original"}
	inputVector := auto.Vectors["embedding"].([]float32)
	if err := tenants.AddDocument(ctx, "t", "docs", &auto); err != nil {
		t.Fatal(err)
	}
	if auto.ID != 41 {
		t.Fatalf("ID after explicit 40 = %d, want 41", auto.ID)
	}
	inputVector[0] = 999
	auto.Metadata["nested"].(map[string]interface{})["key"] = "mutated"
	stored, ok := durableTestStoredDocument(t, store, "t", "docs", 41)
	if !ok {
		t.Fatal("stored document missing")
	}
	if got := stored.Metadata["nested"].(map[string]interface{})["key"]; got != "original" {
		t.Fatalf("caller metadata mutation leaked into store: %v", got)
	}
	if err := tenants.AddDocument(ctx, "t", "docs", &explicit); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("duplicate explicit ID error = %v", err)
	}
}

func TestDurableStoreRejectsOutOfScopeIndexAndOversizePayloadWithoutFault(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ivf := durableTestSchema("ivf")
	ivf.Fields[0].Index.Type = IndexTypeIVF
	if _, err := store.Tenants().CreateCollection(ctx, "t", ivf); err == nil || !strings.Contains(err.Error(), "index type ivf was retired") {
		t.Fatalf("IVF create error = %v", err)
	}
	invalidName := durableTestSchema("contains/slash")
	if _, err := store.Tenants().CreateCollection(ctx, "t", invalidName); err == nil || !strings.Contains(err.Error(), "collection name") {
		t.Fatalf("invalid collection name error = %v", err)
	}
	for _, tc := range []struct {
		name      string
		indexType IndexType
		params    map[string]interface{}
	}{
		{
			name:      "flat-pq",
			indexType: IndexTypeFLAT,
			params: map[string]interface{}{
				"quantization": map[string]interface{}{"type": "pq"},
			},
		},
		{
			name:      "hnsw-float16",
			indexType: IndexTypeHNSW,
			params: map[string]interface{}{
				"quantization": map[string]interface{}{"type": "float16"},
			},
		},
		{
			name:      "flat-gpu",
			indexType: IndexTypeFLAT,
			params:    map[string]interface{}{"gpu": true},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			schema := durableTestSchema(tc.name)
			schema.Fields[0].Index = IndexConfig{Type: tc.indexType, Params: tc.params}
			if _, err := store.Tenants().CreateCollection(ctx, "t", schema); err == nil || !strings.Contains(err.Error(), "outside the canonical release contract") {
				t.Fatalf("out-of-scope params create error = %v", err)
			}
		})
	}
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	large := durableTestDocument(1)
	large.Metadata["oversize"] = strings.Repeat("x", int(collectionJournalMaxPayload)+1)
	if err := store.Tenants().AddDocument(ctx, "t", "docs", &large); err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("oversize mutation error = %v", err)
	}
	if err := store.Err(); err != nil {
		t.Fatalf("pre-append size rejection faulted store: %v", err)
	}
	small := durableTestDocument(2)
	if err := store.Tenants().AddDocument(ctx, "t", "docs", &small); err != nil {
		t.Fatalf("store unusable after oversize rejection: %v", err)
	}
}

func TestDurableStoreRejectsRawCollectionHandlesAndChecksLists(t *testing.T) {
	store, err := OpenDurableStore(filepath.Join(t.TempDir(), "collections"), "")
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	tenants := store.Tenants()
	handle, err := tenants.CreateCollection(context.Background(), "tenant", durableTestSchema("docs"))
	if err != nil {
		t.Fatal(err)
	}
	if handle != nil {
		t.Fatal("durable create returned a raw collection handle")
	}
	if handle, err := tenants.GetCollection("tenant", "docs"); handle != nil || !errors.Is(err, ErrDurableCollectionHandle) {
		t.Fatalf("durable raw get = (%v, %v), want nil/raw-handle error", handle, err)
	}
	names, err := tenants.ListCollectionsChecked("tenant")
	if err != nil || len(names) != 1 || names[0] != "docs" {
		t.Fatalf("checked collection names = %v, err=%v", names, err)
	}
	tenantIDs, err := tenants.ListTenantsChecked()
	if err != nil || len(tenantIDs) != 1 || tenantIDs[0] != "tenant" {
		t.Fatalf("checked tenants = %v, err=%v", tenantIDs, err)
	}
	count, err := tenants.TenantCountChecked()
	if err != nil || count != 1 {
		t.Fatalf("checked tenant count = %d, err=%v", count, err)
	}
}

func TestDurableStoreApplyFailureLatchesFaultAndReplays(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	store.apply = func(context.Context, canonicalMutation) error { return errors.New("injected apply failure") }
	doc := durableTestDocument(1)
	if err := store.Tenants().AddDocument(ctx, "t", "docs", &doc); err == nil || !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("apply failure error = %v", err)
	}
	store.apply = store.applyMutationDirect
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("blocked")); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault mutation error = %v", err)
	}
	if _, err := store.Tenants().GetCollection("t", "docs"); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault get error = %v", err)
	}
	if _, err := store.Tenants().GetCollectionInfo("t", "docs"); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault info error = %v", err)
	}
	if _, err := store.Tenants().ListCollectionInfosChecked("t"); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault list error = %v", err)
	}
	if _, err := store.Tenants().ListCollectionsChecked("t"); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault name list error = %v", err)
	}
	if _, err := store.Tenants().ListTenantsChecked(); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault tenant list error = %v", err)
	}
	if _, err := store.Tenants().TenantCountChecked(); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault tenant count error = %v", err)
	}
	if _, err := store.Tenants().GetTenantStats("t"); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault stats error = %v", err)
	}
	if _, err := store.Tenants().SearchCollection(ctx, "t", SearchRequest{
		CollectionName: "docs", Queries: map[string]interface{}{"embedding": []float32{1, 0, 0, 0}}, TopK: 1,
	}); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("post-fault search error = %v", err)
	}
	if err := store.Close(); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("faulted close error = %v", err)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("replay faulted append: %v", err)
	}
	defer reopened.Close()
	info := durableTestCollectionInfo(t, reopened, "t", "docs")
	if info.DocCount != 1 {
		t.Fatalf("replayed count = %d, want 1", info.DocCount)
	}
}

func TestDurableStoreCommittedApplyIgnoresRequestCancellation(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	applyDirect := store.apply
	store.apply = func(applyCtx context.Context, mutation canonicalMutation) error {
		// Model a disconnect after the journal append has completed but before
		// the in-memory mutation begins.
		cancel()
		if err := applyCtx.Err(); err != nil {
			return fmt.Errorf("committed apply inherited request cancellation: %w", err)
		}
		return applyDirect(applyCtx, mutation)
	}
	doc := durableTestDocument(1)
	if err := store.Tenants().AddDocument(ctx, "t", "docs", &doc); err != nil {
		t.Fatalf("committed insert failed after request cancellation: %v", err)
	}
	store.apply = applyDirect
	if err := store.Err(); err != nil {
		t.Fatalf("request cancellation faulted store: %v", err)
	}
	if doc.ID != 1 {
		t.Fatalf("assigned ID = %d, want 1", doc.ID)
	}
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen after canceled committed request: %v", err)
	}
	defer reopened.Close()
	info := durableTestCollectionInfo(t, reopened, "t", "docs")
	if got := info.DocCount; got != 1 {
		t.Fatalf("replayed document count = %d, want 1", got)
	}
}

func TestDurableStoreReadBarrierSpansAppendAndApply(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	applyDirect := store.apply
	applyEntered := make(chan struct{})
	releaseApply := make(chan struct{})
	store.apply = func(ctx context.Context, mutation canonicalMutation) error {
		close(applyEntered)
		<-releaseApply
		return applyDirect(ctx, mutation)
	}
	writeDone := make(chan error, 1)
	go func() {
		doc := durableTestDocument(1)
		writeDone <- store.Tenants().AddDocument(ctx, "t", "docs", &doc)
	}()
	<-applyEntered

	readDone := make(chan error, 1)
	go func() {
		_, err := store.Tenants().SearchCollection(ctx, "t", SearchRequest{
			CollectionName: "docs",
			Queries:        map[string]interface{}{"embedding": []float32{1, 0, 0, 0}},
			TopK:           1,
		})
		readDone <- err
	}()
	readReturnedEarly := false
	var earlyReadErr error
	select {
	case earlyReadErr = <-readDone:
		readReturnedEarly = true
	case <-time.After(100 * time.Millisecond):
	}
	close(releaseApply)
	writeErr := <-writeDone
	if readReturnedEarly {
		if writeErr != nil {
			t.Fatalf("write failed (%v) and read crossed append/apply boundary: %v", writeErr, earlyReadErr)
		}
		t.Fatalf("read crossed append/apply boundary: %v", earlyReadErr)
	}
	readErr := <-readDone
	if writeErr != nil {
		t.Fatal(writeErr)
	}
	if readErr != nil {
		t.Fatal(readErr)
	}
}

func TestDurableStoreCheckpointSerializesWithMutations(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	const writers = 4
	const perWriter = 20
	var wg sync.WaitGroup
	errCh := make(chan error, writers+1)
	for writer := 0; writer < writers; writer++ {
		wg.Add(1)
		go func(writer int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				doc := durableTestDocument(float32(writer*perWriter + i + 1))
				if err := store.Tenants().AddDocument(ctx, "t", "docs", &doc); err != nil {
					errCh <- err
					return
				}
			}
		}(writer)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			if err := store.Checkpoint(); err != nil {
				errCh <- err
				return
			}
		}
	}()
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("concurrent operation failed: %v", err)
	}
	info := durableTestCollectionInfo(t, store, "t", "docs")
	if got := info.DocCount; got != writers*perWriter {
		t.Fatalf("document count = %d, want %d", got, writers*perWriter)
	}
	if err := store.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("covered journal remained after checkpoint: %s: %v", path, err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	info = durableTestCollectionInfo(t, reopened, "t", "docs")
	if got := info.DocCount; got != writers*perWriter {
		t.Fatalf("reopened document count = %d, want %d", got, writers*perWriter)
	}
}

func TestDurableStoreCheckpointRecoversFrozenAndCurrentJournals(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	originalSync := collectionSnapshotFileSync
	collectionSnapshotFileSync = func(*os.File) error { return errors.New("injected snapshot sync failure") }
	err = store.Checkpoint()
	collectionSnapshotFileSync = originalSync
	if err == nil || !strings.Contains(err.Error(), "injected snapshot sync failure") {
		t.Fatalf("checkpoint error = %v", err)
	}
	if _, err := os.Stat(base + ".journal.frozen"); err != nil {
		t.Fatalf("failed checkpoint did not retain frozen journal: %v", err)
	}

	doc := durableTestDocument(1)
	if err := store.Tenants().AddDocument(ctx, "t", "docs", &doc); err != nil {
		t.Fatalf("append current journal after failed checkpoint: %v", err)
	}
	if _, err := os.Stat(base + ".journal"); err != nil {
		t.Fatalf("current journal missing after new mutation: %v", err)
	}
	if err := store.Checkpoint(); err != nil {
		t.Fatalf("retry checkpoint with frozen and current journals: %v", err)
	}
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("covered artifact remained after retry: %s: %v", path, err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	info := durableTestCollectionInfo(t, reopened, "t", "docs")
	if info.DocCount != 1 {
		t.Fatalf("checkpoint retry state: count=%d", info.DocCount)
	}
}

func TestDurableMutationEncodingStrictRoundTrip(t *testing.T) {
	m := canonicalMutation{
		typeName:       mutationDeleteDocument,
		tenantID:       "tenant",
		collectionName: "docs",
		documentID:     42,
	}
	data, err := encodeDurableMutation(m)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeDurableMutation(data)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(decoded.typeName, decoded.tenantID, decoded.collectionName, decoded.documentID) !=
		fmt.Sprint(m.typeName, m.tenantID, m.collectionName, m.documentID) {
		t.Fatalf("decoded mutation = %+v, want %+v", decoded, m)
	}
}
