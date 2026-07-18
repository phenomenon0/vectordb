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

func abandonDurableStoreForTest(t *testing.T, store *DurableStore) {
	t.Helper()
	store.mu.Lock()
	if !store.closed {
		store.closed = true
		store.manager.closeAll()
		store.tenants.closeAll()
		if err := store.lock.release(); err != nil {
			store.mu.Unlock()
			t.Fatalf("release store lock: %v", err)
		}
	}
	store.mu.Unlock()
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

func TestDurableStoreReplayAndRecoveredCheckpoint(t *testing.T) {
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
	coll, err := reopened.Tenants().GetCollection("tenant-a", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if got := coll.Count(); got != 2 {
		t.Fatalf("recovered document count = %d, want 2", got)
	}
	if _, ok := coll.GetDocument(2); ok {
		t.Fatal("deleted document reappeared after replay")
	}
	if _, err := reopened.Tenants().GetCollection("tenant-a", "temporary"); err == nil {
		t.Fatal("deleted collection reappeared after replay")
	}
	for _, path := range []string{base + ".journal", base + ".journal.frozen"} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("recovery did not clean covered journal %s: %v", path, err)
		}
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
	coll, err := reopened.Tenants().GetCollection("t", "hybrid")
	if err != nil || coll.Count() != 1 {
		t.Fatalf("hybrid collection after replay: count=%d err=%v", coll.Count(), err)
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
	managerPtr, tenantPtr := store.Manager(), store.Tenants()
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
	if store.Manager() != managerPtr || store.Tenants() != tenantPtr {
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
	if got := reopened.Manager().CollectionCount(); got != 1 {
		t.Fatalf("V2 collection count = %d, want 1", got)
	}
	if _, err := reopened.Tenants().GetCollection("existing", "existing-v3"); err != nil {
		t.Fatalf("existing V3 state lost: %v", err)
	}
	if _, err := reopened.Tenants().GetCollection("new", "docs"); err != nil {
		t.Fatalf("new V3 state lost: %v", err)
	}
}

func TestDurableStoreRejectsMalformedMutationBeforeReplay(t *testing.T) {
	cases := map[string]string{
		"unknown version": `{"version":2,"type":"delete_collection","payload":{"tenant_id":"t","collection_name":"c"}}`,
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
	retrieved, err := tenants.GetCollection("t", "docs")
	if err != nil {
		t.Fatal(err)
	}
	stored, ok := retrieved.GetDocument(41)
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
	if _, err := store.Tenants().CreateCollection(ctx, "t", ivf); err == nil || !strings.Contains(err.Error(), "only HNSW or Flat") {
		t.Fatalf("IVF create error = %v", err)
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
	if err := store.Close(); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("faulted close error = %v", err)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("replay faulted append: %v", err)
	}
	defer reopened.Close()
	coll, err := reopened.Tenants().GetCollection("t", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if coll.Count() != 1 {
		t.Fatalf("replayed count = %d, want 1", coll.Count())
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
	coll, err := store.Tenants().GetCollection("t", "docs")
	if err != nil {
		t.Fatal(err)
	}
	if got := coll.Count(); got != writers*perWriter {
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
	coll, _ = reopened.Tenants().GetCollection("t", "docs")
	if got := coll.Count(); got != writers*perWriter {
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
	coll, err := reopened.Tenants().GetCollection("t", "docs")
	if err != nil || coll.Count() != 1 {
		t.Fatalf("checkpoint retry state: count=%d err=%v", coll.Count(), err)
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
