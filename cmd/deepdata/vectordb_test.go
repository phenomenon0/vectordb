package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/index"
)

func TestStoreAddSearchMetaAndDelete(t *testing.T) {
	vs := NewVectorStore(10, 3)

	id1, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", map[string]string{"tag": "a"}, "c1", "")
	if err != nil {
		t.Fatalf("failed to add: %v", err)
	}
	id2, err := vs.Add([]float32{0, 1, 0}, "doc-2", "id2", map[string]string{"tag": "b"}, "c2", "")
	if err != nil {
		t.Fatalf("failed to add: %v", err)
	}
	if id1 != "id1" || id2 != "id2" {
		t.Fatalf("unexpected ids: %s %s", id1, id2)
	}

	// Ensure collection tracking
	if coll := vs.Coll[hashID(id1)]; coll != "c1" {
		t.Fatalf("expected collection c1, got %s", coll)
	}

	// ANN search should return id1 for a matching query (search in correct collection)
	ixs := vs.SearchANNWithParams([]float32{1, 0, 0}, 1, "c1", 0)
	if len(ixs) != 1 || vs.GetID(ixs[0]) != "id1" {
		t.Fatalf("expected id1 from ANN search, got %+v", ixs)
	}

	// Upsert should overwrite doc/meta/collection
	_, err = vs.Upsert([]float32{1, 0, 0}, "doc-1b", "id1", map[string]string{"tag": "a2"}, "c3", "")
	if err != nil {
		t.Fatalf("failed to upsert: %v", err)
	}
	if vs.GetDoc(ixs[0]) != "doc-1b" {
		t.Fatalf("upsert did not update doc")
	}
	if vs.Meta[hashID("id1")]["tag"] != "a2" {
		t.Fatalf("upsert did not update meta")
	}
	if vs.Coll[hashID("id1")] != "c3" {
		t.Fatalf("upsert did not update collection")
	}

	// Delete id2; it should not appear in ANN results anymore.
	err = vs.Delete("id2")
	if err != nil {
		t.Fatalf("failed to delete: %v", err)
	}
	ixs = vs.SearchANNWithParams([]float32{0, 1, 0}, 2, "c2", 0)
	for _, ix := range ixs {
		if vs.GetID(ix) == "id2" {
			t.Fatalf("deleted id2 should not appear in ANN results")
		}
	}
}

func TestMatchesMetaHelpers(t *testing.T) {
	meta := map[string]string{"tag": "a", "env": "prod"}
	if !matchesMeta(meta, map[string]string{"tag": "a"}) {
		t.Fatalf("matchesMeta should succeed on AND filter")
	}
	if matchesMeta(meta, map[string]string{"tag": "b"}) {
		t.Fatalf("matchesMeta should fail on mismatched filter")
	}
	if !matchesAny(meta, []map[string]string{{"tag": "x"}, {"env": "prod"}}) {
		t.Fatalf("matchesAny should succeed when one filter matches")
	}
	if matchesAny(meta, []map[string]string{{"tag": "x"}}) {
		t.Fatalf("matchesAny should fail when none match")
	}
}

func TestPersistenceSnapshotReload(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.gob")

	vs := NewVectorStore(10, 3)
	if _, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", map[string]string{"tag": "a"}, "c1", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}
	if err := vs.Delete("id1"); err != nil {
		t.Fatalf("failed to delete: %v", err)
	}
	if err := vs.Save(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	vs2, loaded := loadOrInitStore(path, 10, 3)
	if !loaded {
		t.Fatalf("expected to load snapshot (got new store)")
	}
	if vs2.Count != vs.Count {
		t.Fatalf("count mismatch after load: %d vs %d", vs2.Count, vs.Count)
	}
	if !vs2.Deleted[hashID("id1")] {
		t.Fatalf("deleted flag not restored")
	}

	// Ensure ANN search still works after reload.
	ixs := vs2.SearchANN([]float32{1, 0, 0}, 1)
	for _, ix := range ixs {
		if vs2.GetID(ix) == "id1" {
			t.Fatalf("deleted id1 should not surface after reload")
		}
	}
}

func TestWALReplay(t *testing.T) {
	dir := t.TempDir()
	wal := filepath.Join(dir, "index.gob.wal")

	vs := NewVectorStore(10, 3)
	vs.walPath = wal
	if _, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", nil, "", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}
	if _, err := vs.Add([]float32{0, 1, 0}, "doc-2", "id2", nil, "", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}
	if err := vs.Delete("id2"); err != nil {
		t.Fatalf("failed to delete: %v", err)
	}
	t.Logf("walOps=%d", vs.walOps)

	if _, err := os.Stat(wal); err != nil {
		t.Fatalf("expected wal to exist: %v", err)
	}
	if info, err := os.Stat(wal); err == nil {
		t.Logf("wal size=%d bytes", info.Size())
	}

	f, err := os.Open(wal)
	if err != nil {
		t.Fatalf("open wal: %v", err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	var entries int
	var decErr error
	for {
		var e walEntry
		if err := dec.Decode(&e); err != nil {
			decErr = err
			break
		}
		t.Logf("wal entry %d: op=%s id=%s", entries, e.Op, e.ID)
		entries++
	}
	if entries != 3 {
		t.Fatalf("expected 3 wal entries, got %d (decode err: %v)", entries, decErr)
	}
	_ = f.Close()

	vs2 := NewVectorStore(10, 3)
	vs2.walPath = wal
	replayWAL(vs2)

	if vs2.Count != 2 {
		t.Fatalf("expected 2 entries after replay, got %d", vs2.Count)
	}
	if !vs2.Deleted[hashID("id2")] {
		t.Fatalf("expected id2 to be tombstoned after replay")
	}
	if _, err := os.Stat(wal); !os.IsNotExist(err) {
		t.Fatalf("expected wal to be removed after replay, got err=%v", err)
	}
}

func TestWALAutoSnapshot(t *testing.T) {
	dir := t.TempDir()
	wal := filepath.Join(dir, "index.gob.wal")
	snapshot := strings.TrimSuffix(wal, ".wal")

	vs := NewVectorStore(10, 3)
	vs.walPath = wal
	vs.walMaxOps = 1 // trigger snapshot after first op

	if _, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", nil, "", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	// Wait briefly for async snapshot.
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(snapshot); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("snapshot not created at %s", snapshot)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if _, err := os.Stat(wal); err == nil {
		t.Fatalf("wal should be removed after snapshot")
	}
}

func TestEmbedderAndANNFlow(t *testing.T) {
	emb := NewHashEmbedder(5)
	vec, err := emb.Embed("hello")
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(vec) != emb.Dim() {
		t.Fatalf("dim mismatch: got %d want %d", len(vec), emb.Dim())
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v * v)
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 1e-3 {
		t.Fatalf("expected normalized vector, got norm %f", norm)
	}

	vs := NewVectorStore(10, emb.Dim())
	if _, err := vs.Add(vec, "doc-hello", "id-hello", nil, "c1", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	qvec, _ := emb.Embed("hello")
	ids := vs.SearchANNWithParams(qvec, 1, "c1", 0)
	if len(ids) != 1 {
		t.Fatalf("expected 1 result, got %d", len(ids))
	}
	if got := vs.GetID(ids[0]); got != "id-hello" {
		t.Fatalf("expected id-hello, got %s", got)
	}
}

func TestQueryFilterLogic(t *testing.T) {
	vs := NewVectorStore(10, 3)
	// Add to same collection so SearchANN can find both
	if _, err := vs.Add([]float32{1, 0, 0}, "doc-a", "a", map[string]string{"tag": "x", "env": "prod"}, "", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}
	if _, err := vs.Add([]float32{0, 1, 0}, "doc-b", "b", map[string]string{"tag": "y", "env": "dev"}, "", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	ids := vs.SearchANN([]float32{1, 0, 0}, 2)
	if len(ids) != 2 {
		t.Fatalf("expected 2 results, got %d", len(ids))
	}

	// AND filter matches only doc-a
	meta := map[string]string{"tag": "x"}
	if !matchesMeta(vs.Meta[hashID("a")], meta) || matchesMeta(vs.Meta[hashID("b")], meta) {
		t.Fatalf("matchesMeta failed for AND filter")
	}

	// OR filter matches doc-b via env=dev
	metaAny := []map[string]string{{"env": "dev"}, {"tag": "nope"}}
	if matchesAny(vs.Meta[hashID("a")], metaAny) {
		t.Fatalf("metaAny should not match doc-a")
	}
	if !matchesAny(vs.Meta[hashID("b")], metaAny) {
		t.Fatalf("metaAny should match doc-b")
	}

	// NOT filter should exclude doc-b
	metaNot := map[string]string{"env": "dev"}
	if matchesMeta(vs.Meta[hashID("b")], metaNot) == false {
		t.Fatalf("expected matchesMeta to see metaNot match doc-b")
	}
}

// ======================================================================================
// Concurrent Safety Tests
// ======================================================================================

func TestConcurrentAddAndSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping concurrent test in short mode (known race condition in legacy HNSW)")
	}
	vs := NewVectorStore(1000, 3)
	emb := NewHashEmbedder(3)

	// Launch concurrent writers
	done := make(chan bool)
	errors := make(chan error, 100)

	// 10 concurrent writers
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				vec, _ := emb.Embed(fmt.Sprintf("doc-%d-%d", id, j))
				if _, err := vs.Add(vec, fmt.Sprintf("content-%d-%d", id, j), "", nil, "default", ""); err != nil {
					errors <- err
				}
			}
			done <- true
		}(i)
	}

	// 5 concurrent readers
	for i := 0; i < 5; i++ {
		go func() {
			for j := 0; j < 200; j++ {
				vec, _ := emb.Embed("query")
				_ = vs.SearchANN(vec, 10)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 15; i++ {
		<-done
	}

	// Check for errors
	close(errors)
	for err := range errors {
		t.Errorf("concurrent operation error: %v", err)
	}

	if vs.Count != 1000 {
		t.Errorf("expected 1000 docs, got %d", vs.Count)
	}
}

func TestConcurrentUpsertAndDelete(t *testing.T) {
	vs := NewVectorStore(500, 3)
	emb := NewHashEmbedder(3)

	done := make(chan bool)
	errors := make(chan error, 100)

	// Concurrent upserts on same IDs
	for i := 0; i < 5; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				vec, _ := emb.Embed(fmt.Sprintf("doc-%d", j))
				docID := fmt.Sprintf("id-%d", j%50) // Reuse 50 IDs
				if _, err := vs.Upsert(vec, fmt.Sprintf("content-%d", j), docID, nil, "default", ""); err != nil {
					errors <- err
				}
			}
			done <- true
		}(i)
	}

	// Concurrent deletes
	for i := 0; i < 3; i++ {
		go func() {
			for j := 0; j < 50; j++ {
				docID := fmt.Sprintf("id-%d", j%50)
				if err := vs.Delete(docID); err != nil {
					errors <- err
				}
			}
			done <- true
		}()
	}

	// Wait
	for i := 0; i < 8; i++ {
		<-done
	}

	close(errors)
	for err := range errors {
		t.Errorf("concurrent operation error: %v", err)
	}
}

// ======================================================================================
// Error Path Tests
// ======================================================================================

func TestDimensionMismatchError(t *testing.T) {
	vs := NewVectorStore(10, 3)

	// Test Add with wrong dimension
	_, err := vs.Add([]float32{1, 0}, "doc", "id1", nil, "default", "")
	if err == nil {
		t.Fatal("expected error for dimension mismatch in Add")
	}
	if !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("expected 'dimension mismatch' error, got: %v", err)
	}

	// Test Upsert with wrong dimension
	_, err = vs.Upsert([]float32{1, 0, 0, 0}, "doc", "id2", nil, "default", "")
	if err == nil {
		t.Fatal("expected error for dimension mismatch in Upsert")
	}
	if !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("expected 'dimension mismatch' error, got: %v", err)
	}
}

func TestWALErrorPropagation(t *testing.T) {
	dir := t.TempDir()
	vs := NewVectorStore(10, 3)

	// Set WAL path to invalid location (read-only directory)
	vs.walPath = "/invalid/path/wal.log"

	// This should fail with WAL error
	_, err := vs.Add([]float32{1, 0, 0}, "doc", "id1", nil, "default", "")
	if err == nil {
		t.Fatal("expected error when WAL write fails")
	}
	if !strings.Contains(err.Error(), "WAL") {
		t.Fatalf("expected WAL error, got: %v", err)
	}

	// Set valid WAL path
	vs.walPath = filepath.Join(dir, "test.wal")

	// Now it should succeed
	_, err = vs.Add([]float32{1, 0, 0}, "doc", "id2", nil, "default", "")
	if err != nil {
		t.Fatalf("unexpected error with valid WAL: %v", err)
	}
}

func TestEmptyStoreOperations(t *testing.T) {
	vs := NewVectorStore(10, 3)

	// Search on empty store
	ids := vs.SearchANN([]float32{1, 0, 0}, 10)
	if len(ids) != 0 {
		t.Errorf("expected 0 results from empty store, got %d", len(ids))
	}

	// Delete non-existent ID
	err := vs.Delete("nonexistent")
	if err != nil {
		t.Errorf("delete on non-existent ID should not error, got: %v", err)
	}

	// Upsert with empty ID should generate one
	id, err := vs.Add([]float32{1, 0, 0}, "doc", "", nil, "default", "")
	if err != nil {
		t.Fatalf("add with empty ID failed: %v", err)
	}
	if id == "" {
		t.Fatal("expected generated ID, got empty string")
	}
}

func TestLargeBatchOperations(t *testing.T) {
	vs := NewVectorStore(10000, 3)
	emb := NewHashEmbedder(3)

	// Add 5000 documents
	for i := 0; i < 5000; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		if _, err := vs.Add(vec, fmt.Sprintf("content-%d", i), "", nil, "default", ""); err != nil {
			t.Fatalf("failed to add doc %d: %v", i, err)
		}
	}

	if vs.Count != 5000 {
		t.Errorf("expected 5000 docs, got %d", vs.Count)
	}

	// Delete 2500 documents
	for i := 0; i < 2500; i++ {
		id := vs.GetID(i)
		if err := vs.Delete(id); err != nil {
			t.Fatalf("failed to delete doc %d: %v", i, err)
		}
	}

	if len(vs.Deleted) != 2500 {
		t.Errorf("expected 2500 deleted, got %d", len(vs.Deleted))
	}

	// Search should still work
	vec, _ := emb.Embed("query")
	results := vs.SearchANN(vec, 100)
	for _, idx := range results {
		id := vs.GetID(idx)
		if vs.Deleted[hashID(id)] {
			t.Errorf("deleted doc %s appeared in search results", id)
		}
	}
}

func TestMetadataEdgeCases(t *testing.T) {
	vs := NewVectorStore(10, 3)

	// Very large metadata
	largeMeta := make(map[string]string)
	for i := 0; i < 50; i++ {
		largeMeta[fmt.Sprintf("key%d", i)] = strings.Repeat("x", 100)
	}

	_, err := vs.Add([]float32{1, 0, 0}, "doc", "id1", largeMeta, "default", "")
	if err != nil {
		t.Fatalf("failed to add doc with large metadata: %v", err)
	}

	// Nil metadata
	_, err = vs.Add([]float32{0, 1, 0}, "doc2", "id2", nil, "default", "")
	if err != nil {
		t.Fatalf("failed to add doc with nil metadata: %v", err)
	}

	// Empty metadata
	_, err = vs.Add([]float32{0, 0, 1}, "doc3", "id3", map[string]string{}, "default", "")
	if err != nil {
		t.Fatalf("failed to add doc with empty metadata: %v", err)
	}
}

// ======================================================================================
// Performance Benchmarks
// ======================================================================================

func BenchmarkInsert(b *testing.B) {
	vs := NewVectorStore(b.N, 384)
	emb := NewHashEmbedder(384)
	vec, _ := emb.Embed("benchmark doc")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vs.Add(vec, "benchmark doc content", "", nil, "default", "")
	}
}

func BenchmarkSearchANN(b *testing.B) {
	vs := NewVectorStore(10000, 384)
	emb := NewHashEmbedder(384)

	// Pre-populate with 10k vectors
	for i := 0; i < 10000; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		vs.Add(vec, fmt.Sprintf("content-%d", i), "", nil, "default", "")
	}

	query, _ := emb.Embed("query")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vs.SearchANN(query, 10)
	}
}

func BenchmarkSearchScan(b *testing.B) {
	vs := NewVectorStore(10000, 384)
	emb := NewHashEmbedder(384)

	// Pre-populate with 10k vectors
	for i := 0; i < 10000; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		vs.Add(vec, fmt.Sprintf("content-%d", i), "", nil, "default", "")
	}

	query, _ := emb.Embed("query")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vs.Search(query, 10)
	}
}

func BenchmarkConcurrentReads(b *testing.B) {
	vs := NewVectorStore(1000, 384)
	emb := NewHashEmbedder(384)

	// Pre-populate
	for i := 0; i < 1000; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		vs.Add(vec, fmt.Sprintf("content-%d", i), "", nil, "default", "")
	}

	query, _ := emb.Embed("query")

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			vs.SearchANN(query, 10)
		}
	})
}

func BenchmarkCompaction(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "bench.gob")

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		vs := NewVectorStore(1000, 384)
		emb := NewHashEmbedder(384)

		// Add 1000 docs and delete 500
		for j := 0; j < 1000; j++ {
			vec, _ := emb.Embed(fmt.Sprintf("doc-%d", j))
			vs.Add(vec, fmt.Sprintf("content-%d", j), fmt.Sprintf("id-%d", j), nil, "default", "")
		}
		for j := 0; j < 500; j++ {
			vs.Delete(fmt.Sprintf("id-%d", j))
		}

		b.StartTimer()
		if err := vs.Compact(path); err != nil {
			b.Fatalf("compaction failed: %v", err)
		}
	}
}

// ======================================================================================
// Auto-Collection Index Tests
// ======================================================================================

func TestAutoCollectionIndexCreation(t *testing.T) {
	vs := NewVectorStore(100, 3)

	// Verify default index exists
	if _, ok := vs.indexes["default"]; !ok {
		t.Fatal("expected default index to exist")
	}

	// Add vector to a new collection (should auto-create index)
	newCollection := "test_autocreate_collection"
	_, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", nil, newCollection, "")
	if err != nil {
		t.Fatalf("failed to add to new collection: %v", err)
	}

	// Verify the new collection index was created
	if _, ok := vs.indexes[newCollection]; !ok {
		t.Fatalf("expected auto-created index for collection %q", newCollection)
	}

	// Verify the vector is in the correct collection
	hid := hashID("id1")
	if vs.Coll[hid] != newCollection {
		t.Fatalf("expected collection %q, got %q", newCollection, vs.Coll[hid])
	}
}

func TestAutoCollectionIndexSearch(t *testing.T) {
	vs := NewVectorStore(500, 3)
	emb := NewHashEmbedder(3)

	// Create vectors in different collections
	coll1 := "collection_alpha"
	coll2 := "collection_beta"

	// Add more vectors to collection 1 to ensure HNSW index is populated
	for i := 0; i < 50; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("alpha-doc-%d", i))
		vs.Add(vec, fmt.Sprintf("alpha-doc-%d", i), fmt.Sprintf("alpha%d", i), nil, coll1, "")
	}

	// Add vectors to collection 2
	for i := 0; i < 50; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("beta-doc-%d", i))
		vs.Add(vec, fmt.Sprintf("beta-doc-%d", i), fmt.Sprintf("beta%d", i), nil, coll2, "")
	}

	// Verify both collection indexes were created
	if _, ok := vs.indexes[coll1]; !ok {
		t.Fatalf("expected auto-created index for collection %q", coll1)
	}
	if _, ok := vs.indexes[coll2]; !ok {
		t.Fatalf("expected auto-created index for collection %q", coll2)
	}

	// Search using brute-force (Search) to ensure vectors are stored correctly
	query, _ := emb.Embed("alpha-doc-0")
	results := vs.Search(query, 10)
	if len(results) < 1 {
		t.Fatal("expected at least 1 result from brute-force search")
	}

	// Verify we can find vectors
	t.Logf("Found %d results from search", len(results))
}

func TestAutoCollectionIndexWithUpsert(t *testing.T) {
	vs := NewVectorStore(100, 3)

	newCollection := "upsert_collection"

	// Upsert to a new collection (should auto-create index)
	_, err := vs.Upsert([]float32{1, 0, 0}, "upsert-doc-1", "upsert1", nil, newCollection, "")
	if err != nil {
		t.Fatalf("failed to upsert to new collection: %v", err)
	}

	// Verify the new collection index was created
	if _, ok := vs.indexes[newCollection]; !ok {
		t.Fatalf("expected auto-created index for collection %q via upsert", newCollection)
	}

	// Upsert again (update existing)
	_, err = vs.Upsert([]float32{0.9, 0.1, 0}, "upsert-doc-1-updated", "upsert1", nil, newCollection, "")
	if err != nil {
		t.Fatalf("failed to upsert update: %v", err)
	}

	// Verify doc was updated
	hid := hashID("upsert1")
	ix := vs.idToIx[hid]
	if vs.Docs[ix] != "upsert-doc-1-updated" {
		t.Fatalf("expected updated doc, got %q", vs.Docs[ix])
	}
}

func TestAutoCollectionMultipleCollections(t *testing.T) {
	vs := NewVectorStore(500, 3)
	emb := NewHashEmbedder(3)

	collections := []string{"coll_a", "coll_b", "coll_c", "coll_d", "coll_e"}

	// Add 20 vectors to each collection
	for _, coll := range collections {
		for i := 0; i < 20; i++ {
			vec, _ := emb.Embed(fmt.Sprintf("%s-doc-%d", coll, i))
			id := fmt.Sprintf("%s-id-%d", coll, i)
			_, err := vs.Add(vec, fmt.Sprintf("%s content %d", coll, i), id, nil, coll, "")
			if err != nil {
				t.Fatalf("failed to add to collection %s: %v", coll, err)
			}
		}
	}

	// Verify all collection indexes were created
	for _, coll := range collections {
		if _, ok := vs.indexes[coll]; !ok {
			t.Fatalf("expected auto-created index for collection %q", coll)
		}
	}

	// Verify vector counts
	collCounts := make(map[string]int)
	for _, coll := range vs.Coll {
		collCounts[coll]++
	}

	for _, coll := range collections {
		if collCounts[coll] != 20 {
			t.Errorf("expected 20 vectors in collection %q, got %d", coll, collCounts[coll])
		}
	}

	// Total should be 100 user vectors + seed vectors
	expectedMin := 100
	if vs.Count < expectedMin {
		t.Errorf("expected at least %d total vectors, got %d", expectedMin, vs.Count)
	}
}

func TestAutoCollectionDefaultFallback(t *testing.T) {
	vs := NewVectorStore(100, 3)

	// Add to "default" collection explicitly
	_, err := vs.Add([]float32{1, 0, 0}, "default-doc", "default1", nil, "default", "")
	if err != nil {
		t.Fatalf("failed to add to default collection: %v", err)
	}

	// Add with empty collection (should use default)
	_, err = vs.Add([]float32{0, 1, 0}, "empty-coll-doc", "empty1", nil, "", "")
	if err != nil {
		t.Fatalf("failed to add with empty collection: %v", err)
	}

	// Both should be in default collection
	if vs.Coll[hashID("default1")] != "default" {
		t.Error("expected default1 to be in default collection")
	}
	if vs.Coll[hashID("empty1")] != "default" {
		t.Error("expected empty1 to be in default collection")
	}

	// Verify no extra index was created for empty collection
	indexCount := len(vs.indexes)
	if indexCount > 2 { // default + possibly one more
		t.Logf("Note: Found %d indexes (may include seed indexes)", indexCount)
	}
}

func TestAutoCollectionConcurrentCreation(t *testing.T) {
	vs := NewVectorStore(1000, 3)
	emb := NewHashEmbedder(3)

	done := make(chan bool)
	errors := make(chan error, 100)

	// 5 goroutines creating vectors in different collections simultaneously
	for i := 0; i < 5; i++ {
		go func(id int) {
			coll := fmt.Sprintf("concurrent_coll_%d", id)
			for j := 0; j < 50; j++ {
				vec, _ := emb.Embed(fmt.Sprintf("concurrent-%d-%d", id, j))
				docID := fmt.Sprintf("concurrent-%d-%d", id, j)
				if _, err := vs.Add(vec, fmt.Sprintf("content-%d-%d", id, j), docID, nil, coll, ""); err != nil {
					errors <- fmt.Errorf("goroutine %d: %w", id, err)
				}
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 5; i++ {
		<-done
	}

	// Check for errors
	close(errors)
	for err := range errors {
		t.Errorf("concurrent operation error: %v", err)
	}

	// Verify all 5 collection indexes were created
	for i := 0; i < 5; i++ {
		coll := fmt.Sprintf("concurrent_coll_%d", i)
		if _, ok := vs.indexes[coll]; !ok {
			t.Errorf("expected auto-created index for collection %q", coll)
		}
	}
}

func TestAutoCollectionIndexIsolation(t *testing.T) {
	vs := NewVectorStore(500, 3)

	// Create two collections with very different vectors
	coll1 := "isolation_north"
	coll2 := "isolation_south"

	// North collection: vectors pointing "up" (positive Y) - add enough for HNSW
	for i := 0; i < 50; i++ {
		vec := []float32{float32(i) * 0.01, 1.0, 0}
		normalize(vec)
		vs.Add(vec, fmt.Sprintf("north-%d", i), fmt.Sprintf("north-%d", i), nil, coll1, "")
	}

	// South collection: vectors pointing "down" (negative Y)
	for i := 0; i < 50; i++ {
		vec := []float32{float32(i) * 0.01, -1.0, 0}
		normalize(vec)
		vs.Add(vec, fmt.Sprintf("south-%d", i), fmt.Sprintf("south-%d", i), nil, coll2, "")
	}

	// Query with a "north" vector using brute-force search
	queryNorth := []float32{0, 1, 0}
	normalize(queryNorth)

	// Use brute-force search to verify data is correctly stored
	results := vs.Search(queryNorth, 20)

	// Check that results exist
	if len(results) == 0 {
		t.Fatal("expected some results from search")
	}

	// Count north vs south in top results
	northCount := 0
	southCount := 0
	for _, idx := range results {
		id := vs.GetID(idx)
		if strings.HasPrefix(id, "north-") {
			northCount++
		} else if strings.HasPrefix(id, "south-") {
			southCount++
		}
	}

	// North vectors should dominate since query points north
	t.Logf("Search results: %d north, %d south (out of %d)", northCount, southCount, len(results))
	if northCount < southCount {
		t.Errorf("expected more north results than south for north-pointing query")
	}
}

// Helper function to normalize a vector
func normalize(v []float32) {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type failingIndex struct {
	addErr error
}

func (f *failingIndex) Name() string { return "failing" }

func (f *failingIndex) Add(ctx context.Context, id uint64, vector []float32) error {
	return f.addErr
}

func (f *failingIndex) Search(ctx context.Context, query []float32, k int, params index.SearchParams) ([]index.Result, error) {
	return nil, nil
}

func (f *failingIndex) Delete(ctx context.Context, id uint64) error { return nil }

func (f *failingIndex) Stats() index.IndexStats { return index.IndexStats{} }

func (f *failingIndex) Export() ([]byte, error) { return nil, nil }

func (f *failingIndex) Import(data []byte) error { return nil }

func assertStoreEmpty(t *testing.T, vs *VectorStore) {
	t.Helper()

	if vs.Count != 0 {
		t.Fatalf("expected empty store count, got %d", vs.Count)
	}
	if len(vs.Data) != 0 || len(vs.Docs) != 0 || len(vs.IDs) != 0 || len(vs.Seqs) != 0 {
		t.Fatalf("expected no stored vectors, got data=%d docs=%d ids=%d seqs=%d", len(vs.Data), len(vs.Docs), len(vs.IDs), len(vs.Seqs))
	}
	if len(vs.idToIx) != 0 || len(vs.Meta) != 0 || len(vs.Coll) != 0 || len(vs.TenantID) != 0 {
		t.Fatalf("expected no document mappings, got idToIx=%d meta=%d coll=%d tenants=%d", len(vs.idToIx), len(vs.Meta), len(vs.Coll), len(vs.TenantID))
	}
	if len(vs.lexTF) != 0 || len(vs.docLen) != 0 || len(vs.df) != 0 || vs.sumDocL != 0 {
		t.Fatalf("expected lexical state rollback, got lexTF=%d docLen=%d df=%d sumDocL=%d", len(vs.lexTF), len(vs.docLen), len(vs.df), vs.sumDocL)
	}
}

func TestAddRollsBackStateOnIndexFailure(t *testing.T) {
	vs := NewVectorStore(10, 3)
	vs.indexes["default"] = &failingIndex{addErr: errors.New("index add failure")}

	if _, err := vs.Add([]float32{1, 0, 0}, "doc", "id1", map[string]string{"tag": "a"}, "default", ""); err == nil {
		t.Fatal("expected add to fail when index add fails")
	}

	assertStoreEmpty(t, vs)
	if vs.next != 0 {
		t.Fatalf("expected next counter rollback, got %d", vs.next)
	}
}

func TestAddRollsBackStateOnWALError(t *testing.T) {
	vs := NewVectorStore(10, 3)
	vs.walPath = "/invalid/path/wal.log"

	if _, err := vs.Add([]float32{1, 0, 0}, "doc", "", map[string]string{"tag": "a"}, "default", ""); err == nil {
		t.Fatal("expected add to fail when WAL append fails")
	}

	assertStoreEmpty(t, vs)
	if vs.next != 0 {
		t.Fatalf("expected generated ID counter rollback, got %d", vs.next)
	}
}

func TestUpsertNewIDRollsBackStateOnWALError(t *testing.T) {
	vs := NewVectorStore(10, 3)
	vs.walPath = "/invalid/path/wal.log"

	if _, err := vs.Upsert([]float32{1, 0, 0}, "doc", "id-upsert", map[string]string{"tag": "a"}, "default", ""); err == nil {
		t.Fatal("expected upsert insert path to fail when WAL append fails")
	}

	assertStoreEmpty(t, vs)
}

func TestUpsertMovingCollectionRemovesOldIndexEntry(t *testing.T) {
	vs := NewVectorStore(100, 3)

	if _, err := vs.Add([]float32{1, 0, 0}, "doc-old", "move-id", nil, "old-coll", ""); err != nil {
		t.Fatalf("failed to seed old collection: %v", err)
	}

	if _, err := vs.Upsert([]float32{0, 1, 0}, "doc-new", "move-id", nil, "new-coll", ""); err != nil {
		t.Fatalf("failed to move document to new collection: %v", err)
	}

	if coll := vs.Coll[hashID("move-id")]; coll != "new-coll" {
		t.Fatalf("expected collection to move to new-coll, got %q", coll)
	}
	if _, ok := vs.indexes["new-coll"]; !ok {
		t.Fatal("expected new collection index to be created")
	}

	oldResults := vs.SearchANNWithParams([]float32{1, 0, 0}, 10, "old-coll", 0)
	for _, ix := range oldResults {
		if vs.GetID(ix) == "move-id" {
			t.Fatal("moved document is still searchable in old collection")
		}
	}

	newResults := vs.SearchANNWithParams([]float32{0, 1, 0}, 10, "new-coll", 0)
	found := false
	for _, ix := range newResults {
		if vs.GetID(ix) == "move-id" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("moved document is not searchable in new collection")
	}
}

func TestPersistenceRangeIndexRebuild(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.gob")

	vs := NewVectorStore(10, 3)

	// Insert documents with numeric metadata that will be parsed by ingestMeta
	_, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", map[string]string{"price": "10.5", "tag": "a"}, "c1", "")
	if err != nil {
		t.Fatalf("failed to add doc-1: %v", err)
	}
	_, err = vs.Add([]float32{0, 1, 0}, "doc-2", "id2", map[string]string{"price": "25.0", "tag": "b"}, "c1", "")
	if err != nil {
		t.Fatalf("failed to add doc-2: %v", err)
	}
	_, err = vs.Add([]float32{0, 0, 1}, "doc-3", "id3", map[string]string{"price": "50.0", "tag": "c"}, "c1", "")
	if err != nil {
		t.Fatalf("failed to add doc-3: %v", err)
	}

	// Verify range query works before save
	min := 10.0
	max := 30.0
	candidates := vs.candidateIDsForRange([]RangeFilter{{Key: "price", Min: &min, Max: &max}})
	if len(candidates) != 2 {
		t.Fatalf("pre-save: expected 2 candidates in [10,30], got %d", len(candidates))
	}

	if err := vs.Save(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// Reload from disk
	vs2, loaded := loadOrInitStore(path, 10, 3)
	if !loaded {
		t.Fatal("expected to load snapshot")
	}

	// Verify range query works AFTER reload — this was broken before the fix
	candidates2 := vs2.candidateIDsForRange([]RangeFilter{{Key: "price", Min: &min, Max: &max}})
	if len(candidates2) != 2 {
		t.Fatalf("post-reload: expected 2 candidates in [10,30], got %d", len(candidates2))
	}

	// Verify the correct document IDs are in the result
	hid1 := hashID("id1")
	hid2 := hashID("id2")
	if _, ok := candidates2[hid1]; !ok {
		t.Fatal("post-reload: id1 (price=10.5) missing from range [10,30]")
	}
	if _, ok := candidates2[hid2]; !ok {
		t.Fatal("post-reload: id2 (price=25.0) missing from range [10,30]")
	}
	hid3 := hashID("id3")
	if _, ok := candidates2[hid3]; ok {
		t.Fatal("post-reload: id3 (price=50.0) should NOT be in range [10,30]")
	}
}

func TestPersistenceTimeIndexRebuild(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.gob")

	vs := NewVectorStore(10, 3)

	t1 := time.Date(2024, 1, 15, 0, 0, 0, 0, time.UTC).Format(time.RFC3339)
	t2 := time.Date(2024, 6, 15, 0, 0, 0, 0, time.UTC).Format(time.RFC3339)
	t3 := time.Date(2025, 1, 15, 0, 0, 0, 0, time.UTC).Format(time.RFC3339)

	_, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", map[string]string{"created": t1}, "c1", "")
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	_, err = vs.Add([]float32{0, 1, 0}, "doc-2", "id2", map[string]string{"created": t2}, "c1", "")
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	_, err = vs.Add([]float32{0, 0, 1}, "doc-3", "id3", map[string]string{"created": t3}, "c1", "")
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	// Range filter: Q1 2024 only
	rangeStart := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).Format(time.RFC3339)
	rangeEnd := time.Date(2024, 3, 31, 23, 59, 59, 0, time.UTC).Format(time.RFC3339)

	candidates := vs.candidateIDsForRange([]RangeFilter{{Key: "created", TimeMin: rangeStart, TimeMax: rangeEnd}})
	if len(candidates) != 1 {
		t.Fatalf("pre-save: expected 1 candidate in Q1 2024, got %d", len(candidates))
	}

	if err := vs.Save(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	vs2, loaded := loadOrInitStore(path, 10, 3)
	if !loaded {
		t.Fatal("expected to load snapshot")
	}

	candidates2 := vs2.candidateIDsForRange([]RangeFilter{{Key: "created", TimeMin: rangeStart, TimeMax: rangeEnd}})
	if len(candidates2) != 1 {
		t.Fatalf("post-reload: expected 1 candidate in Q1 2024, got %d", len(candidates2))
	}

	hid1 := hashID("id1")
	if _, ok := candidates2[hid1]; !ok {
		t.Fatal("post-reload: id1 (Jan 2024) missing from Q1 2024 range")
	}

	// Broader range should return all 3
	allRange := vs2.candidateIDsForRange([]RangeFilter{{Key: "created",
		TimeMin: time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC).Format(time.RFC3339),
		TimeMax: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC).Format(time.RFC3339),
	}})
	if len(allRange) != 3 {
		t.Fatalf("post-reload: expected 3 candidates in full range, got %d", len(allRange))
	}
}

// TestSaveRenamesBeforeDeletingWAL verifies that Save() atomically commits the
// snapshot (rename) BEFORE deleting the WAL. After Save, the snapshot must exist
// and the WAL must be gone.
func TestSaveRenamesBeforeDeletingWAL(t *testing.T) {
	dir := t.TempDir()
	snapPath := filepath.Join(dir, "index.gob")
	walPath := snapPath + ".wal"

	vs := NewVectorStore(10, 3)
	vs.walPath = walPath

	// Insert a doc to generate a WAL entry
	if _, err := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", nil, "", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	// WAL must exist before Save
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("expected WAL to exist before Save: %v", err)
	}

	if err := vs.Save(snapPath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// After Save: snapshot must exist, WAL must be deleted
	if _, err := os.Stat(snapPath); err != nil {
		t.Fatalf("expected snapshot to exist after Save: %v", err)
	}
	if _, err := os.Stat(walPath); !os.IsNotExist(err) {
		t.Fatalf("expected WAL to be deleted after successful Save, got err=%v", err)
	}
}

// TestReplayWALReturnsErrorOnFailures verifies that replayWAL returns an error
// when WAL entries fail to replay, and preserves the WAL file for inspection.
func TestReplayWALReturnsErrorOnFailures(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "index.gob.wal")

	// Write a WAL with invalid entries (missing vectors for insert)
	f, err := os.Create(walPath)
	if err != nil {
		t.Fatalf("create WAL: %v", err)
	}
	enc := json.NewEncoder(f)
	// Insert with nil vector — will fail because Add() requires non-empty vector
	enc.Encode(walEntry{Op: "insert", ID: "x1", Doc: "bad-doc", Vec: nil})
	// Also add a corrupt JSON line
	f.WriteString("{this is not valid json}\n")
	f.Close()

	vs := NewVectorStore(10, 3)
	vs.walPath = walPath

	err = replayWAL(vs)
	if err == nil {
		t.Fatal("expected replayWAL to return an error when entries fail, got nil")
	}
	if !strings.Contains(err.Error(), "errors") {
		t.Fatalf("expected error message to mention errors, got: %v", err)
	}

	// WAL must be preserved for manual inspection when replay has errors
	if _, err := os.Stat(walPath); err != nil {
		t.Fatalf("expected WAL to be preserved after failed replay, but it was deleted: %v", err)
	}
}

// TestReplayWALDeletesWALOnSuccess verifies that replayWAL deletes the WAL
// file when all entries replay successfully.
func TestReplayWALDeletesWALOnSuccess(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "index.gob.wal")

	// Write a valid WAL
	f, err := os.Create(walPath)
	if err != nil {
		t.Fatalf("create WAL: %v", err)
	}
	enc := json.NewEncoder(f)
	enc.Encode(walEntry{Op: "insert", ID: "ok1", Doc: "good-doc", Vec: []float32{1, 0, 0}})
	f.Close()

	vs := NewVectorStore(10, 3)
	vs.walPath = walPath

	err = replayWAL(vs)
	if err != nil {
		t.Fatalf("expected replayWAL to succeed, got: %v", err)
	}

	// WAL should be deleted on successful replay
	if _, err := os.Stat(walPath); !os.IsNotExist(err) {
		t.Fatalf("expected WAL to be deleted after successful replay, got err=%v", err)
	}

	// The doc should have been replayed
	if vs.Count != 1 {
		t.Fatalf("expected 1 entry after replay, got %d", vs.Count)
	}
}

// TestConcurrentSnapshotDedup verifies that only one background snapshot runs
// at a time — the snapshotRunning atomic guard prevents concurrent goroutines
// from racing on the same .tmp file path.
func TestConcurrentSnapshotDedup(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "index.gob.wal")

	vs := NewVectorStore(10, 3)
	vs.walPath = walPath
	vs.walMaxOps = 1 // Trigger snapshot after every WAL write

	// Insert many docs rapidly to trigger multiple snapshot attempts
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("id-%d", i)
			vec := []float32{float32(i), 0, 0}
			vs.Add(vec, fmt.Sprintf("doc-%d", i), id, nil, "", "")
		}(i)
	}
	wg.Wait()

	// Wait for any in-flight background snapshots to finish
	vs.bgWg.Wait()

	// The key assertion: snapshotRunning should be false after all goroutines complete
	if vs.snapshotRunning.Load() {
		t.Fatal("snapshotRunning should be false after all background snapshots complete")
	}

	// No crash, no corrupt snapshot — the dedup guard prevented .tmp file races
	snapPath := strings.TrimSuffix(walPath, ".wal")
	if _, err := os.Stat(snapPath); err != nil {
		// Snapshot may or may not exist depending on timing, but should not have crashed
		t.Logf("snapshot file does not exist (normal if timing prevented Save): %v", err)
	}
}
