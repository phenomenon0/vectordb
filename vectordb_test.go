package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

	// ANN search should return id1 for a matching query
	ixs := vs.SearchANN([]float32{1, 0, 0}, 1)
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
	ixs = vs.SearchANN([]float32{0, 1, 0}, 2)
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
	ids := vs.SearchANN(qvec, 1)
	if len(ids) != 1 {
		t.Fatalf("expected 1 result, got %d", len(ids))
	}
	if got := vs.GetID(ids[0]); got != "id-hello" {
		t.Fatalf("expected id-hello, got %s", got)
	}
}

func TestQueryFilterLogic(t *testing.T) {
	vs := NewVectorStore(10, 3)
	if _, err := vs.Add([]float32{1, 0, 0}, "doc-a", "a", map[string]string{"tag": "x", "env": "prod"}, "c1", ""); err != nil {
		t.Fatalf("failed to add: %v", err)
	}
	if _, err := vs.Add([]float32{0, 1, 0}, "doc-b", "b", map[string]string{"tag": "y", "env": "dev"}, "c2", ""); err != nil {
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
