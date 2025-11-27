package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStoreAddSearchMetaAndDelete(t *testing.T) {
	vs := NewVectorStore(10, 3)

	id1 := vs.Add([]float32{1, 0, 0}, "doc-1", "id1", map[string]string{"tag": "a"}, "c1")
	id2 := vs.Add([]float32{0, 1, 0}, "doc-2", "id2", map[string]string{"tag": "b"}, "c2")
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
	vs.Upsert([]float32{1, 0, 0}, "doc-1b", "id1", map[string]string{"tag": "a2"}, "c3")
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
	vs.Delete("id2")
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
	vs.Add([]float32{1, 0, 0}, "doc-1", "id1", map[string]string{"tag": "a"}, "c1")
	vs.Delete("id1")
	if err := vs.Save(path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	vs2, loaded := loadOrInitStore(path, 10, 3)
	if !loaded {
		t.Fatalf("expected to load snapshot")
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
	vs.Add([]float32{1, 0, 0}, "doc-1", "id1", nil, "")
	vs.Add([]float32{0, 1, 0}, "doc-2", "id2", nil, "")
	vs.Delete("id2")
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

	vs.Add([]float32{1, 0, 0}, "doc-1", "id1", nil, "")

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
	vs.Add(vec, "doc-hello", "id-hello", nil, "c1")

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
	vs.Add([]float32{1, 0, 0}, "doc-a", "a", map[string]string{"tag": "x", "env": "prod"}, "c1")
	vs.Add([]float32{0, 1, 0}, "doc-b", "b", map[string]string{"tag": "y", "env": "dev"}, "c2")

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
