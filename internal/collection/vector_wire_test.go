package collection

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

// This file pins the three JSON byte streams that must survive the
// Document.Vectors interface{}->typed refactor unchanged: HTTP/generic
// json.Marshal(Document), the durable journal mutation envelope, and the
// snapshot v2 document frame. Fixtures are golden files regenerated only via
// DEEPDATA_UPDATE_GOLDEN=1; ordinary runs must compare byte-for-byte.

const (
	wireGoldenDocID = 101
	wireStressDocID = 102
)

// wireStressDocument stresses float32 JSON formatting: subnormal-ish small
// magnitude, negative zero, near float32-max, a value above the 2^24 exact
// integer boundary, and a repeating fraction.
func wireStressDocument(id uint64) Document {
	return Document{
		ID: id,
		Vectors: map[string]interface{}{
			"embedding": []float32{0.1, -0.0, 1e-8, 3.4028235e38, 16777217, 1.0 / 3.0},
			"keywords": &sparse.SparseVector{
				Indices: []uint32{2, 4},
				Values:  []float32{1e-8, 3.4028235e38},
				Dim:     2048,
			},
		},
	}
}

func wireStressSchema() CollectionSchema {
	return CollectionSchema{
		Name: "wire-stress-docs",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  6,
				Index: IndexConfig{
					Type:   IndexTypeHNSW,
					Params: map[string]interface{}{"m": float64(16), "ef_construction": float64(200)},
				},
			},
			{
				Name: "keywords",
				Type: VectorTypeSparse,
				Dim:  2048,
				Index: IndexConfig{
					Type:   IndexTypeInverted,
					Params: map[string]interface{}{"k1": float64(1.2), "b": float64(0.75)},
				},
			},
		},
	}
}

func wireGoldenDocuments() []Document {
	return []Document{goldenTestDocument(wireGoldenDocID), wireStressDocument(wireStressDocID)}
}

func wireGoldenSchemas() []CollectionSchema {
	return []CollectionSchema{goldenTestSchema(), wireStressSchema()}
}

// assertGoldenFixture compares got against testdata/name, or (re)writes the
// fixture when DEEPDATA_UPDATE_GOLDEN=1 is set.
func assertGoldenFixture(t *testing.T, name string, got []byte) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if os.Getenv("DEEPDATA_UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
		}
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatalf("write golden fixture %s: %v", path, err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden fixture %s: %v (rerun with DEEPDATA_UPDATE_GOLDEN=1 to generate)", path, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("golden fixture %s mismatch:\n got:  %s\nwant: %s", path, got, want)
	}
}

func TestDocumentWireFormat(t *testing.T) {
	docs := wireGoldenDocuments()
	schemas := wireGoldenSchemas()

	// Stream 1: json.Marshal(Document) — HTTP responses.
	docLines := make([][]byte, len(docs))
	for i, doc := range docs {
		b, err := json.Marshal(doc)
		if err != nil {
			t.Fatalf("marshal document %d: %v", i, err)
		}
		docLines[i] = b
	}
	assertGoldenFixture(t, "document_wire_v2.json", bytes.Join(docLines, []byte("\n")))

	// Stream 2: durable journal mutation envelope (batch insert of both docs).
	mutation := canonicalMutation{
		typeName:       mutationBatchInsert,
		tenantID:       "tenant-wire",
		collectionName: "wire-docs",
		documents:      docs,
	}
	mutBytes, err := encodeDurableMutation(mutation)
	if err != nil {
		t.Fatalf("encodeDurableMutation: %v", err)
	}
	assertGoldenFixture(t, "mutation_batch_insert_v2.json", mutBytes)

	// Stream 3: snapshot v2 document frame.
	snapLines := make([][]byte, len(docs))
	for i, doc := range docs {
		doc := doc
		b, err := json.Marshal(collectionSnapshotV2Document{ID: doc.ID, Document: &doc})
		if err != nil {
			t.Fatalf("marshal snapshot frame %d: %v", i, err)
		}
		snapLines[i] = b
	}
	assertGoldenFixture(t, "snapshot_document_frame_v2.json", bytes.Join(snapLines, []byte("\n")))

	// (b) round-trip: decode stream 1, normalize against a matching schema,
	// re-marshal — bytes must be identical to what was marshaled above.
	for i, line := range docLines {
		var decoded Document
		if err := decodeCollectionJSON(line, &decoded); err != nil {
			t.Fatalf("decodeCollectionJSON document %d: %v", i, err)
		}
		if err := normalizeDocumentVectorTypes(&decoded, &schemas[i]); err != nil {
			t.Fatalf("normalizeDocumentVectorTypes document %d: %v", i, err)
		}
		reencoded, err := json.Marshal(decoded)
		if err != nil {
			t.Fatalf("re-marshal document %d: %v", i, err)
		}
		if !bytes.Equal(reencoded, line) {
			t.Fatalf("document %d round-trip mismatch:\n got:  %s\nwant: %s", i, reencoded, line)
		}
	}

	// (c) round-trip: decode stream 2, normalize (mirrors the real journal
	// replay path in prepareReplayMutation/normalizeReplayDocuments, which
	// always normalizes before reapplying decoded documents), re-encode —
	// bytes must be identical.
	decodedMutation, err := decodeDurableMutation(mutBytes)
	if err != nil {
		t.Fatalf("decodeDurableMutation: %v", err)
	}
	for i := range decodedMutation.documents {
		if err := normalizeDocumentVectorTypes(&decodedMutation.documents[i], &schemas[i]); err != nil {
			t.Fatalf("normalizeDocumentVectorTypes mutation document %d: %v", i, err)
		}
	}
	reencodedMutation, err := encodeDurableMutation(decodedMutation)
	if err != nil {
		t.Fatalf("re-encode mutation: %v", err)
	}
	if !bytes.Equal(reencodedMutation, mutBytes) {
		t.Fatalf("mutation round-trip mismatch:\n got:  %s\nwant: %s", reencodedMutation, mutBytes)
	}
}

// BenchmarkDecodeSnapshotDocument128d measures the allocation cost of the
// snapshot replay decode path (decodeCollectionJSON + normalizeDocumentVectorTypes)
// for a single document carrying one 128-d dense vector. This is the
// baseline the Vectors interface{}->typed refactor must improve on.
func BenchmarkDecodeSnapshotDocument128d(b *testing.B) {
	dense := make([]float32, 128)
	for i := range dense {
		dense[i] = float32(i) * 0.01
	}
	schema := CollectionSchema{
		Name: "bench-docs",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  128,
				Index: IndexConfig{
					Type:   IndexTypeHNSW,
					Params: map[string]interface{}{"m": float64(16), "ef_construction": float64(200)},
				},
			},
		},
	}
	doc := Document{ID: 1, Vectors: map[string]interface{}{"embedding": dense}}
	frame, err := json.Marshal(collectionSnapshotV2Document{ID: doc.ID, Document: &doc})
	if err != nil {
		b.Fatalf("marshal frame: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var record collectionSnapshotV2Document
		if err := decodeCollectionJSON(frame, &record); err != nil {
			b.Fatalf("decodeCollectionJSON: %v", err)
		}
		if err := normalizeDocumentVectorTypes(record.Document, &schema); err != nil {
			b.Fatalf("normalizeDocumentVectorTypes: %v", err)
		}
	}
}

// TestHeapFootprint100kDocuments measures the resident heap cost of holding
// 100k Documents, each with one 128-d dense vector, in memory at once. Skipped
// by default; opt in with DEEPDATA_HEAP_FOOTPRINT=1.
func TestHeapFootprint100kDocuments(t *testing.T) {
	if os.Getenv("DEEPDATA_HEAP_FOOTPRINT") != "1" {
		t.Skip("set DEEPDATA_HEAP_FOOTPRINT=1 to run")
	}
	const n = 100_000
	docs := make([]Document, n)
	for i := 0; i < n; i++ {
		dense := make([]float32, 128)
		for j := range dense {
			dense[j] = float32(i+j) * 0.001
		}
		docs[i] = Document{
			ID:      uint64(i + 1),
			Vectors: map[string]interface{}{"embedding": dense},
		}
	}

	runtime.GC()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	t.Logf("HeapAlloc=%d HeapObjects=%d", stats.HeapAlloc, stats.HeapObjects)
	runtime.KeepAlive(docs)
}
