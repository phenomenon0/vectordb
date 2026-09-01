package storage

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// generateTestPayload creates a realistic VectorStore payload for testing
func generateTestPayload(numVectors, dim int) *Payload {
	// Generate embeddings
	data := make([]float32, numVectors*dim)
	for i := range data {
		data[i] = rand.Float32()
	}

	// Generate docs and IDs
	docs := make([]string, numVectors)
	ids := make([]string, numVectors)
	for i := 0; i < numVectors; i++ {
		docs[i] = fmt.Sprintf("Document %d with some content for testing.", i)
		ids[i] = fmt.Sprintf("doc_%d", i)
	}

	// Generate metadata
	meta := make(map[uint64]map[string]string)
	for i := 0; i < numVectors; i++ {
		meta[uint64(i)] = map[string]string{
			"author":   fmt.Sprintf("author_%d", i%10),
			"category": fmt.Sprintf("cat_%d", i%5),
		}
	}

	// Generate lexical index data
	lexTF := make(map[uint64]map[string]int)
	docLen := make(map[uint64]int)
	for i := 0; i < numVectors; i++ {
		lexTF[uint64(i)] = map[string]int{"word1": 3, "word2": 5}
		docLen[uint64(i)] = 50 + i
	}

	df := map[string]int{"word1": numVectors / 2, "word2": numVectors / 3}

	return &Payload{
		Dim:       dim,
		Data:      data,
		Docs:      docs,
		IDs:       ids,
		Meta:      meta,
		Deleted:   map[uint64]bool{0: true, 5: true},
		Coll:      map[uint64]string{1: "collection_a", 2: "collection_b"},
		Next:      int64(numVectors),
		Count:     numVectors,
		HNSW:      []byte("mock_hnsw_data_here"),
		Checksum:  "abc123",
		LastSaved: time.Now().Truncate(time.Nanosecond),
		LexTF:     lexTF,
		DocLen:    docLen,
		DF:        df,
		SumDocL:   numVectors * 50,
		NumMeta:   map[uint64]map[string]float64{0: {"score": 0.95}},
		TimeMeta:  map[uint64]map[string]time.Time{0: {"created": time.Now().Truncate(time.Nanosecond)}},
	}
}

func TestGobRoundTrip(t *testing.T) {
	gob := &GobFormat{}
	payload := generateTestPayload(100, 384)

	var buf bytes.Buffer
	if err := gob.Save(&buf, payload); err != nil {
		t.Fatalf("gob save failed: %v", err)
	}

	loaded, err := gob.Load(&buf)
	if err != nil {
		t.Fatalf("gob load failed: %v", err)
	}

	// Verify key fields
	if loaded.Dim != payload.Dim {
		t.Errorf("Dim mismatch: got %d, want %d", loaded.Dim, payload.Dim)
	}
	if loaded.Count != payload.Count {
		t.Errorf("Count mismatch: got %d, want %d", loaded.Count, payload.Count)
	}
	if len(loaded.Data) != len(payload.Data) {
		t.Errorf("Data length mismatch: got %d, want %d", len(loaded.Data), len(payload.Data))
	}
	if len(loaded.Docs) != len(payload.Docs) {
		t.Errorf("Docs length mismatch: got %d, want %d", len(loaded.Docs), len(payload.Docs))
	}
}

func TestFormatRegistry(t *testing.T) {
	// The Cowrie formats were retired under SYS-03; gob is the only registered
	// format left, so it must also be the default the engine picks silently.
	formats := List()
	if len(formats) != 1 || formats[0] != "gob" {
		t.Errorf("Expected only the gob format to be registered, got %v", formats)
	}

	if Default() == nil {
		t.Fatal("Default format should not be nil")
	}
	if Default().Name() != "gob" {
		t.Errorf("Default format should be gob, got %s", Default().Name())
	}

	// Check Get works
	if Get("gob") == nil {
		t.Error("Get(gob) should not return nil")
	}
}

// Benchmarks

var benchSizes = []struct {
	vectors int
	dim     int
}{
	{100, 384},
	{1000, 384},
	{5000, 384},
}

func BenchmarkGobSave(b *testing.B) {
	gob := &GobFormat{}

	for _, sz := range benchSizes {
		payload := generateTestPayload(sz.vectors, sz.dim)

		b.Run(fmt.Sprintf("%dx%d", sz.vectors, sz.dim), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				var buf bytes.Buffer
				gob.Save(&buf, payload)
			}
		})
	}
}
func BenchmarkGobLoad(b *testing.B) {
	gob := &GobFormat{}

	for _, sz := range benchSizes {
		payload := generateTestPayload(sz.vectors, sz.dim)
		var buf bytes.Buffer
		gob.Save(&buf, payload)
		data := buf.Bytes()

		b.Run(fmt.Sprintf("%dx%d", sz.vectors, sz.dim), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			for i := 0; i < b.N; i++ {
				gob.Load(bytes.NewReader(data))
			}
		})
	}
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
