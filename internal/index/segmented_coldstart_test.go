package index

// Cold-start memory envelope of SegmentedIndex against one plain HNSWIndex
// (gate RCV-06). The durable store rebuilds every dense index from scratch on
// open: snapshot v2 restore feeds one Index.Add per document
// (internal/collection/snapshot.go restoreCollectionSnapshotV2Document ->
// Collection.addToIndex), then journal replay feeds one BatchAdder.BatchAdd
// per record (internal/collection/durable_store.go applyMutationDirect ->
// Collection.addPreparedDocumentsLocked). Neither path calls Export: a v2
// snapshot persists documents, not index blobs (snapshot.go
// validateCollectionSnapshotV2RepresentableLocked). The three non-test
// Index.Export call sites are all off that path: manager.go
// capturePersistedState (legacy state file; a Collection there can hold a
// SegmentedIndex, but its only entry points CollectionManager.Save and
// TenantManager.Save have no non-test callers), collection.go ExportIndexes
// (no callers), and cmd/deepdata/main.go VectorStore.Save (its indexes come
// from NewHNSWIndex or the index.Create factory, which registers only
// hnsw/flat/sparse, so it never holds a SegmentedIndex). Merge-on-export is
// therefore not measured here.

import (
	"context"
	"runtime"
	"runtime/debug"
	"testing"
)

const (
	coldStartDocs     = 4000
	coldStartDim      = 64
	coldStartBatch    = 100 // journal records of 100 documents, as durable_bench_test.go batches
	coldStartSegments = 4
)

func coldStartCorpus() [][]float32 {
	lcg := newSegLCG(42)
	docs := make([][]float32, coldStartDocs)
	for i := range docs {
		docs[i] = lcg.vec(coldStartDim)
	}
	return docs
}

func newColdStartHNSW() (Index, error) { return NewHNSWIndex(coldStartDim, nil) }

func newColdStartSegmented() (Index, error) {
	return NewSegmentedIndex(coldStartSegments, newColdStartHNSW)
}

// coldStartRestore rebuilds a fresh index from corpus along the cold-start
// path: the first half as one Add per document (snapshot restore), the second
// half as BatchAdd records of coldStartBatch documents (journal replay).
func coldStartRestore(tb testing.TB, newIndex func() (Index, error), corpus [][]float32) Index {
	ctx := context.Background()
	idx, err := newIndex()
	if err != nil {
		tb.Fatal(err)
	}
	half := len(corpus) / 2
	for id := 0; id < half; id++ {
		if err := idx.Add(ctx, uint64(id), corpus[id]); err != nil {
			tb.Fatalf("Add(%d): %v", id, err)
		}
	}
	batcher := idx.(BatchAdder)
	for start := half; start < len(corpus); start += coldStartBatch {
		end := min(start+coldStartBatch, len(corpus))
		batch := make(map[uint64][]float32, end-start)
		for id := start; id < end; id++ {
			batch[uint64(id)] = corpus[id]
		}
		if err := batcher.BatchAdd(ctx, batch); err != nil {
			tb.Fatalf("BatchAdd(%d..%d): %v", start, end, err)
		}
	}
	return idx
}

// measureColdStart returns the bytes allocated by the rebuild and the bytes
// still live after a forced GC with the index held. The collector is off for
// the duration so TotalAlloc is exact and no cycle runs mid-build. The corpus
// is pinned until both readings are taken: the index copies every vector, so
// a caller whose last use of corpus is this call would otherwise see it
// collected between the two readings and the retained figure understated.
func measureColdStart(tb testing.TB, newIndex func() (Index, error), corpus [][]float32) (totalAlloc, retained uint64) {
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	idx := coldStartRestore(tb, newIndex, corpus)
	runtime.GC()
	runtime.ReadMemStats(&after)
	runtime.KeepAlive(idx)
	runtime.KeepAlive(corpus)
	return after.TotalAlloc - before.TotalAlloc, after.HeapAlloc - before.HeapAlloc
}

// BenchmarkSegmentedColdStart rebuilds the same 4000x64 corpus into one plain
// HNSWIndex and into a 4-segment SegmentedIndex along the cold-start path.
// retained-B/op is the heap still live after runtime.GC() with the index held.
func BenchmarkSegmentedColdStart(b *testing.B) {
	corpus := coldStartCorpus()
	for _, tc := range []struct {
		name     string
		newIndex func() (Index, error)
	}{
		{"HNSW", newColdStartHNSW},
		{"Segmented4", newColdStartSegmented},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			var retained uint64
			for i := 0; i < b.N; i++ {
				_, r := measureColdStart(b, tc.newIndex, corpus)
				retained += r
			}
			b.ReportMetric(float64(retained)/float64(b.N), "retained-B/op")
		})
	}
}

// TestSegmentedColdStartEnvelope answers the 2026-08-28 review question
// (tasks/journal/2026-08-28-bounded-recovery.md:37-40): does splitting one
// HNSW graph into N widen the cold-start memory envelope? Measured 2026-09-01
// on the 4000x64 corpus, 4 segments against 1 graph (BenchmarkSegmentedColdStart,
// 16-core host): total allocation ratio 0.60 (smaller graphs mean cheaper
// insert searches), retained heap ratio 1.00-1.02; at 16 segments 0.39 and
// 1.05. K = 1.25 leaves headroom for level-RNG and map-growth noise but fails
// if segmentation ever costs more than a small constant factor — a per-segment
// corpus copy, or per-Add scratch retained in the wrapper — which is exactly
// what would make segments unsafe to enable on a store that already recovers
// near its memory cap. The lower bound on the plain figure keeps the ratio from
// passing vacuously when nothing was measured: the index keeps a copy of every
// vector.
func TestSegmentedColdStartEnvelope(t *testing.T) {
	const K = 1.25
	corpus := coldStartCorpus()

	plainAlloc, plainRetained := measureColdStart(t, newColdStartHNSW, corpus)
	segAlloc, segRetained := measureColdStart(t, newColdStartSegmented, corpus)
	t.Logf("plain: alloc=%d retained=%d; segmented(%d): alloc=%d retained=%d; ratios alloc=%.3f retained=%.3f",
		plainAlloc, plainRetained, coldStartSegments, segAlloc, segRetained,
		float64(segAlloc)/float64(plainAlloc), float64(segRetained)/float64(plainRetained))

	if minRetained := uint64(coldStartDocs * coldStartDim * 4); plainRetained < minRetained {
		t.Fatalf("plain retained heap %d < %d bytes of stored vectors: measurement is broken", plainRetained, minRetained)
	}
	if float64(segAlloc) > K*float64(plainAlloc) {
		t.Errorf("segmented cold start allocated %d bytes, plain %d: ratio %.3f exceeds K=%.2f",
			segAlloc, plainAlloc, float64(segAlloc)/float64(plainAlloc), K)
	}
	if float64(segRetained) > K*float64(plainRetained) {
		t.Errorf("segmented cold start retained %d bytes, plain %d: ratio %.3f exceeds K=%.2f",
			segRetained, plainRetained, float64(segRetained)/float64(plainRetained), K)
	}
}
