package collection

import (
	"context"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func medianDuration(ds []time.Duration) time.Duration {
	sorted := append([]time.Duration(nil), ds...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[len(sorted)/2]
}

// TestSearchProceedsDuringBatchAdd is the collection-level instrument for the
// two-lock fix (see post-rc-cleanup phase 3): Collection.BatchAdd used to
// hold c.mu.Lock() for the whole batch (prepare, ID reservation, AND every
// field's index insert), so a concurrent Search — which takes c.mu.RLock()
// for its whole body — blocked until the entire batch finished. After the
// fix, BatchAdd only holds c.mu for the short prepare+reserve phase; the
// index inserts run without c.mu, so Search can interleave (and the index
// layer's own writeMu/RLock split, from the prior commit, keeps the graph
// safe to search while it is being built).
func TestSearchProceedsDuringBatchAdd(t *testing.T) {
	const dim = 64
	schema := CollectionSchema{
		Name: "concurrency_test",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  dim,
				Index: IndexConfig{
					Type: IndexTypeHNSW,
					Params: map[string]interface{}{
						"m":               16,
						"ef_construction": 200,
					},
				},
			},
		},
	}
	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("failed to create collection: %v", err)
	}
	ctx := context.Background()

	rng := rand.New(rand.NewSource(42))
	makeDocs := func(n int) []Document {
		docs := make([]Document, n)
		for i := range docs {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rng.Float32()
			}
			docs[i] = Document{Vectors: map[string]Vector{"embedding": {Dense: vec}}}
		}
		return docs
	}

	// Warm the collection so both the idle baseline and the concurrent batch
	// search a real graph, not an empty one.
	if err := coll.BatchAdd(ctx, makeDocs(2000)); err != nil {
		t.Fatalf("warm batch add failed: %v", err)
	}

	query := make([]float32, dim)
	for j := range query {
		query[j] = rng.Float32()
	}
	req := SearchRequest{
		CollectionName: "concurrency_test",
		Queries:        map[string]interface{}{"embedding": query},
		TopK:           10,
	}

	doSearch := func() time.Duration {
		start := time.Now()
		if _, err := coll.Search(ctx, req); err != nil {
			t.Fatalf("search failed: %v", err)
		}
		return time.Since(start)
	}

	// Idle baseline: median of 50 sequential searches with no writer active.
	idleLatencies := make([]time.Duration, 50)
	for i := range idleLatencies {
		idleLatencies[i] = doSearch()
	}
	idleMedian := medianDuration(idleLatencies)

	// Generate the concurrent batch's docs BEFORE starting the timing window:
	// doing this inside the goroutine would let random-vector generation (not
	// the lock) be the delay that lets early searches race ahead of
	// BatchAdd's c.mu.Lock() and get miscounted as "in-flight".
	bigBatch := makeDocs(20000)

	// Fire a large batch add concurrently and keep searching until it's done.
	//
	// Hold c.mu.RLock() ourselves until the writer goroutine has issued its
	// BatchAdd call: sync.RWMutex is writer-fair (a pending Lock() blocks
	// later RLock() callers behind it), so once we RUnlock, the writer's
	// already-pending Lock() is guaranteed to win the race against this
	// goroutine's very next Search() call. Without this, goroutine-launch
	// scheduling jitter (and, pre-fix, doc-generation time) is itself a race
	// window that can let a handful of searches slip in "before" the batch's
	// lock is actually held, corrupting the in-flight count.
	coll.mu.RLock()
	var done atomic.Bool
	var batchEndNanos atomic.Int64
	var wg sync.WaitGroup
	wg.Add(1)
	started := make(chan struct{})
	go func() {
		defer wg.Done()
		close(started)
		if err := coll.BatchAdd(ctx, bigBatch); err != nil {
			t.Errorf("concurrent batch add failed: %v", err)
		}
		batchEndNanos.Store(time.Now().UnixNano())
		done.Store(true)
	}()
	<-started
	time.Sleep(2 * time.Millisecond) // let BatchAdd reach c.mu.Lock() and start blocking
	coll.mu.RUnlock()

	type sample struct {
		end     time.Time
		latency time.Duration
	}
	var samples []sample
	batchStart := time.Now()
	for !done.Load() {
		start := time.Now()
		if _, err := coll.Search(ctx, req); err != nil {
			t.Fatalf("search during batch failed: %v", err)
		}
		end := time.Now()
		samples = append(samples, sample{end: end, latency: end.Sub(start)})
	}
	wg.Wait()
	elapsed := time.Since(batchStart)

	if elapsed < 200*time.Millisecond {
		t.Fatalf("concurrent batch add finished in %v — too fast to observe in-flight searches; raise the batch size", elapsed)
	}

	batchEnd := time.Unix(0, batchEndNanos.Load())
	var duringLatencies []time.Duration
	for _, s := range samples {
		if s.end.Before(batchEnd) {
			duringLatencies = append(duringLatencies, s.latency)
		}
	}

	if len(duringLatencies) < 50 {
		t.Fatalf("in-flight searches=%d idle=%v — want >=50 searches to complete while the batch is in flight", len(duringLatencies), idleMedian)
	}
	duringMedian := medianDuration(duringLatencies)
	ratio := float64(duringMedian) / float64(idleMedian)
	t.Logf("in-flight searches=%d idle=%v during=%v ratio=%.1fx", len(duringLatencies), idleMedian, duringMedian, ratio)
	if ratio > 5.0 {
		t.Fatalf("search latency during batch add degraded too much: idle=%v during=%v ratio=%.1fx (want <=5x)", idleMedian, duringMedian, ratio)
	}
}
