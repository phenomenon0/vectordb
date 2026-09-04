package index

// Tests for SegmentedIndex: routing determinism, search-merge correctness
// against a brute-force oracle, export byte-equality with a single plain
// HNSWIndex, batch atomicity, filtered-search escalation, and result-order
// determinism. All randomness comes from a fixed-seed LCG so failures are
// reproducible; no sleeps, no global rand state.

import (
	"bytes"
	"context"
	"errors"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/filter"
)

const segTestDim = 8

// segLCG is a deterministic 64-bit linear congruential generator producing
// floats in [0,1). Fixed seed => reproducible corpora without global state.
type segLCG struct{ s uint64 }

func newSegLCG(seed uint64) *segLCG { return &segLCG{s: seed} }

func (l *segLCG) f32() float32 {
	l.s = l.s*6364136223846793005 + 1442695040888963407
	return float32(l.s>>40) / float32(1<<24)
}

func (l *segLCG) vec(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = l.f32()
	}
	return v
}

func segCorpus(n int) map[uint64][]float32 {
	lcg := newSegLCG(42)
	docs := make(map[uint64][]float32, n)
	for id := 0; id < n; id++ {
		docs[uint64(id)] = lcg.vec(segTestDim)
	}
	return docs
}

// segUnitNorm mirrors HNSW pre-normalization (default true): stored vectors
// are L2-normalized and cosine distance becomes 1 - dot(a,b).
func segUnitNorm(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func newSegFactory(t *testing.T) func() (Index, error) {
	t.Helper()
	return func() (Index, error) { return NewHNSWIndex(segTestDim, nil) }
}

// TestSegmentedRoutingAndStats verifies id%N routing determinism: counts sum
// across segments, a delete lands on exactly the owning segment even while the
// other segments hold its nearest neighbors, and Delete's not-found contract
// propagates verbatim.
func TestSegmentedRoutingAndStats(t *testing.T) {
	ctx := context.Background()
	idx, err := NewSegmentedIndex(3, newSegFactory(t))
	if err != nil {
		t.Fatalf("NewSegmentedIndex: %v", err)
	}

	lcg := newSegLCG(7)
	for id := uint64(0); id < 9; id++ {
		if err := idx.Add(ctx, id, lcg.vec(segTestDim)); err != nil {
			t.Fatalf("Add(%d): %v", id, err)
		}
	}

	stats := idx.Stats()
	if stats.Count != 9 || stats.Active != 9 || stats.Deleted != 0 {
		t.Errorf("aggregated stats = %+v, want Count=9 Active=9 Deleted=0", stats)
	}
	if stats.Name != "SegmentedHNSW" {
		t.Errorf("Name/Stats name = %q, want %q", stats.Name, "SegmentedHNSW")
	}
	for i, seg := range idx.segments {
		if got := seg.Stats().Count; got != 3 {
			t.Errorf("segment %d count = %d, want 3 (ids %d..%d route here)", i, got, i, i+3)
		}
	}

	// Replay the same LCG stream: ID 4's vector is the 5th draw, used as the
	// query after deletion.
	lcgForQ := newSegLCG(7)
	q := lcgForQ.vec(segTestDim)
	for id := 1; id <= 4; id++ {
		q = lcgForQ.vec(segTestDim)
	}

	if err := idx.Delete(ctx, 4); err != nil {
		t.Fatalf("Delete(4): %v", err)
	}
	if got := idx.segments[4%3].Stats().Deleted; got != 1 {
		t.Errorf("owning segment (id 4 -> segment 1) Deleted = %d, want 1", got)
	}
	for i, seg := range idx.segments {
		if i != 1 && seg.Stats().Deleted != 0 {
			t.Errorf("non-owning segment %d saw a tombstone", i)
		}
	}

	res, err := idx.Search(ctx, q, 3, DefaultSearchParams{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range res {
		if r.ID == 4 {
			t.Errorf("deleted ID 4 returned by Search: %+v", res)
		}
	}

	err = idx.Delete(ctx, 999)
	if err == nil || !strings.Contains(err.Error(), "999") {
		t.Errorf("Delete(missing 999) = %v, want segment not-found error naming the ID", err)
	}
}

// TestSegmentedSearchMatchesBruteForceOracle computes the true Euclidean-free
// top-k (cosine on unit vectors, matching HNSW prenormalization) and requires
// SegmentedIndex(N=4) at high ef_search to return the identical ID set: on 400
// docs with ef >= corpus size per segment, HNSW is effectively exact, so any
// merge/routing bug changes the set.
func TestSegmentedSearchMatchesBruteForceOracle(t *testing.T) {
	ctx := context.Background()
	idx, err := NewSegmentedIndex(4, newSegFactory(t))
	if err != nil {
		t.Fatalf("NewSegmentedIndex: %v", err)
	}

	const nDocs, k = 400, 10
	docs := segCorpus(nDocs)
	if err := idx.BatchAdd(ctx, docs); err != nil {
		t.Fatalf("BatchAdd: %v", err)
	}

	query := newSegLCG(nDocs + 1).vec(segTestDim)

	type scored struct {
		id   uint64
		dist float32
	}
	nq := segUnitNorm(query)
	scores := make([]scored, 0, nDocs)
	for id, v := range docs {
		nv := segUnitNorm(v)
		var dot float32
		for i := range nv {
			dot += nv[i] * nq[i]
		}
		scores = append(scores, scored{id, 1 - dot})
	}
	sort.Slice(scores, func(a, b int) bool {
		if scores[a].dist != scores[b].dist {
			return scores[a].dist < scores[b].dist
		}
		return scores[a].id < scores[b].id
	})
	wantIDs := make(map[uint64]bool, k)
	for _, s := range scores[:k] {
		wantIDs[s.id] = true
	}

	res, err := idx.Search(ctx, query, k, HNSWSearchParams{EfSearch: 400})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res) != k {
		t.Fatalf("len(results) = %d, want %d", len(res), k)
	}
	gotIDs := make(map[uint64]bool, k)
	prevDist := float32(-1)
	for _, r := range res {
		gotIDs[r.ID] = true
		if r.Distance < prevDist {
			t.Errorf("results not sorted ascending by distance: %f after %f", r.Distance, prevDist)
		}
		prevDist = r.Distance
	}
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Errorf("top-%d ID set mismatch:\n got %v\nwant %v", k, gotIDs, wantIDs)
	}
}

// TestSegmentedExportByteIdenticalAndRoundTrip requires the merged segmented
// export to be byte-identical to one plain HNSWIndex export of the same docs,
// and requires Import into a fresh SegmentedIndex to restore searchable state.
func TestSegmentedExportByteIdenticalAndRoundTrip(t *testing.T) {
	ctx := context.Background()
	docs := segCorpus(200)

	plain, err := NewHNSWIndex(segTestDim, nil)
	if err != nil {
		t.Fatalf("NewHNSWIndex: %v", err)
	}
	segged, err := NewSegmentedIndex(4, newSegFactory(t))
	if err != nil {
		t.Fatalf("NewSegmentedIndex: %v", err)
	}
	if err := plain.(*HNSWIndex).BatchAdd(ctx, docs); err != nil {
		t.Fatalf("plain BatchAdd: %v", err)
	}
	if err := segged.BatchAdd(ctx, docs); err != nil {
		t.Fatalf("segmented BatchAdd: %v", err)
	}

	a, err := plain.Export()
	if err != nil {
		t.Fatalf("plain Export: %v", err)
	}
	b, err := segged.Export()
	if err != nil {
		t.Fatalf("segmented Export: %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("exports differ: plain=%d bytes segmented=%d bytes\nplain:     %s\nsegmented: %s",
			len(a), len(b), truncate(a), truncate(b))
	}

	// Determinism of the wrapper's own output.
	b2, err := segged.Export()
	if err != nil {
		t.Fatalf("second Export: %v", err)
	}
	if !bytes.Equal(b, b2) {
		t.Error("repeated segmented exports differ")
	}

	// Round-trip: import into fresh segments, verify count + exact self-match.
	fresh, err := NewSegmentedIndex(4, newSegFactory(t))
	if err != nil {
		t.Fatalf("fresh NewSegmentedIndex: %v", err)
	}
	if err := fresh.Import(b); err != nil {
		t.Fatalf("Import: %v", err)
	}
	if got := fresh.Stats().Active; got != len(docs) {
		t.Errorf("post-import Active = %d, want %d", got, len(docs))
	}
	res, err := fresh.Search(ctx, docs[42], 1, HNSWSearchParams{EfSearch: 200})
	if err != nil || len(res) != 1 {
		t.Fatalf("post-import search: res=%v err=%v", res, err)
	}
	if res[0].ID != 42 {
		t.Errorf("self-match top-1 = %d, want 42 (distance %f)", res[0].ID, res[0].Distance)
	}
}

// TestSegmentedBatchAtomicity forces a mid-batch failure via a duplicate live
// ID in one segment's subset (rejected before mutation inside that segment,
// hnsw.go:313-340) while sibling segments complete, then verifies all-or-
// nothing behavior: completed segments' inserts are rolled back to tombstones,
// Active is unchanged, rolled-back IDs are re-addable (resurrection path),
// and the failed segment's original state is untouched.
func TestSegmentedBatchAtomicity(t *testing.T) {
	ctx := context.Background()
	idx, err := NewSegmentedIndex(4, newSegFactory(t))
	if err != nil {
		t.Fatalf("NewSegmentedIndex: %v", err)
	}

	lcg := newSegLCG(11)
	if err := idx.Add(ctx, 7, lcg.vec(segTestDim)); err != nil { // owner: segment 3
		t.Fatalf("seed Add(7): %v", err)
	}

	// ID 7 duplicates the live vector (fails segment 3 pre-mutation);
	// 12/13/14 land on healthy segments 0/1/2 and complete.
	batch := map[uint64][]float32{
		7:  newSegLCG(99).vec(segTestDim),
		12: lcg.vec(segTestDim),
		13: lcg.vec(segTestDim),
		14: lcg.vec(segTestDim),
	}
	if err := idx.BatchAdd(ctx, batch); err == nil {
		t.Fatal("BatchAdd with duplicate live ID: expected error, got nil")
	}

	stats := idx.Stats()
	if stats.Active != 1 || stats.Count != 4 || stats.Deleted != 3 {
		t.Errorf("stats after failed batch = %+v, want Active=1 Count=4 Deleted=3 (rollback tombstones)", stats)
	}
	if err := idx.Add(ctx, 12, batch[12]); err != nil {
		t.Errorf("re-add rolled-back ID 12: %v (rollback should have tombstoned it, enabling resurrection)", err)
	}
	err = idx.Add(ctx, 7, batch[7])
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Errorf("re-add untouched live ID 7 = %v, want 'already exists'", err)
	}
}

// TestSegmentedFilteredEscalation spreads ~10 matches of a selective filter
// across 4 segments (~2-3 each). The first fetch round cannot reach k=10, some
// segments saturate their limit, and escalation must recover every match.
func TestSegmentedFilteredEscalation(t *testing.T) {
	ctx := context.Background()
	idx, err := NewSegmentedIndex(4, newSegFactory(t))
	if err != nil {
		t.Fatalf("NewSegmentedIndex: %v", err)
	}

	lcg := newSegLCG(13)
	want := make(map[uint64]bool)
	for id := uint64(0); id < 200; id++ {
		vec := lcg.vec(segTestDim)
		if err := idx.Add(ctx, id, vec); err != nil {
			t.Fatalf("Add(%d): %v", id, err)
		}
		tier := int(id % 20)
		meta := map[string]interface{}{"tier": tier}
		if err := idx.SetMetadata(id, meta); err != nil {
			t.Fatalf("SetMetadata(%d): %v", id, err)
		}
		if tier == 13 {
			want[id] = true // 10 matches total, 2-3 per segment
		}
	}

	f := &filter.ComparisonFilter{Field: "tier", Operator: filter.OpEqual, Value: 13}
	res, err := idx.Search(ctx, lcg.vec(segTestDim), 10, HNSWSearchParams{EfSearch: 200, Filter: f})
	if err != nil {
		t.Fatalf("filtered Search: %v", err)
	}
	if len(res) != len(want) {
		t.Errorf("escalated filtered search returned %d hits, want %d", len(res), len(want))
	}
	for _, r := range res {
		if !want[r.ID] {
			t.Errorf("non-matching ID %d in results", r.ID)
		}
	}
}

// TestSegmentedSearchDeterminism runs the same query twice (unfiltered and
// filtered) and deep-compares results including order — the wrapper's total
// (distance, docID) comparator must make repeated calls identical.
func TestSegmentedSearchDeterminism(t *testing.T) {
	ctx := context.Background()
	idx, err := NewSegmentedIndex(4, newSegFactory(t))
	if err != nil {
		t.Fatalf("NewSegmentedIndex: %v", err)
	}

	lcg := newSegLCG(21)
	for id := uint64(0); id < 80; id++ {
		if err := idx.Add(ctx, id, lcg.vec(segTestDim)); err != nil {
			t.Fatalf("Add(%d): %v", id, err)
		}
		if err := idx.SetMetadata(id, map[string]interface{}{"even": id%2 == 0}); err != nil {
			t.Fatalf("SetMetadata(%d): %v", id, err)
		}
	}
	q := lcg.vec(segTestDim)

	first, err := idx.Search(ctx, q, 10, HNSWSearchParams{EfSearch: 100})
	if err != nil {
		t.Fatalf("first Search: %v", err)
	}
	second, err := idx.Search(ctx, q, 10, HNSWSearchParams{EfSearch: 100})
	if err != nil {
		t.Fatalf("second Search: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Errorf("unfiltered repeat differs:\n%+v\n%+v", first, second)
	}

	even := &filter.ComparisonFilter{Field: "even", Operator: filter.OpEqual, Value: true}
	ffirst, err := idx.Search(ctx, q, 10, HNSWSearchParams{EfSearch: 100, Filter: even})
	if err != nil {
		t.Fatalf("first filtered Search: %v", err)
	}
	fsecond, err := idx.Search(ctx, q, 10, HNSWSearchParams{EfSearch: 100, Filter: even})
	if err != nil {
		t.Fatalf("second filtered Search: %v", err)
	}
	if !reflect.DeepEqual(ffirst, fsecond) {
		t.Errorf("filtered repeat differs:\n%+v\n%+v", ffirst, fsecond)
	}
	for _, r := range ffirst {
		if r.ID%2 != 0 {
			t.Errorf("filter leaked odd ID %d", r.ID)
		}
	}
}

// TestSegmentsForNewCollection pins the host-independent durable default.
func TestSegmentsForNewCollection(t *testing.T) {
	got := SegmentsForNewCollection()
	if got != 1 {
		t.Errorf("SegmentsForNewCollection() = %d, want 1", got)
	}
}

// TestSegmentedConstructorErrors covers constructor validation.
func TestSegmentedConstructorErrors(t *testing.T) {
	if _, err := NewSegmentedIndex(0, newSegFactory(t)); err == nil {
		t.Error("NewSegmentedIndex(0): expected error")
	}
	if _, err := NewSegmentedIndex(2, nil); err == nil {
		t.Error("NewSegmentedIndex(2, nil factory): expected error")
	}
	_, err := NewSegmentedIndex(2, func() (Index, error) { return nil, errors.New("boom") })
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("failing factory: err = %v, want wrapped factory error", err)
	}
	_, err = NewSegmentedIndex(2, func() (Index, error) { return nil, nil })
	if err == nil || !strings.Contains(err.Error(), "nil index") {
		t.Errorf("nil-returning factory: err = %v, want a nil-index error", err)
	}
}

func truncate(b []byte) string {
	if len(b) > 300 {
		return string(b[:300]) + "..."
	}
	return string(b)
}
