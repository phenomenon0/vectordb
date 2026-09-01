package collection

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"
)

func newAgentRetrievalCollection(t *testing.T) *Collection {
	t.Helper()
	coll, err := NewCollection(CollectionSchema{
		Name: "docs",
		Fields: []VectorField{{
			Name: "dense", Type: VectorTypeDense, Dim: 4,
			Index: IndexConfig{Type: IndexTypeFLAT},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return coll
}

func addVec(t *testing.T, coll *Collection, id uint64, field string, vec []float32) {
	t.Helper()
	if err := coll.Add(context.Background(), &Document{
		ID:      id,
		Vectors: map[string]interface{}{field: vec},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestSearchScoreFloorAndWeakMatch(t *testing.T) {
	ctx := context.Background()
	coll := newAgentRetrievalCollection(t)
	// Dense scores are cosine distances: doc 1 is an exact match (0),
	// docs 2 and 3 are orthogonal (1).
	addVec(t, coll, 1, "dense", []float32{1, 0, 0, 0})
	addVec(t, coll, 2, "dense", []float32{0, 1, 0, 0})
	addVec(t, coll, 3, "dense", []float32{0, 0, 1, 0})

	// Baseline: no floor — nothing is weak, BestScore is the best raw
	// score (the minimum distance).
	base, err := coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
		TopK:           10,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(base.Documents) != 3 {
		t.Fatalf("baseline docs = %d, want 3", len(base.Documents))
	}
	if base.WeakMatch {
		t.Fatal("no floor set, WeakMatch must be false")
	}
	if base.BestScore != 0 {
		t.Fatalf("BestScore = %v, want 0 (the exact-match distance)", base.BestScore)
	}

	// Distance floor 0.5 keeps only the exact-match document.
	kept, err := coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
		TopK:           10,
		ScoreFloor:     0.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(kept.Documents) != 1 || kept.Documents[0].ID != 1 {
		t.Fatalf("floor kept %v, want exactly doc 1 (distance 0)", kept.Documents)
	}
	if kept.WeakMatch {
		t.Fatal("a document survived the floor, WeakMatch must be false")
	}
	if kept.BestScore != 0 {
		t.Fatalf("kept BestScore = %v, want 0", kept.BestScore)
	}

	// Floor tighter than every distance drops everything and signals weak
	// match: query at 45 degrees, closest distance ~0.293 > floor 0.2.
	weak, err := coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": []float32{0.5, 0.5, 0, 0}},
		TopK:           10,
		ScoreFloor:     0.2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(weak.Documents) != 0 || weak.BestScore != 0 {
		t.Fatalf("expected empty weak response, got %d docs best=%v", len(weak.Documents), weak.BestScore)
	}
	if !weak.WeakMatch {
		t.Fatal("everything beyond the floor, WeakMatch must be true")
	}

	// Invalid floors are rejected, never clamped.
	for _, bad := range []float64{-1, math.NaN()} {
		if _, err := coll.Search(ctx, SearchRequest{
			CollectionName: "docs",
			Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
			TopK:           1,
			ScoreFloor:     bad,
		}); err == nil {
			t.Fatalf("score_floor %v accepted, want error", bad)
		}
	}
}

func TestSearchFallbackLadder(t *testing.T) {
	ctx := context.Background()
	schema := CollectionSchema{
		Name: "docs",
		Fields: []VectorField{
			{Name: "primary", Type: VectorTypeDense, Dim: 4, Index: IndexConfig{Type: IndexTypeFLAT}},
			{Name: "secondary", Type: VectorTypeDense, Dim: 4, Index: IndexConfig{Type: IndexTypeFLAT}},
		},
	}
	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatal(err)
	}

	// Documents must carry both fields. Primary distances are 1.0 (bad),
	// secondary distances are 0.2 / 0.293 (good) against q.
	if err := coll.Add(ctx, &Document{ID: 1, Vectors: map[string]interface{}{
		"primary":   []float32{0, 0, 1, 0},
		"secondary": []float32{0.8, 0.6, 0, 0},
	}}); err != nil {
		t.Fatal(err)
	}
	if err := coll.Add(ctx, &Document{ID: 2, Vectors: map[string]interface{}{
		"primary":   []float32{0, 0, 0, 1},
		"secondary": []float32{0.7, 0.7, 0, 0},
	}}); err != nil {
		t.Fatal(err)
	}

	both := map[string]interface{}{
		"primary":   []float32{1, 0, 0, 0},
		"secondary": []float32{1, 0, 0, 0},
	}

	// Primary best distance 1.0 exceeds threshold 0.5 -> fall back.
	resp, err := coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        both,
		TopK:           5,
		Fallback:       &FallbackParams{Primary: "primary", Secondary: "secondary", Threshold: 0.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FellBackTo != "secondary" {
		t.Fatalf("FellBackTo = %q, want secondary", resp.FellBackTo)
	}
	if len(resp.Documents) != 2 || resp.Documents[0].ID != 1 {
		t.Fatalf("fallback response = %v, want doc 1 first (secondary distance 0.2)", resp.Documents)
	}
	if math.Abs(float64(resp.BestScore)-0.2) > 1e-3 {
		t.Fatalf("BestScore = %v, want ~0.2", resp.BestScore)
	}
	if resp.CandidatesExamined != 4 {
		t.Fatalf("CandidatesExamined = %d, want 2 (primary) + 2 (secondary)", resp.CandidatesExamined)
	}
	if resp.WeakMatch {
		t.Fatal("fallback returned real results, WeakMatch must be false")
	}

	// Primary best distance 1.0 within threshold 1.5 -> no fallback.
	resp, err = coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        both,
		TopK:           5,
		Fallback:       &FallbackParams{Primary: "primary", Secondary: "secondary", Threshold: 1.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FellBackTo != "" {
		t.Fatalf("primary met the threshold, FellBackTo = %q, want empty", resp.FellBackTo)
	}
	if len(resp.Documents) != 2 || resp.BestScore != 1 {
		t.Fatalf("primary response = %v best=%v, want 2 docs best distance 1.0", resp.Documents, resp.BestScore)
	}

	// The floor still applies to the fallback answer, and weak match can
	// fire on the secondary field: both secondary distances (~0.2, ~0.293)
	// exceed floor 0.1.
	resp, err = coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        both,
		TopK:           5,
		ScoreFloor:     0.1,
		Fallback:       &FallbackParams{Primary: "primary", Secondary: "secondary", Threshold: 0.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FellBackTo != "secondary" || len(resp.Documents) != 0 || !resp.WeakMatch {
		t.Fatalf("floor+fallback = fell=%q docs=%d weak=%v", resp.FellBackTo, len(resp.Documents), resp.WeakMatch)
	}

	// Contract violations.
	cases := []struct {
		name string
		req  SearchRequest
		want string
	}{
		{
			name: "mutually exclusive with hybrid",
			req: SearchRequest{
				CollectionName: "docs", Queries: both, TopK: 1,
				Fallback:     &FallbackParams{Primary: "primary", Secondary: "secondary"},
				HybridParams: &HybridSearchParams{Strategy: "rrf"},
			},
			want: "mutually exclusive",
		},
		{
			name: "same primary and secondary",
			req: SearchRequest{
				CollectionName: "docs", Queries: both, TopK: 1,
				Fallback: &FallbackParams{Primary: "primary", Secondary: "primary"},
			},
			want: "must differ",
		},
		{
			name: "field not in queries",
			req: SearchRequest{
				CollectionName: "docs",
				Queries:        both, TopK: 1,
				Fallback: &FallbackParams{Primary: "primary", Secondary: "missing"},
			},
			want: "not in queries",
		},
	}
	for _, tc := range cases {
		if _, err := coll.Search(ctx, tc.req); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want containing %q", tc.name, err, tc.want)
		}
	}
}

func TestSearchUsageBoostReranksCloseScores(t *testing.T) {
	ctx := context.Background()
	coll := newAgentRetrievalCollection(t)
	// Dense scores are distances: doc 1 is an exact match (0), doc 2
	// is 1 - cos(10deg) ~= 0.015 away from q.
	const theta = 10.0 * math.Pi / 180.0
	addVec(t, coll, 1, "dense", []float32{1, 0, 0, 0})
	addVec(t, coll, 2, "dense", []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0})

	q := func(boost float64) *SearchResponse {
		t.Helper()
		resp, err := coll.Search(ctx, SearchRequest{
			CollectionName: "docs",
			Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
			TopK:           2,
			UsageBoost:     boost,
		})
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	base := q(0)
	if base.Documents[0].ID != 1 {
		t.Fatal("baseline order: doc 1 (raw 1.0) must lead")
	}
	bScore := float64(base.Scores[1])

	// Repeated direct fetches build a usage signal for doc 2.
	for i := 0; i < 10; i++ {
		if _, ok := coll.GetDocument(2); !ok {
			t.Fatal("doc 2 vanished")
		}
	}

	// With a strong blend, doc 2's order value (quality ~0.985 x 1.9)
	// beats doc 1's (quality 1.0 x ~1.08), but reported scores stay raw.
	boosted := q(0.9)
	if boosted.Documents[0].ID != 2 {
		t.Fatalf("usage blend did not promote doc 2: top = %d", boosted.Documents[0].ID)
	}
	if got := float64(boosted.Scores[0]); math.Abs(got-bScore) > 1e-6 {
		t.Fatalf("top raw score = %v, want doc 2's raw %v (scores must stay raw)", got, bScore)
	}

	// A fresh collection with no usage never reorders.
	fresh := newAgentRetrievalCollection(t)
	addVec(t, fresh, 1, "dense", []float32{1, 0, 0, 0})
	addVec(t, fresh, 2, "dense", []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0})
	resp, err := fresh.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
		TopK:           2,
		UsageBoost:     0.9,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Documents[0].ID != 1 {
		t.Fatal("no usage recorded: order must be unchanged")
	}

	// The blend is a nudge, never a takeover: 1.0 is rejected.
	if _, err := coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
		TopK:           1,
		UsageBoost:     1.0,
	}); err == nil {
		t.Fatal("usage_boost 1.0 accepted, want error")
	}
	if _, err := coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
		TopK:           1,
		UsageBoost:     -0.1,
	}); err == nil {
		t.Fatal("negative usage_boost accepted, want error")
	}
}

func TestUsageTrackerDecay(t *testing.T) {
	var now time.Time
	tr := &UsageTracker{
		entries: make(map[uint64]*usageEntry),
		now:     func() time.Time { return now },
	}
	tr.Record(7)
	if got := tr.Score(7); got != 1 {
		t.Fatalf("fresh score = %v, want 1", got)
	}
	now = now.Add(time.Hour)
	if got := tr.Score(7); math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("score after one half-life = %v, want 0.5", got)
	}
	tr.Record(7) // second hit at t+1h
	now = now.Add(time.Hour)
	if got := tr.Score(7); math.Abs(got-1.0) > 1e-9 {
		t.Fatalf("score after second hit + one half-life = %v, want 1.0", got)
	}
	if got := tr.Score(8); got != 0 {
		t.Fatalf("unknown id score = %v, want 0", got)
	}

	// Entries that decay to noise are pruned. A single-hit entry crosses
	// the 1e-6 threshold once it is ~142 years stale (the max reachable
	// time.Duration). Use a fresh tracker: the two-hit entry above is
	// still above the threshold at every representable time.
	var now2 time.Time
	tr2 := &UsageTracker{
		entries: make(map[uint64]*usageEntry),
		now:     func() time.Time { return now2 },
	}
	tr2.Record(7)
	now2 = now2.Add(1 << 62)
	if removed := tr2.Prune(); removed != 1 {
		t.Fatalf("prune removed %d, want 1", removed)
	}
	if got := tr2.Len(); got != 0 {
		t.Fatalf("len after prune = %d, want 0", got)
	}
	if got := tr2.Score(7); got != 0 {
		t.Fatalf("pruned entry still scores %v, want 0", got)
	}
}

func TestSearchRecordsUsageForReturnedDocs(t *testing.T) {
	ctx := context.Background()
	coll := newAgentRetrievalCollection(t)
	addVec(t, coll, 1, "dense", []float32{1, 0, 0, 0})
	addVec(t, coll, 2, "dense", []float32{0, 1, 0, 0})

	if _, err := coll.Search(ctx, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": []float32{1, 0, 0, 0}},
		TopK:           2,
	}); err != nil {
		t.Fatal(err)
	}
	if got := coll.usage.Len(); got != 2 {
		t.Fatalf("usage entries = %d, want 2 (returned docs)", got)
	}
}

// TestSearchErrorsAreTypedSentinels pins the engine's error classification.
// The transports (cmd/deepdata) map these sentinels to HTTP statuses and gRPC
// codes through internal/apierror; a bare fmt.Errorf here reaches an agent
// as a 500 with no hint, which is what forced both transports to duplicate
// the engine's validation before CTL-01.
func TestSearchErrorsAreTypedSentinels(t *testing.T) {
	ctx := context.Background()
	tenants := NewTenantManager("")
	field := func(name string) VectorField {
		return VectorField{Name: name, Type: VectorTypeDense, Dim: 4, Index: IndexConfig{Type: IndexTypeFLAT}}
	}
	schema := CollectionSchema{Name: "docs", Fields: []VectorField{field("dense"), field("other")}}
	if _, err := tenants.CreateCollection(ctx, "acme", schema); err != nil {
		t.Fatal(err)
	}
	if _, err := tenants.CreateCollection(ctx, "acme", schema); !errors.Is(err, ErrCollectionExists) {
		t.Fatalf("duplicate create = %v, want ErrCollectionExists", err)
	}
	if _, err := tenants.GetCollection("acme", "missing"); !errors.Is(err, ErrCollectionNotFound) {
		t.Fatalf("missing collection = %v, want ErrCollectionNotFound", err)
	}
	if _, err := tenants.SearchCollection(ctx, "nobody", SearchRequest{CollectionName: "docs", TopK: 1}); !errors.Is(err, ErrCollectionNotFound) {
		t.Fatalf("unknown tenant = %v, want ErrCollectionNotFound", err)
	}

	q := []float32{1, 0, 0, 0}
	cases := []struct {
		name string
		req  SearchRequest
		want error
	}{
		{"no queries", SearchRequest{TopK: 1}, ErrInvalidArgument},
		{"top_k above the cap", SearchRequest{Queries: map[string]interface{}{"dense": q}, TopK: CanonicalMaxSearchTopK + 1}, ErrInvalidArgument},
		{"two fields without a fusion rule", SearchRequest{Queries: map[string]interface{}{"dense": q, "other": q}, TopK: 1}, ErrInvalidSearchArgument},
	}
	for _, tc := range cases {
		tc.req.CollectionName = "docs"
		_, err := tenants.SearchCollection(ctx, "acme", tc.req)
		if !errors.Is(err, tc.want) {
			t.Fatalf("%s: err = %v, want %v", tc.name, err, tc.want)
		}
	}

	// The message names the wire fields an agent can actually send, not the
	// Go identifiers of the request struct.
	_, err := tenants.SearchCollection(ctx, "acme", SearchRequest{
		CollectionName: "docs", Queries: map[string]interface{}{"dense": q, "other": q}, TopK: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "hybrid_params or fallback") {
		t.Fatalf("multi-field message = %v, want it to name hybrid_params or fallback", err)
	}
}
