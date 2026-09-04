package main

import (
	"math"
	"path/filepath"
	"strings"
	"testing"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// TestCanonicalGRPCAgentRetrievalMapping proves the v3 gRPC Search
// handler maps the agent-oriented retrieval fields (score_floor, fallback,
// usage_boost) into the engine request and surfaces best_score, weak_match
// and fell_back_to in the response.
func TestCanonicalGRPCAgentRetrievalMapping(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close durable store: %v", err)
		}
	})

	server := &CollectionGRPCServer{tenants: store.Tenants(), persistenceHealth: store.Err}
	ctx := canonicalGRPCAdminContext("acme")

	if _, err := server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId: "acme",
		Name:     "docs",
		Fields: []*deepdatav3.VectorFieldConfig{
			{Name: "primary", Type: int32(vcollection.VectorTypeDense), Dim: 4, IndexType: "flat"},
			{Name: "secondary", Type: int32(vcollection.VectorTypeDense), Dim: 4, IndexType: "flat"},
		},
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	insert := func(id uint64, primary, secondary []float32) {
		t.Helper()
		if _, err := server.Insert(ctx, &deepdatav3.InsertRequest{
			TenantId: "acme", Collection: "docs", Id: id,
			Vectors: map[string]*deepdatav3.VectorData{
				"primary":   denseProtoVector(primary...),
				"secondary": denseProtoVector(secondary...),
			},
		}); err != nil {
			t.Fatalf("insert doc %d: %v", id, err)
		}
	}
	insert(41, []float32{0, 0, 1, 0}, []float32{0.8, 0.6, 0, 0})
	insert(42, []float32{0, 0, 0, 1}, []float32{0.7, 0.7, 0, 0})

	both := map[string]*deepdatav3.VectorData{
		"primary":   denseProtoVector(1, 0, 0, 0),
		"secondary": denseProtoVector(1, 0, 0, 0),
	}
	search := func(req *deepdatav3.SearchRequest) (*deepdatav3.SearchResponse, error) {
		t.Helper()
		req.TenantId = "acme"
		req.Collection = "docs"
		return server.Search(ctx, req)
	}

	// Fallback fires: primary best distance 1.0 > threshold 0.5.
	fb, err := search(&deepdatav3.SearchRequest{
		Queries: both, TopK: 5,
		Fallback: &deepdatav3.FallbackParams{Primary: "primary", Secondary: "secondary", Threshold: 0.5},
	})
	if err != nil {
		t.Fatalf("fallback search: %v", err)
	}
	if fb.FellBackTo != "secondary" || len(fb.Results) != 2 || fb.Results[0].Id != 41 {
		t.Fatalf("fallback = fell=%q results=%d top=%d, want secondary / 2 / 41", fb.FellBackTo, len(fb.Results), fb.Results[0].Id)
	}
	if math.Abs(float64(fb.BestScore)-0.2) > 1e-3 || fb.WeakMatch {
		t.Fatalf("fallback best=%v weak=%v, want ~0.2 / false", fb.BestScore, fb.WeakMatch)
	}

	// Score floor applies to the fallback answer and weak match fires.
	weak, err := search(&deepdatav3.SearchRequest{
		Queries: both, TopK: 5, ScoreFloor: 0.1,
		Fallback: &deepdatav3.FallbackParams{Primary: "primary", Secondary: "secondary", Threshold: 0.5},
	})
	if err != nil {
		t.Fatalf("floor+fallback search: %v", err)
	}
	if weak.FellBackTo != "secondary" || len(weak.Results) != 0 || !weak.WeakMatch {
		t.Fatalf("floor+fallback = fell=%q results=%d weak=%v", weak.FellBackTo, len(weak.Results), weak.WeakMatch)
	}

	// Loose threshold: primary answers, no fallback marker.
	primary, err := search(&deepdatav3.SearchRequest{
		Queries: both, TopK: 5,
		Fallback: &deepdatav3.FallbackParams{Primary: "primary", Secondary: "secondary", Threshold: 1.5},
	})
	if err != nil {
		t.Fatalf("primary search: %v", err)
	}
	if primary.FellBackTo != "" || len(primary.Results) != 2 || math.Abs(float64(primary.BestScore)-1.0) > 1e-6 {
		t.Fatalf("primary = fell=%q results=%d best=%v", primary.FellBackTo, len(primary.Results), primary.BestScore)
	}

	// Usage boost is accepted; reported scores stay raw.
	boosted, err := search(&deepdatav3.SearchRequest{
		Queries: map[string]*deepdatav3.VectorData{"primary": denseProtoVector(1, 0, 0, 0)},
		TopK:    5, UsageBoost: 0.9,
	})
	if err != nil {
		t.Fatalf("usage_boost search: %v", err)
	}
	if len(boosted.Results) != 2 || boosted.WeakMatch {
		t.Fatalf("usage_boost = results=%d weak=%v, want 2 / false", len(boosted.Results), boosted.WeakMatch)
	}

	// Engine-level contract violations surface as InvalidArgument.
	for name, req := range map[string]*deepdatav3.SearchRequest{
		"usage_boost out of range": {
			Queries: map[string]*deepdatav3.VectorData{"primary": denseProtoVector(1, 0, 0, 0)},
			TopK:    1, UsageBoost: 1.0,
		},
		"negative score_floor": {
			Queries: both, TopK: 1, ScoreFloor: -0.5,
		},
		"fallback same field": {
			Queries: both, TopK: 1,
			Fallback: &deepdatav3.FallbackParams{Primary: "primary", Secondary: "primary"},
		},
		"fallback unknown field": {
			Queries: both, TopK: 1,
			Fallback: &deepdatav3.FallbackParams{Primary: "primary", Secondary: "missing"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := search(req)
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("err = %v, want InvalidArgument", err)
			}
		})
	}

	// Handler-level admission: fallback + hybrid is rejected before the
	// engine, and multiple fields still require hybrid_params or fallback.
	for name, req := range map[string]*deepdatav3.SearchRequest{
		"fallback and hybrid mutually exclusive": {
			Queries: both, TopK: 1,
			Fallback:     &deepdatav3.FallbackParams{Primary: "primary", Secondary: "secondary"},
			HybridParams: &deepdatav3.HybridSearchParams{Strategy: "rrf"},
		},
		"multiple fields without hybrid or fallback": {
			Queries: both, TopK: 1,
		},
		"fallback with one query field": {
			Queries:  map[string]*deepdatav3.VectorData{"primary": denseProtoVector(1, 0, 0, 0)},
			TopK:     1,
			Fallback: &deepdatav3.FallbackParams{Primary: "primary", Secondary: "secondary"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := search(req)
			if status.Code(err) != codes.InvalidArgument || !strings.Contains(err.Error(), "fallback") && !strings.Contains(err.Error(), "hybrid") {
				t.Fatalf("err = %v, want InvalidArgument mentioning fallback/hybrid", err)
			}
		})
	}
}
