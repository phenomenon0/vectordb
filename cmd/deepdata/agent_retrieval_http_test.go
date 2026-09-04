package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// TestCanonicalHTTPAgentRetrievalSearchContract drives the new
// agent-oriented retrieval fields (score_floor, fallback, usage_boost)
// through the canonical V3 tenant HTTP surface end-to-end, and proves the
// zero-value identity: requests without the new fields behave exactly as
// before.
func TestCanonicalHTTPAgentRetrievalSearchContract(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)

	schema := vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{
			{Name: "primary", Type: vcollection.VectorTypeDense, Dim: 4, Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT}},
			{Name: "secondary", Type: vcollection.VectorTypeDense, Dim: 4, Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT}},
		},
	}
	createBody, err := json.Marshal(schema)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections", bytes.NewReader(createBody))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("create collection returned %d: %s", response.Code, response.Body.String())
	}

	// Primary distances to q=[1,0,0,0] are 1.0 (bad); secondary distances
	// are 0.2 / ~0.293 (good).
	docs := map[uint64]string{
		41: `{"vectors":{"primary":[0,0,1,0],"secondary":[0.8,0.6,0,0]}}`,
		42: `{"vectors":{"primary":[0,0,0,1],"secondary":[0.7,0.7,0,0]}}`,
	}
	for id, body := range docs {
		path := "/v3/tenants/acme/collections/docs/docs"
		if id == 41 {
			path += "/41"
		} else {
			path += "/42"
		}
		request = httptest.NewRequest(http.MethodPut, path, bytes.NewReader([]byte(body)))
		request.Header.Set("Content-Type", "application/json")
		response = httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("upsert doc %d returned %d: %s", id, response.Code, response.Body.String())
		}
	}

	search := func(t *testing.T, payload map[string]interface{}, wantStatus int) tenantSearchJSONResponse {
		t.Helper()
		body, err := json.Marshal(payload)
		if err != nil {
			t.Fatal(err)
		}
		request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/docs/search", bytes.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		response = httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != wantStatus {
			t.Fatalf("search returned %d, want %d: %s", response.Code, wantStatus, response.Body.String())
		}
		var out tenantSearchJSONResponse
		if err := json.NewDecoder(response.Body).Decode(&out); err != nil {
			t.Fatalf("decode search response: %v body=%s", err, response.Body.String())
		}
		return out
	}

	both := map[string]interface{}{
		"primary":   []float32{1, 0, 0, 0},
		"secondary": []float32{1, 0, 0, 0},
	}

	// Zero-value identity: no new fields -> classic behavior, no weak
	// match, no fallback marker, raw best score surfaced.
	base := search(t, map[string]interface{}{"queries": map[string]interface{}{"primary": []float32{1, 0, 0, 0}}, "top_k": 5}, http.StatusOK)
	if base.WeakMatch || base.FellBackTo != "" {
		t.Fatalf("baseline must be zero-value-identical, got weak=%v fell_back_to=%q", base.WeakMatch, base.FellBackTo)
	}
	if math.Abs(float64(base.BestScore)-1.0) > 1e-6 {
		t.Fatalf("baseline BestScore = %v, want 1.0 (primary distance)", base.BestScore)
	}

	// Fallback ladder fires: primary best 1.0 > threshold 0.5 -> secondary.
	fb := search(t, map[string]interface{}{
		"queries":  both,
		"top_k":    5,
		"fallback": map[string]interface{}{"primary": "primary", "secondary": "secondary", "threshold": 0.5},
	}, http.StatusOK)
	if fb.FellBackTo != "secondary" {
		t.Fatalf("FellBackTo = %q, want secondary", fb.FellBackTo)
	}
	if len(fb.Documents) != 2 || fb.Documents[0].ID != 41 {
		t.Fatalf("fallback docs = %v, want doc 41 first", fb.Documents)
	}
	if math.Abs(float64(fb.BestScore)-0.2) > 1e-3 {
		t.Fatalf("fallback BestScore = %v, want ~0.2", fb.BestScore)
	}
	if fb.WeakMatch {
		t.Fatal("fallback returned real results, WeakMatch must be false")
	}

	// Floor applies to the fallback answer and weak match fires when
	// nothing survives the floor.
	weak := search(t, map[string]interface{}{
		"queries":     both,
		"top_k":       5,
		"score_floor": 0.1,
		"fallback":    map[string]interface{}{"primary": "primary", "secondary": "secondary", "threshold": 0.5},
	}, http.StatusOK)
	if weak.FellBackTo != "secondary" || len(weak.Documents) != 0 || !weak.WeakMatch {
		t.Fatalf("floor+fallback = fell=%q docs=%d weak=%v", weak.FellBackTo, len(weak.Documents), weak.WeakMatch)
	}

	// Loose threshold: primary answers, no fallback marker.
	primary := search(t, map[string]interface{}{
		"queries":  both,
		"top_k":    5,
		"fallback": map[string]interface{}{"primary": "primary", "secondary": "secondary", "threshold": 1.5},
	}, http.StatusOK)
	if primary.FellBackTo != "" || len(primary.Documents) != 2 {
		t.Fatalf("primary answer = fell=%q docs=%d, want no fallback with 2 docs", primary.FellBackTo, len(primary.Documents))
	}
	if math.Abs(float64(primary.BestScore)-1.0) > 1e-6 {
		t.Fatalf("primary BestScore = %v, want 1.0", primary.BestScore)
	}

	// Usage boost is accepted and reported scores stay raw.
	boosted := search(t, map[string]interface{}{
		"queries":     map[string]interface{}{"primary": []float32{1, 0, 0, 0}},
		"top_k":       5,
		"usage_boost": 0.9,
	}, http.StatusOK)
	if boosted.WeakMatch || len(boosted.Documents) != 2 {
		t.Fatalf("usage_boost response = docs=%d weak=%v, want 2 docs no weak", len(boosted.Documents), boosted.WeakMatch)
	}

	// Contract violations are client errors (400), not server faults.
	for name, payload := range map[string]map[string]interface{}{
		"usage_boost out of range": {
			"queries": map[string]interface{}{"primary": []float32{1, 0, 0, 0}}, "top_k": 1, "usage_boost": 1.0,
		},
		"negative score_floor": {
			"queries": map[string]interface{}{"primary": []float32{1, 0, 0, 0}}, "top_k": 1, "score_floor": -0.5,
		},
		"fallback and hybrid mutually exclusive": {
			"queries": both, "top_k": 1,
			"fallback":      map[string]interface{}{"primary": "primary", "secondary": "secondary"},
			"hybrid_params": map[string]interface{}{"strategy": "rrf"},
		},
		"fallback same field": {
			"queries": both, "top_k": 1,
			"fallback": map[string]interface{}{"primary": "primary", "secondary": "primary"},
		},
		"fallback unknown field": {
			"queries": both, "top_k": 1,
			"fallback": map[string]interface{}{"primary": "primary", "secondary": "missing"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			body, err := json.Marshal(payload)
			if err != nil {
				t.Fatal(err)
			}
			request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/docs/search", bytes.NewReader(body))
			request.Header.Set("Content-Type", "application/json")
			response = httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("returned %d, want 400: %s", response.Code, response.Body.String())
			}
		})
	}

	// Unknown fields remain rejected by DisallowUnknownFields.
	unknown := `{"queries":{"primary":[1,0,0,0]},"top_k":1,"score_flor":0.5}`
	request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/docs/search", strings.NewReader(unknown))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("unknown field returned %d, want 400: %s", response.Code, response.Body.String())
	}
}
