package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

type collectionSearchResponse struct {
	Status             string                 `json:"status"`
	Documents          []vcollection.Document `json:"documents"`
	Scores             []float32              `json:"scores"`
	CandidatesExamined int                    `json:"candidates_examined"`
}

func setupCollectionHTTPServerForSearch(t *testing.T) *CollectionHTTPServer {
	t.Helper()

	server := NewCollectionHTTPServer(t.TempDir())
	schema := vcollection.CollectionSchema{
		Name: "test",
		Fields: []vcollection.VectorField{
			{
				Name:  "embedding",
				Type:  vcollection.VectorTypeDense,
				Dim:   4,
				Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
			},
		},
	}

	ctx := context.Background()
	if _, err := server.manager.CreateCollection(ctx, schema); err != nil {
		t.Fatalf("create collection failed: %v", err)
	}

	docs := []vcollection.Document{
		{Vectors: map[string]interface{}{"embedding": []float32{1.0, 0.0, 0.0, 0.0}}},
		{Vectors: map[string]interface{}{"embedding": []float32{0.95, 0.05, 0.0, 0.0}}},
		{Vectors: map[string]interface{}{"embedding": []float32{0.90, 0.10, 0.0, 0.0}}},
		{Vectors: map[string]interface{}{"embedding": []float32{0.85, 0.15, 0.0, 0.0}}},
	}
	for i := range docs {
		if err := server.manager.AddDocument(ctx, "test", &docs[i]); err != nil {
			t.Fatalf("add document failed: %v", err)
		}
	}

	return server
}

func runCollectionSearch(t *testing.T, server *CollectionHTTPServer, payload map[string]interface{}, wantStatus int) (collectionSearchResponse, string) {
	t.Helper()

	var out collectionSearchResponse

	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal request failed: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/search", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.handleSearch(w, req)

	if w.Code != wantStatus {
		t.Fatalf("expected status %d, got %d: %s", wantStatus, w.Code, w.Body.String())
	}

	rawBody := w.Body.String()
	if wantStatus == http.StatusOK {
		if err := json.NewDecoder(strings.NewReader(rawBody)).Decode(&out); err != nil {
			t.Fatalf("decode response failed: %v body=%s", err, rawBody)
		}
	}
	return out, rawBody
}

func TestCollectionHTTPHandleSearchOffsetPagination(t *testing.T) {
	server := setupCollectionHTTPServerForSearch(t)
	query := []float32{1.0, 0.0, 0.0, 0.0}

	full, _ := runCollectionSearch(t, server, map[string]interface{}{
		"collection": "test",
		"queries": map[string]interface{}{
			"embedding": query,
		},
		"top_k": 4,
	}, http.StatusOK)

	if len(full.Documents) < 3 {
		t.Fatalf("expected at least 3 documents in baseline response, got %d", len(full.Documents))
	}

	paged, _ := runCollectionSearch(t, server, map[string]interface{}{
		"collection": "test",
		"queries": map[string]interface{}{
			"embedding": query,
		},
		"top_k":  2,
		"offset": 1,
	}, http.StatusOK)

	if len(paged.Documents) != 2 {
		t.Fatalf("expected 2 documents for paged response, got %d", len(paged.Documents))
	}
	if len(paged.Scores) != 2 {
		t.Fatalf("expected 2 scores for paged response, got %d", len(paged.Scores))
	}

	if paged.Documents[0].ID != full.Documents[1].ID || paged.Documents[1].ID != full.Documents[2].ID {
		t.Fatalf(
			"offset pagination mismatch: got ids [%d %d], want [%d %d]",
			paged.Documents[0].ID, paged.Documents[1].ID,
			full.Documents[1].ID, full.Documents[2].ID,
		)
	}

	empty, _ := runCollectionSearch(t, server, map[string]interface{}{
		"collection": "test",
		"queries": map[string]interface{}{
			"embedding": query,
		},
		"top_k":  2,
		"offset": 100,
	}, http.StatusOK)

	if len(empty.Documents) != 0 {
		t.Fatalf("expected empty documents for large offset, got %d", len(empty.Documents))
	}
	if len(empty.Scores) != 0 {
		t.Fatalf("expected empty scores for large offset, got %d", len(empty.Scores))
	}
}

func TestCollectionHTTPHandleSearchRejectsNegativeOffset(t *testing.T) {
	server := setupCollectionHTTPServerForSearch(t)

	_, body := runCollectionSearch(t, server, map[string]interface{}{
		"collection": "test",
		"queries": map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
		"top_k":  2,
		"offset": -1,
	}, http.StatusBadRequest)

	if !strings.Contains(body, "offset must be >= 0") {
		t.Fatalf("expected negative offset validation error, got %q", body)
	}
}

func TestCollectionHTTPHandleSearchOmitsVectorsByDefault(t *testing.T) {
	server := setupCollectionHTTPServerForSearch(t)

	resp, _ := runCollectionSearch(t, server, map[string]interface{}{
		"collection": "test",
		"queries": map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
		"top_k": 1,
	}, http.StatusOK)

	if len(resp.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(resp.Documents))
	}
	if resp.Documents[0].Vectors != nil {
		t.Fatal("expected vectors to be omitted by default")
	}
}

func TestCollectionHTTPHandleSearchIncludesVectorsWhenRequested(t *testing.T) {
	server := setupCollectionHTTPServerForSearch(t)

	resp, _ := runCollectionSearch(t, server, map[string]interface{}{
		"collection": "test",
		"queries": map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
		"top_k":           1,
		"include_vectors": true,
	}, http.StatusOK)

	if len(resp.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(resp.Documents))
	}
	if len(resp.Documents[0].Vectors) == 0 {
		t.Fatal("expected vectors when include_vectors=true")
	}
}

func TestCollectionHTTPHandleSearchCanOmitVectors(t *testing.T) {
	server := setupCollectionHTTPServerForSearch(t)

	resp, _ := runCollectionSearch(t, server, map[string]interface{}{
		"collection": "test",
		"queries": map[string]interface{}{
			"embedding": []float32{1.0, 0.0, 0.0, 0.0},
		},
		"top_k":           1,
		"include_vectors": false,
	}, http.StatusOK)

	if len(resp.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(resp.Documents))
	}
	if resp.Documents[0].Vectors != nil {
		t.Fatal("expected vectors to be omitted when include_vectors=false")
	}
}

func TestCollectionHTTPHandleGetDocuments(t *testing.T) {
	server := setupCollectionHTTPServerForSearch(t)

	body, err := json.Marshal(map[string]interface{}{
		"ids": []uint64{2, 4, 999},
	})
	if err != nil {
		t.Fatalf("marshal request failed: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v2/collections/test/get", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.handleCollectionOps(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		Status    string                 `json:"status"`
		Documents []vcollection.Document `json:"documents"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if resp.Status != "success" {
		t.Fatalf("unexpected status payload: %+v", resp)
	}
	if len(resp.Documents) != 2 {
		t.Fatalf("expected 2 found documents, got %d", len(resp.Documents))
	}
	if resp.Documents[0].ID != 2 || resp.Documents[1].ID != 4 {
		t.Fatalf("unexpected document ids: [%d %d]", resp.Documents[0].ID, resp.Documents[1].ID)
	}
}
