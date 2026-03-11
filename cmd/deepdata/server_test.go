package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/phenomenon0/vectordb/client"
	"github.com/phenomenon0/vectordb/internal/testutil"
)

type testEmbedder struct {
	dim     int
	vectors map[string][]float32
}

func (e *testEmbedder) Embed(text string) ([]float32, error) {
	vec, ok := e.vectors[text]
	if !ok {
		return nil, fmt.Errorf("missing vector for %q", text)
	}
	out := make([]float32, len(vec))
	copy(out, vec)
	return out, nil
}

func (e *testEmbedder) Dim() int {
	return e.dim
}

type identityReranker struct{}

func (identityReranker) Rerank(_ string, docs []string, topK int) ([]string, []float32, string, error) {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	out := append([]string(nil), docs[:topK]...)
	scores := make([]float32, len(out))
	return out, scores, "identity", nil
}

type reverseReranker struct{}

func (reverseReranker) Rerank(_ string, docs []string, topK int) ([]string, []float32, string, error) {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	out := make([]string, 0, topK)
	scores := make([]float32, 0, topK)
	for i := len(docs) - 1; i >= 0 && len(out) < topK; i-- {
		out = append(out, docs[i])
		scores = append(scores, float32(len(out)))
	}
	return out, scores, "reverse", nil
}

func TestHTTPHandlersInsertQueryDelete(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "index.gob")

	store := NewVectorStore(10, 3)
	embedder := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: embedder}

	handler, _ := newHTTPHandler(store, embedder, reranker, indexPath)
	srv := testutil.NewLoopbackServer(t, handler)

	cli := client.New(srv.URL)

	// insert
	if _, err := cli.Insert(context.Background(), client.InsertRequest{Doc: "hello world", Meta: map[string]string{"tag": "a", "score": "0.8", "ts": "2024-01-01T00:00:00Z"}}); err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	// query first page
	min := 0.5
	qr, err := cli.Query(context.Background(), client.QueryRequest{
		Query:       "hello",
		TopK:        2,
		PageSize:    1,
		IncludeMeta: true,
		ScoreMode:   "hybrid",
		MetaRanges: []client.RangeFilter{
			{Key: "score", Min: &min},
			{Key: "ts", TimeMin: "2023-12-31T00:00:00Z", TimeMax: "2024-12-31T00:00:00Z"},
		},
	})
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}
	if len(qr.IDs) != 1 || len(qr.Docs) != 1 {
		t.Fatalf("unexpected query response: %+v", qr)
	}
	if len(qr.Scores) != 1 {
		t.Fatalf("expected score in response")
	}
	if qr.Next != "" {
		t.Fatalf("expected no next page token for single doc")
	}

	// delete
	if _, err := cli.Delete(context.Background(), client.DeleteRequest{ID: qr.IDs[0]}); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
}

func TestHTTPQueryPaginationUsesPageToken(t *testing.T) {
	emb := &testEmbedder{
		dim: 2,
		vectors: map[string][]float32{
			"apple alpha": {1, 0},
			"apple beta":  {1, 0},
			"apple gamma": {1, 0},
			"apple":       {1, 0},
		},
	}
	store := NewVectorStore(10, emb.Dim())
	handler, _ := newHTTPHandler(store, emb, identityReranker{}, "")

	docs := []struct {
		id   string
		doc  string
		meta map[string]string
	}{
		{id: "doc-1", doc: "apple alpha", meta: map[string]string{"rank": "1"}},
		{id: "doc-2", doc: "apple beta", meta: map[string]string{"rank": "2"}},
		{id: "doc-3", doc: "apple gamma", meta: map[string]string{"rank": "3"}},
	}
	for _, doc := range docs {
		vec, err := emb.Embed(doc.doc)
		if err != nil {
			t.Fatalf("embed failed: %v", err)
		}
		if _, err := store.Add(vec, doc.doc, doc.id, doc.meta, "default", ""); err != nil {
			t.Fatalf("add failed: %v", err)
		}
	}

	type queryResponse struct {
		IDs  []string            `json:"ids"`
		Docs []string            `json:"docs"`
		Meta []map[string]string `json:"meta"`
		Next string              `json:"next"`
	}

	query := func(pageToken string) queryResponse {
		t.Helper()

		reqBody := map[string]any{
			"query":        "apple",
			"top_k":        3,
			"mode":         "lex",
			"page_size":    1,
			"limit":        1,
			"include_meta": true,
		}
		if pageToken != "" {
			reqBody["page_token"] = pageToken
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			t.Fatalf("marshal failed: %v", err)
		}

		req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("unexpected status %d: %s", w.Code, w.Body.String())
		}

		var resp queryResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatalf("decode failed: %v", err)
		}
		return resp
	}

	page1 := query("")
	if len(page1.IDs) != 1 || page1.IDs[0] != "doc-1" {
		t.Fatalf("unexpected page1 ids: %+v", page1.IDs)
	}
	if len(page1.Docs) != 1 || page1.Docs[0] != "apple alpha" {
		t.Fatalf("unexpected page1 docs: %+v", page1.Docs)
	}
	if len(page1.Meta) != 1 || page1.Meta[0]["rank"] != "1" {
		t.Fatalf("unexpected page1 meta: %+v", page1.Meta)
	}
	if page1.Next == "" {
		t.Fatal("expected next page token on page1")
	}

	page2 := query(page1.Next)
	if len(page2.IDs) != 1 || page2.IDs[0] != "doc-2" {
		t.Fatalf("unexpected page2 ids: %+v", page2.IDs)
	}
	if len(page2.Docs) != 1 || page2.Docs[0] != "apple beta" {
		t.Fatalf("unexpected page2 docs: %+v", page2.Docs)
	}
	if len(page2.Meta) != 1 || page2.Meta[0]["rank"] != "2" {
		t.Fatalf("unexpected page2 meta: %+v", page2.Meta)
	}
	if page2.Next == "" {
		t.Fatal("expected next page token on page2")
	}

	page3 := query(page2.Next)
	if len(page3.IDs) != 1 || page3.IDs[0] != "doc-3" {
		t.Fatalf("unexpected page3 ids: %+v", page3.IDs)
	}
	if len(page3.Docs) != 1 || page3.Docs[0] != "apple gamma" {
		t.Fatalf("unexpected page3 docs: %+v", page3.Docs)
	}
	if len(page3.Meta) != 1 || page3.Meta[0]["rank"] != "3" {
		t.Fatalf("unexpected page3 meta: %+v", page3.Meta)
	}
	if page3.Next != "" {
		t.Fatalf("expected final page to have empty next token, got %q", page3.Next)
	}
}

func TestHTTPQueryRerankKeepsResponseFieldsAligned(t *testing.T) {
	emb := &testEmbedder{
		dim: 2,
		vectors: map[string][]float32{
			"query": {1, 0},
			"alpha": {0.9, 0},
			"beta":  {0.5, 0},
			"gamma": {0.1, 0},
		},
	}
	store := NewVectorStore(10, emb.Dim())
	handler, _ := newHTTPHandler(store, emb, reverseReranker{}, "")

	docs := []struct {
		id   string
		doc  string
		meta map[string]string
	}{
		{id: "id-alpha", doc: "alpha", meta: map[string]string{"name": "alpha"}},
		{id: "id-beta", doc: "beta", meta: map[string]string{"name": "beta"}},
		{id: "id-gamma", doc: "gamma", meta: map[string]string{"name": "gamma"}},
	}
	for _, doc := range docs {
		vec, err := emb.Embed(doc.doc)
		if err != nil {
			t.Fatalf("embed failed: %v", err)
		}
		if _, err := store.Add(vec, doc.doc, doc.id, doc.meta, "default", ""); err != nil {
			t.Fatalf("add failed: %v", err)
		}
	}

	reqBody := map[string]any{
		"query":        "query",
		"top_k":        3,
		"mode":         "scan",
		"include_meta": true,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		IDs    []string            `json:"ids"`
		Docs   []string            `json:"docs"`
		Scores []float32           `json:"scores"`
		Meta   []map[string]string `json:"meta"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	if len(resp.IDs) != 3 || len(resp.Docs) != 3 || len(resp.Scores) != 3 || len(resp.Meta) != 3 {
		t.Fatalf("unaligned response lengths: ids=%d docs=%d scores=%d meta=%d", len(resp.IDs), len(resp.Docs), len(resp.Scores), len(resp.Meta))
	}

	wantIDs := []string{"id-gamma", "id-beta", "id-alpha"}
	wantDocs := []string{"gamma", "beta", "alpha"}
	wantScores := []float32{0.1, 0.5, 0.9}
	wantMeta := []string{"gamma", "beta", "alpha"}

	for i := range wantIDs {
		if resp.IDs[i] != wantIDs[i] {
			t.Fatalf("id[%d] = %q, want %q", i, resp.IDs[i], wantIDs[i])
		}
		if resp.Docs[i] != wantDocs[i] {
			t.Fatalf("doc[%d] = %q, want %q", i, resp.Docs[i], wantDocs[i])
		}
		if resp.Meta[i]["name"] != wantMeta[i] {
			t.Fatalf("meta[%d] = %+v, want name=%q", i, resp.Meta[i], wantMeta[i])
		}
		if resp.Scores[i] != wantScores[i] {
			t.Fatalf("score[%d] = %v, want %v", i, resp.Scores[i], wantScores[i])
		}
	}
}
