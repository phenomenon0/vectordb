package main

import (
	"context"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"agentscope/vectordb/client"
)

func TestHTTPHandlersInsertQueryDelete(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "index.gob")

	store := NewVectorStore(10, 3)
	embedder := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: embedder}

	handler := newHTTPHandler(store, embedder, reranker, indexPath)
	srv := httptest.NewServer(handler)
	defer srv.Close()

	cli := client.New(srv.URL)

	// insert
	if _, err := cli.Insert(context.Background(), client.InsertRequest{Doc: "hello world", Meta: map[string]string{"tag": "a", "score": "0.8", "ts": "2024-01-01T00:00:00Z"}}); err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	// query
	min := 0.5
	qr, err := cli.Query(context.Background(), client.QueryRequest{
		Query:       "hello",
		TopK:        1,
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

	// delete
	if _, err := cli.Delete(context.Background(), client.DeleteRequest{ID: qr.IDs[0]}); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	qr2, err := cli.Query(context.Background(), client.QueryRequest{Query: "hello", TopK: 1})
	if err != nil {
		t.Fatalf("query2 failed: %v", err)
	}
	if len(qr2.IDs) != 0 {
		t.Fatalf("expected no results after delete, got %+v", qr2.IDs)
	}
}
