package client

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestInsertSetsHeadersAndParsesResponse(t *testing.T) {
	var sawAuth, sawCustom bool

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/insert" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		if got := r.Header.Get("Authorization"); got == "Bearer tok" {
			sawAuth = true
		}
		if got := r.Header.Get("X-Custom"); got == "yes" {
			sawCustom = true
		}
		defer r.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"abc"}`))
	}))
	defer ts.Close()

	c := New(ts.URL, WithToken("tok"), WithHeader("X-Custom", "yes"))
	resp, err := c.Insert(context.Background(), InsertRequest{Doc: "hello"})
	if err != nil {
		t.Fatalf("insert error: %v", err)
	}
	if resp.ID != "abc" {
		t.Fatalf("unexpected id: %s", resp.ID)
	}
	if !sawAuth || !sawCustom {
		t.Fatalf("expected auth/custom headers to be set")
	}
}

func TestQueryParsesBody(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/query" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ids":["1"],"docs":["d"],"scores":[0.9],"stats":"ok"}`))
	}))
	defer ts.Close()

	c := New(ts.URL)
	resp, err := c.Query(context.Background(), QueryRequest{Query: "q"})
	if err != nil {
		t.Fatalf("query error: %v", err)
	}
	if len(resp.IDs) != 1 || resp.IDs[0] != "1" {
		t.Fatalf("unexpected ids: %+v", resp.IDs)
	}
	if len(resp.Scores) != 1 || resp.Scores[0] != 0.9 {
		t.Fatalf("unexpected scores: %+v", resp.Scores)
	}
	if resp.Stats != "ok" {
		t.Fatalf("unexpected stats: %s", resp.Stats)
	}
}

func TestHTTPErrorSurface(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	}))
	defer ts.Close()

	c := New(ts.URL)
	_, err := c.Delete(context.Background(), DeleteRequest{ID: "x"})
	if err == nil {
		t.Fatalf("expected error")
	}
	if httpErr, ok := err.(*HTTPError); ok {
		if httpErr.StatusCode != http.StatusInternalServerError {
			t.Fatalf("unexpected status: %d", httpErr.StatusCode)
		}
		if httpErr.Body != "boom" {
			t.Fatalf("unexpected body: %s", httpErr.Body)
		}
	} else {
		t.Fatalf("expected HTTPError, got %T", err)
	}
}
