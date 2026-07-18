package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

type failingEmbedder struct {
	called bool
}

func (e *failingEmbedder) Embed(text string) ([]float32, error) {
	e.called = true
	return nil, http.ErrHandlerTimeout
}

func (e *failingEmbedder) EmbedQuery(text string) ([]float32, error) { return e.Embed(text) }

func (e *failingEmbedder) Dim() int { return 3 }

func TestCORSDefaultDoesNotAllowCredentials(t *testing.T) {
	t.Setenv("CORS_ALLOWED_ORIGINS", "")

	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	req := httptest.NewRequest(http.MethodOptions, "/health", nil)
	req.Header.Set("Origin", "https://example.com")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("expected wildcard origin, got %q", got)
	}
	if got := w.Header().Get("Access-Control-Allow-Credentials"); got != "" {
		t.Fatalf("expected no credentials header by default, got %q", got)
	}
}

func TestCORSAllowlistAllowsOnlyConfiguredOrigins(t *testing.T) {
	t.Setenv("CORS_ALLOWED_ORIGINS", "https://allowed.example")

	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	allowedReq := httptest.NewRequest(http.MethodOptions, "/health", nil)
	allowedReq.Header.Set("Origin", "https://allowed.example")
	allowedW := httptest.NewRecorder()
	handler.ServeHTTP(allowedW, allowedReq)

	if allowedW.Code != http.StatusOK {
		t.Fatalf("expected status 200 for allowed origin, got %d", allowedW.Code)
	}
	if got := allowedW.Header().Get("Access-Control-Allow-Origin"); got != "https://allowed.example" {
		t.Fatalf("expected allowed origin header, got %q", got)
	}
	if got := allowedW.Header().Get("Access-Control-Allow-Credentials"); got != "true" {
		t.Fatalf("expected credentials header for allowlist origin, got %q", got)
	}

	disallowedReq := httptest.NewRequest(http.MethodOptions, "/health", nil)
	disallowedReq.Header.Set("Origin", "https://forbidden.example")
	disallowedW := httptest.NewRecorder()
	handler.ServeHTTP(disallowedW, disallowedReq)

	if disallowedW.Code != http.StatusForbidden {
		t.Fatalf("expected status 403 for disallowed origin, got %d", disallowedW.Code)
	}
}

func TestEmbedEndpointsRequireAuthWhenEnabled(t *testing.T) {
	store := NewVectorStore(100, 3)
	store.requireAuth = true
	store.apiToken = "secret-token"
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	body, _ := json.Marshal(map[string]string{"text": "hello"})

	reqNoAuth := httptest.NewRequest(http.MethodPost, "/api/embed", bytes.NewReader(body))
	reqNoAuth.Header.Set("Content-Type", "application/json")
	wNoAuth := httptest.NewRecorder()
	handler.ServeHTTP(wNoAuth, reqNoAuth)
	if wNoAuth.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without auth, got %d", wNoAuth.Code)
	}

	reqAuth := httptest.NewRequest(http.MethodPost, "/api/embed", bytes.NewReader(body))
	reqAuth.Header.Set("Content-Type", "application/json")
	reqAuth.Header.Set("Authorization", "Bearer secret-token")
	wAuth := httptest.NewRecorder()
	handler.ServeHTTP(wAuth, reqAuth)
	if wAuth.Code != http.StatusOK {
		t.Fatalf("expected 200 with valid auth, got %d: %s", wAuth.Code, wAuth.Body.String())
	}
}

func TestReadyzDoesNotCallEmbedder(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := &failingEmbedder{}
	reranker := &SimpleReranker{Embedder: NewHashEmbedder(3)}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected readyz status 200, got %d: %s", w.Code, w.Body.String())
	}
	if emb.called {
		t.Fatal("expected readyz to avoid embedder calls")
	}
}

func TestReadyzFailsClosedAfterWALFault(t *testing.T) {
	store := NewVectorStore(100, 3)
	store.walFault = errors.New("indeterminate append")
	emb := NewHashEmbedder(3)
	handler, _ := newHTTPHandler(store, emb, &SimpleReranker{Embedder: emb}, "")

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected readyz status 503 after WAL fault, got %d: %s", w.Code, w.Body.String())
	}
}

func TestCollectionPersistenceCorruptionIsRetainedForStartupRefusal(t *testing.T) {
	for _, test := range []struct {
		name  string
		setup func(t *testing.T, basePath string)
	}{
		{
			name: "manager",
			setup: func(t *testing.T, basePath string) {
				t.Helper()
				if err := os.WriteFile(basePath+".manager", []byte(`{"collections":{"broken":null}}`), 0o600); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "tenants",
			setup: func(t *testing.T, basePath string) {
				t.Helper()
				seed := NewCollectionHTTPServer(basePath)
				if err := seed.manager.Save(basePath + ".manager"); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(basePath+".tenants", []byte(`{"tenants":{"broken":null}}`), 0o600); err != nil {
					t.Fatal(err)
				}
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			indexPath := filepath.Join(dir, "index.gob")
			basePath := indexPath + ".collections"
			test.setup(t, basePath)

			store := NewVectorStore(100, 3)
			emb := NewHashEmbedder(3)
			_, collections := newHTTPHandler(store, emb, &SimpleReranker{Embedder: emb}, indexPath)
			if err := collections.PersistenceError(); err == nil {
				t.Fatal("expected corrupt collection persistence to block production startup")
			}
			if _, err := os.Stat(basePath + ".snapshot"); !os.IsNotExist(err) {
				t.Fatalf("failed load replaced corrupt state: %v", err)
			}
		})
	}
}

func TestOnlineSnapshotImportDisabledInRC(t *testing.T) {
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "index.gob")
	store := NewVectorStore(100, 3)
	store.walPath = indexPath + ".wal"
	if _, err := store.Add([]float32{1, 0, 0}, "existing", "id-1", nil, "default", "default"); err != nil {
		t.Fatalf("seed store: %v", err)
	}
	wantWAL, err := os.ReadFile(store.walPath)
	if err != nil {
		t.Fatalf("read seed WAL: %v", err)
	}
	wantHighWater := store.appliedWALSeq
	emb := NewHashEmbedder(3)
	handler, _ := newHTTPHandler(store, emb, &SimpleReranker{Embedder: emb}, indexPath)

	req := httptest.NewRequest(http.MethodPost, "/import", bytes.NewBufferString("not-a-snapshot"))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if w.Code != http.StatusNotImplemented {
		t.Fatalf("expected online import status 501, got %d: %s", w.Code, w.Body.String())
	}
	if store.Count != 1 || store.GetDoc(0) != "existing" {
		t.Fatalf("disabled online import mutated store: count=%d", store.Count)
	}
	if store.appliedWALSeq != wantHighWater {
		t.Fatalf("disabled online import changed WAL high-water: got=%d want=%d", store.appliedWALSeq, wantHighWater)
	}
	gotWAL, err := os.ReadFile(store.walPath)
	if err != nil {
		t.Fatalf("read WAL after disabled import: %v", err)
	}
	if !bytes.Equal(gotWAL, wantWAL) {
		t.Fatal("disabled online import changed WAL artifact")
	}
	if _, err := os.Stat(indexPath); !os.IsNotExist(err) {
		t.Fatalf("disabled online import created snapshot: %v", err)
	}
}

func TestVaultEndpointsRequireAdmin(t *testing.T) {
	store := NewVectorStore(100, 3)
	store.requireAuth = true
	store.apiToken = "secret-token"
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, filepath.Join(t.TempDir(), "index.gob"))

	req := httptest.NewRequest(http.MethodGet, "/vault/browse", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for non-admin vault access, got %d: %s", w.Code, w.Body.String())
	}
}

func TestObsidianEnableStoresConfigBesideIndex(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	dir := t.TempDir()
	indexPath := filepath.Join(dir, "index.gob")
	handler, _ := newHTTPHandler(store, emb, reranker, indexPath)

	vaultDir := filepath.Join(dir, "vault")
	if err := os.MkdirAll(vaultDir, 0o755); err != nil {
		t.Fatalf("mkdir vault failed: %v", err)
	}

	body, _ := json.Marshal(map[string]string{
		"vault":      vaultDir,
		"collection": "obsidian",
		"interval":   "1m",
	})
	req := httptest.NewRequest(http.MethodPost, "/admin/obsidian/enable", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 enabling obsidian, got %d: %s", w.Code, w.Body.String())
	}

	if _, err := os.Stat(filepath.Join(dir, "obsidian.json")); err != nil {
		t.Fatalf("expected obsidian config in data directory: %v", err)
	}
}

func TestVaultAnnotationsStoreBesideIndex(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	dir := t.TempDir()
	handler, _ := newHTTPHandler(store, emb, reranker, filepath.Join(dir, "index.gob"))

	body, _ := json.Marshal(map[string]any{
		"id":     "ann-1",
		"doc_id": "doc-1",
		"text":   "note",
	})
	req := httptest.NewRequest(http.MethodPost, "/vault/annotations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 storing annotation, got %d: %s", w.Code, w.Body.String())
	}

	if _, err := os.Stat(filepath.Join(dir, "annotations.json")); err != nil {
		t.Fatalf("expected annotations file in data directory: %v", err)
	}
}
