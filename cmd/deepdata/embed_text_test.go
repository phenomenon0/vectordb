package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// hashServerEmbedder is the CI stand-in for a real text embedder: the hash
// embedder is deterministic, so identical text lands on identical vectors and
// a text search for a stored text must rank that document first.
func hashServerEmbedder() *serverEmbedder {
	return &serverEmbedder{Embedder: NewHashEmbedder(4), Provider: "hash", Model: "4"}
}

func newTextTestHandler(t *testing.T, emb Embedder) http.Handler {
	t.Helper()
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	store := NewVectorStore(8, 4)
	indexPath := filepath.Join(t.TempDir(), "index.gob")
	handler, collections := newCanonicalHTTPHandler(store, emb, nil, indexPath)
	if err := collections.PersistenceError(); err != nil {
		t.Fatalf("open canonical persistence: %v", err)
	}
	t.Cleanup(func() {
		if err := collections.Close(); err != nil {
			t.Errorf("close canonical persistence: %v", err)
		}
	})
	return handler
}

func doJSON(t *testing.T, handler http.Handler, method, path string, payload interface{}) (int, map[string]interface{}) {
	t.Helper()
	var body []byte
	if payload != nil {
		var err error
		if body, err = json.Marshal(payload); err != nil {
			t.Fatal(err)
		}
	}
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	out := map[string]interface{}{}
	if response.Body.Len() > 0 {
		if err := json.Unmarshal(response.Body.Bytes(), &out); err != nil {
			t.Fatalf("%s %s: non-JSON body (%d): %s", method, path, response.Code, response.Body.String())
		}
	}
	return response.Code, out
}

func wantAPIError(t *testing.T, got map[string]interface{}, code, field string) {
	t.Helper()
	if got["code"] != code {
		t.Fatalf("code = %v, want %q (body %v)", got["code"], code, got)
	}
	if field != "" && got["field"] != field {
		t.Fatalf("field = %v, want %q (body %v)", got["field"], field, got)
	}
}

var textSchema = map[string]interface{}{
	"name": "notes",
	"fields": []map[string]interface{}{
		// dim omitted on purpose: the server fills it from the embedder.
		{"name": "text", "type": "dense", "index": map[string]string{"type": "flat"}, "embedding": map[string]string{"provider": "hash"}},
		{"name": "terms", "type": "sparse", "dim": 256, "index": map[string]string{"type": "inverted"}, "embedding": map[string]string{"provider": "bm25"}},
	},
}

// rawSchema binds text but leaves raw to the caller; every document must
// still carry every field, so texts and vectors are mixed per document.
var rawSchema = map[string]interface{}{
	"name": "mixed",
	"fields": []map[string]interface{}{
		{"name": "text", "type": "dense", "index": map[string]string{"type": "flat"}, "embedding": map[string]string{"provider": "hash"}},
		{"name": "raw", "type": "dense", "dim": 4, "index": map[string]string{"type": "flat"}},
	},
}

// TestCanonicalHTTPTextsRoundTrip is CTL-02's contract on the HTTP surface:
// create with an embedding binding, insert texts, search texts, get the
// stored document back with embedded_by naming the embedder.
func TestCanonicalHTTPTextsRoundTrip(t *testing.T) {
	handler := newTextTestHandler(t, hashServerEmbedder())
	const base = "/v3/tenants/acme/collections"

	if code, body := doJSON(t, handler, http.MethodPost, base, textSchema); code != http.StatusCreated {
		t.Fatalf("create returned %d: %v", code, body)
	}
	code, info := doJSON(t, handler, http.MethodGet, base+"/notes", nil)
	if code != http.StatusOK {
		t.Fatalf("get collection returned %d: %v", code, info)
	}
	if nested, ok := info["collection"].(map[string]interface{}); ok {
		info = nested
	}
	fields, _ := info["fields"].([]interface{})
	if len(fields) != 2 {
		t.Fatalf("collection info fields = %v", info["fields"])
	}
	text, _ := fields[0].(map[string]interface{})
	embedding, _ := text["embedding"].(map[string]interface{})
	if text["dim"] != float64(4) || embedding["provider"] != "hash" || embedding["model"] != "4" {
		t.Fatalf("bound field must carry the filled dim and model: %v", text)
	}

	docs := base + "/notes/docs"
	code, body := doJSON(t, handler, http.MethodPost, docs, map[string]interface{}{
		"id":       1,
		"texts":    map[string]string{"text": "durable vector storage", "terms": "durable vector storage"},
		"metadata": map[string]string{"text": "durable vector storage"},
	})
	if code != http.StatusOK {
		t.Fatalf("insert texts returned %d: %v", code, body)
	}
	code, body = doJSON(t, handler, http.MethodPut, docs+"/2", map[string]interface{}{
		"texts": map[string]string{"text": "unrelated cooking recipe", "terms": "unrelated cooking recipe"},
	})
	if code != http.StatusOK {
		t.Fatalf("upsert texts returned %d: %v", code, body)
	}
	code, body = doJSON(t, handler, http.MethodPost, docs+"/batch", map[string]interface{}{
		"documents": []map[string]interface{}{{"id": 3, "texts": map[string]string{"text": "batch inserted terms", "terms": "batch inserted terms"}}},
	})
	if code != http.StatusOK {
		t.Fatalf("batch texts returned %d: %v", code, body)
	}

	code, result := doJSON(t, handler, http.MethodPost, base+"/notes/search", map[string]interface{}{
		"texts": map[string]string{"text": "durable vector storage"},
		"top_k": 2,
	})
	if code != http.StatusOK {
		t.Fatalf("search texts returned %d: %v", code, result)
	}
	documents, _ := result["documents"].([]interface{})
	if len(documents) == 0 {
		t.Fatalf("text search found nothing: %v", result)
	}
	if top, _ := documents[0].(map[string]interface{}); top["id"] != float64(1) {
		t.Fatalf("same text must rank its own document first, got %v", documents[0])
	}
	embeddedBy, _ := result["embedded_by"].(map[string]interface{})
	if embeddedBy["text"] != "hash:4" {
		t.Fatalf("embedded_by = %v, want text -> hash:4", result["embedded_by"])
	}

	code, result = doJSON(t, handler, http.MethodPost, base+"/notes/search", map[string]interface{}{
		"texts": map[string]string{"terms": "batch inserted terms"},
		"top_k": 3,
	})
	if code != http.StatusOK {
		t.Fatalf("sparse text search returned %d: %v", code, result)
	}
	embeddedBy, _ = result["embedded_by"].(map[string]interface{})
	if embeddedBy["terms"] != "bm25" {
		t.Fatalf("sparse embedded_by = %v, want terms -> bm25", result["embedded_by"])
	}

	// A vectors-only search reports no embedder: the caller did the embedding.
	code, result = doJSON(t, handler, http.MethodPost, base+"/notes/search", map[string]interface{}{
		"queries": map[string]interface{}{"text": []float32{0, 1, 0, 0}},
		"top_k":   1,
	})
	if code != http.StatusOK {
		t.Fatalf("vector search returned %d: %v", code, result)
	}
	if _, present := result["embedded_by"]; present {
		t.Fatalf("vector-only search must omit embedded_by: %v", result)
	}
}

// Each rejection tells the agent which field to fix and how.
func TestCanonicalHTTPTextsRejections(t *testing.T) {
	handler := newTextTestHandler(t, hashServerEmbedder())
	const base = "/v3/tenants/acme/collections"
	if code, body := doJSON(t, handler, http.MethodPost, base, textSchema); code != http.StatusCreated {
		t.Fatalf("create returned %d: %v", code, body)
	}
	docs := base + "/notes/docs"

	code, body := doJSON(t, handler, http.MethodPost, docs, map[string]interface{}{
		"texts":   map[string]string{"text": "twice"},
		"vectors": map[string]interface{}{"text": []float32{1, 0, 0, 0}},
	})
	if code != http.StatusBadRequest {
		t.Fatalf("text+vector on one field returned %d: %v", code, body)
	}
	wantAPIError(t, body, apierror.CodeInvalidArgument, "texts.text")

	if code, body := doJSON(t, handler, http.MethodPost, base, rawSchema); code != http.StatusCreated {
		t.Fatalf("create mixed returned %d: %v", code, body)
	}
	code, body = doJSON(t, handler, http.MethodPost, base+"/mixed/docs", map[string]interface{}{
		"texts": map[string]string{"text": "bound", "raw": "unbound"},
	})
	if code != http.StatusBadRequest {
		t.Fatalf("text on unbound field returned %d: %v", code, body)
	}
	wantAPIError(t, body, apierror.CodeInvalidArgument, "texts.raw")
	if hint, _ := body["hint"].(string); !strings.Contains(hint, "bind an embedding") {
		t.Fatalf("unbound-field hint must say how to fix it: %v", body)
	}

	code, body = doJSON(t, handler, http.MethodPost, docs, map[string]interface{}{"metadata": map[string]string{"k": "v"}})
	if code != http.StatusBadRequest {
		t.Fatalf("document with neither vectors nor texts returned %d: %v", code, body)
	}
	wantAPIError(t, body, apierror.CodeInvalidArgument, "")

	// A binding the running server cannot honor is a state conflict, not a
	// malformed request.
	code, body = doJSON(t, handler, http.MethodPost, base, map[string]interface{}{
		"name":   "elsewhere",
		"fields": []map[string]interface{}{{"name": "text", "type": "dense", "index": map[string]string{"type": "flat"}, "embedding": map[string]string{"provider": "ollama"}}},
	})
	if code != http.StatusConflict {
		t.Fatalf("mismatched provider returned %d: %v", code, body)
	}
	wantAPIError(t, body, apierror.CodeEmbeddingMismatch, "fields.text.embedding")

	code, body = doJSON(t, handler, http.MethodPost, base, map[string]interface{}{
		"name":   "wrongdim",
		"fields": []map[string]interface{}{{"name": "text", "type": "dense", "dim": 8, "index": map[string]string{"type": "flat"}, "embedding": map[string]string{"provider": "hash"}}},
	})
	if code != http.StatusBadRequest {
		t.Fatalf("dim disagreeing with the embedder returned %d: %v", code, body)
	}
	wantAPIError(t, body, apierror.CodeInvalidArgument, "fields.text.dim")
}

// With DEEPDATA_EMBEDDER=none the server still runs; it just refuses text
// with a 503 that names the fix, and /readyz says which embedder it has.
func TestCanonicalHTTPTextsWithoutEmbedder(t *testing.T) {
	handler := newTextTestHandler(t, nil)
	code, body := doJSON(t, handler, http.MethodPost, "/v3/tenants/acme/collections", textSchema)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("binding an embedder on a server without one returned %d: %v", code, body)
	}
	wantAPIError(t, body, apierror.CodeEmbedderUnavailable, "fields.text.embedding")
	if body["retryable"] != true {
		t.Fatalf("embedder_unavailable must be retryable (operator can configure one): %v", body)
	}

	code, ready := doJSON(t, handler, http.MethodGet, "/readyz", nil)
	if code != http.StatusOK || ready["embedder"] != "none" {
		t.Fatalf("readyz = %d %v, want 200 with embedder none", code, ready)
	}
	withHash := newTextTestHandler(t, hashServerEmbedder())
	if code, ready = doJSON(t, withHash, http.MethodGet, "/readyz", nil); code != http.StatusOK || ready["embedder"] != "hash:4" {
		t.Fatalf("readyz = %d %v, want 200 with embedder hash:4", code, ready)
	}
}

// A collection bound while an embedder was configured can outlive that
// configuration (restart with DEEPDATA_EMBEDDER=none, or a different one);
// resolveTexts is the shared HTTP/gRPC gate that must then refuse text.
func TestResolveTextsAgainstChangedEmbedder(t *testing.T) {
	fields := []vcollection.VectorField{{
		Name: "text", Type: vcollection.VectorTypeDense, Dim: 4,
		Index:     vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		Embedding: &vcollection.EmbeddingConfig{Provider: "hash", Model: "4"},
	}}
	texts := map[string]string{"text": "hello"}

	_, aerr := resolveTexts(fields, nil, texts, map[string]interface{}{}, false)
	if aerr == nil || aerr.Code != apierror.CodeEmbedderUnavailable {
		t.Fatalf("no embedder: got %v, want %s", aerr, apierror.CodeEmbedderUnavailable)
	}
	other := &serverEmbedder{Embedder: NewHashEmbedder(4), Provider: "ollama", Model: "nomic-embed-text"}
	_, aerr = resolveTexts(fields, other, texts, map[string]interface{}{}, false)
	if aerr == nil || aerr.Code != apierror.CodeEmbeddingMismatch {
		t.Fatalf("different embedder: got %v, want %s", aerr, apierror.CodeEmbeddingMismatch)
	}
	vectors := map[string]interface{}{}
	embeddedBy, aerr := resolveTexts(fields, hashServerEmbedder(), texts, vectors, true)
	if aerr != nil {
		t.Fatalf("matching embedder: %v", aerr)
	}
	if embeddedBy["text"] != "hash:4" || len(vectors["text"].([]float32)) != 4 {
		t.Fatalf("resolved = %v vectors=%v", embeddedBy, vectors)
	}
}

// DEEPDATA_EMBEDDER is the one switch; hash is never implicit and an unknown
// or unusable choice is a startup error, not a silent fallback. The ollama and
// openai probes are network calls and are not exercised here.
func TestServerEmbedderFromEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("DEEPDATA_EMBED_DIM", "")

	for _, value := range []string{"", "none", " NONE "} {
		t.Setenv("DEEPDATA_EMBEDDER", value)
		emb, err := newServerEmbedderFromEnv()
		if err != nil || emb != nil {
			t.Fatalf("DEEPDATA_EMBEDDER=%q: got %v, %v; want no embedder", value, emb, err)
		}
		if emb.Label() != "none" {
			t.Fatalf("nil embedder label = %q", emb.Label())
		}
	}

	t.Setenv("DEEPDATA_EMBEDDER", "hash")
	t.Setenv("DEEPDATA_EMBED_DIM", "16")
	emb, err := newServerEmbedderFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if emb.Label() != "hash:16" || emb.Dim() != 16 {
		t.Fatalf("hash embedder = %s dim %d", emb.Label(), emb.Dim())
	}
	t.Setenv("DEEPDATA_EMBED_DIM", "0")
	if _, err := newServerEmbedderFromEnv(); err == nil {
		t.Fatal("dim 0 must be rejected")
	}

	t.Setenv("DEEPDATA_EMBEDDER", "openai")
	if _, err := newServerEmbedderFromEnv(); err == nil || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Fatalf("openai without a key: %v", err)
	}
	t.Setenv("DEEPDATA_EMBEDDER", "bogus")
	if _, err := newServerEmbedderFromEnv(); err == nil || !strings.Contains(err.Error(), "unknown DEEPDATA_EMBEDDER") {
		t.Fatalf("unknown embedder: %v", err)
	}
}

// The gRPC surface mirrors HTTP field for field.
func TestCanonicalGRPCTextsRoundTrip(t *testing.T) {
	base := t.TempDir()
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("close durable store: %v", err)
		}
	})
	server := &CollectionGRPCServer{tenants: store.Tenants(), persistenceHealth: store.Err, embedder: hashServerEmbedder()}
	ctx := canonicalGRPCAdminContext("acme")

	if _, err := server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId: "acme", Name: "notes",
		Fields: []*deepdatav3.VectorFieldConfig{
			{Name: "text", Type: int32(vcollection.VectorTypeDense), IndexType: "flat", Embedding: &deepdatav3.EmbeddingConfig{Provider: "hash"}},
		},
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	info, err := server.GetCollection(ctx, &deepdatav3.GetCollectionRequest{TenantId: "acme", Name: "notes"})
	if err != nil {
		t.Fatalf("get collection: %v", err)
	}
	text := info.Collection.Fields[0]
	if text.Dim != 4 || text.Embedding == nil || text.Embedding.Provider != "hash" || text.Embedding.Model != "4" {
		t.Fatalf("bound field must carry the filled dim and model: %v", text)
	}

	if _, err := server.Insert(ctx, &deepdatav3.InsertRequest{
		TenantId: "acme", Collection: "notes", Id: 1,
		Texts: map[string]string{"text": "durable vector storage"},
	}); err != nil {
		t.Fatalf("insert texts: %v", err)
	}
	if _, err := server.Upsert(ctx, &deepdatav3.UpsertRequest{
		TenantId: "acme", Collection: "notes", Id: 2,
		Texts: map[string]string{"text": "unrelated cooking recipe"},
	}); err != nil {
		t.Fatalf("upsert texts: %v", err)
	}
	found, err := server.Search(ctx, &deepdatav3.SearchRequest{
		TenantId: "acme", Collection: "notes", TopK: 2,
		Texts: map[string]string{"text": "durable vector storage"},
	})
	if err != nil {
		t.Fatalf("search texts: %v", err)
	}
	if len(found.Results) == 0 || found.Results[0].Id != 1 || found.EmbeddedBy["text"] != "hash:4" {
		t.Fatalf("search = %v", found)
	}

	_, err = server.Insert(ctx, &deepdatav3.InsertRequest{
		TenantId: "acme", Collection: "notes", Id: 3,
		Texts:   map[string]string{"text": "twice"},
		Vectors: map[string]*deepdatav3.VectorData{"text": denseProtoVector(1, 0, 0, 0)},
	})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("text+vector on one field: %v", err)
	}
	_, err = server.Insert(ctx, &deepdatav3.InsertRequest{TenantId: "acme", Collection: "notes", Id: 4})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("neither vectors nor texts: %v", err)
	}
	_, err = server.CreateCollection(ctx, &deepdatav3.CreateCollectionRequest{
		TenantId: "acme", Name: "elsewhere",
		Fields: []*deepdatav3.VectorFieldConfig{
			{Name: "text", Type: int32(vcollection.VectorTypeDense), IndexType: "flat", Embedding: &deepdatav3.EmbeddingConfig{Provider: "ollama"}},
		},
	})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("mismatched provider: %v", err)
	}
}
