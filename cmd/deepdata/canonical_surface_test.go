package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/security"
)

func newCanonicalSurfaceTestHandler(t *testing.T) http.Handler {
	t.Helper()
	return newCanonicalSurfaceTestHandlerAt(t, filepath.Join(t.TempDir(), "index.gob"))
}

// newCanonicalSurfaceTestHandlerAt is the same handler over a caller-chosen
// data directory, so a test can seed on-disk state the server must find at
// open.
func newCanonicalSurfaceTestHandlerAt(t *testing.T, indexPath string) http.Handler {
	t.Helper()
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	rt := testServerRuntime(t)
	embedder := NewHashEmbedder(4)
	handler, collections := newCanonicalHTTPHandler(rt, embedder, indexPath)
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

func TestCanonicalRCSurfaceRejectsUnsupportedHandlers(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	unsupported := []string{
		"/",
		"/insert",
		"/query",
		"/v2/collections",
		"/v2/import",
		"/v2/recommend",
		"/v2/discover",
		"/v2/feedback",
		"/v2/extract",
		"/api/embed",
		"/api/config/embedder",
	}
	for _, path := range unsupported {
		for _, method := range []string{http.MethodGet, http.MethodPost, http.MethodOptions} {
			t.Run(method+" "+path, func(t *testing.T) {
				req := httptest.NewRequest(method, path, nil)
				resp := httptest.NewRecorder()
				handler.ServeHTTP(resp, req)
				if resp.Code != http.StatusNotFound {
					t.Fatalf("%s %s returned %d, want 404", method, path, resp.Code)
				}
			})
		}
	}

	for _, path := range []string{"/healthz", "/livez", "/readyz", "/metrics"} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			resp := httptest.NewRecorder()
			handler.ServeHTTP(resp, req)
			if resp.Code == http.StatusNotFound {
				t.Fatalf("canonical operational endpoint %s was not registered", path)
			}
		})
	}
}

func TestCanonicalRCSurfaceTenantBatchContract(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	schema := vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name:  "embedding",
			Type:  vcollection.VectorTypeDense,
			Dim:   2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
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

	batchBody := []byte(`{"documents":[{"id":41,"vectors":{"embedding":[1,0]},"metadata":{"kind":"a"}},{"vectors":{"embedding":[0,1]},"metadata":{"kind":"b"}}]}`)
	request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/docs/docs/batch", bytes.NewReader(batchBody))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("batch insert returned %d: %s", response.Code, response.Body.String())
	}
	var result struct {
		IDs      []uint64 `json:"ids"`
		Inserted int      `json:"inserted"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Inserted != 2 || len(result.IDs) != 2 || result.IDs[0] != 41 || result.IDs[1] != 42 {
		t.Fatalf("unexpected batch result: %+v", result)
	}
}

func TestCanonicalHTTPUpsertAndGetDocContract(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	schema := vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name:  "embedding",
			Type:  vcollection.VectorTypeDense,
			Dim:   2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
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

	// PUT upserts a document under the path ID.
	putBody := []byte(`{"vectors":{"embedding":[1,0]},"metadata":{"kind":"replaced"}}`)
	request = httptest.NewRequest(http.MethodPut, "/v3/tenants/acme/collections/docs/docs/55", bytes.NewReader(putBody))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("upsert returned %d: %s", response.Code, response.Body.String())
	}

	// Replacing the same ID must not duplicate storage.
	putBody = []byte(`{"vectors":{"embedding":[0,1]},"metadata":{"kind":"replaced-again"}}`)
	request = httptest.NewRequest(http.MethodPut, "/v3/tenants/acme/collections/docs/docs/55", bytes.NewReader(putBody))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("second upsert returned %d: %s", response.Code, response.Body.String())
	}

	// GET reads the replacement back.
	request = httptest.NewRequest(http.MethodGet, "/v3/tenants/acme/collections/docs/docs/55", nil)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("get doc returned %d: %s", response.Code, response.Body.String())
	}
	var doc struct {
		ID       uint64                 `json:"id"`
		Metadata map[string]interface{} `json:"metadata"`
	}
	if err := json.NewDecoder(response.Body).Decode(&doc); err != nil {
		t.Fatal(err)
	}
	if doc.ID != 55 || doc.Metadata["kind"] != "replaced-again" {
		t.Fatalf("get after upsert returned %+v", doc)
	}

	// GET of a never-written ID is 404.
	request = httptest.NewRequest(http.MethodGet, "/v3/tenants/acme/collections/docs/docs/57", nil)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("get of missing doc returned %d, want 404", response.Code)
	}

	// A non-numeric document id segment is routed as 404, not parsed.
	request = httptest.NewRequest(http.MethodGet, "/v3/tenants/acme/collections/docs/docs/not-a-number", nil)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("non-numeric doc id returned %d, want 404", response.Code)
	}
}

func TestCanonicalHTTPRejectsUnaddressableCollectionName(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v3/tenants/acme/collections",
		strings.NewReader(`{"name":"not/addressable","fields":[{"name":"dense","type":"dense","dim":1,"index":{"type":"flat"}}]}`),
	)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("invalid collection name returned %d: %s", response.Code, response.Body.String())
	}
}

func TestCanonicalRCSurfaceSearchAdmissionIsBounded(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	// The bounds are checked by the engine, which needs the collection to
	// exist; a missing collection is a 404 and would mask the check.
	if response := canonicalHTTPCreateCollection(t, handler, "acme", "docs", ""); response.Code/100 != 2 {
		t.Fatalf("create collection returned %d: %s", response.Code, response.Body.String())
	}
	path := "/v3/tenants/acme/collections/docs/search"
	for _, tc := range []struct {
		name string
		body string
	}{
		{
			name: "top k",
			body: fmt.Sprintf(`{"queries":{"dense":[1]},"top_k":%d}`, vcollection.MaxSearchTopK+1),
		},
		{
			name: "query fields",
			body: `{"queries":{"a":[1],"b":[1],"c":[1]},"top_k":1,"hybrid_params":{"strategy":"rrf"}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(tc.body))
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("bounded search returned %d: %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestCanonicalRCSurfaceDoesNotAcceptBearerTokenInURL(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "secret")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	embedder := NewHashEmbedder(4)
	handler, collections := newCanonicalHTTPHandler(rt, embedder, filepath.Join(t.TempDir(), "index.gob"))
	t.Cleanup(func() { _ = collections.Close() })

	request := httptest.NewRequest(http.MethodGet, "/v3/tenants/default?token=secret", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("URL bearer token returned %d, want 401", response.Code)
	}
}

type canonicalReadTracker struct {
	read bool
}

func (body *canonicalReadTracker) Read([]byte) (int, error) {
	body.read = true
	return 0, io.EOF
}

func TestCanonicalCreateRejectsUnauthorizedCallerBeforeReadingBody(t *testing.T) {
	t.Setenv("JWT_SECRET", "canonical-http-jwt-test-secret")
	t.Setenv("JWT_ISSUER", "canonical-test")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(
		rt,
		NewHashEmbedder(4),
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })

	readOnly, err := rt.jwtMgr.GenerateTenantToken("acme", []string{"read"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	crossTenantAdmin, err := rt.jwtMgr.GenerateTenantToken("other", []string{"admin"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name  string
		token string
	}{
		{name: "missing admin permission", token: readOnly},
		{name: "cross tenant admin", token: crossTenantAdmin},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := new(canonicalReadTracker)
			request := httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections", body)
			request.Header.Set("Authorization", "Bearer "+test.token)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusForbidden {
				t.Fatalf("create returned %d: %s", response.Code, response.Body.String())
			}
			if body.read {
				t.Fatal("unauthorized create request body was read")
			}
		})
	}
}

func TestCanonicalHTTPJWTTenantAdminCannotEscapeTenantOrCollectionScope(t *testing.T) {
	t.Setenv("JWT_SECRET", "canonical-http-jwt-test-secret")
	t.Setenv("JWT_ISSUER", "canonical-test")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(
		rt,
		NewHashEmbedder(4),
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })

	scopedAdmin, err := rt.jwtMgr.GenerateTenantToken(
		"acme",
		[]string{"admin"},
		[]string{"allowed"},
		time.Hour,
	)
	if err != nil {
		t.Fatal(err)
	}
	unscopedAdmin, err := rt.jwtMgr.GenerateTenantToken(
		"acme",
		[]string{"admin"},
		nil,
		time.Hour,
	)
	if err != nil {
		t.Fatal(err)
	}

	request := func(method, path, token, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		return response
	}
	schema := func(name string) string {
		return fmt.Sprintf(`{"name":%q,"fields":[{"name":"dense","type":"dense","dim":1,"index":{"type":"flat"}}]}`, name)
	}

	if response := request(http.MethodPost, "/v3/tenants/acme/collections", scopedAdmin, schema("allowed")); response.Code != http.StatusCreated {
		t.Fatalf("scoped tenant admin could not create allowed collection: %d %s", response.Code, response.Body.String())
	}
	for _, tc := range []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{name: "cross tenant", method: http.MethodGet, path: "/v3/tenants/other/collections/allowed"},
		{name: "out of collection scope", method: http.MethodPost, path: "/v3/tenants/acme/collections", body: schema("secret")},
		{name: "tenant-wide list", method: http.MethodGet, path: "/v3/tenants/acme/collections"},
		{name: "tenant-wide info", method: http.MethodGet, path: "/v3/tenants/acme"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := request(tc.method, tc.path, scopedAdmin, tc.body)
			if response.Code != http.StatusForbidden {
				t.Fatalf("scope escape returned %d: %s", response.Code, response.Body.String())
			}
		})
	}
	if response := request(http.MethodGet, "/v3/tenants/other/collections/allowed", unscopedAdmin, ""); response.Code != http.StatusForbidden {
		t.Fatalf("unscoped tenant admin crossed tenant: %d %s", response.Code, response.Body.String())
	}
	if response := request(http.MethodGet, "/v3/tenants/acme/collections", unscopedAdmin, ""); response.Code != http.StatusOK {
		t.Fatalf("unscoped tenant admin list failed: %d %s", response.Code, response.Body.String())
	}
}

func TestCanonicalHTTPJWTRejectsMissingMalformedAndExpiredCredentials(t *testing.T) {
	t.Setenv("JWT_SECRET", "canonical-http-jwt-test-secret")
	t.Setenv("JWT_ISSUER", "canonical-test")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(
		rt,
		NewHashEmbedder(4),
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })
	expired, err := rt.jwtMgr.GenerateTenantToken("acme", []string{"read"}, nil, -time.Hour)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name  string
		token string
	}{
		{name: "missing"},
		{name: "malformed", token: "not-a-jwt"},
		{name: "expired", token: expired},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "/v3/tenants/acme/collections/docs", nil)
			if tc.token != "" {
				request.Header.Set("Authorization", "Bearer "+tc.token)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("credential rejection returned %d: %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestCanonicalRCSurfaceRequiresDurablePersistence(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(rt, NewHashEmbedder(4), "")

	request := httptest.NewRequest(http.MethodGet, "/v3/tenants/acme", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("canonical route without durable store returned %d, want 503", response.Code)
	}
	request = httptest.NewRequest(http.MethodGet, "/readyz", nil)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("readiness without durable store returned %d, want 503", response.Code)
	}
	if collections.IsDurable() {
		t.Fatal("empty persistence path unexpectedly opened a durable store")
	}
}

func TestCanonicalHTTPAcknowledgementSurvivesRestart(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	indexPath := filepath.Join(t.TempDir(), "index.gob")
	newHandler := func() (http.Handler, *CollectionHTTPServer) {
		return newCanonicalHTTPHandler(testServerRuntime(t), NewHashEmbedder(4), indexPath)
	}

	handler, collections := newHandler()
	if err := collections.PersistenceError(); err != nil {
		t.Fatal(err)
	}
	schema := []byte(`{"name":"docs","fields":[{"name":"embedding","type":"dense","dim":2,"index":{"type":"flat"}}]}`)
	request := httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections", bytes.NewReader(schema))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("create returned %d: %s", response.Code, response.Body.String())
	}

	document := []byte(`{"id":77,"vectors":{"embedding":[1,0]},"metadata":{"kind":"kept"}}`)
	request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/docs/docs", bytes.NewReader(document))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("insert returned %d: %s", response.Code, response.Body.String())
	}
	journal := indexPath + ".collections.journal"
	if info, err := os.Stat(journal); err != nil || info.Size() == 0 {
		t.Fatalf("acknowledged mutation not present in journal: info=%v err=%v", info, err)
	}
	if err := collections.Close(); err != nil {
		t.Fatal(err)
	}

	handler, reopened := newHandler()
	t.Cleanup(func() { _ = reopened.Close() })
	if err := reopened.PersistenceError(); err != nil {
		t.Fatalf("reopen canonical persistence: %v", err)
	}
	search := []byte(`{"queries":{"embedding":[1,0]},"top_k":1,"include_vectors":true}`)
	request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/docs/search", bytes.NewReader(search))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("search after restart returned %d: %s", response.Code, response.Body.String())
	}
	var result tenantSearchJSONResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if len(result.Documents) != 1 || result.Documents[0].ID != 77 {
		t.Fatalf("restarted search returned %+v", result.Documents)
	}
}

// TestCanonicalTenantLifecycleHTTP drives the full tenant lifecycle with a
// static server-admin credential: create, list, suspend (data-plane writes
// blocked with a hint pointing back at reactivation), reactivate, usage
// tracking, then delete and re-delete.
func TestCanonicalTenantLifecycleHTTP(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "server-admin-token")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(rt, NewHashEmbedder(4), filepath.Join(t.TempDir(), "index.gob"))
	t.Cleanup(func() { _ = collections.Close() })

	request := func(method, path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer server-admin-token")
		req.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		return response
	}

	if response := request(http.MethodPost, "/v3/tenants", `{"tenant_id":"acme"}`); response.Code != http.StatusCreated {
		t.Fatalf("create tenant returned %d: %s", response.Code, response.Body.String())
	}

	schema := `{"name":"docs","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`
	if response := request(http.MethodPost, "/v3/tenants/acme/collections", schema); response.Code != http.StatusCreated {
		t.Fatalf("create collection returned %d: %s", response.Code, response.Body.String())
	}

	response := request(http.MethodGet, "/v3/tenants", "")
	if response.Code != http.StatusOK {
		t.Fatalf("list tenants returned %d: %s", response.Code, response.Body.String())
	}
	var list struct {
		Tenants []vcollection.TenantInfo `json:"tenants"`
	}
	if err := json.NewDecoder(response.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, info := range list.Tenants {
		if info.TenantID == "acme" {
			found = true
			if info.Status != vcollection.TenantStatusActive {
				t.Fatalf("acme status = %q, want active", info.Status)
			}
		}
	}
	if !found {
		t.Fatalf("acme missing from tenant list: %+v", list.Tenants)
	}

	if response := request(http.MethodPut, "/v3/tenants/acme", `{"status":"suspended"}`); response.Code != http.StatusOK {
		t.Fatalf("suspend tenant returned %d: %s", response.Code, response.Body.String())
	}

	doc := `{"vectors":{"dense":[0,1]}}`
	response = request(http.MethodPost, "/v3/tenants/acme/collections/docs/docs", doc)
	if response.Code != http.StatusForbidden {
		t.Fatalf("insert on suspended tenant returned %d: %s", response.Code, response.Body.String())
	}
	var apiErr apierror.Error
	if err := json.NewDecoder(response.Body).Decode(&apiErr); err != nil {
		t.Fatal(err)
	}
	if apiErr.Code != apierror.CodePermissionDenied {
		t.Fatalf("suspended insert code = %q, want permission_denied", apiErr.Code)
	}
	if !strings.Contains(apiErr.Hint, "PUT /v3/tenants/{tenant}") {
		t.Fatalf("suspended insert hint = %q, want it to mention PUT /v3/tenants/{tenant}", apiErr.Hint)
	}

	if response := request(http.MethodPut, "/v3/tenants/acme", `{"status":"active"}`); response.Code != http.StatusOK {
		t.Fatalf("reactivate tenant returned %d: %s", response.Code, response.Body.String())
	}

	if response := request(http.MethodPost, "/v3/tenants/acme/collections/docs/docs", doc); response.Code != http.StatusOK {
		t.Fatalf("insert after reactivation returned %d: %s", response.Code, response.Body.String())
	}

	response = request(http.MethodGet, "/v3/tenants/acme", "")
	if response.Code != http.StatusOK {
		t.Fatalf("tenant info returned %d: %s", response.Code, response.Body.String())
	}
	var info struct {
		Tenant vcollection.TenantInfo `json:"tenant"`
	}
	if err := json.NewDecoder(response.Body).Decode(&info); err != nil {
		t.Fatal(err)
	}
	if info.Tenant.Usage.Documents != 1 {
		t.Fatalf("tenant usage.documents = %d, want 1", info.Tenant.Usage.Documents)
	}

	if response := request(http.MethodDelete, "/v3/tenants/acme", ""); response.Code != http.StatusOK {
		t.Fatalf("delete tenant returned %d: %s", response.Code, response.Body.String())
	}

	response = request(http.MethodGet, "/v3/tenants", "")
	if response.Code != http.StatusOK {
		t.Fatalf("list tenants after delete returned %d: %s", response.Code, response.Body.String())
	}
	var afterDelete struct {
		Count int `json:"count"`
	}
	if err := json.NewDecoder(response.Body).Decode(&afterDelete); err != nil {
		t.Fatal(err)
	}
	if afterDelete.Count != 0 {
		t.Fatalf("tenant count after delete = %d, want 0", afterDelete.Count)
	}

	if response := request(http.MethodDelete, "/v3/tenants/acme", ""); response.Code != http.StatusNotFound {
		t.Fatalf("delete of already-deleted tenant returned %d, want 404: %s", response.Code, response.Body.String())
	}
}

// TestTenantUpdateHTTPPreservesOmittedFields pins the review-round-1 fix for
// the exact sequence the Python SDK's own documented usage produces
// (README.md: create with a quota, then .update(status=...) without a
// quota): PUT /v3/tenants/{tenant} with only "status" must not zero out a
// previously-set quota, and PUT with only "quota" must not reactivate a
// suspended tenant.
func TestTenantUpdateHTTPPreservesOmittedFields(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "server-admin-token")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(rt, NewHashEmbedder(4), filepath.Join(t.TempDir(), "index.gob"))
	t.Cleanup(func() { _ = collections.Close() })

	request := func(method, path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer server-admin-token")
		req.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		return response
	}
	tenantInfo := func() vcollection.TenantInfo {
		t.Helper()
		response := request(http.MethodGet, "/v3/tenants/acme", "")
		if response.Code != http.StatusOK {
			t.Fatalf("get tenant info returned %d: %s", response.Code, response.Body.String())
		}
		var info struct {
			Tenant vcollection.TenantInfo `json:"tenant"`
		}
		if err := json.NewDecoder(response.Body).Decode(&info); err != nil {
			t.Fatal(err)
		}
		return info.Tenant
	}

	if response := request(http.MethodPost, "/v3/tenants", `{"tenant_id":"acme","quota":{"max_documents":1000}}`); response.Code != http.StatusCreated {
		t.Fatalf("create tenant returned %d: %s", response.Code, response.Body.String())
	}

	// SDK's client.tenant("acme").update(status="suspended") sends exactly
	// this body: status only, no "quota" key at all.
	if response := request(http.MethodPut, "/v3/tenants/acme", `{"status":"suspended"}`); response.Code != http.StatusOK {
		t.Fatalf("status-only update returned %d: %s", response.Code, response.Body.String())
	}
	info := tenantInfo()
	if info.Status != vcollection.TenantStatusSuspended {
		t.Fatalf("status after status-only update = %q, want suspended", info.Status)
	}
	if info.Quota.MaxDocuments != 1000 {
		t.Fatalf("quota after status-only update = %+v, want max_documents=1000 (unchanged)", info.Quota)
	}

	// Quota-only update must not reactivate the tenant it's scoped to.
	if response := request(http.MethodPut, "/v3/tenants/acme", `{"quota":{"max_documents":500}}`); response.Code != http.StatusOK {
		t.Fatalf("quota-only update returned %d: %s", response.Code, response.Body.String())
	}
	info = tenantInfo()
	if info.Status != vcollection.TenantStatusSuspended {
		t.Fatalf("status after quota-only update = %q, want suspended (unchanged)", info.Status)
	}
	if info.Quota.MaxDocuments != 500 {
		t.Fatalf("quota after quota-only update = %+v, want max_documents=500", info.Quota)
	}
}

// TestCanonicalTenantLifecycleRequiresServerAdmin checks that a tenant-scoped
// admin JWT — the credential that manages one tenant's collections — cannot
// touch tenant lifecycle routes; only the server_admin claim can.
func TestCanonicalTenantLifecycleRequiresServerAdmin(t *testing.T) {
	t.Setenv("JWT_SECRET", "canonical-http-jwt-test-secret")
	t.Setenv("JWT_ISSUER", "canonical-test")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(rt, NewHashEmbedder(4), filepath.Join(t.TempDir(), "index.gob"))
	t.Cleanup(func() { _ = collections.Close() })

	tenantAdmin, err := rt.jwtMgr.SignTenantClaims(security.TenantClaims{
		TenantID:    "acme",
		Permissions: []string{"admin"},
	}, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	serverAdmin, err := rt.jwtMgr.SignTenantClaims(security.TenantClaims{
		TenantID:    "acme",
		Permissions: []string{"admin"},
		ServerAdmin: true,
	}, time.Hour)
	if err != nil {
		t.Fatal(err)
	}

	request := func(method, path, token, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		return response
	}

	for _, tc := range []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{name: "create", method: http.MethodPost, path: "/v3/tenants", body: `{"tenant_id":"acme"}`},
		{name: "list", method: http.MethodGet, path: "/v3/tenants"},
		{name: "update", method: http.MethodPut, path: "/v3/tenants/acme", body: `{"status":"active"}`},
		{name: "delete", method: http.MethodDelete, path: "/v3/tenants/acme"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := request(tc.method, tc.path, tenantAdmin, tc.body)
			if response.Code != http.StatusForbidden {
				t.Fatalf("tenant admin without server_admin returned %d, want 403: %s", response.Code, response.Body.String())
			}
		})
	}

	if response := request(http.MethodPost, "/v3/tenants", serverAdmin, `{"tenant_id":"acme"}`); response.Code != http.StatusCreated {
		t.Fatalf("server_admin create tenant returned %d: %s", response.Code, response.Body.String())
	}
}

func TestCanonicalGRPCDescriptorExcludesAdvancedMethods(t *testing.T) {
	if got := deepdatav3.DeepData_ServiceDesc.ServiceName; got != "deepdata.v3.DeepData" {
		t.Fatalf("canonical gRPC service name = %q, want deepdata.v3.DeepData", got)
	}
	want := map[string]bool{
		"GetTenantInfo":    true,
		"CreateTenant":     true,
		"ListTenants":      true,
		"UpdateTenant":     true,
		"DeleteTenant":     true,
		"ListCollections":  true,
		"GetCollection":    true,
		"CreateCollection": true,
		"DeleteCollection": true,
		"Insert":           true,
		"BatchInsert":      true,
		"Search":           true,
		"DeleteDoc":        true,
		"Upsert":           true,
		"GetDoc":           true,
	}
	for _, method := range deepdatav3.DeepData_ServiceDesc.Methods {
		if !want[method.MethodName] {
			t.Fatalf("unsupported gRPC method remains registered: %s", method.MethodName)
		}
		delete(want, method.MethodName)
	}
	if len(want) != 0 {
		t.Fatalf("canonical gRPC methods missing from descriptor: %v", want)
	}
}
