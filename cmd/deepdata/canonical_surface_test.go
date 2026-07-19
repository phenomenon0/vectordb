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

	deepdatav1 "github.com/phenomenon0/vectordb/api/gen/deepdata/v1"
	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

func newCanonicalSurfaceTestHandler(t *testing.T) http.Handler {
	t.Helper()
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	store := NewVectorStore(8, 4)
	embedder := NewHashEmbedder(4)
	indexPath := filepath.Join(t.TempDir(), "index.gob")
	handler, collections := newCanonicalHTTPHandler(store, embedder, nil, indexPath)
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
	path := "/v3/tenants/acme/collections/docs/search"
	for _, tc := range []struct {
		name string
		body string
	}{
		{
			name: "top k",
			body: fmt.Sprintf(`{"queries":{"dense":[1]},"top_k":%d}`, vcollection.CanonicalMaxSearchTopK+1),
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
	store := NewVectorStore(8, 4)
	embedder := NewHashEmbedder(4)
	handler, collections := newCanonicalHTTPHandler(store, embedder, nil, filepath.Join(t.TempDir(), "index.gob"))
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
	store := NewVectorStore(8, 4)
	handler, collections := newCanonicalHTTPHandler(
		store,
		NewHashEmbedder(4),
		nil,
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })

	readOnly, err := store.jwtMgr.GenerateTenantToken("acme", []string{"read"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	crossTenantAdmin, err := store.jwtMgr.GenerateTenantToken("other", []string{"admin"}, nil, time.Hour)
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
	store := NewVectorStore(8, 4)
	handler, collections := newCanonicalHTTPHandler(
		store,
		NewHashEmbedder(4),
		nil,
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })

	scopedAdmin, err := store.jwtMgr.GenerateTenantToken(
		"acme",
		[]string{"admin"},
		[]string{"allowed"},
		time.Hour,
	)
	if err != nil {
		t.Fatal(err)
	}
	unscopedAdmin, err := store.jwtMgr.GenerateTenantToken(
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
	store := NewVectorStore(8, 4)
	handler, collections := newCanonicalHTTPHandler(
		store,
		NewHashEmbedder(4),
		nil,
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })
	expired, err := store.jwtMgr.GenerateTenantToken("acme", []string{"read"}, nil, -time.Hour)
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
	store := NewVectorStore(8, 4)
	handler, collections := newCanonicalHTTPHandler(store, NewHashEmbedder(4), nil, "")

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
		return newCanonicalHTTPHandler(NewVectorStore(8, 4), NewHashEmbedder(4), nil, indexPath)
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

func TestCanonicalGRPCDescriptorExcludesAdvancedMethods(t *testing.T) {
	if got := deepdatav3.DeepData_ServiceDesc.ServiceName; got != "deepdata.v3.DeepData" {
		t.Fatalf("canonical gRPC service name = %q, want deepdata.v3.DeepData", got)
	}
	want := map[string]bool{
		"GetTenantInfo":    true,
		"ListCollections":  true,
		"GetCollection":    true,
		"CreateCollection": true,
		"DeleteCollection": true,
		"Insert":           true,
		"BatchInsert":      true,
		"Search":           true,
		"DeleteDoc":        true,
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

func TestHistoricalV1GRPCDescriptorRemainsFrozen(t *testing.T) {
	if got := deepdatav1.DeepData_ServiceDesc.ServiceName; got != "deepdata.v1.DeepData" {
		t.Fatalf("historical gRPC service name = %q, want deepdata.v1.DeepData", got)
	}
	wantMethods := map[string]bool{
		"CreateCollection": true,
		"DeleteCollection": true,
		"Insert":           true,
		"BatchInsert":      true,
		"Search":           true,
		"DeleteDoc":        true,
		"Recommend":        true,
		"Discover":         true,
	}
	for _, method := range deepdatav1.DeepData_ServiceDesc.Methods {
		if !wantMethods[method.MethodName] {
			t.Fatalf("unexpected historical v1 method: %s", method.MethodName)
		}
		delete(wantMethods, method.MethodName)
	}
	if len(wantMethods) != 0 {
		t.Fatalf("historical v1 methods missing: %v", wantMethods)
	}

	indexParams := (&deepdatav1.VectorFieldConfig{}).ProtoReflect().Descriptor().Fields().ByName("index_params")
	metadata := (&deepdatav1.InsertRequest{}).ProtoReflect().Descriptor().Fields().ByName("metadata")
	textField := (&deepdatav1.InsertRequest{}).ProtoReflect().Descriptor().Fields().ByNumber(4)
	if indexParams == nil || !indexParams.IsMap() || metadata == nil || !metadata.IsMap() || textField == nil || textField.Name() != "text" {
		t.Fatal("historical v1 map fields or text field changed wire shape")
	}
}
