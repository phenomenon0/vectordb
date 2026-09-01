package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/security"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func canonicalResourceSchema(name string) vcollection.CollectionSchema {
	return vcollection.CollectionSchema{
		Name: name,
		Fields: []vcollection.VectorField{{
			Name:  "embedding",
			Type:  vcollection.VectorTypeDense,
			Dim:   2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}
}

func canonicalHTTPCreateCollection(t *testing.T, handler http.Handler, tenantID, name, token string) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(canonicalResourceSchema(name))
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v3/tenants/"+tenantID+"/collections", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func canonicalGRPCCreateCollectionRequest(tenantID, name string) *deepdatav3.CreateCollectionRequest {
	return &deepdatav3.CreateCollectionRequest{
		TenantId: tenantID,
		Name:     name,
		Fields: []*deepdatav3.VectorFieldConfig{{
			Name:      "embedding",
			Type:      int32(vcollection.VectorTypeDense),
			Dim:       2,
			IndexType: "flat",
		}},
	}
}

func TestCanonicalDurableLimitsAreSharedAcrossHTTPAndGRPC(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	t.Setenv("MAX_TENANTS", "1")
	t.Setenv("MAX_COLLECTIONS", "2")
	t.Setenv("TENANT_RPS", "100")
	t.Setenv("TENANT_BURST", "100")

	rt := newServerRuntime()
	handler, collections := newCanonicalHTTPHandler(
		rt,
		NewHashEmbedder(4),
		nil,
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })

	response := canonicalHTTPCreateCollection(t, handler, "one", "first", "")
	if response.Code != http.StatusCreated {
		t.Fatalf("HTTP create returned %d: %s", response.Code, response.Body.String())
	}

	grpcServer := &CollectionGRPCServer{
		tenants:           collections.TenantManager(),
		persistenceHealth: collections.PersistenceError,
	}
	if _, err := grpcServer.CreateCollection(
		canonicalGRPCAdminContext("two"),
		canonicalGRPCCreateCollectionRequest("two", "blocked"),
	); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("second tenant gRPC create error = %v, want FailedPrecondition (a fixed limit, not a retryable exhaustion)", err)
	}
	if _, err := grpcServer.CreateCollection(
		canonicalGRPCAdminContext("one"),
		canonicalGRPCCreateCollectionRequest("one", "second"),
	); err != nil {
		t.Fatalf("same-tenant gRPC create failed: %v", err)
	}

	response = canonicalHTTPCreateCollection(t, handler, "one", "third", "")
	if response.Code != http.StatusConflict {
		t.Fatalf("N+1 HTTP collection create returned %d, want 409 (quota_exceeded is permanent, never 429): %s", response.Code, response.Body.String())
	}
}

func TestCanonicalTenantRateLimitIsSharedAcrossHTTPAndGRPCJWTs(t *testing.T) {
	t.Setenv("JWT_SECRET", "canonical-shared-tenant-rate-limit-secret")
	t.Setenv("JWT_ISSUER", "canonical-resource-test")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "1")
	t.Setenv("TENANT_RPS", "1")
	t.Setenv("TENANT_BURST", "1")
	t.Setenv("MAX_TENANTS", "1")
	t.Setenv("MAX_RATE_LIMIT_KEYS", "100")

	rt := newServerRuntime()
	handler, collections := newCanonicalHTTPHandler(
		rt,
		NewHashEmbedder(4),
		nil,
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })

	firstToken, err := rt.jwtMgr.GenerateTenantToken("acme", []string{"admin"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	secondToken, err := rt.jwtMgr.GenerateTenantToken("acme", []string{"admin", "read"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	otherToken, err := rt.jwtMgr.GenerateTenantToken("other", []string{"read"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/v3/tenants/acme", nil)
	request.Header.Set("Authorization", "Bearer "+firstToken)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("first authorized HTTP request returned %d: %s", response.Code, response.Body.String())
	}

	interceptor := grpcAuthInterceptorWithTenantLimiter(
		rt.jwtMgr,
		"",
		true,
		testLogger(),
		rt.canonicalTenantRL,
	)
	grpcContext := metadata.NewIncomingContext(
		context.Background(),
		metadata.Pairs("authorization", "Bearer "+secondToken),
	)
	if _, err := interceptor(grpcContext, nil, dummyServerInfo("/deepdata.v3.DeepData/GetTenantInfo"), passThroughHandler); status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("second JWT for same tenant error = %v, want ResourceExhausted", err)
	}

	grpcContext = metadata.NewIncomingContext(
		context.Background(),
		metadata.Pairs("authorization", "Bearer "+otherToken),
	)
	if _, err := interceptor(grpcContext, nil, dummyServerInfo("/deepdata.v3.DeepData/GetTenantInfo"), passThroughHandler); err != nil {
		t.Fatalf("different tenant was not independently admitted: %v", err)
	}
}

func TestCanonicalResponseBudgetErrorMappings(t *testing.T) {
	err := fmt.Errorf("search admission: %w", vcollection.ErrSearchResponseBudgetExceeded)
	response := httptest.NewRecorder()
	writeCanonicalOperationError(response, err, apierror.CodeInternal)
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("HTTP response budget error mapped to %d, want 413", response.Code)
	}
	if got := status.Code(canonicalGRPCError(context.Background(), err, apierror.CodeInternal)); got != codes.ResourceExhausted {
		t.Fatalf("gRPC response budget error mapped to %v, want ResourceExhausted", got)
	}
}

func TestCanonicalSearchResponseBudgetAcrossHTTPAndGRPC(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	t.Setenv("TENANT_RPS", "100")
	t.Setenv("TENANT_BURST", "100")
	t.Setenv("MAX_TENANTS", "10")
	t.Setenv("MAX_COLLECTIONS", "10")

	rt := newServerRuntime()
	handler, collections := newCanonicalHTTPHandler(
		rt,
		NewHashEmbedder(4),
		nil,
		filepath.Join(t.TempDir(), "index.gob"),
	)
	t.Cleanup(func() { _ = collections.Close() })

	schema := canonicalResourceSchema("wide")
	schema.Fields[0].Dim = vcollection.CanonicalMaxVectorDimension
	createBody, err := json.Marshal(schema)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections", bytes.NewReader(createBody))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("create maximum-dimension collection returned %d: %s", response.Code, response.Body.String())
	}

	vector := make([]float32, vcollection.CanonicalMaxVectorDimension)
	vector[0] = 1
	insertBody, err := json.Marshal(map[string]interface{}{
		"id":       1,
		"vectors":  map[string]interface{}{"embedding": vector},
		"metadata": map[string]interface{}{"kind": "wide"},
	})
	if err != nil {
		t.Fatal(err)
	}
	request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/wide/docs", bytes.NewReader(insertBody))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("insert maximum-dimension document returned %d: %s", response.Code, response.Body.String())
	}

	searchBody := func(includeVectors bool) []byte {
		body, err := json.Marshal(map[string]interface{}{
			"queries":         map[string]interface{}{"embedding": vector},
			"top_k":           vcollection.CanonicalMaxSearchTopK,
			"include_vectors": includeVectors,
		})
		if err != nil {
			t.Fatal(err)
		}
		return body
	}
	request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/wide/search", bytes.NewReader(searchBody(true)))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("HTTP include_vectors search returned %d: %s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodPost, "/v3/tenants/acme/collections/wide/search", bytes.NewReader(searchBody(false)))
	request.Header.Set("Content-Type", "application/json")
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("HTTP no-vectors search returned %d: %s", response.Code, response.Body.String())
	}
	var httpSearch struct {
		Documents []vcollection.Document `json:"documents"`
	}
	if err := json.NewDecoder(response.Body).Decode(&httpSearch); err != nil {
		t.Fatal(err)
	}
	if len(httpSearch.Documents) != 1 || httpSearch.Documents[0].Vectors != nil {
		t.Fatalf("HTTP no-vectors response materialized vectors: %+v", httpSearch.Documents)
	}

	grpcServer := &CollectionGRPCServer{
		tenants:           collections.TenantManager(),
		persistenceHealth: collections.PersistenceError,
	}
	grpcRequest := &deepdatav3.SearchRequest{
		TenantId:       "acme",
		Collection:     "wide",
		TopK:           int32(vcollection.CanonicalMaxSearchTopK),
		IncludeVectors: true,
		Queries: map[string]*deepdatav3.VectorData{
			"embedding": denseProtoVector(vector...),
		},
	}
	if _, err := grpcServer.Search(canonicalGRPCAdminContext("acme"), grpcRequest); status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("gRPC include_vectors search error = %v, want ResourceExhausted", err)
	}
	grpcRequest.IncludeVectors = false
	grpcSearch, err := grpcServer.Search(canonicalGRPCAdminContext("acme"), grpcRequest)
	if err != nil {
		t.Fatalf("gRPC no-vectors search failed: %v", err)
	}
	if len(grpcSearch.Results) != 1 || len(grpcSearch.Results[0].Vectors) != 0 {
		t.Fatalf("gRPC no-vectors response materialized vectors: %+v", grpcSearch.Results)
	}
}

func TestCanonicalResourceControlEnvironmentFailsFast(t *testing.T) {
	for _, key := range []string{
		"TENANT_RPS",
		"TENANT_BURST",
		"MAX_TENANTS",
		"MAX_COLLECTIONS",
		"MAX_RATE_LIMIT_KEYS",
	} {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, "0")
			errs := validateEnvConfig(testLogger())
			found := false
			for _, err := range errs {
				if strings.HasPrefix(err, key+"=") {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("validateEnvConfig did not reject %s=0: %v", key, errs)
			}
		})
	}
}

func TestCanonicalRateLimitTenantUsesAdminTargetWithoutJWTClaimEscape(t *testing.T) {
	serverAdmin := &security.TenantContext{TenantID: "default", IsServerAdmin: true}
	if got := canonicalRateLimitTenant(serverAdmin, "acme"); got != "acme" {
		t.Fatalf("server-admin rate key = %q, want acme", got)
	}

	tenantJWT := &security.TenantContext{TenantID: "acme", IsAdmin: true}
	if got := canonicalRateLimitTenant(tenantJWT, "other"); got != "acme" {
		t.Fatalf("tenant JWT escaped to target rate key %q, want acme", got)
	}
}
