package main

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/security"
)

// metricLineWith returns the first exposition line for family whose label
// set contains every substring in subs, or ok=false.
func metricLineWith(body, family string, subs ...string) (string, bool) {
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, family+"{") {
			continue
		}
		match := true
		for _, s := range subs {
			if !strings.Contains(line, s) {
				match = false
				break
			}
		}
		if match {
			return line, true
		}
	}
	return "", false
}

// TestTenantMetricsAreEmitted pins that the per-tenant metrics wiring
// actually produces samples: creating a collection and inserting a document
// under tenant "acme" must show up in the /metrics exposition, both as a
// tagged request count and as the tenant's document gauge.
func TestTenantMetricsAreEmitted(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	call := func(method, path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, req)
		return resp
	}

	schema := `{"name":"docs","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`
	if resp := call(http.MethodPost, "/v3/tenants/acme/collections", schema); resp.Code != http.StatusCreated {
		t.Fatalf("create collection returned %d: %s", resp.Code, resp.Body.String())
	}

	doc := `{"vectors":{"dense":[0,1]}}`
	insertResp := call(http.MethodPost, "/v3/tenants/acme/collections/docs/docs", doc)
	if insertResp.Code != http.StatusOK {
		t.Fatalf("insert document returned %d: %s", insertResp.Code, insertResp.Body.String())
	}

	metricsResp := call(http.MethodGet, "/metrics", "")
	if metricsResp.Code != http.StatusOK {
		t.Fatalf("GET /metrics returned %d: %s", metricsResp.Code, metricsResp.Body.String())
	}
	body := metricsResp.Body.String()

	// Literal, not normalizeMetricsPath(insertPath): the point is to pin the
	// wire label, not to re-derive it from the function under test.
	const wantOperation = "/v3/tenants/:id/collections/:name/docs"
	if _, ok := metricLineWith(body, "vectordb_tenant_requests_total",
		`tenant="acme"`,
		`operation="`+wantOperation+`"`,
		`code="`+strconv.Itoa(http.StatusOK)+`"`,
	); !ok {
		t.Fatalf("no vectordb_tenant_requests_total sample for tenant=acme operation=%s code=200:\n%s", wantOperation, body)
	}

	line, ok := metricLineWith(body, "vectordb_tenant_documents", `tenant="acme"`)
	if !ok {
		t.Fatalf("no vectordb_tenant_documents{tenant=\"acme\"} sample:\n%s", body)
	}
	if !strings.HasSuffix(strings.TrimSpace(line), " 1") {
		t.Fatalf("vectordb_tenant_documents{tenant=\"acme\"} sample = %q, want value 1", line)
	}
}

// TestMetricsRequiresServerAdmin pins the review-round-1 fix: /metrics
// exposes vectordb_tenant_documents/vectordb_tenant_requests_total for every
// tenant on the server, so an authenticated-but-merely-tenant-scoped caller
// must not be able to read it — only the server-administrator credential can.
func TestMetricsRequiresServerAdmin(t *testing.T) {
	t.Setenv("JWT_SECRET", "canonical-http-jwt-metrics-test-secret")
	t.Setenv("JWT_ISSUER", "canonical-test")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "1")
	rt := testServerRuntime(t)
	handler, collections := newCanonicalHTTPHandler(rt, NewHashEmbedder(4), filepath.Join(t.TempDir(), "index.gob"))
	t.Cleanup(func() { _ = collections.Close() })

	tenantScoped, err := rt.jwtMgr.SignTenantClaims(security.TenantClaims{
		TenantID:    "acme",
		Permissions: []string{"read", "write"},
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

	get := func(token string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
		req.Header.Set("Authorization", "Bearer "+token)
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, req)
		return resp
	}

	if resp := get(tenantScoped); resp.Code != http.StatusForbidden {
		t.Fatalf("tenant-scoped GET /metrics returned %d, want 403: %s", resp.Code, resp.Body.String())
	}
	if resp := get(serverAdmin); resp.Code != http.StatusOK {
		t.Fatalf("server-admin GET /metrics returned %d, want 200: %s", resp.Code, resp.Body.String())
	}
}
