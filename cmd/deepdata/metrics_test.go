package main

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
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
