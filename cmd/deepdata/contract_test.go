package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/api/contract"
	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// ── the operation list is the source ─────────────────────────────────────

func testOperations(t *testing.T) []contract.Operation {
	t.Helper()
	ops, err := contract.Operations()
	if err != nil {
		t.Fatal(err)
	}
	if len(ops) == 0 {
		t.Fatal("operations.json is empty")
	}
	return ops
}

// The routes subcommand is the only rendering of the operation list, so the
// docs linter's generated table (DOC-03) can never say something the
// contract does not.
func TestRoutesSubcommandPrintsTheOperationList(t *testing.T) {
	var out bytes.Buffer
	if err := printRoutes(&out); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimRight(out.String(), "\n"), "\n")
	ops := testOperations(t)
	if len(lines) != len(ops)+1 {
		t.Fatalf("routes printed %d lines, want %d operations plus a header", len(lines), len(ops))
	}
	if lines[0] != "method\tpath\tpermission\tgrpc_rpc" {
		t.Fatalf("header = %q", lines[0])
	}
	for i, op := range ops {
		want := fmt.Sprintf("%s\t%s\t%s\t%s", op.Method, op.Path, op.Permission, op.GRPCRPC)
		if lines[i+1] != want {
			t.Fatalf("line %d = %q, want %q", i+1, lines[i+1], want)
		}
	}
}

// Every operation names a gRPC method that exists, and every RPC the service
// exposes is reachable over HTTP. A drift in either direction means one
// transport gained a call the other cannot make.
func TestOperationsMatchTheProtoServiceMethods(t *testing.T) {
	service := deepdatav3.File_deepdata_v3_deepdata_proto.Services().ByName("DeepData")
	if service == nil {
		t.Fatal("service deepdata.v3.DeepData not found in the descriptor")
	}
	proto := map[string]bool{}
	for i := 0; i < service.Methods().Len(); i++ {
		proto[string(service.Methods().Get(i).Name())] = true
	}
	listed := map[string]bool{}
	for _, op := range testOperations(t) {
		if op.GRPCRPC == "" {
			continue
		}
		if !proto[op.GRPCRPC] {
			t.Errorf("operation %s names grpc_rpc %q, which the service does not expose", op.Name, op.GRPCRPC)
		}
		listed[op.GRPCRPC] = true
	}
	for rpc := range proto {
		if !listed[rpc] {
			t.Errorf("rpc %s has no HTTP operation in operations.json", rpc)
		}
	}
}

func TestOperationNamesAreUniqueAndWellFormed(t *testing.T) {
	seen := map[string]bool{}
	permissions := map[string]bool{"read": true, "write": true, "admin": true}
	for _, op := range testOperations(t) {
		if seen[op.Name] {
			t.Errorf("duplicate operation name %q", op.Name)
		}
		seen[op.Name] = true
		if !strings.HasPrefix(op.Path, "/v3/") {
			t.Errorf("operation %s serves %q, which is not on the v3 surface", op.Name, op.Path)
		}
		if !permissions[op.Permission] {
			t.Errorf("operation %s requires unknown permission %q", op.Name, op.Permission)
		}
		if op.MCPTool != "" {
			found := false
			for _, name := range contract.Names() {
				if name == op.MCPTool {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("operation %s names mcp_tool %q, which has no contract schema", op.Name, op.MCPTool)
			}
		}
	}
}

// ── the dispatcher serves exactly the listed operations ──────────────────

// Each operation is routed through the canonical handler in an order that
// leaves the fixture usable; a 404 means the dispatcher does not serve a
// route the contract advertises. Paths absent from the list must 404.
func TestCanonicalDispatcherServesEveryListedOperation(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	call := func(method, path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, req)
		return resp
	}
	create := `{"name":"docs","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`
	if resp := call(http.MethodPost, "/v3/tenants/acme/collections", create); resp.Code != http.StatusCreated {
		t.Fatalf("fixture create returned %d: %s", resp.Code, resp.Body.String())
	}
	if resp := call(http.MethodPut, "/v3/tenants/acme/collections/docs/docs/1", `{"vectors":{"dense":[1,0]}}`); resp.Code != http.StatusOK {
		t.Fatalf("fixture upsert returned %d: %s", resp.Code, resp.Body.String())
	}

	// name → the request this operation is exercised with, in an order that
	// keeps the fixture alive (deletes last).
	bodies := map[string]string{
		"create_collection": `{"name":"other","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`,
		"insert_doc":        `{"vectors":{"dense":[0,1]}}`,
		"batch_insert_docs": `{"documents":[{"vectors":{"dense":[1,1]}}]}`,
		"upsert_doc":        `{"vectors":{"dense":[1,0]}}`,
		"delete_doc":        `{"doc_id":1}`,
		"search":            `{"queries":{"dense":[1,0]},"top_k":1}`,
	}
	order := map[string]int{"delete_doc": 1, "delete_collection": 2}
	ops := testOperations(t)
	sort.SliceStable(ops, func(i, j int) bool { return order[ops[i].Name] < order[ops[j].Name] })

	replacer := strings.NewReplacer("{tenant}", "acme", "{collection}", "docs", "{doc_id}", "1")
	for _, op := range ops {
		path := replacer.Replace(op.Path)
		resp := call(op.Method, path, bodies[op.Name])
		if resp.Code == http.StatusNotFound || resp.Code == http.StatusMethodNotAllowed {
			t.Errorf("%s %s (%s) returned %d: %s", op.Method, path, op.Name, resp.Code, resp.Body.String())
		}
	}

	for _, path := range []string{"/v3/nope", "/v3/tenants/acme/collections/docs/explain", "/v3/statuses"} {
		if resp := call(http.MethodGet, path, ""); resp.Code != http.StatusNotFound {
			t.Errorf("unlisted %s returned %d, want 404", path, resp.Code)
		}
	}
}

// ── limits and json tags against the contract schemas ────────────────────

func schemaSection(t *testing.T, tool, section string) map[string]any {
	t.Helper()
	raw, err := contract.Schema(tool)
	if err != nil {
		t.Fatal(err)
	}
	var pair map[string]json.RawMessage
	if err := json.Unmarshal(raw, &pair); err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(pair[section], &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func schemaProperties(t *testing.T, node map[string]any) map[string]any {
	t.Helper()
	props, _ := node["properties"].(map[string]any)
	if props == nil {
		t.Fatalf("node has no properties: %v", node)
	}
	return props
}

// The agent-facing caps are deliberately tighter than the engine's, but they
// may never exceed it: a schema that advertises more than the engine accepts
// invites a 400 the agent cannot see coming.
func TestContractCapsStayWithinCanonicalLimits(t *testing.T) {
	recall := schemaProperties(t, schemaSection(t, "deepdata_recall", "input"))
	topK, _ := recall["top_k"].(map[string]any)
	maximum, _ := topK["maximum"].(float64)
	if maximum <= 0 || int(maximum) > vcollection.CanonicalMaxSearchTopK {
		t.Errorf("recall top_k maximum %v exceeds CanonicalMaxSearchTopK %d", maximum, vcollection.CanonicalMaxSearchTopK)
	}

	remember := schemaProperties(t, schemaSection(t, "deepdata_remember", "input"))
	items, _ := remember["items"].(map[string]any)
	maxItems, _ := items["maxItems"].(float64)
	if maxItems <= 0 || int(maxItems) > vcollection.CanonicalMaxBatchDocuments {
		t.Errorf("remember maxItems %v exceeds CanonicalMaxBatchDocuments %d", maxItems, vcollection.CanonicalMaxBatchDocuments)
	}

	// The operator set /v3/status advertises is the one the recall schema
	// documents; the parser has no registry, so this is the only thing
	// holding the two lists together.
	defs, _ := schemaSection(t, "deepdata_recall", "input")["$defs"].(map[string]any)
	filterDef, _ := defs["filter"].(map[string]any)
	described, _ := filterDef["description"].(string)
	if len(canonicalFilterOperators) != 15 {
		t.Errorf("status reports %d filter operators, want the 15 filter.Op* constants", len(canonicalFilterOperators))
	}
	for _, op := range canonicalFilterOperators {
		if !strings.Contains(described, op) {
			t.Errorf("status advertises filter operator %q, which the recall schema does not document", op)
		}
	}
}

// jsonTags returns the json names a struct serializes, minus "-".
func jsonTags(t *testing.T, v any) []string {
	t.Helper()
	rt := reflect.TypeOf(v)
	var names []string
	for i := 0; i < rt.NumField(); i++ {
		tag := rt.Field(i).Tag.Get("json")
		if tag == "" || tag == "-" {
			if rt.Field(i).Anonymous {
				names = append(names, jsonTags(t, reflect.New(rt.Field(i).Type).Elem().Interface())...)
			}
			continue
		}
		names = append(names, strings.Split(tag, ",")[0])
	}
	return names
}

// Every json name the transports and the engine emit is either a property of
// the agent-facing schema or an explicitly listed transport concern. The list
// is the documentation: a new field lands in one column or the other, never
// silently.
func TestSearchAndCollectionJSONTagsAreCoveredByTheContract(t *testing.T) {
	recallOut := schemaProperties(t, schemaSection(t, "deepdata_recall", "output"))
	// The HTTP/gRPC search answer is richer than the MCP projection: it
	// names the tenant, returns whole documents rather than elided hits,
	// and carries the operational fields MCP does not forward.
	searchTransportOnly := map[string]string{
		"status":              "HTTP envelope",
		"tenant_id":           "HTTP envelope",
		"documents":           "projected onto hits[]",
		"scores":              "projected onto hits[].score",
		"query_time_ms":       "operational, not forwarded over MCP",
		"candidates_examined": "operational, not forwarded over MCP",
		"request_id":          "operational, not forwarded over MCP",
	}
	for _, name := range jsonTags(t, tenantSearchJSONResponse{}) {
		if _, ok := recallOut[name]; ok {
			continue
		}
		if _, ok := searchTransportOnly[name]; !ok {
			t.Errorf("search response field %q is neither in the recall output schema nor a listed transport concern", name)
		}
	}
	if _, ok := recallOut["score_direction"]; !ok {
		t.Error("recall output schema must publish score_direction")
	}

	collectionsOut := schemaProperties(t, schemaSection(t, "deepdata_collections", "output"))
	collections, _ := collectionsOut["collections"].(map[string]any)
	item, _ := collections["items"].(map[string]any)
	itemProps := schemaProperties(t, item)
	infoOnly := map[string]string{"metadata": "free-form; the schema leaves it to additionalProperties"}
	for _, name := range jsonTags(t, vcollection.CollectionInfo{}) {
		if _, ok := itemProps[name]; ok {
			continue
		}
		if _, ok := infoOnly[name]; !ok {
			t.Errorf("CollectionInfo field %q is not a property of the collections output schema", name)
		}
	}
	fields, _ := itemProps["fields"].(map[string]any)
	fieldItem, _ := fields["items"].(map[string]any)
	fieldProps := schemaProperties(t, fieldItem)
	for _, name := range jsonTags(t, vcollection.FieldInfo{}) {
		if _, ok := fieldProps[name]; !ok {
			t.Errorf("FieldInfo field %q is not a property of the collections output schema", name)
		}
	}
}

// ── GET /v3/status ───────────────────────────────────────────────────────

func TestStatusDescribesTheServerFromTheContract(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/v3/status", nil)
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("status returned %d: %s", resp.Code, resp.Body.String())
	}
	var body struct {
		Version  string `json:"version"`
		Contract struct {
			HTTP []struct {
				Method     string `json:"method"`
				Path       string `json:"path"`
				Permission string `json:"permission"`
			} `json:"http"`
			GRPC []string `json:"grpc"`
			MCP  []string `json:"mcp"`
		} `json:"contract"`
		Embedding struct {
			Provider  string `json:"provider"`
			Model     string `json:"model"`
			Dim       int    `json:"dim"`
			Available bool   `json:"available"`
		} `json:"embedding"`
		Limits       map[string]any `json:"limits"`
		Capabilities struct {
			Texts      bool     `json:"texts"`
			Filters    []string `json:"filters"`
			Hybrid     []string `json:"hybrid"`
			Fallback   bool     `json:"fallback"`
			UsageBoost bool     `json:"usage_boost"`
			IndexTypes []string `json:"index_types"`
		} `json:"capabilities"`
		Signals struct {
			Usage struct {
				Loaded *bool `json:"loaded"`
			} `json:"usage"`
		} `json:"signals"`
		RequestID string `json:"request_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Version == "" {
		t.Error("status must name the build version")
	}
	ops := testOperations(t)
	if len(body.Contract.HTTP) != len(ops) {
		t.Errorf("status lists %d HTTP operations, operations.json has %d", len(body.Contract.HTTP), len(ops))
	}
	for i, op := range ops {
		got := body.Contract.HTTP[i]
		if got.Method != op.Method || got.Path != op.Path || got.Permission != op.Permission {
			t.Errorf("status operation %d = %+v, want %s %s %s", i, got, op.Method, op.Path, op.Permission)
		}
	}
	if len(body.Contract.GRPC) == 0 || len(body.Contract.MCP) == 0 {
		t.Errorf("status must project the gRPC and MCP surfaces: %+v", body.Contract)
	}
	// newCanonicalSurfaceTestHandler runs with DEEPDATA_EMBEDDER unset, so
	// the honest answer is "no embedder, so no texts".
	if body.Embedding.Available || body.Embedding.Provider != "none" || body.Capabilities.Texts {
		t.Errorf("embedding = %+v, texts = %v; want none/false without DEEPDATA_EMBEDDER", body.Embedding, body.Capabilities.Texts)
	}
	if got := body.Limits["max_search_top_k"]; got != float64(vcollection.CanonicalMaxSearchTopK) {
		t.Errorf("limits.max_search_top_k = %v, want %d", got, vcollection.CanonicalMaxSearchTopK)
	}
	if len(body.Capabilities.Filters) != 15 {
		t.Errorf("capabilities.filters = %v, want the 15 filter operators", body.Capabilities.Filters)
	}
	if !reflect.DeepEqual(body.Capabilities.Hybrid, []string{"rrf", "weighted", "linear"}) {
		t.Errorf("capabilities.hybrid = %v", body.Capabilities.Hybrid)
	}
	if !reflect.DeepEqual(body.Capabilities.IndexTypes, []string{"hnsw", "flat", "inverted"}) {
		t.Errorf("capabilities.index_types = %v", body.Capabilities.IndexTypes)
	}
	if !body.Capabilities.Fallback || !body.Capabilities.UsageBoost {
		t.Error("fallback and usage_boost are engine capabilities and must be reported")
	}
	// signals.usage.loaded is always present, so an operator never has to
	// tell "no ranking hints were lost" apart from "this build cannot say".
	// This handler opens a fresh store with no sidecar to discard, so the
	// honest answer is true.
	if body.Signals.Usage.Loaded == nil {
		t.Fatal("signals.usage.loaded missing; the class B signal must always be emitted")
	}
	if !*body.Signals.Usage.Loaded {
		t.Error("signals.usage.loaded = false on a store that had no sidecar to lose")
	}
	if body.RequestID == "" {
		t.Error("status must echo the request id it answered under")
	}
}

// A discarded usage sidecar is the one case signals.usage.loaded exists to
// report. The server must come up and keep answering searches — class B is a
// ranking hint, not data — while telling an operator, through the contract
// surface and not just a log line, that the hints are gone (CTL-05).
func TestStatusReportsDiscardedUsageSidecar(t *testing.T) {
	indexPath := filepath.Join(t.TempDir(), "index.gob")
	sidecar := indexPath + ".collections.usage.json"
	if err := os.WriteFile(sidecar, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	handler := newCanonicalSurfaceTestHandlerAt(t, indexPath)
	call := func(method, path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, req)
		return resp
	}

	resp := call(http.MethodGet, "/v3/status", "")
	if resp.Code != http.StatusOK {
		t.Fatalf("GET /v3/status returned %d: %s", resp.Code, resp.Body.String())
	}
	var status struct {
		Signals struct {
			Usage struct {
				Loaded *bool `json:"loaded"`
			} `json:"usage"`
		} `json:"signals"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if status.Signals.Usage.Loaded == nil || *status.Signals.Usage.Loaded {
		t.Errorf("signals.usage.loaded = %v, want false after the sidecar was discarded", status.Signals.Usage.Loaded)
	}

	create := `{"name":"docs","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`
	if resp := call(http.MethodPost, "/v3/tenants/acme/collections", create); resp.Code != http.StatusCreated {
		t.Fatalf("create returned %d: %s", resp.Code, resp.Body.String())
	}
	if resp := call(http.MethodPut, "/v3/tenants/acme/collections/docs/docs/1", `{"vectors":{"dense":[1,0]}}`); resp.Code != http.StatusOK {
		t.Fatalf("upsert returned %d: %s", resp.Code, resp.Body.String())
	}
	if resp := call(http.MethodPost, "/v3/tenants/acme/collections/docs/search", `{"queries":{"dense":[1,0]},"top_k":1}`); resp.Code != http.StatusOK {
		t.Fatalf("search returned %d after a discarded sidecar: %s", resp.Code, resp.Body.String())
	}
	if resp := call(http.MethodGet, "/readyz", ""); resp.Code != http.StatusOK {
		t.Errorf("/readyz returned %d; a class B discard must not fault the store", resp.Code)
	}
}

// /v3/status is on the RC allowlist and is metered under its own literal
// path, not a rewritten one.
func TestStatusIsOnTheCanonicalSurfaceAndMetricsPath(t *testing.T) {
	if got := normalizeMetricsPath("/v3/status"); got != "/v3/status" {
		t.Errorf("normalizeMetricsPath(/v3/status) = %q", got)
	}
	handler := newCanonicalSurfaceTestHandler(t)
	req := httptest.NewRequest(http.MethodPost, "/v3/status", nil)
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	if resp.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /v3/status returned %d, want 405", resp.Code)
	}
}

// ── read-gated discovery ─────────────────────────────────────────────────

// A least-privilege agent must be able to find out what it may search. Both
// list and describe are read; neither needs admin, and an unauthenticated
// caller still gets 401 through the shared envelope.
func TestDiscoveryAndStatusAreReadGated(t *testing.T) {
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
	admin, err := store.jwtMgr.GenerateTenantToken("acme", []string{"admin", "read", "write"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	call := func(method, path, token, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, req)
		return resp
	}
	create := `{"name":"docs","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`
	if resp := call(http.MethodPost, "/v3/tenants/acme/collections", admin, create); resp.Code != http.StatusCreated {
		t.Fatalf("create returned %d: %s", resp.Code, resp.Body.String())
	}

	for _, path := range []string{"/v3/tenants/acme/collections", "/v3/tenants/acme/collections/docs", "/v3/status"} {
		if resp := call(http.MethodGet, path, readOnly, ""); resp.Code != http.StatusOK {
			t.Errorf("read token got %d on %s: %s", resp.Code, path, resp.Body.String())
		}
		resp := call(http.MethodGet, path, "", "")
		if resp.Code != http.StatusUnauthorized {
			t.Errorf("anonymous got %d on %s, want 401", resp.Code, path)
		}
		var envelope struct {
			Code      string `json:"code"`
			Message   string `json:"message"`
			RequestID string `json:"request_id"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
			t.Fatal(err)
		}
		if envelope.Code == "" || envelope.Message == "" {
			t.Errorf("401 on %s is not the shared envelope: %+v", path, envelope)
		}
	}
}

// ── per-result confidence on both transports ─────────────────────────────

// A dense field answers with distances, a sparse field with BM25 scores; the
// caller cannot tell which without being told, so every answer says so and
// reports how long it took.
func TestSearchResponseCarriesScoreDirectionAndQueryTime(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	call := func(method, path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, req)
		return resp
	}
	create := `{"name":"docs","fields":[` +
		`{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}},` +
		`{"name":"keywords","type":"sparse","dim":16,"index":{"type":"inverted"}}]}`
	if resp := call(http.MethodPost, "/v3/tenants/acme/collections", create); resp.Code != http.StatusCreated {
		t.Fatalf("create returned %d: %s", resp.Code, resp.Body.String())
	}
	doc := `{"vectors":{"dense":[1,0],"keywords":{"indices":[1],"values":[1],"dim":16}}}`
	if resp := call(http.MethodPost, "/v3/tenants/acme/collections/docs/docs", doc); resp.Code != http.StatusCreated && resp.Code != http.StatusOK {
		t.Fatalf("insert returned %d: %s", resp.Code, resp.Body.String())
	}

	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{"dense", `{"queries":{"dense":[1,0]},"top_k":1}`, vcollection.ScoreDirectionLowerIsBetter},
		{"sparse", `{"queries":{"keywords":{"indices":[1],"values":[1],"dim":16}},"top_k":1}`, vcollection.ScoreDirectionHigherIsBetter},
		{
			"hybrid fused",
			`{"queries":{"dense":[1,0],"keywords":{"indices":[1],"values":[1],"dim":16}},"top_k":1,"hybrid_params":{"strategy":"rrf"}}`,
			vcollection.ScoreDirectionHigherIsBetter,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp := call(http.MethodPost, "/v3/tenants/acme/collections/docs/search", tc.body)
			if resp.Code != http.StatusOK {
				t.Fatalf("search returned %d: %s", resp.Code, resp.Body.String())
			}
			var body struct {
				ScoreDirection string  `json:"score_direction"`
				QueryTimeMs    float64 `json:"query_time_ms"`
				RequestID      string  `json:"request_id"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body.ScoreDirection != tc.want {
				t.Errorf("score_direction = %q, want %q", body.ScoreDirection, tc.want)
			}
			// The engine stamps wall time around the whole call, so a real
			// search is never free.
			if body.QueryTimeMs <= 0 {
				t.Errorf("query_time_ms = %v, want > 0", body.QueryTimeMs)
			}
			if body.RequestID == "" {
				t.Error("search answer must carry the request id")
			}
		})
	}

	resp := call(http.MethodGet, "/v3/tenants/acme/collections/docs", "")
	if resp.Code != http.StatusOK {
		t.Fatalf("get collection returned %d: %s", resp.Code, resp.Body.String())
	}
	var info struct {
		Collection struct {
			Fields []struct {
				Name           string `json:"name"`
				ScoreDirection string `json:"score_direction"`
			} `json:"fields"`
		} `json:"collection"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"dense":    vcollection.ScoreDirectionLowerIsBetter,
		"keywords": vcollection.ScoreDirectionHigherIsBetter,
	}
	if len(info.Collection.Fields) != len(want) {
		t.Fatalf("collection reported %d fields", len(info.Collection.Fields))
	}
	for _, f := range info.Collection.Fields {
		if f.ScoreDirection != want[f.Name] {
			t.Errorf("field %s score_direction = %q, want %q", f.Name, f.ScoreDirection, want[f.Name])
		}
	}
}
