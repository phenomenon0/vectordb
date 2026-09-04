package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/api/contract"
)

// newTestServer builds an MCP server pointed at the given base URL.
func newTestServer(base string) *mcpServer {
	return &mcpServer{
		base:       base,
		tenant:     "mcp",
		collection: "memory",
		client:     &http.Client{},
		infos:      map[string]*collectionInfo{},
	}
}

// memoryInfo is GET /collections/memory for a preset "memory" collection:
// a dense field bound to the server embedder plus a sparse bm25 field.
const memoryInfo = `{"status":"success","tenant_id":"mcp","collection":{"name":"memory","fields":[` +
	`{"name":"text","type":"dense","dim":4,"index":{"type":"hnsw"},"score_direction":"lower_is_better","embedding":{"provider":"hash","model":"4"}},` +
	`{"name":"keywords","type":"sparse","dim":10000,"index":{"type":"inverted"},"score_direction":"higher_is_better","embedding":{"provider":"bm25"}}],"doc_count":3}}`

// vectorsOnlyInfo has no embedding binding, so text cannot be routed.
const vectorsOnlyInfo = `{"status":"success","tenant_id":"mcp","collection":{"name":"raw","fields":[` +
	`{"name":"dense","type":"dense","dim":4,"index":{"type":"flat"}}],"doc_count":0}}`

// feedLines runs the stdio loop over the given JSON-RPC lines and returns
// the decoded response frames (notifications produce no frame).
func feedLines(t *testing.T, s *mcpServer, lines ...string) []map[string]any {
	t.Helper()
	var out bytes.Buffer
	input := strings.Join(lines, "\n") + "\n"
	if err := serveStdio(s, strings.NewReader(input), &out); err != nil {
		t.Fatalf("serveStdio: %v", err)
	}
	var frames []map[string]any
	scanner := bufio.NewScanner(&out)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		var frame map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &frame); err != nil {
			t.Fatalf("decode frame %q: %v", scanner.Text(), err)
		}
		frames = append(frames, frame)
	}
	return frames
}

func mustCall(t *testing.T, method string, params any) string {
	t.Helper()
	payload := map[string]any{"jsonrpc": "2.0", "id": 1, "method": method}
	if params != nil {
		encoded, err := json.Marshal(params)
		if err != nil {
			t.Fatal(err)
		}
		payload["params"] = json.RawMessage(encoded)
	}
	line, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return string(line)
}

func toolCall(t *testing.T, name string, args any) string {
	t.Helper()
	return mustCall(t, "tools/call", map[string]any{"name": name, "arguments": args})
}

func resultOf(t *testing.T, frame map[string]any) map[string]any {
	t.Helper()
	if frame["error"] != nil {
		t.Fatalf("unexpected rpc error: %v", frame["error"])
	}
	result, ok := frame["result"].(map[string]any)
	if !ok {
		t.Fatalf("no result object in frame: %v", frame)
	}
	return result
}

func toolText(t *testing.T, frame map[string]any) (string, bool) {
	t.Helper()
	result := resultOf(t, frame)
	content, _ := result["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected one content part, got %v", content)
	}
	part, _ := content[0].(map[string]any)
	text, _ := part["text"].(string)
	isError, _ := result["isError"].(bool)
	return text, isError
}

// structured returns structuredContent, failing on isError results.
func structured(t *testing.T, frame map[string]any) map[string]any {
	t.Helper()
	text, isError := toolText(t, frame)
	if isError {
		t.Fatalf("tool reported isError: %s", text)
	}
	sc, _ := resultOf(t, frame)["structuredContent"].(map[string]any)
	if sc == nil {
		t.Fatalf("no structuredContent in %v", frame)
	}
	return sc
}

func rpcErrorCode(t *testing.T, frame map[string]any) int {
	t.Helper()
	rpcErr, _ := frame["error"].(map[string]any)
	if rpcErr == nil {
		t.Fatalf("expected an rpc error, got %v", frame)
	}
	code, _ := rpcErr["code"].(float64)
	return int(code)
}

// recorder is a fake DeepData server that logs every request body by
// "METHOD path" and answers from a per-route table.
type recorder struct {
	t       *testing.T
	bodies  map[string][]map[string]any
	answers map[string]func(w http.ResponseWriter)
}

func newRecorder(t *testing.T) (*recorder, *httptest.Server) {
	rec := &recorder{t: t, bodies: map[string][]map[string]any{}, answers: map[string]func(http.ResponseWriter){}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.Method + " " + r.URL.Path
		body := map[string]any{}
		raw, _ := io.ReadAll(r.Body)
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &body)
		}
		rec.bodies[key] = append(rec.bodies[key], body)
		w.Header().Set("Content-Type", "application/json")
		if answer, ok := rec.answers[key]; ok {
			answer(w)
			return
		}
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, `{"code":"not_found","message":"no route %s","retryable":false}`, key)
	}))
	t.Cleanup(server.Close)
	return rec, server
}

func (r *recorder) on(method, path string, status int, body string) {
	r.answers[method+" "+path] = func(w http.ResponseWriter) {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}
}

func (r *recorder) calls(method, path string) int {
	return len(r.bodies[method+" "+path])
}

func (r *recorder) body(method, path string) map[string]any {
	r.t.Helper()
	bodies := r.bodies[method+" "+path]
	if len(bodies) == 0 {
		r.t.Fatalf("server never saw %s %s; saw %v", method, path, keys(r.bodies))
	}
	return bodies[len(bodies)-1]
}

func keys(m map[string][]map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

const memoryPath = "/v3/tenants/mcp/collections/memory"

// ── tools/list ───────────────────────────────────────────────────────────

// The tool list is the agent's whole view of the system: every verb must be
// namespaced, self-describing (input and output schema), and annotated so a
// host can tell reads from writes from deletes without running anything.
func TestInitializeAndToolsList(t *testing.T) {
	s := newTestServer("http://127.0.0.1:1")
	frames := feedLines(t, s,
		mustCall(t, "initialize", map[string]any{"protocolVersion": mcpProtocolVersion}),
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		mustCall(t, "tools/list", map[string]any{}),
	)
	if len(frames) != 2 {
		t.Fatalf("expected 2 response frames (notification silent), got %d", len(frames))
	}
	init := resultOf(t, frames[0])
	caps, _ := init["capabilities"].(map[string]any)
	if caps["tools"] == nil || caps["resources"] == nil {
		t.Fatalf("capabilities must advertise tools and resources: %v", caps)
	}
	list := resultOf(t, frames[1])
	tools, _ := list["tools"].([]any)
	if len(tools) != 6 {
		t.Fatalf("expected 6 tools, got %d", len(tools))
	}
	var names []string
	for _, raw := range tools {
		tool, _ := raw.(map[string]any)
		name, _ := tool["name"].(string)
		names = append(names, name)
		for _, key := range []string{"title", "description", "inputSchema", "outputSchema", "annotations"} {
			if tool[key] == nil {
				t.Errorf("tool %s has no %s", name, key)
			}
		}
		ann, _ := tool["annotations"].(map[string]any)
		if ann["openWorldHint"] != false {
			t.Errorf("tool %s must declare openWorldHint=false (it talks to one server)", name)
		}
		switch name {
		case "deepdata_recall", "deepdata_get", "deepdata_collections":
			if ann["readOnlyHint"] != true {
				t.Errorf("%s must be readOnlyHint", name)
			}
		case "deepdata_forget":
			if ann["destructiveHint"] != true || ann["idempotentHint"] != true {
				t.Errorf("forget must be destructive and idempotent: %v", ann)
			}
		default:
			if ann["destructiveHint"] != false {
				t.Errorf("%s must declare destructiveHint=false explicitly (the MCP default is true): %v", name, ann)
			}
		}
	}
	want := contract.Names()
	got := append([]string(nil), names...)
	sort.Strings(got)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tool names %v != contract schemas %v", got, want)
	}
}

// inputSchema and outputSchema are the contract files byte-for-byte, so the
// contract cannot drift from what agents are shown.
func TestToolSchemasAreTheContractFiles(t *testing.T) {
	compact := func(raw json.RawMessage) string {
		var buf bytes.Buffer
		if err := json.Compact(&buf, raw); err != nil {
			t.Fatal(err)
		}
		return buf.String()
	}
	for _, def := range newTestServer("").tools() {
		raw, err := contract.Schema(def.Name)
		if err != nil {
			t.Fatal(err)
		}
		var pair struct {
			Input  json.RawMessage `json:"input"`
			Output json.RawMessage `json:"output"`
		}
		if err := json.Unmarshal(raw, &pair); err != nil {
			t.Fatal(err)
		}
		if compact(def.InputSchema) != compact(pair.Input) {
			t.Errorf("%s inputSchema differs from contract", def.Name)
		}
		if compact(def.OutputSchema) != compact(pair.Output) {
			t.Errorf("%s outputSchema differs from contract", def.Name)
		}
		if def.Description == "" {
			t.Errorf("%s has no description in its contract input schema", def.Name)
		}
	}
}

// ── deepdata_recall ──────────────────────────────────────────────────────

// An agent speaks text. With the memory preset (dense + sparse bound) the
// server must embed the query for both fields and install the dense→sparse
// fallback ladder, without the agent knowing either field name.
func TestRecallTextRoutesToBoundFieldsWithDefaultFallback(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, memoryPath, 200, memoryInfo)
	rec.on(http.MethodPost, memoryPath+"/search", 200, `{"status":"success","tenant_id":"mcp",`+
		`"documents":[{"id":7,"vectors":{"text":[0.1,0.2,0.3,0.4]},"metadata":{"text":"the durable journal","tag":"ops"}}],`+
		`"scores":[0.12],"candidates_examined":3,"best_score":0.12,"weak_match":false,"fell_back_to":"keywords",`+
		`"score_direction":"higher_is_better","query_time_ms":1.5,"request_id":"abc","embedded_by":{"text":"hash:4","keywords":"bm25"}}`)

	s := newTestServer(server.URL)
	frames := feedLines(t, s,
		toolCall(t, "deepdata_recall", map[string]any{"query": "journal", "top_k": 3}),
		toolCall(t, "deepdata_recall", map[string]any{"query": "journal", "fallback": map[string]any{"primary": "keywords", "secondary": "text", "threshold": 0.5}}),
	)
	out := structured(t, frames[0])
	hits, _ := out["hits"].([]any)
	if len(hits) != 1 {
		t.Fatalf("hits = %v", out)
	}
	hit, _ := hits[0].(map[string]any)
	if hit["id"] != float64(7) || hit["score"] != 0.12 {
		t.Fatalf("hit = %v", hit)
	}
	if _, leaked := hit["vectors"]; leaked {
		t.Fatal("vectors must never be returned over MCP")
	}
	if meta, _ := hit["metadata"].(map[string]any); meta["text"] != "the durable journal" {
		t.Fatalf("metadata not passed through: %v", hit)
	}
	if out["fell_back_to"] != "keywords" || out["best_score"] != 0.12 || out["weak_match"] != false {
		t.Fatalf("confidence fields missing: %v", out)
	}
	// The ladder answered from the sparse field, so the score reads upward;
	// without this the agent compares a BM25 score as if it were a distance.
	if out["score_direction"] != "higher_is_better" {
		t.Fatalf("score_direction = %v, want higher_is_better", out["score_direction"])
	}
	if by, _ := out["embedded_by"].(map[string]any); by["text"] != "hash:4" {
		t.Fatalf("embedded_by missing: %v", out)
	}

	body := rec.bodies["POST "+memoryPath+"/search"][0]
	texts, _ := body["texts"].(map[string]any)
	if texts["text"] != "journal" || texts["keywords"] != "journal" {
		t.Fatalf("query text must be embedded for every bound field: %v", body)
	}
	if body["queries"] != nil || body["include_vectors"] != false || body["top_k"] != float64(3) {
		t.Fatalf("search body = %v", body)
	}
	fallback, _ := body["fallback"].(map[string]any)
	if fallback["primary"] != "text" || fallback["secondary"] != "keywords" {
		t.Fatalf("default fallback must be dense→sparse: %v", body["fallback"])
	}

	// The caller's own ladder is forwarded untouched and not overridden.
	second := rec.bodies["POST "+memoryPath+"/search"][1]
	fallback, _ = second["fallback"].(map[string]any)
	if fallback["primary"] != "keywords" || fallback["threshold"] != 0.5 {
		t.Fatalf("caller fallback overridden: %v", second["fallback"])
	}
	if rec.calls(http.MethodGet, memoryPath) != 1 {
		t.Fatalf("collection schema must be fetched once and cached, got %d GETs", rec.calls(http.MethodGet, memoryPath))
	}
}

// Vectors the agent computed itself go straight through: no schema lookup.
func TestRecallVectorsSkipSchemaLookup(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodPost, "/v3/tenants/mcp/collections/notes/search", 200,
		`{"status":"success","tenant_id":"mcp","documents":[],"scores":[],"candidates_examined":0,"weak_match":true}`)
	frames := feedLines(t, newTestServer(server.URL), toolCall(t, "deepdata_recall", map[string]any{
		"collection":  "notes",
		"queries":     map[string]any{"dense": []float32{0.1, 0.2}},
		"score_floor": 0.4,
		"usage_boost": 0.25,
		"filters":     map[string]any{"tag": map[string]any{"$in": []string{"a", "b"}}},
	}))
	out := structured(t, frames[0])
	if out["weak_match"] != true || !strings.Contains(out["hint"].(string), "score_floor") {
		t.Fatalf("a weak match must explain itself: %v", out)
	}
	body := rec.body(http.MethodPost, "/v3/tenants/mcp/collections/notes/search")
	if body["score_floor"] != 0.4 || body["usage_boost"] != 0.25 || body["texts"] != nil {
		t.Fatalf("agent-retrieval params not forwarded: %v", body)
	}
	if filters, _ := body["filters"].(map[string]any); filters["tag"] == nil {
		t.Fatalf("filters not forwarded: %v", body)
	}
	if rec.calls(http.MethodGet, "/v3/tenants/mcp/collections/notes") != 0 {
		t.Fatal("vector queries must not fetch the collection schema")
	}
}

// Text against a vectors-only collection cannot be embedded; the failure must
// be a tool result that names the next call, and nothing may reach /search.
func TestRecallTextWithoutBindingIsErrorWithHint(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, "/v3/tenants/mcp/collections/raw", 200, vectorsOnlyInfo)
	frames := feedLines(t, newTestServer(server.URL), toolCall(t, "deepdata_recall", map[string]any{"collection": "raw", "query": "x"}))
	text, isError := toolText(t, frames[0])
	if !isError || !strings.Contains(text, "invalid_argument") || !strings.Contains(text, "deepdata_create_collection") {
		t.Fatalf("expected a local invalid_argument envelope with a hint, got %q", text)
	}
	sc, _ := resultOf(t, frames[0])["structuredContent"].(map[string]any)
	if sc["code"] != "invalid_argument" || sc["hint"] == "" {
		t.Fatalf("structuredContent = %v", sc)
	}
	if rec.calls(http.MethodPost, "/v3/tenants/mcp/collections/raw/search") != 0 {
		t.Fatal("no search may be issued when text cannot be embedded")
	}
}

// Token budget: concise mode elides long strings; both modes drop tail hits
// until the result fits max_chars and say which ids were dropped, so the
// agent can narrow the query instead of silently missing memories.
func TestRecallConciseElisionAndTruncation(t *testing.T) {
	long := strings.Repeat("é", 400)
	var docs []string
	for i := 1; i <= 5; i++ {
		docs = append(docs, fmt.Sprintf(`{"id":%d,"metadata":{"text":%q}}`, i, long))
	}
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, memoryPath, 200, memoryInfo)
	rec.on(http.MethodPost, memoryPath+"/search", 200,
		`{"status":"success","tenant_id":"mcp","documents":[`+strings.Join(docs, ",")+`],"scores":[0.1,0.2,0.3,0.4,0.5],"candidates_examined":5,"best_score":0.1,"weak_match":false}`)

	s := newTestServer(server.URL)
	frames := feedLines(t, s,
		toolCall(t, "deepdata_recall", map[string]any{"query": "q"}),
		toolCall(t, "deepdata_recall", map[string]any{"query": "q", "response_format": "detailed", "max_chars": 100000}),
		toolCall(t, "deepdata_recall", map[string]any{"query": "q", "max_chars": 900}),
	)
	concise := structured(t, frames[0])
	hits, _ := concise["hits"].([]any)
	if len(hits) != 5 || concise["truncated"] != nil {
		t.Fatalf("5 elided hits fit the default budget: %v", concise)
	}
	meta, _ := hits[0].(map[string]any)["metadata"].(map[string]any)
	if text, _ := meta["text"].(string); len([]rune(text)) != conciseMaxRunes+1 || !strings.HasSuffix(text, "…") {
		t.Fatalf("concise mode must elide to %d runes + ellipsis, got %d runes", conciseMaxRunes, len([]rune(text)))
	}

	detailed := structured(t, frames[1])
	hits, _ = detailed["hits"].([]any)
	meta, _ = hits[0].(map[string]any)["metadata"].(map[string]any)
	if meta["text"] != long {
		t.Fatal("detailed mode must not elide")
	}

	tight := structured(t, frames[2])
	hits, _ = tight["hits"].([]any)
	if tight["truncated"] != true || len(hits) == 0 || len(hits) >= 5 {
		t.Fatalf("expected tail hits dropped under max_chars=900: %v", tight)
	}
	hint, _ := tight["hint"].(string)
	if !strings.Contains(hint, "5") || !strings.Contains(hint, "max_chars") {
		t.Fatalf("hint must name the dropped ids and the knob: %q", hint)
	}
	if text, _ := toolText(t, frames[2]); len(text) > 900 {
		t.Fatalf("result exceeds max_chars: %d bytes", len(text))
	}
}

// ── deepdata_remember ────────────────────────────────────────────────────

// Routing: items with an id replace via PUT; one fresh item inserts via
// POST /docs; several fresh items go through the atomic batch. Text is
// embedded for every bound field and kept at metadata.text so recall can
// return words, not ids.
func TestRememberRoutesPutPostBatch(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, memoryPath, 200, memoryInfo)
	rec.on(http.MethodPut, memoryPath+"/docs/42", 200, `{"status":"success","tenant_id":"mcp","id":42,"message":"document upserted"}`)
	rec.on(http.MethodPost, memoryPath+"/docs", 200, `{"status":"success","tenant_id":"mcp","id":9,"message":"document added"}`)
	rec.on(http.MethodPost, memoryPath+"/docs/batch", 200, `{"status":"success","tenant_id":"mcp","ids":[100,101],"inserted":2}`)

	s := newTestServer(server.URL)
	frames := feedLines(t, s,
		toolCall(t, "deepdata_remember", map[string]any{"items": []map[string]any{{"text": "alpha", "metadata": map[string]any{"tag": "a"}}}}),
		toolCall(t, "deepdata_remember", map[string]any{"items": []map[string]any{
			{"text": "replace me", "id": 42},
			{"text": "beta"},
			{"vectors": map[string]any{"text": []float32{1, 0, 0, 0}}},
		}}),
	)
	single := structured(t, frames[0])
	if ids, _ := single["ids"].([]any); len(ids) != 1 || ids[0] != float64(9) || single["count"] != float64(1) {
		t.Fatalf("single insert must return the server-assigned id: %v", single)
	}
	post := rec.body(http.MethodPost, memoryPath+"/docs")
	if _, hasID := post["id"]; hasID {
		t.Fatalf("fresh items must not carry an id: %v", post)
	}
	texts, _ := post["texts"].(map[string]any)
	if texts["text"] != "alpha" || texts["keywords"] != "alpha" {
		t.Fatalf("text must be embedded for every bound field: %v", post)
	}
	if meta, _ := post["metadata"].(map[string]any); meta["text"] != "alpha" || meta["tag"] != "a" {
		t.Fatalf("metadata.text must hold the raw text alongside caller metadata: %v", post)
	}

	mixed := structured(t, frames[1])
	if ids, _ := mixed["ids"].([]any); !reflect.DeepEqual(ids, []any{float64(42), float64(100), float64(101)}) {
		t.Fatalf("ids must follow item order across PUT and batch: %v", mixed)
	}
	put := rec.body(http.MethodPut, memoryPath+"/docs/42")
	if _, hasID := put["id"]; hasID {
		t.Fatalf("upsert body must not carry the id (path owns it): %v", put)
	}
	batch := rec.body(http.MethodPost, memoryPath+"/docs/batch")
	documents, _ := batch["documents"].([]any)
	if len(documents) != 2 {
		t.Fatalf("batch must carry exactly the fresh items: %v", batch)
	}
	vectorDoc, _ := documents[1].(map[string]any)
	if vectorDoc["vectors"] == nil || vectorDoc["texts"] != nil || vectorDoc["metadata"] != nil {
		t.Fatalf("vector items are forwarded as-is without metadata.text: %v", vectorDoc)
	}
	if rec.calls(http.MethodGet, memoryPath) != 1 {
		t.Fatalf("schema fetched %d times, want 1", rec.calls(http.MethodGet, memoryPath))
	}
}

// ── deepdata_forget / deepdata_get ───────────────────────────────────────

// Forget is idempotent: a document that is already gone is a success, but a
// missing collection is still an error the agent must act on.
func TestForgetIsIdempotentOnMissingDocument(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodDelete, memoryPath+"/docs", 404, `{"code":"not_found","message":"document not found: 5 in collection memory","retryable":false}`)
	rec.on(http.MethodDelete, "/v3/tenants/mcp/collections/nope/docs", 404, `{"code":"not_found","message":"collection not found: nope","hint":"call deepdata_collections","retryable":false}`)

	s := newTestServer(server.URL)
	frames := feedLines(t, s,
		toolCall(t, "deepdata_forget", map[string]any{"id": 5}),
		toolCall(t, "deepdata_forget", map[string]any{"id": 5, "collection": "nope"}),
	)
	if out := structured(t, frames[0]); out["deleted"] != float64(5) {
		t.Fatalf("forget of a gone document must succeed: %v", out)
	}
	if body := rec.body(http.MethodDelete, memoryPath+"/docs"); body["doc_id"] != float64(5) {
		t.Fatalf("delete body = %v", body)
	}
	if text, isError := toolText(t, frames[1]); !isError || !strings.Contains(text, "collection not found") {
		t.Fatalf("missing collection must surface: %q isError=%v", text, isError)
	}
}

func TestGetCollectsMissingIdsAndStripsVectors(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, memoryPath+"/docs/1", 200, `{"status":"success","tenant_id":"mcp","id":1,"vectors":{"text":[1,0,0,0]},"metadata":{"text":"one"}}`)
	rec.on(http.MethodGet, memoryPath+"/docs/2", 404, `{"code":"not_found","message":"document 2 not found","retryable":false}`)
	frames := feedLines(t, newTestServer(server.URL), toolCall(t, "deepdata_get", map[string]any{"ids": []int{1, 2}}))
	out := structured(t, frames[0])
	documents, _ := out["documents"].([]any)
	if len(documents) != 1 {
		t.Fatalf("documents = %v", out)
	}
	doc, _ := documents[0].(map[string]any)
	if doc["id"] != float64(1) || doc["vectors"] != nil {
		t.Fatalf("document must carry id and metadata only: %v", doc)
	}
	if missing, _ := out["missing"].([]any); !reflect.DeepEqual(missing, []any{float64(2)}) {
		t.Fatalf("missing = %v", out["missing"])
	}
}

// ── deepdata_collections / deepdata_create_collection ────────────────────

func TestCollectionsListAndDescribe(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, "/v3/tenants/mcp/collections", 200, `{"status":"success","tenant_id":"mcp","count":1,"collections":[`+
		`{"name":"memory","fields":[{"name":"text","type":"dense","dim":4,"index":{"type":"hnsw"},"score_direction":"lower_is_better","embedding":{"provider":"hash","model":"4"}}],"doc_count":3}]}`)
	rec.on(http.MethodGet, memoryPath, 200, memoryInfo)
	frames := feedLines(t, newTestServer(server.URL),
		toolCall(t, "deepdata_collections", map[string]any{}),
		toolCall(t, "deepdata_collections", map[string]any{"name": "memory"}),
	)
	list := structured(t, frames[0])
	cols, _ := list["collections"].([]any)
	if len(cols) != 1 {
		t.Fatalf("list = %v", list)
	}
	col, _ := cols[0].(map[string]any)
	fields, _ := col["fields"].([]any)
	if col["name"] != "memory" || len(fields) != 1 || col["doc_count"] != float64(3) {
		t.Fatalf("collection = %v", col)
	}
	field, _ := fields[0].(map[string]any)
	if emb, _ := field["embedding"].(map[string]any); emb["provider"] != "hash" {
		t.Fatalf("embedding binding must be visible to the agent: %v", field)
	}
	// Per-field score direction is how the agent knows which way to read the
	// scores it will get back from this field.
	if field["score_direction"] != "lower_is_better" {
		t.Fatalf("field score_direction = %v, want lower_is_better", field["score_direction"])
	}
	one := structured(t, frames[1])
	cols, _ = one["collections"].([]any)
	if len(cols) != 1 || cols[0].(map[string]any)["name"] != "memory" {
		t.Fatalf("describe = %v", one)
	}
}

// The memory preset is built from the server's own embedder (via /readyz),
// so the agent never has to know provider, model or dimension. Without an
// embedder the preset must refuse with a hint instead of creating a
// collection nobody can write text into.
func TestCreateCollectionPresetFollowsServerEmbedder(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, "/readyz", 200, `{"ready":true,"checks":["collection_snapshot"],"embedder":"hash:4","version":"test"}`)
	rec.on(http.MethodPost, "/v3/tenants/mcp/collections", 201, `{"status":"success","tenant_id":"mcp","message":"created"}`)
	rec.on(http.MethodGet, memoryPath, 200, memoryInfo)

	frames := feedLines(t, newTestServer(server.URL), toolCall(t, "deepdata_create_collection", map[string]any{"name": "memory", "preset": "memory"}))
	out := structured(t, frames[0])
	if out["created"] != "memory" {
		t.Fatalf("created = %v", out)
	}
	if fields, _ := out["fields"].([]any); len(fields) != 2 {
		t.Fatalf("resolved fields must be returned: %v", out)
	}
	body := rec.body(http.MethodPost, "/v3/tenants/mcp/collections")
	fields, _ := body["fields"].([]any)
	if body["name"] != "memory" || len(fields) != 2 {
		t.Fatalf("create body = %v", body)
	}
	dense, _ := fields[0].(map[string]any)
	emb, _ := dense["embedding"].(map[string]any)
	if dense["name"] != "text" || dense["type"] != "dense" || emb["provider"] != "hash" || emb["model"] != "4" {
		t.Fatalf("dense field must bind the server embedder: %v", dense)
	}
	if _, hasDim := dense["dim"]; hasDim {
		t.Fatalf("dense dim must be left for the server to fill: %v", dense)
	}
	sparse, _ := fields[1].(map[string]any)
	emb, _ = sparse["embedding"].(map[string]any)
	if sparse["name"] != "keywords" || sparse["type"] != "sparse" || emb["provider"] != "bm25" || sparse["dim"] != float64(sparseMemoryDim) {
		t.Fatalf("sparse field = %v", sparse)
	}

	// No embedder: refuse before POST.
	rec2, server2 := newRecorder(t)
	rec2.on(http.MethodGet, "/readyz", 200, `{"ready":true,"checks":[],"embedder":"none","version":"test"}`)
	frames = feedLines(t, newTestServer(server2.URL), toolCall(t, "deepdata_create_collection", map[string]any{"name": "memory", "preset": "memory"}))
	text, isError := toolText(t, frames[0])
	if !isError || !strings.Contains(text, "embedder_unavailable") || !strings.Contains(text, "DEEPDATA_EMBEDDER") {
		t.Fatalf("expected embedder_unavailable with a hint, got %q", text)
	}
	if rec2.calls(http.MethodPost, "/v3/tenants/mcp/collections") != 0 {
		t.Fatal("preset must not create a collection without an embedder")
	}
}

// ── cache ────────────────────────────────────────────────────────────────

// Any 4xx for a collection means our cached schema may be stale (deleted,
// recreated, renamed); the next text call must re-fetch it.
func TestSchemaCacheDropsOn4xx(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, memoryPath, 200, memoryInfo)
	rec.on(http.MethodPost, memoryPath+"/search", 404, `{"code":"not_found","message":"collection not found: memory","retryable":false}`)
	s := newTestServer(server.URL)
	frames := feedLines(t, s,
		toolCall(t, "deepdata_recall", map[string]any{"query": "a"}),
		toolCall(t, "deepdata_recall", map[string]any{"query": "b"}),
	)
	for _, frame := range frames {
		if _, isError := toolText(t, frame); !isError {
			t.Fatal("search 404 must surface as isError")
		}
	}
	if rec.calls(http.MethodGet, memoryPath) != 2 {
		t.Fatalf("schema must be re-fetched after a 4xx, got %d GETs", rec.calls(http.MethodGet, memoryPath))
	}
}

// ── resources ────────────────────────────────────────────────────────────

func TestResourcesServeContractAndStatus(t *testing.T) {
	rec, server := newRecorder(t)
	rec.on(http.MethodGet, "/v3/status", 200, `{"version":"test","contract":{"http":[{"method":"GET","path":"/v3/status","permission":"read"}]},"embedding":{"provider":"none","available":false},"capabilities":{"texts":false}}`)
	frames := feedLines(t, newTestServer(server.URL),
		mustCall(t, "resources/list", map[string]any{}),
		mustCall(t, "resources/read", map[string]any{"uri": contractURI}),
		mustCall(t, "resources/read", map[string]any{"uri": statusURI}),
		mustCall(t, "resources/read", map[string]any{"uri": "deepdata://nope"}),
	)
	resources, _ := resultOf(t, frames[0])["resources"].([]any)
	if len(resources) != 2 {
		t.Fatalf("resources = %v", resources)
	}
	contents, _ := resultOf(t, frames[1])["contents"].([]any)
	doc, _ := contents[0].(map[string]any)
	text, _ := doc["text"].(string)
	if doc["mimeType"] != "text/markdown" || !strings.Contains(text, "deepdata_recall") || !strings.Contains(text, "deepdata_forget") {
		t.Fatalf("contract resource must be the agent contract naming the verbs: %v", doc)
	}
	contents, _ = resultOf(t, frames[2])["contents"].([]any)
	status, _ := contents[0].(map[string]any)
	// deepdata://status is the server's own self-description, not a liveness
	// probe: the agent reads the operation list and the embedder from it.
	statusText, _ := status["text"].(string)
	if !strings.Contains(statusText, `"contract"`) || !strings.Contains(statusText, `"provider":"none"`) {
		t.Fatalf("status resource must proxy GET /v3/status: %v", status)
	}
	if rpcErrorCode(t, frames[3]) != -32002 {
		t.Fatalf("unknown resource must be -32002: %v", frames[3])
	}
}

// ── errors ───────────────────────────────────────────────────────────────

func TestToolErrorsAreIsErrorNotRPCErrors(t *testing.T) {
	// The server answers with the structured envelope; the tool result must
	// hand the model the code, message and hint as text (so it can decide
	// its next call) and the raw envelope as structuredContent (so a host
	// can branch on the code without parsing prose).
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"code":"not_found","message":"collection not found: nope","hint":"list the collections first","retryable":false,"docs":"internal/collection/API.md#errors"}`))
	}))
	defer server.Close()

	s := newTestServer(server.URL)
	getMissing := toolCall(t, "deepdata_get", map[string]any{"collection": "nope", "ids": []int{1}})
	frames := feedLines(t, s, getMissing)
	text, isError := toolText(t, frames[0])
	if !isError || !strings.Contains(text, "not_found: collection not found: nope Hint: list the collections first") {
		t.Fatalf("expected isError with code, message and hint, got %q isError=%v", text, isError)
	}
	sc, _ := resultOf(t, frames[0])["structuredContent"].(map[string]any)
	if sc["code"] != "not_found" || sc["hint"] != "list the collections first" || sc["docs"] == nil {
		t.Fatalf("structuredContent must carry the whole envelope, got %v", sc)
	}

	// A server that still answers in plain text is forwarded verbatim.
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "collection not found", http.StatusNotFound)
	}))
	defer plain.Close()
	frames = feedLines(t, newTestServer(plain.URL), getMissing)
	text, isError = toolText(t, frames[0])
	if !isError || text != "collection not found" {
		t.Fatalf("expected plain-text passthrough, got %q isError=%v", text, isError)
	}
	if _, has := resultOf(t, frames[0])["structuredContent"]; has {
		t.Fatal("plain-text errors carry no structuredContent")
	}

	// Unknown tool and argument-shape violations are RPC-level errors and
	// never reach the network.
	frames = feedLines(t, newTestServer("http://127.0.0.1:1"),
		toolCall(t, "nope", map[string]any{}),
		toolCall(t, "deepdata_recall", map[string]any{"collection": "memory"}),
		toolCall(t, "deepdata_recall", map[string]any{"query": "x", "queries": map[string]any{"a": []int{1}}}),
		toolCall(t, "deepdata_recall", map[string]any{"query": "x", "top_k": 500}),
		toolCall(t, "deepdata_recall", map[string]any{"query": "x", "tenant": "other"}),
		toolCall(t, "deepdata_remember", map[string]any{"items": []map[string]any{{"metadata": map[string]any{}}}}),
		toolCall(t, "deepdata_create_collection", map[string]any{"name": "bad name", "preset": "memory"}),
		toolCall(t, "deepdata_get", map[string]any{"ids": []int{}}),
	)
	if rpcErrorCode(t, frames[0]) != -32601 {
		t.Fatalf("unknown tool must be -32601: %v", frames[0])
	}
	for i, frame := range frames[1:] {
		if rpcErrorCode(t, frame) != -32602 {
			t.Fatalf("case %d must be invalid params (-32602): %v", i, frame)
		}
	}
	if msg, _ := frames[4]["error"].(map[string]any)["message"].(string); !strings.Contains(msg, "tenant") {
		t.Fatalf("tenant is process-bound and must be rejected as an unknown argument: %q", msg)
	}
}

func TestParseErrorAndUnknownMethod(t *testing.T) {
	s := newTestServer("http://127.0.0.1:1")
	frames := feedLines(t, s, `{nope`, mustCall(t, "bogus/method", nil))
	if rpcErrorCode(t, frames[0]) != -32700 {
		t.Fatalf("parse error expected: %v", frames[0])
	}
	if rpcErrorCode(t, frames[1]) != -32601 {
		t.Fatalf("method-not-found error expected: %v", frames[1])
	}
}
