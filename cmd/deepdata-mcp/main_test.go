package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestServer builds an MCP server pointed at the given base URL.
func newTestServer(base string) *mcpServer {
	return &mcpServer{
		base:   base,
		tenant: "mcp",
		client: &http.Client{},
	}
}

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
	info, _ := init["serverInfo"].(map[string]any)
	if info["name"] != mcpServerName {
		t.Fatalf("serverInfo.name = %v", info["name"])
	}
	list := resultOf(t, frames[1])
	tools, _ := list["tools"].([]any)
	if len(tools) != 5 {
		t.Fatalf("expected 5 tools, got %d", len(tools))
	}
	seen := map[string]bool{}
	for _, raw := range tools {
		tool, _ := raw.(map[string]any)
		seen[tool["name"].(string)] = true
		if tool["inputSchema"] == nil {
			t.Fatalf("tool %v has no inputSchema", tool["name"])
		}
	}
	for _, name := range []string{"search", "insert", "upsert", "get_document", "list_collections"} {
		if !seen[name] {
			t.Fatalf("tool %s missing from tools/list", name)
		}
	}
}

func TestToolsCallSearchWithAgentRetrieval(t *testing.T) {
	var gotPath, gotMethod string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotMethod = r.URL.Path, r.Method
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"success","tenant_id":"mcp","documents":[],"scores":[],"candidates_examined":7,"best_score":0.9,"weak_match":true,"fell_back_to":"sparse"}`))
	}))
	defer server.Close()

	s := newTestServer(server.URL)
	frames := feedLines(t, s, mustCall(t, "tools/call", map[string]any{
		"name": "search",
		"arguments": map[string]any{
			"collection":  "notes",
			"queries":     map[string]any{"dense": []float32{0.1, 0.2}, "sparse": map[string]int{"1": 1}},
			"top_k":       3,
			"score_floor": 0.4,
			"fallback":    map[string]any{"primary": "dense", "secondary": "sparse", "threshold": 0.6},
			"usage_boost": 0.25,
		},
	}))
	text, isError := toolText(t, frames[0])
	if isError {
		t.Fatalf("search reported isError: %s", text)
	}
	if !strings.Contains(text, `"weak_match":true`) || !strings.Contains(text, `"fell_back_to":"sparse"`) {
		t.Fatalf("unexpected payload: %s", text)
	}
	if gotMethod != http.MethodPost || gotPath != "/v3/tenants/mcp/collections/notes/search" {
		t.Fatalf("server saw %s %s", gotMethod, gotPath)
	}
	if gotBody["score_floor"] != 0.4 || gotBody["usage_boost"] != 0.25 {
		t.Fatalf("agent-retrieval params not forwarded: %v", gotBody)
	}
	fallback, _ := gotBody["fallback"].(map[string]any)
	if fallback["primary"] != "dense" || fallback["secondary"] != "sparse" {
		t.Fatalf("fallback not forwarded: %v", gotBody["fallback"])
	}
}

func TestToolsCallUpsertAndInsertUseCanonicalEndpoints(t *testing.T) {
	var upsertPath, insertPath string
	var upsertBody, insertBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch {
		case r.Method == http.MethodPut && strings.HasSuffix(r.URL.Path, "/docs/42"):
			upsertPath = r.URL.Path
			upsertBody = map[string]any{}
			_ = json.Unmarshal(body, &upsertBody)
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/docs"):
			insertPath = r.URL.Path
			insertBody = map[string]any{}
			_ = json.Unmarshal(body, &insertBody)
		}
		w.Write([]byte(`{"status":"success"}`))
	}))
	defer server.Close()

	s := newTestServer(server.URL)
	frames := feedLines(t, s,
		mustCall(t, "tools/call", map[string]any{
			"name":      "upsert",
			"arguments": map[string]any{"collection": "notes", "id": 42, "vectors": map[string]any{"dense": []float32{1, 0}}, "metadata": map[string]any{"k": "v"}},
		}),
		mustCall(t, "tools/call", map[string]any{
			"name":      "insert",
			"arguments": map[string]any{"collection": "notes", "id": 7, "vectors": map[string]any{"dense": []float32{0, 1}}},
		}),
	)
	if _, isError := toolText(t, frames[0]); isError {
		t.Fatal("upsert reported isError")
	}
	if _, isError := toolText(t, frames[1]); isError {
		t.Fatal("insert reported isError")
	}
	if upsertPath != "/v3/tenants/mcp/collections/notes/docs/42" {
		t.Fatalf("upsert path = %q", upsertPath)
	}
	if _, hasID := upsertBody["id"]; hasID {
		t.Fatalf("upsert body must not carry the id (path owns it): %v", upsertBody)
	}
	if insertPath != "/v3/tenants/mcp/collections/notes/docs" {
		t.Fatalf("insert path = %q", insertPath)
	}
	if insertBody["id"] != float64(7) {
		t.Fatalf("insert body missing caller id: %v", insertBody)
	}
}

func TestToolErrorsAreIsErrorNotRPCErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "collection not found", http.StatusNotFound)
	}))
	defer server.Close()

	s := newTestServer(server.URL)
	frames := feedLines(t, s, mustCall(t, "tools/call", map[string]any{
		"name":      "get_document",
		"arguments": map[string]any{"collection": "nope", "id": 1},
	}))
	text, isError := toolText(t, frames[0])
	if !isError || !strings.Contains(text, "not found") {
		t.Fatalf("expected isError with server message, got %q isError=%v", text, isError)
	}

	// Unknown tool and missing argument are RPC-level errors.
	unknown := feedLines(t, s, mustCall(t, "tools/call", map[string]any{"name": "nope"}))
	if unknown[0]["error"] == nil {
		t.Fatal("unknown tool must yield an rpc error")
	}
	missing := feedLines(t, s, mustCall(t, "tools/call", map[string]any{
		"name": "search", "arguments": map[string]any{"queries": map[string]any{}},
	}))
	if missing[0]["error"] == nil {
		t.Fatal("missing collection must yield an rpc error")
	}
}

func TestParseErrorAndUnknownMethod(t *testing.T) {
	s := newTestServer("http://127.0.0.1:1")
	frames := feedLines(t, s, `{nope`, mustCall(t, "bogus/method", nil))
	if frames[0]["error"] == nil {
		t.Fatal("parse error expected")
	}
	code, _ := frames[0]["error"].(map[string]any)["code"].(float64)
	if int(code) != -32700 {
		t.Fatalf("parse error code = %v", code)
	}
	if frames[1]["error"] == nil {
		t.Fatal("method-not-found error expected")
	}
}
