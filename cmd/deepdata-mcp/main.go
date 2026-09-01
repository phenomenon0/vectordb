// Command deepdata-mcp exposes a running DeepData server as MCP (Model
// Context Protocol) tools over stdio.
//
// It is a self-contained JSON-RPC 2.0 implementation over newline-delimited
// JSON with zero external dependencies: it wraps the canonical tenant HTTP
// protocol, so it requires no vectordb import and no engine build.
//
// Configuration (environment):
//
//	DEEPDATA_URL     base URL of the DeepData server (default http://127.0.0.1:8080)
//	DEEPDATA_TENANT  tenant id to operate on (default "mcp")
//	DEEPDATA_API_KEY optional bearer token
//
// Exposed tools:
//
//	search         hybrid/agent search with score_floor, fallback, usage_boost
//	insert         insert one document (caller-supplied id)
//	upsert         insert or replace one document by id
//	get_document   fetch one document by id
//	list_collections
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	mcpProtocolVersion = "2025-06-18"
	mcpServerName      = "deepdata-mcp"
	mcpServerVersion   = "0.1.0"
	defaultServerURL   = "http://127.0.0.1:8080"
	defaultTenant      = "mcp"
)

// ── JSON-RPC 2.0 wire types ──────────────────────────────────────────────

type rpcRequest struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string           `json:"method"`
	Params  json.RawMessage  `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id"`
	Result  any              `json:"result,omitempty"`
	Error   *rpcError        `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func newRPCError(code int, message string) *rpcError {
	return &rpcError{Code: code, Message: message}
}

// ── MCP result shapes ────────────────────────────────────────────────────

type toolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type toolCallResult struct {
	Content           []contentPart `json:"content"`
	IsError           bool          `json:"isError,omitempty"`
	StructuredContent any           `json:"structuredContent,omitempty"`
}

type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// serverError is the DeepData error envelope (internal/collection/API.md,
// section Errors). The text part tells the model what went wrong and what to
// do next; the raw envelope rides along as structuredContent for hosts that
// branch on the code.
type serverError struct {
	Code    string          `json:"code"`
	Message string          `json:"message"`
	Hint    string          `json:"hint,omitempty"`
	Raw     json.RawMessage `json:"-"`
}

func (e *serverError) Error() string {
	if e.Hint == "" {
		return e.Code + ": " + e.Message
	}
	return e.Code + ": " + e.Message + " Hint: " + e.Hint
}

// ── Server ───────────────────────────────────────────────────────────────

type mcpServer struct {
	base   string
	tenant string
	apiKey string
	client *http.Client
}

func newMCPServer() *mcpServer {
	base := os.Getenv("DEEPDATA_URL")
	if base == "" {
		base = defaultServerURL
	}
	tenant := os.Getenv("DEEPDATA_TENANT")
	if tenant == "" {
		tenant = defaultTenant
	}
	return &mcpServer{
		base:   strings.TrimRight(base, "/"),
		tenant: tenant,
		apiKey: os.Getenv("DEEPDATA_API_KEY"),
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (s *mcpServer) tenantPath(collection string, suffix string) string {
	return fmt.Sprintf("/v3/tenants/%s/collections/%s%s", s.tenant, collection, suffix)
}

// doHTTP issues a request against the DeepData server and returns the
// decoded JSON body (or an error string suitable for an isError result).
func (s *mcpServer) doHTTP(method, path string, body any) (json.RawMessage, error) {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("encode request: %w", err)
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequest(method, s.base+path, reader)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.apiKey)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		var envelope serverError
		if json.Unmarshal(raw, &envelope) == nil && envelope.Code != "" {
			envelope.Raw = raw
			return nil, &envelope
		}
		return nil, errors.New(strings.TrimSpace(string(raw)))
	}
	return raw, nil
}

func textResult(payload json.RawMessage) any {
	return toolCallResult{Content: []contentPart{{Type: "text", Text: string(payload)}}}
}

func errorResult(err error) any {
	result := toolCallResult{
		Content: []contentPart{{Type: "text", Text: err.Error()}},
		IsError: true,
	}
	var envelope *serverError
	if errors.As(err, &envelope) {
		result.StructuredContent = envelope.Raw
	}
	return result
}

// ── Dispatch ─────────────────────────────────────────────────────────────

func (s *mcpServer) handle(req rpcRequest) rpcResponse {
	resp := rpcResponse{JSONRPC: "2.0", ID: req.ID}
	var result any
	var rpcErr *rpcError

	switch req.Method {
	case "initialize":
		result = map[string]any{
			"protocolVersion": mcpProtocolVersion,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": mcpServerName, "version": mcpServerVersion},
		}
	case "ping":
		result = map[string]any{}
	case "tools/list":
		result = map[string]any{"tools": s.tools()}
	case "tools/call":
		var params struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &params); err != nil || params.Name == "" {
			return rpcResponse{JSONRPC: "2.0", ID: req.ID, Error: newRPCError(-32602, "invalid params: name required")}
		}
		out, ok, err := s.callTool(params.Name, params.Arguments)
		if err != nil {
			rpcErr = newRPCError(-32602, err.Error())
		} else if !ok {
			rpcErr = newRPCError(-32601, fmt.Sprintf("unknown tool: %s", params.Name))
		} else {
			result = out
		}
	case "notifications/initialized":
		// Notification: no response is sent by the caller loop.
		return rpcResponse{}
	default:
		rpcErr = newRPCError(-32601, fmt.Sprintf("method not found: %s", req.Method))
	}

	if rpcErr != nil {
		resp.Error = rpcErr
	} else {
		resp.Result = result
	}
	return resp
}

// hasResponse reports whether a response frame must be written for this
// request (notifications carry no id).
func (req rpcRequest) hasResponse() bool {
	return req.ID != nil
}

// ── Tools ────────────────────────────────────────────────────────────────

func objectSchema(required []string, properties map[string]any) map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}
}

func (s *mcpServer) tools() []toolDefinition {
	return []toolDefinition{
		{
			Name: "search",
			Description: "Search a DeepData collection (dense, sparse, or multi-field). " +
				"Give queries (field name -> vector) or texts (field name -> query text, " +
				"embedded by the server for fields created with an embedding binding; " +
				"the response reports embedded_by). " +
				"Agent-retrieval options: score_floor (confidence filter; the response " +
				"reports weak_match=true when nothing survives), fallback (auto-fallback " +
				"ladder to a secondary field when the primary is weak), usage_boost " +
				"(0..1 frecency blend; reported scores stay raw).",
			InputSchema: objectSchema([]string{"collection"}, map[string]any{
				"collection": map[string]any{"type": "string"},
				"queries": map[string]any{
					"type": "object",
				},
				"texts": map[string]any{
					"type": "object",
				},
				"top_k":       map[string]any{"type": "integer", "minimum": 1},
				"ef_search":   map[string]any{"type": "integer", "minimum": 0},
				"filters":     map[string]any{"type": "object"},
				"score_floor": map[string]any{"type": "number", "minimum": 0},
				"fallback":    map[string]any{"type": "object"},
				"usage_boost": map[string]any{"type": "number", "minimum": 0, "maximum": 0.999999},
			}),
		},
		{
			Name: "insert",
			Description: "Insert one document into a DeepData collection. " +
				"The document id is caller-supplied and must be non-zero. Give vectors " +
				"(field name -> vector) or texts (field name -> text, embedded by the " +
				"server for fields with an embedding binding); the server stores only " +
				"the vector, so keep the text in metadata if you want it back.",
			InputSchema: objectSchema([]string{"collection", "id"}, map[string]any{
				"collection": map[string]any{"type": "string"},
				"id":         map[string]any{"type": "integer", "minimum": 1},
				"vectors":    map[string]any{"type": "object"},
				"texts":      map[string]any{"type": "object"},
				"metadata":   map[string]any{"type": "object"},
			}),
		},
		{
			Name: "upsert",
			Description: "Insert or replace a document by caller-supplied id in a " +
				"DeepData collection. Takes vectors or texts as insert does.",
			InputSchema: objectSchema([]string{"collection", "id"}, map[string]any{
				"collection": map[string]any{"type": "string"},
				"id":         map[string]any{"type": "integer", "minimum": 1},
				"vectors":    map[string]any{"type": "object"},
				"texts":      map[string]any{"type": "object"},
				"metadata":   map[string]any{"type": "object"},
			}),
		},
		{
			Name:        "get_document",
			Description: "Fetch one document by id from a DeepData collection.",
			InputSchema: objectSchema([]string{"collection", "id"}, map[string]any{
				"collection": map[string]any{"type": "string"},
				"id":         map[string]any{"type": "integer", "minimum": 1},
			}),
		},
		{
			Name:        "list_collections",
			Description: "List collections available to the configured tenant.",
			InputSchema: objectSchema(nil, map[string]any{}),
		},
	}
}

func (s *mcpServer) callTool(name string, args map[string]any) (any, bool, error) {
	if args == nil {
		args = map[string]any{}
	}
	strArg := func(key string) (string, error) {
		value, ok := args[key].(string)
		if !ok || value == "" {
			return "", fmt.Errorf("%s is a required string argument", key)
		}
		return value, nil
	}

	switch name {
	case "search":
		collection, err := strArg("collection")
		if err != nil {
			return nil, true, err
		}
		payload, ok := fieldMaps(args, "queries", "texts")
		if !ok {
			return nil, true, errors.New("queries or texts is required: a non-empty object keyed by field name")
		}
		if v, ok := args["top_k"].(float64); ok {
			payload["top_k"] = int(v)
		}
		for _, key := range []string{"ef_search", "filters", "score_floor", "fallback", "usage_boost", "include_vectors"} {
			if value, present := args[key]; present {
				payload[key] = value
			}
		}
		raw, err := s.doHTTP(http.MethodPost, s.tenantPath(collection, "/search"), payload)
		if err != nil {
			return errorResult(err), true, nil
		}
		return textResult(raw), true, nil
	case "insert":
		collection, err := strArg("collection")
		if err != nil {
			return nil, true, err
		}
		id, ok := args["id"].(float64)
		if !ok || id < 1 {
			return nil, true, errors.New("id is a required positive integer argument")
		}
		payload, ok := docPayload(args)
		if !ok {
			return nil, true, errors.New("vectors or texts is required: a non-empty object keyed by field name")
		}
		payload["id"] = int64(id)
		raw, err := s.doHTTP(http.MethodPost, s.tenantPath(collection, "/docs"), payload)
		if err != nil {
			return errorResult(err), true, nil
		}
		return textResult(raw), true, nil
	case "upsert":
		collection, err := strArg("collection")
		if err != nil {
			return nil, true, err
		}
		id, ok := args["id"].(float64)
		if !ok || id < 1 {
			return nil, true, errors.New("id is a required positive integer argument")
		}
		// The upsert endpoint takes the id in the path, never the body.
		payload, ok := docPayload(args)
		if !ok {
			return nil, true, errors.New("vectors or texts is required: a non-empty object keyed by field name")
		}
		raw, err := s.doHTTP(http.MethodPut, s.tenantPath(collection, fmt.Sprintf("/docs/%d", int64(id))), payload)
		if err != nil {
			return errorResult(err), true, nil
		}
		return textResult(raw), true, nil
	case "get_document":
		collection, err := strArg("collection")
		if err != nil {
			return nil, true, err
		}
		id, ok := args["id"].(float64)
		if !ok || id < 1 {
			return nil, true, errors.New("id is a required positive integer argument")
		}
		raw, err := s.doHTTP(http.MethodGet, s.tenantPath(collection, fmt.Sprintf("/docs/%d", int64(id))), nil)
		if err != nil {
			return errorResult(err), true, nil
		}
		return textResult(raw), true, nil
	case "list_collections":
		raw, err := s.doHTTP(http.MethodGet, fmt.Sprintf("/v3/tenants/%s/collections", s.tenant), nil)
		if err != nil {
			return errorResult(err), true, nil
		}
		return textResult(raw), true, nil
	default:
		return nil, false, nil
	}
}

// fieldMaps copies the named field-keyed object arguments (vectors/queries and
// texts) that are present and non-empty; ok is false when none is.
func fieldMaps(args map[string]any, keys ...string) (map[string]any, bool) {
	payload := map[string]any{}
	for _, key := range keys {
		if m, ok := args[key].(map[string]any); ok && len(m) > 0 {
			payload[key] = m
		}
	}
	return payload, len(payload) > 0
}

// docPayload builds the {vectors, texts, metadata} document body, forwarding
// metadata only when the caller provided it. The id is added by insert and
// carried in the path by upsert.
func docPayload(args map[string]any) (map[string]any, bool) {
	payload, ok := fieldMaps(args, "vectors", "texts")
	if metadata, has := args["metadata"].(map[string]any); has {
		payload["metadata"] = metadata
	}
	return payload, ok
}

// ── stdio transport ──────────────────────────────────────────────────────

func main() {
	log.SetFlags(0)
	log.SetPrefix("deepdata-mcp: ")
	server := newMCPServer()
	if err := serveStdio(server, os.Stdin, os.Stdout); err != nil {
		log.Fatalf("%v", err)
	}
}

// serveStdio reads newline-delimited JSON-RPC messages from r and writes
// responses to w. It is separate from main for testability.
func serveStdio(server *mcpServer, r io.Reader, w io.Writer) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	encoder := json.NewEncoder(w)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var req rpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			_ = encoder.Encode(rpcResponse{
				JSONRPC: "2.0",
				ID:      nil,
				Error:   newRPCError(-32700, "parse error"),
			})
			continue
		}
		if req.JSONRPC != "2.0" {
			_ = encoder.Encode(rpcResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error:   newRPCError(-32600, "invalid request: jsonrpc must be 2.0"),
			})
			continue
		}
		resp := server.handle(req)
		if req.hasResponse() {
			if err := encoder.Encode(resp); err != nil {
				return err
			}
		}
	}
	return scanner.Err()
}
