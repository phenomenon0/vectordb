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
	"regexp"
	"strings"
	"time"

	"github.com/phenomenon0/vectordb/api/contract"
)

const (
	mcpProtocolVersion = "2025-06-18"
	mcpServerName      = "deepdata-mcp"
	mcpServerVersion   = "0.2.0"
	defaultServerURL   = "http://127.0.0.1:8080"
	defaultTenant      = "mcp"
	defaultCollection  = "memory"

	maxRecallTopK    = 50
	defaultTopK      = 10
	defaultMaxChars  = 8000
	minMaxChars      = 500
	maxRememberItems = 100
	maxGetIDs        = 50
	conciseMaxRunes  = 300
	sparseMemoryDim  = 10000

	contractURI = "deepdata://contract"
	statusURI   = "deepdata://status"
)

var identifierRE = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

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
	Name         string          `json:"name"`
	Title        string          `json:"title,omitempty"`
	Description  string          `json:"description"`
	InputSchema  json.RawMessage `json:"inputSchema"`
	OutputSchema json.RawMessage `json:"outputSchema,omitempty"`
	Annotations  map[string]any  `json:"annotations,omitempty"`
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

// toolSchema is one api/contract/v3/schemas/<tool>.json split into the two
// halves MCP serves, plus the description lifted from the input schema.
type toolSchema struct {
	Input       json.RawMessage
	Output      json.RawMessage
	Description string
}

var schemas = loadSchemas()

func loadSchemas() map[string]toolSchema {
	out := map[string]toolSchema{}
	for _, name := range contract.Names() {
		raw, err := contract.Schema(name)
		if err != nil {
			panic(err)
		}
		var pair struct {
			Input  json.RawMessage `json:"input"`
			Output json.RawMessage `json:"output"`
		}
		if err := json.Unmarshal(raw, &pair); err != nil {
			panic(fmt.Sprintf("contract schema %s: %v", name, err))
		}
		var head struct {
			Description string `json:"description"`
		}
		_ = json.Unmarshal(pair.Input, &head)
		out[name] = toolSchema{Input: pair.Input, Output: pair.Output, Description: head.Description}
	}
	return out
}

// serverError is the DeepData error envelope (internal/collection/API.md,
// section Errors). The text part tells the model what went wrong and what to
// do next; the envelope rides along as structuredContent for hosts that
// branch on the code. Local semantic failures use the same shape.
type serverError struct {
	Code      string          `json:"code"`
	Message   string          `json:"message"`
	Hint      string          `json:"hint,omitempty"`
	Retryable bool            `json:"retryable"`
	Status    int             `json:"-"`
	Raw       json.RawMessage `json:"-"`
}

func (e *serverError) Error() string {
	if e.Hint == "" {
		return e.Code + ": " + e.Message
	}
	return e.Code + ": " + e.Message + " Hint: " + e.Hint
}

func localError(code, message, hint string) *serverError {
	return &serverError{Code: code, Message: message, Hint: hint}
}

// isDocNotFound distinguishes a missing document (which forget and get treat
// as a normal outcome) from a missing collection (which is an error).
func isDocNotFound(err error) bool {
	var envelope *serverError
	return errors.As(err, &envelope) && envelope.Code == "not_found" && strings.HasPrefix(envelope.Message, "document")
}

// ── Server ───────────────────────────────────────────────────────────────

type mcpServer struct {
	base       string
	tenant     string
	collection string
	apiKey     string
	client     *http.Client
	// infos caches collection schemas for text routing; stdio serves one
	// request at a time, so no lock. Any 4xx for a collection drops its entry.
	infos map[string]*collectionInfo
}

type collectionInfo struct {
	Name        string          `json:"name"`
	Fields      []fieldInfo     `json:"fields"`
	Description string          `json:"description,omitempty"`
	Metadata    json.RawMessage `json:"metadata,omitempty"`
	DocCount    int             `json:"doc_count"`
}

type fieldInfo struct {
	Name  string          `json:"name"`
	Type  string          `json:"type"`
	Dim   int             `json:"dim"`
	Index json.RawMessage `json:"index,omitempty"`
	// ScoreDirection tells the agent how to read this field's scores:
	// "lower_is_better" on dense distances, "higher_is_better" on sparse.
	ScoreDirection string           `json:"score_direction,omitempty"`
	Embedding      *embeddingConfig `json:"embedding,omitempty"`
}

type embeddingConfig struct {
	Provider string `json:"provider"`
	Model    string `json:"model,omitempty"`
}

func newMCPServer() *mcpServer {
	env := func(key, fallback string) string {
		if v := os.Getenv(key); v != "" {
			return v
		}
		return fallback
	}
	return &mcpServer{
		base:       strings.TrimRight(env("DEEPDATA_URL", defaultServerURL), "/"),
		tenant:     env("DEEPDATA_TENANT", defaultTenant),
		collection: env("DEEPDATA_COLLECTION", defaultCollection),
		apiKey:     os.Getenv("DEEPDATA_API_KEY"),
		client:     &http.Client{Timeout: 30 * time.Second},
		infos:      map[string]*collectionInfo{},
	}
}

func (s *mcpServer) tenantPath(collection string, suffix string) string {
	return fmt.Sprintf("/v3/tenants/%s/collections/%s%s", s.tenant, collection, suffix)
}

// doHTTP issues a request against the DeepData server and returns the JSON
// body and status. Non-2xx bodies become a *serverError when they carry the
// envelope, otherwise a plain error with the trimmed body.
func (s *mcpServer) doHTTP(method, path string, body any) (json.RawMessage, int, error) {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, 0, fmt.Errorf("encode request: %w", err)
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequest(method, s.base+path, reader)
	if err != nil {
		return nil, 0, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.apiKey)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		var envelope serverError
		if json.Unmarshal(raw, &envelope) == nil && envelope.Code != "" {
			envelope.Raw = raw
			envelope.Status = resp.StatusCode
			return nil, resp.StatusCode, &envelope
		}
		return nil, resp.StatusCode, errors.New(strings.TrimSpace(string(raw)))
	}
	return raw, resp.StatusCode, nil
}

// call is doHTTP scoped to one collection: a 4xx means our cached view of
// that collection may be wrong, so the cache entry goes.
func (s *mcpServer) call(collection, method, suffix string, body any) (json.RawMessage, error) {
	raw, status, err := s.doHTTP(method, s.tenantPath(collection, suffix), body)
	if status >= 400 && status < 500 {
		delete(s.infos, collection)
	}
	return raw, err
}

func (s *mcpServer) info(collection string) (*collectionInfo, error) {
	if cached := s.infos[collection]; cached != nil {
		return cached, nil
	}
	raw, err := s.call(collection, http.MethodGet, "", nil)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Collection *collectionInfo `json:"collection"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil || resp.Collection == nil {
		return nil, fmt.Errorf("decode collection %q: unexpected response %s", collection, raw)
	}
	s.infos[collection] = resp.Collection
	return resp.Collection, nil
}

func (s *mcpServer) collectionOr(name string) string {
	if name == "" {
		return s.collection
	}
	return name
}

// boundFields returns the embedding-bound field names, split by type.
func boundFields(info *collectionInfo) (dense, sparse, all []string) {
	for _, f := range info.Fields {
		if f.Embedding == nil {
			continue
		}
		all = append(all, f.Name)
		switch f.Type {
		case "dense":
			dense = append(dense, f.Name)
		case "sparse":
			sparse = append(sparse, f.Name)
		}
	}
	return dense, sparse, all
}

func noBoundFieldError(info *collectionInfo, verb string) *serverError {
	return localError("invalid_argument",
		fmt.Sprintf("collection %q has no embedding-bound field, so text cannot be embedded for %s", info.Name, verb),
		"pass vectors (field name -> vector) instead, or create a collection with preset \"memory\" via deepdata_create_collection")
}

func structuredResult(v any) any {
	text, _ := json.Marshal(v)
	return toolCallResult{Content: []contentPart{{Type: "text", Text: string(text)}}, StructuredContent: v}
}

func errorResult(err error) any {
	result := toolCallResult{
		Content: []contentPart{{Type: "text", Text: err.Error()}},
		IsError: true,
	}
	var envelope *serverError
	if errors.As(err, &envelope) {
		if envelope.Raw != nil {
			result.StructuredContent = envelope.Raw
		} else {
			result.StructuredContent = envelope
		}
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
			"capabilities":    map[string]any{"tools": map[string]any{}, "resources": map[string]any{}},
			"serverInfo":      map[string]any{"name": mcpServerName, "version": mcpServerVersion},
		}
	case "ping":
		result = map[string]any{}
	case "tools/list":
		result = map[string]any{"tools": s.tools()}
	case "tools/call":
		var params struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
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
	case "resources/list":
		result = map[string]any{"resources": []map[string]any{
			{"uri": contractURI, "name": "DeepData agent contract", "mimeType": "text/markdown",
				"description": "The six verbs, texts vs vectors, how to read a recall result, error codes, limits."},
			{"uri": statusURI, "name": "DeepData server status", "mimeType": "application/json",
				"description": "Live readiness, named checks, configured embedder (provider:model or none), version."},
		}}
	case "resources/read":
		var params struct {
			URI string `json:"uri"`
		}
		_ = json.Unmarshal(req.Params, &params)
		switch params.URI {
		case contractURI:
			result = resourceContents(contractURI, "text/markdown", contract.Markdown)
		case statusURI:
			raw, _, err := s.doHTTP(http.MethodGet, "/v3/status", nil)
			text := string(raw)
			if err != nil {
				text = err.Error() // a 503 body is still the status the agent asked for
			}
			result = resourceContents(statusURI, "application/json", text)
		default:
			rpcErr = newRPCError(-32002, fmt.Sprintf("resource not found: %s", params.URI))
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

func resourceContents(uri, mimeType, text string) map[string]any {
	return map[string]any{"contents": []map[string]any{{"uri": uri, "mimeType": mimeType, "text": text}}}
}

// hasResponse reports whether a response frame must be written for this
// request (notifications carry no id).
func (req rpcRequest) hasResponse() bool {
	return req.ID != nil
}

// ── Tools ────────────────────────────────────────────────────────────────

// tools lists the six verbs. Schemas and descriptions come from api/contract;
// only names, titles and safety annotations live here. Keep one Name per line:
// scripts/check_docs_contract.py reads them for the docs/mcp.md tool list.
func (s *mcpServer) tools() []toolDefinition {
	readOnly := map[string]any{"readOnlyHint": true, "idempotentHint": true, "openWorldHint": false}
	additive := map[string]any{"destructiveHint": false, "openWorldHint": false}
	defs := []toolDefinition{
		{
			Name:        "deepdata_recall",
			Title:       "Recall memories",
			Annotations: readOnly,
		},
		{
			Name:        "deepdata_remember",
			Title:       "Remember items",
			Annotations: additive,
		},
		{
			Name:        "deepdata_forget",
			Title:       "Forget one document",
			Annotations: map[string]any{"destructiveHint": true, "idempotentHint": true, "openWorldHint": false},
		},
		{
			Name:        "deepdata_get",
			Title:       "Get documents by id",
			Annotations: readOnly,
		},
		{
			Name:        "deepdata_collections",
			Title:       "Describe or list collections",
			Annotations: readOnly,
		},
		{
			Name:        "deepdata_create_collection",
			Title:       "Create a collection",
			Annotations: additive,
		},
	}
	for i := range defs {
		schema, ok := schemas[defs[i].Name]
		if !ok {
			panic("no contract schema for tool " + defs[i].Name)
		}
		defs[i].Description = schema.Description
		defs[i].InputSchema = schema.Input
		defs[i].OutputSchema = schema.Output
	}
	return defs
}

// decodeArgs decodes tool arguments strictly: unknown or mistyped fields are
// an invalid-params RPC error, not a tool result.
func decodeArgs(raw json.RawMessage, into any) error {
	if len(raw) == 0 || string(raw) == "null" {
		raw = []byte("{}")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(into); err != nil {
		return fmt.Errorf("invalid arguments: %v", err)
	}
	return nil
}

func (s *mcpServer) callTool(name string, args json.RawMessage) (any, bool, error) {
	var out any
	var err error
	switch name {
	case "deepdata_recall":
		out, err = s.recall(args)
	case "deepdata_remember":
		out, err = s.remember(args)
	case "deepdata_forget":
		out, err = s.forget(args)
	case "deepdata_get":
		out, err = s.get(args)
	case "deepdata_collections":
		out, err = s.collections(args)
	case "deepdata_create_collection":
		out, err = s.createCollection(args)
	default:
		return nil, false, nil
	}
	return out, true, err
}

// ── deepdata_recall ──────────────────────────────────────────────────────

type fallbackParams struct {
	Primary   string  `json:"primary"`
	Secondary string  `json:"secondary"`
	Threshold float64 `json:"threshold,omitempty"`
}

type recallArgs struct {
	Query          string                     `json:"query"`
	Queries        map[string]json.RawMessage `json:"queries"`
	Collection     string                     `json:"collection"`
	TopK           int                        `json:"top_k"`
	Filters        json.RawMessage            `json:"filters"`
	ScoreFloor     float64                    `json:"score_floor"`
	Fallback       *fallbackParams            `json:"fallback"`
	UsageBoost     float64                    `json:"usage_boost"`
	ResponseFormat string                     `json:"response_format"`
	MaxChars       int                        `json:"max_chars"`
}

type recallHit struct {
	ID       uint64         `json:"id"`
	Score    float64        `json:"score"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type recallOutput struct {
	Hits           []recallHit       `json:"hits"`
	BestScore      float64           `json:"best_score"`
	ScoreDirection string            `json:"score_direction,omitempty"`
	WeakMatch      bool              `json:"weak_match"`
	FellBackTo     string            `json:"fell_back_to,omitempty"`
	EmbeddedBy     map[string]string `json:"embedded_by,omitempty"`
	Truncated      bool              `json:"truncated,omitempty"`
	Hint           string            `json:"hint,omitempty"`
}

func (s *mcpServer) recall(raw json.RawMessage) (any, error) {
	var a recallArgs
	if err := decodeArgs(raw, &a); err != nil {
		return nil, err
	}
	if (a.Query == "") == (len(a.Queries) == 0) {
		return nil, errors.New("exactly one of query (text) or queries (field name -> vector) is required")
	}
	if a.TopK == 0 {
		a.TopK = defaultTopK
	}
	if a.TopK < 1 || a.TopK > maxRecallTopK {
		return nil, fmt.Errorf("top_k must be 1..%d", maxRecallTopK)
	}
	if a.MaxChars == 0 {
		a.MaxChars = defaultMaxChars
	}
	if a.MaxChars < minMaxChars {
		return nil, fmt.Errorf("max_chars must be at least %d", minMaxChars)
	}
	switch a.ResponseFormat {
	case "", "concise", "detailed":
	default:
		return nil, errors.New("response_format must be concise or detailed")
	}
	collection := s.collectionOr(a.Collection)

	body := map[string]any{"top_k": a.TopK, "include_vectors": false}
	if a.Filters != nil {
		body["filters"] = a.Filters
	}
	if a.ScoreFloor > 0 {
		body["score_floor"] = a.ScoreFloor
	}
	if a.UsageBoost > 0 {
		body["usage_boost"] = a.UsageBoost
	}
	if a.Fallback != nil {
		body["fallback"] = a.Fallback
	}
	if a.Query != "" {
		info, err := s.info(collection)
		if err != nil {
			return errorResult(err), nil
		}
		dense, sparse, all := boundFields(info)
		switch {
		case len(all) == 0:
			return errorResult(noBoundFieldError(info, "recall")), nil
		case len(all) > 2:
			return errorResult(localError("invalid_argument",
				fmt.Sprintf("collection %q binds %d fields to embeddings; a search reads at most 2", info.Name, len(all)),
				"pass queries (field name -> vector) naming at most two fields")), nil
		}
		texts := map[string]string{}
		for _, f := range all {
			texts[f] = a.Query
		}
		body["texts"] = texts
		// The memory preset shape: dense answers first, sparse is the safety net.
		if a.Fallback == nil && len(dense) == 1 && len(sparse) == 1 {
			body["fallback"] = fallbackParams{Primary: dense[0], Secondary: sparse[0]}
		}
	} else {
		body["queries"] = a.Queries
	}

	raw, err := s.call(collection, http.MethodPost, "/search", body)
	if err != nil {
		return errorResult(err), nil
	}
	var sr struct {
		Documents []struct {
			ID       uint64         `json:"id"`
			Metadata map[string]any `json:"metadata"`
		} `json:"documents"`
		Scores         []float64         `json:"scores"`
		BestScore      float64           `json:"best_score"`
		ScoreDirection string            `json:"score_direction"`
		WeakMatch      bool              `json:"weak_match"`
		FellBackTo     string            `json:"fell_back_to"`
		EmbeddedBy     map[string]string `json:"embedded_by"`
	}
	if err := json.Unmarshal(raw, &sr); err != nil {
		return nil, fmt.Errorf("decode search response: %v", err)
	}
	out := recallOutput{
		Hits:           make([]recallHit, 0, len(sr.Documents)),
		BestScore:      sr.BestScore,
		ScoreDirection: sr.ScoreDirection,
		WeakMatch:      sr.WeakMatch,
		FellBackTo:     sr.FellBackTo,
		EmbeddedBy:     sr.EmbeddedBy,
	}
	for i, doc := range sr.Documents {
		hit := recallHit{ID: doc.ID, Metadata: doc.Metadata}
		if i < len(sr.Scores) {
			hit.Score = sr.Scores[i]
		}
		if a.ResponseFormat != "detailed" {
			elide(hit.Metadata)
		}
		out.Hits = append(out.Hits, hit)
	}
	if out.WeakMatch {
		out.Hint = "score_floor filtered every hit; best_score shows the closest one. Loosen or drop score_floor to see candidates."
	}
	// Drop tail hits until the result fits max_chars; the first hit always
	// survives so a tight budget still answers.
	var dropped []uint64
	for {
		if len(dropped) > 0 {
			out.Truncated = true
			out.Hint = fmt.Sprintf("%d hit(s) dropped to fit max_chars=%d (ids %v); narrow with filters, lower top_k, or raise max_chars",
				len(dropped), a.MaxChars, dropped)
		}
		encoded, _ := json.Marshal(out)
		if len(encoded) <= a.MaxChars || len(out.Hits) <= 1 {
			break
		}
		last := len(out.Hits) - 1
		dropped = append(dropped, out.Hits[last].ID)
		out.Hits = out.Hits[:last]
	}
	return structuredResult(out), nil
}

// elide shortens long strings in place for concise mode.
func elide(v any) any {
	switch t := v.(type) {
	case string:
		if r := []rune(t); len(r) > conciseMaxRunes {
			return string(r[:conciseMaxRunes]) + "…"
		}
	case map[string]any:
		for k, x := range t {
			t[k] = elide(x)
		}
	case []any:
		for i, x := range t {
			t[i] = elide(x)
		}
	}
	return v
}

// ── deepdata_remember ────────────────────────────────────────────────────

type rememberItem struct {
	Text     string                     `json:"text"`
	Vectors  map[string]json.RawMessage `json:"vectors"`
	Metadata map[string]any             `json:"metadata"`
	ID       uint64                     `json:"id"`
}

type rememberArgs struct {
	Items      []rememberItem `json:"items"`
	Collection string         `json:"collection"`
}

func (s *mcpServer) remember(raw json.RawMessage) (any, error) {
	var a rememberArgs
	if err := decodeArgs(raw, &a); err != nil {
		return nil, err
	}
	if len(a.Items) == 0 || len(a.Items) > maxRememberItems {
		return nil, fmt.Errorf("items must hold 1..%d entries", maxRememberItems)
	}
	collection := s.collectionOr(a.Collection)

	var info *collectionInfo
	docs := make([]map[string]any, len(a.Items))
	for i, item := range a.Items {
		if (item.Text == "") == (len(item.Vectors) == 0) {
			return nil, fmt.Errorf("items[%d]: exactly one of text or vectors is required", i)
		}
		doc := map[string]any{}
		if item.Text != "" {
			if info == nil {
				var err error
				if info, err = s.info(collection); err != nil {
					return errorResult(err), nil
				}
			}
			_, _, all := boundFields(info)
			if len(all) == 0 {
				return errorResult(noBoundFieldError(info, "remember")), nil
			}
			texts := map[string]string{}
			for _, f := range all {
				texts[f] = item.Text
			}
			doc["texts"] = texts
			metadata := map[string]any{}
			for k, v := range item.Metadata {
				metadata[k] = v
			}
			metadata["text"] = item.Text
			doc["metadata"] = metadata
		} else {
			doc["vectors"] = item.Vectors
			if item.Metadata != nil {
				doc["metadata"] = item.Metadata
			}
		}
		docs[i] = doc
	}

	// Replacements first (idempotent), then the one atomic insert; a failure
	// anywhere leaves a state the same call can be re-run against.
	ids := make([]uint64, len(a.Items))
	var fresh []int
	for i, item := range a.Items {
		if item.ID == 0 {
			fresh = append(fresh, i)
			continue
		}
		if _, err := s.call(collection, http.MethodPut, fmt.Sprintf("/docs/%d", item.ID), docs[i]); err != nil {
			return errorResult(err), nil
		}
		ids[i] = item.ID
	}
	switch len(fresh) {
	case 0:
	case 1:
		raw, err := s.call(collection, http.MethodPost, "/docs", docs[fresh[0]])
		if err != nil {
			return errorResult(err), nil
		}
		var resp struct {
			ID uint64 `json:"id"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil {
			return nil, fmt.Errorf("decode insert response: %v", err)
		}
		ids[fresh[0]] = resp.ID
	default:
		batch := make([]map[string]any, len(fresh))
		for j, i := range fresh {
			batch[j] = docs[i]
		}
		raw, err := s.call(collection, http.MethodPost, "/docs/batch", map[string]any{"documents": batch})
		if err != nil {
			return errorResult(err), nil
		}
		var resp struct {
			IDs []uint64 `json:"ids"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil || len(resp.IDs) != len(fresh) {
			return nil, fmt.Errorf("decode batch response: expected %d ids, got %s", len(fresh), raw)
		}
		for j, i := range fresh {
			ids[i] = resp.IDs[j]
		}
	}
	return structuredResult(map[string]any{"ids": ids, "count": len(ids)}), nil
}

// ── deepdata_forget ──────────────────────────────────────────────────────

func (s *mcpServer) forget(raw json.RawMessage) (any, error) {
	var a struct {
		ID         uint64 `json:"id"`
		Collection string `json:"collection"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return nil, err
	}
	if a.ID == 0 {
		return nil, errors.New("id is a required positive integer")
	}
	_, err := s.call(s.collectionOr(a.Collection), http.MethodDelete, "/docs", map[string]any{"doc_id": a.ID})
	if err != nil && !isDocNotFound(err) {
		return errorResult(err), nil
	}
	return structuredResult(map[string]any{"deleted": a.ID}), nil
}

// ── deepdata_get ─────────────────────────────────────────────────────────

func (s *mcpServer) get(raw json.RawMessage) (any, error) {
	var a struct {
		IDs        []uint64 `json:"ids"`
		Collection string   `json:"collection"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return nil, err
	}
	if len(a.IDs) == 0 || len(a.IDs) > maxGetIDs {
		return nil, fmt.Errorf("ids must hold 1..%d entries", maxGetIDs)
	}
	collection := s.collectionOr(a.Collection)
	type document struct {
		ID       uint64         `json:"id"`
		Metadata map[string]any `json:"metadata,omitempty"`
	}
	documents := make([]document, 0, len(a.IDs))
	missing := []uint64{}
	for _, id := range a.IDs {
		if id == 0 {
			return nil, errors.New("ids must be positive integers")
		}
		raw, err := s.call(collection, http.MethodGet, fmt.Sprintf("/docs/%d", id), nil)
		if isDocNotFound(err) {
			missing = append(missing, id)
			continue
		}
		if err != nil {
			return errorResult(err), nil
		}
		var doc document
		if err := json.Unmarshal(raw, &doc); err != nil {
			return nil, fmt.Errorf("decode document %d: %v", id, err)
		}
		documents = append(documents, doc)
	}
	return structuredResult(map[string]any{"documents": documents, "missing": missing}), nil
}

// ── deepdata_collections ─────────────────────────────────────────────────

func (s *mcpServer) collections(raw json.RawMessage) (any, error) {
	var a struct {
		Name string `json:"name"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return nil, err
	}
	var infos []*collectionInfo
	if a.Name != "" {
		if !identifierRE.MatchString(a.Name) {
			return nil, errors.New("name must match [A-Za-z0-9_-]{1,64}")
		}
		delete(s.infos, a.Name) // the caller wants the live view, not the cache
		info, err := s.info(a.Name)
		if err != nil {
			return errorResult(err), nil
		}
		infos = []*collectionInfo{info}
	} else {
		raw, _, err := s.doHTTP(http.MethodGet, fmt.Sprintf("/v3/tenants/%s/collections", s.tenant), nil)
		if err != nil {
			return errorResult(err), nil
		}
		var resp struct {
			Collections []*collectionInfo `json:"collections"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil {
			return nil, fmt.Errorf("decode collections: %v", err)
		}
		infos = resp.Collections
		for _, info := range infos {
			s.infos[info.Name] = info
		}
	}
	if infos == nil {
		infos = []*collectionInfo{}
	}
	return structuredResult(map[string]any{"collections": infos}), nil
}

// ── deepdata_create_collection ───────────────────────────────────────────

func (s *mcpServer) createCollection(raw json.RawMessage) (any, error) {
	var a struct {
		Name        string            `json:"name"`
		Preset      string            `json:"preset"`
		Fields      []json.RawMessage `json:"fields"`
		Description string            `json:"description"`
		Durability  string            `json:"durability"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return nil, err
	}
	if !identifierRE.MatchString(a.Name) {
		return nil, errors.New("name must match [A-Za-z0-9_-]{1,64}")
	}
	if (a.Preset == "") == (len(a.Fields) == 0) {
		return nil, errors.New("exactly one of preset or fields is required")
	}
	var fields any = a.Fields
	if a.Preset != "" {
		if a.Preset != "memory" {
			return nil, errors.New("preset must be \"memory\"")
		}
		raw, _, err := s.doHTTP(http.MethodGet, "/readyz", nil)
		if err != nil {
			return errorResult(err), nil
		}
		var ready struct {
			Embedder string `json:"embedder"`
		}
		if err := json.Unmarshal(raw, &ready); err != nil {
			return nil, fmt.Errorf("decode /readyz: %v", err)
		}
		if ready.Embedder == "" || ready.Embedder == "none" {
			return errorResult(localError("embedder_unavailable",
				"the server has no text embedder, so preset \"memory\" cannot bind its text field",
				"start the server with DEEPDATA_EMBEDDER set, or pass explicit fields and send vectors")), nil
		}
		provider, model, _ := strings.Cut(ready.Embedder, ":")
		fields = []map[string]any{
			{"name": "text", "type": "dense", "index": map[string]any{"type": "hnsw"},
				"embedding": map[string]any{"provider": provider, "model": model}},
			{"name": "keywords", "type": "sparse", "dim": sparseMemoryDim, "index": map[string]any{"type": "inverted"},
				"embedding": map[string]any{"provider": "bm25"}},
		}
	}
	body := map[string]any{"name": a.Name, "fields": fields}
	if a.Description != "" {
		body["description"] = a.Description
	}
	if a.Durability != "" {
		body["durability"] = a.Durability
	}
	if _, _, err := s.doHTTP(http.MethodPost, fmt.Sprintf("/v3/tenants/%s/collections", s.tenant), body); err != nil {
		return errorResult(err), nil
	}
	delete(s.infos, a.Name)
	info, err := s.info(a.Name)
	if err != nil {
		return errorResult(err), nil
	}
	return structuredResult(map[string]any{"created": a.Name, "fields": info.Fields}), nil
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
