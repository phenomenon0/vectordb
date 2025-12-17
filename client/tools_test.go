package client

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"agentscope/core"
)

// InsertTool Tests

func TestInsertToolHappyPath(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/insert" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"doc123"}`))
	}))
	defer ts.Close()

	tool := &InsertTool{Client: New(ts.URL)}
	payload := &InsertRequest{
		Doc:  "test document",
		Meta: map[string]string{"type": "test"},
	}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v err=%s", res.Status, res.Error)
	}
	out, ok := res.Output.(*InsertOutput)
	if !ok {
		t.Fatalf("unexpected output type %T", res.Output)
	}
	if out.ID != "doc123" {
		t.Fatalf("unexpected id: %s", out.ID)
	}
}

func TestInsertToolNilClient(t *testing.T) {
	tool := &InsertTool{Client: nil}
	payload := &InsertRequest{Doc: "test"}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on nil client")
	}
	if res.Error != "nil vectordb client" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
}

func TestInsertToolBadInput(t *testing.T) {
	tool := &InsertTool{Client: New("http://localhost:0")}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: "not an insert request"}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on bad input")
	}
}

func TestInsertToolEmptyDoc(t *testing.T) {
	tool := &InsertTool{Client: New("http://localhost:0")}
	payload := &InsertRequest{Doc: ""} // Empty document
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on empty document")
	}
	if res.Error != "document text is required" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
}

func TestInsertToolWithMetadata(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"meta123"}`))
	}))
	defer ts.Close()

	tool := &InsertTool{Client: New(ts.URL)}
	payload := &InsertRequest{
		Doc: "doc with metadata",
		Meta: map[string]string{
			"agent_id":    "research-001",
			"conv_id":     "conv-123",
			"entity_type": "fact",
		},
	}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v", res.Status)
	}
}

// BatchInsertTool Tests

func TestBatchInsertToolHappyPath(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/batch_insert" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ids":["id1","id2","id3"]}`))
	}))
	defer ts.Close()

	tool := &BatchInsertTool{Client: New(ts.URL)}
	payload := &BatchInsertRequest{
		Docs: []BatchDoc{
			{Doc: "first doc", Meta: map[string]string{"type": "doc1"}},
			{Doc: "second doc", Meta: map[string]string{"type": "doc2"}},
			{Doc: "third doc", Meta: map[string]string{"type": "doc3"}},
		},
	}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v err=%s", res.Status, res.Error)
	}
	out, ok := res.Output.(*BatchInsertOutput)
	if !ok {
		t.Fatalf("unexpected output type %T", res.Output)
	}
	if len(out.IDs) != 3 {
		t.Fatalf("unexpected ids count: %d", len(out.IDs))
	}
	if out.Count != 3 {
		t.Fatalf("unexpected count: %d", out.Count)
	}
}

func TestBatchInsertToolNilClient(t *testing.T) {
	tool := &BatchInsertTool{Client: nil}
	payload := &BatchInsertRequest{Docs: []BatchDoc{{Doc: "test"}}}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on nil client")
	}
}

func TestBatchInsertToolBadInput(t *testing.T) {
	tool := &BatchInsertTool{Client: New("http://localhost:0")}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: "not a batch request"}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on bad input")
	}
}

func TestBatchInsertToolEmptyDocs(t *testing.T) {
	tool := &BatchInsertTool{Client: New("http://localhost:0")}
	payload := &BatchInsertRequest{Docs: []BatchDoc{}} // Empty docs array
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on empty docs")
	}
	if res.Error != "at least one document is required" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
}

func TestBatchInsertToolEmptyDocText(t *testing.T) {
	tool := &BatchInsertTool{Client: New("http://localhost:0")}
	payload := &BatchInsertRequest{
		Docs: []BatchDoc{
			{Doc: "valid doc"},
			{Doc: ""}, // Empty doc text
			{Doc: "another valid doc"},
		},
	}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on empty doc text")
	}
	if res.Error != "document 1: text is required" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
}

func TestBatchInsertToolWithUpsert(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ids":["upsert1","upsert2"]}`))
	}))
	defer ts.Close()

	tool := &BatchInsertTool{Client: New(ts.URL)}
	payload := &BatchInsertRequest{
		Docs: []BatchDoc{
			{ID: "existing1", Doc: "updated doc 1"},
			{ID: "existing2", Doc: "updated doc 2"},
		},
		Upsert: true,
	}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v", res.Status)
	}
}

// DeleteTool Tests

func TestDeleteToolHappyPath(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/delete" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"deleted":"doc123"}`))
	}))
	defer ts.Close()

	tool := &DeleteTool{Client: New(ts.URL)}
	payload := &DeleteRequest{ID: "doc123"}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v err=%s", res.Status, res.Error)
	}
	out, ok := res.Output.(*DeleteOutput)
	if !ok {
		t.Fatalf("unexpected output type %T", res.Output)
	}
	if out.DeletedID != "doc123" {
		t.Fatalf("unexpected deleted id: %s", out.DeletedID)
	}
	if !out.Success {
		t.Fatalf("expected success to be true")
	}
}

func TestDeleteToolNilClient(t *testing.T) {
	tool := &DeleteTool{Client: nil}
	payload := &DeleteRequest{ID: "doc123"}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on nil client")
	}
}

func TestDeleteToolBadInput(t *testing.T) {
	tool := &DeleteTool{Client: New("http://localhost:0")}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: "not a delete request"}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on bad input")
	}
}

func TestDeleteToolEmptyID(t *testing.T) {
	tool := &DeleteTool{Client: New("http://localhost:0")}
	payload := &DeleteRequest{ID: ""} // Empty ID
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on empty ID")
	}
	if res.Error != "document ID is required" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
}

func TestDeleteToolNotFound(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"deleted":""}`)) // Empty deleted ID indicates not found
	}))
	defer ts.Close()

	tool := &DeleteTool{Client: New(ts.URL)}
	payload := &DeleteRequest{ID: "nonexistent"}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v", res.Status)
	}
	out, ok := res.Output.(*DeleteOutput)
	if !ok {
		t.Fatalf("unexpected output type %T", res.Output)
	}
	if out.Success {
		t.Fatalf("expected success to be false for nonexistent document")
	}
}

// Tool Name Tests

func TestToolNames(t *testing.T) {
	insertTool := &InsertTool{}
	if insertTool.Name() != "vectordb_insert" {
		t.Fatalf("unexpected insert tool name: %s", insertTool.Name())
	}

	batchTool := &BatchInsertTool{}
	if batchTool.Name() != "vectordb_batch_insert" {
		t.Fatalf("unexpected batch insert tool name: %s", batchTool.Name())
	}

	deleteTool := &DeleteTool{}
	if deleteTool.Name() != "vectordb_delete" {
		t.Fatalf("unexpected delete tool name: %s", deleteTool.Name())
	}

	fileSearchTool := &FileSearchTool{}
	if fileSearchTool.Name() != "file_search" {
		t.Fatalf("unexpected file search tool name: %s", fileSearchTool.Name())
	}
}

// FileSearchTool Tests

func TestFileSearchToolHappyPath(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/query" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"ids": ["chunk1", "chunk2"],
			"docs": ["def greet():\n    print('hello')", "func main() {\n    fmt.Println(\"hi\")\n}"],
			"scores": [0.92, 0.85],
			"meta": [
				{"path": "/home/user/code/hello.py", "filename": "hello.py", "anchor_type": "line", "anchor_start": "1", "anchor_end": "10", "mtime": "2025-12-17T10:00:00Z"},
				{"path": "/home/user/code/main.go", "filename": "main.go", "anchor_type": "line", "anchor_start": "5", "anchor_end": "15", "mtime": "2025-12-16T08:30:00Z"}
			]
		}`))
	}))
	defer ts.Close()

	tool := &FileSearchTool{Client: New(ts.URL)}
	payload := &FileSearchInput{
		Query: "greeting function",
		TopK:  5,
	}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v err=%s", res.Status, res.Error)
	}
	out, ok := res.Output.(*FileSearchOutput)
	if !ok {
		t.Fatalf("unexpected output type %T", res.Output)
	}
	if out.Total != 2 {
		t.Fatalf("unexpected total: %d", out.Total)
	}
	if len(out.Matches) != 2 {
		t.Fatalf("unexpected matches count: %d", len(out.Matches))
	}

	// Check first match
	m := out.Matches[0]
	if m.Score != 0.92 {
		t.Fatalf("unexpected score: %f", m.Score)
	}
	if m.ID != "chunk1" {
		t.Fatalf("unexpected id: %s", m.ID)
	}
	if m.Meta == nil {
		t.Fatalf("expected meta to be set")
	}
	if m.Meta.Path != "/home/user/code/hello.py" {
		t.Fatalf("unexpected path: %s", m.Meta.Path)
	}
	if m.Meta.Filename != "hello.py" {
		t.Fatalf("unexpected filename: %s", m.Meta.Filename)
	}
	if m.Meta.AnchorType != "line" {
		t.Fatalf("unexpected anchor_type: %s", m.Meta.AnchorType)
	}
	if m.Meta.AnchorStart != 1 {
		t.Fatalf("unexpected anchor_start: %d", m.Meta.AnchorStart)
	}
	if m.Meta.AnchorEnd != 10 {
		t.Fatalf("unexpected anchor_end: %d", m.Meta.AnchorEnd)
	}
}

func TestFileSearchToolNilClient(t *testing.T) {
	tool := &FileSearchTool{Client: nil}
	payload := &FileSearchInput{Query: "test"}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on nil client")
	}
	if res.Error != "nil vectordb client" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
}

func TestFileSearchToolBadInput(t *testing.T) {
	tool := &FileSearchTool{Client: New("http://localhost:0")}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: "not a file search input"}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on bad input")
	}
}

func TestFileSearchToolEmptyQuery(t *testing.T) {
	tool := &FileSearchTool{Client: New("http://localhost:0")}
	payload := &FileSearchInput{Query: ""} // Empty query
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)

	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on empty query")
	}
	if res.Error != "query text is required" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
}

func TestFileSearchToolWithPathPrefix(t *testing.T) {
	var receivedMeta map[string]string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Parse the request to check path_prefix was included
		var req QueryRequest
		if err := decodeJSON(r.Body, &req); err == nil {
			receivedMeta = req.Meta
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ids":[],"docs":[],"scores":[]}`))
	}))
	defer ts.Close()

	tool := &FileSearchTool{Client: New(ts.URL)}
	payload := &FileSearchInput{
		Query:      "test query",
		PathPrefix: "/home/user/Documents/",
	}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	_ = tool.Execute(ctx)

	if receivedMeta == nil {
		t.Fatalf("expected meta to be set")
	}
	if receivedMeta["path_prefix"] != "/home/user/Documents/" {
		t.Fatalf("unexpected path_prefix: %s", receivedMeta["path_prefix"])
	}
}

func TestFileSearchToolDefaults(t *testing.T) {
	var receivedReq QueryRequest
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = decodeJSON(r.Body, &receivedReq)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ids":[],"docs":[],"scores":[]}`))
	}))
	defer ts.Close()

	tool := &FileSearchTool{Client: New(ts.URL)}
	payload := &FileSearchInput{
		Query: "test query",
		// No collection, topK specified - should use defaults
	}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	_ = tool.Execute(ctx)

	if receivedReq.Collection != "localvault" {
		t.Fatalf("expected default collection 'localvault', got: %s", receivedReq.Collection)
	}
	if receivedReq.TopK != 10 {
		t.Fatalf("expected default top_k 10, got: %d", receivedReq.TopK)
	}
	if !receivedReq.IncludeMeta {
		t.Fatalf("expected include_meta to be true")
	}
}

// decodeJSON helper for tests
func decodeJSON(r io.Reader, v any) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}
