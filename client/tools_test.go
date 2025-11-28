package client

import (
	"context"
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
}
