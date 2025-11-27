package client

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"agentscope/core"
)

func TestQueryToolHappyPath(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/query" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ids":["1"],"docs":["doc"],"scores":[0.8]}`))
	}))
	defer ts.Close()

	tool := &QueryTool{Client: New(ts.URL)}
	payload := &QueryRequest{Query: "hi", TopK: 1}

	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: payload}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)
	if res.Status != core.ToolComplete {
		t.Fatalf("unexpected status: %v err=%s", res.Status, res.Error)
	}
	out, ok := res.Output.(*QueryOutput)
	if !ok {
		t.Fatalf("unexpected output type %T", res.Output)
	}
	if len(out.IDs) != 1 || out.IDs[0] != "1" {
		t.Fatalf("unexpected ids: %+v", out.IDs)
	}
}

func TestQueryToolBadInput(t *testing.T) {
	tool := &QueryTool{Client: New("http://localhost:0")}
	req := &core.Message{ToolReq: &core.ToolRequestPayload{Input: "not a query"}}
	ctx := &core.ToolContext{Ctx: context.Background(), Request: req}
	res := tool.Execute(ctx)
	if res.Status != core.ToolFailed {
		t.Fatalf("expected failure on bad input")
	}
}
