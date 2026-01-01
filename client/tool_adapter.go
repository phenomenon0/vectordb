package client

import (
	"fmt"

	"github.com/phenomenon0/Agent-GO/core"
)

// QueryTool is a simple AgentScope Tool that calls the vectordb Query API.
// It expects a QueryRequest as input and returns a QueryResponse on success.
type QueryTool struct {
	Client *Client
}

func (t *QueryTool) Name() string { return "vectordb_query" }

// QueryInput is the input payload agents should pass to the tool.
type QueryInput = QueryRequest

// QueryOutput wraps the response plus an optional debug string.
type QueryOutput struct {
	IDs    []string            `json:"ids"`
	Docs   []string            `json:"docs"`
	Scores []float32           `json:"scores"`
	Meta   []map[string]string `json:"meta,omitempty"`
	Stats  string              `json:"stats,omitempty"`
}

func (t *QueryTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	if t.Client == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "nil vectordb client"}
	}
	var req *QueryRequest
	if ctx != nil && ctx.Request != nil && ctx.Request.ToolReq != nil {
		if input, ok := ctx.Request.ToolReq.Input.(*QueryRequest); ok {
			req = input
		}
	}
	if req == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "expected QueryRequest input"}
	}

	resp, err := t.Client.Query(ctx.Ctx, *req)
	if err != nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: fmt.Sprintf("vectordb query error: %v", err)}
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &QueryOutput{
			IDs:    resp.IDs,
			Docs:   resp.Docs,
			Scores: resp.Scores,
			Meta:   resp.Meta,
			Stats:  resp.Stats,
		},
	}
}
