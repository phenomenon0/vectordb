package client

import (
	"fmt"

	"agentscope/core"
)

// InsertTool stores a single document in the vector database.
// It expects an InsertRequest as input and returns an InsertOutput on success.
type InsertTool struct {
	Client *Client
}

func (t *InsertTool) Name() string { return "vectordb_insert" }

// InsertInput is the input payload agents should pass to the tool.
type InsertInput = InsertRequest

// InsertOutput wraps the insert response.
type InsertOutput struct {
	ID string `json:"id"`
}

func (t *InsertTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	if t.Client == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "nil vectordb client"}
	}
	var req *InsertRequest
	if ctx != nil && ctx.Request != nil && ctx.Request.ToolReq != nil {
		if input, ok := ctx.Request.ToolReq.Input.(*InsertRequest); ok {
			req = input
		}
	}
	if req == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "expected InsertRequest input"}
	}

	// Validate required fields
	if req.Doc == "" {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "document text is required"}
	}

	resp, err := t.Client.Insert(ctx.Ctx, *req)
	if err != nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: fmt.Sprintf("vectordb insert error: %v", err)}
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &InsertOutput{
			ID: resp.ID,
		},
	}
}

// BatchInsertTool stores multiple documents in the vector database efficiently.
// It expects a BatchInsertRequest as input and returns a BatchInsertOutput on success.
type BatchInsertTool struct {
	Client *Client
}

func (t *BatchInsertTool) Name() string { return "vectordb_batch_insert" }

// BatchInsertInput is the input payload agents should pass to the tool.
type BatchInsertInput = BatchInsertRequest

// BatchInsertOutput wraps the batch insert response.
type BatchInsertOutput struct {
	IDs   []string `json:"ids"`
	Count int      `json:"count"`
}

func (t *BatchInsertTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	if t.Client == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "nil vectordb client"}
	}
	var req *BatchInsertRequest
	if ctx != nil && ctx.Request != nil && ctx.Request.ToolReq != nil {
		if input, ok := ctx.Request.ToolReq.Input.(*BatchInsertRequest); ok {
			req = input
		}
	}
	if req == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "expected BatchInsertRequest input"}
	}

	// Validate required fields
	if len(req.Docs) == 0 {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "at least one document is required"}
	}
	for i, doc := range req.Docs {
		if doc.Doc == "" {
			return &core.ToolExecResult{
				Status: core.ToolFailed,
				Error:  fmt.Sprintf("document %d: text is required", i),
			}
		}
	}

	resp, err := t.Client.BatchInsert(ctx.Ctx, *req)
	if err != nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: fmt.Sprintf("vectordb batch insert error: %v", err)}
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &BatchInsertOutput{
			IDs:   resp.IDs,
			Count: len(resp.IDs),
		},
	}
}

// DeleteTool removes a document from the vector database by ID.
// It expects a DeleteRequest as input and returns a DeleteOutput on success.
type DeleteTool struct {
	Client *Client
}

func (t *DeleteTool) Name() string { return "vectordb_delete" }

// DeleteInput is the input payload agents should pass to the tool.
type DeleteInput = DeleteRequest

// DeleteOutput wraps the delete response.
type DeleteOutput struct {
	DeletedID string `json:"deleted_id"`
	Success   bool   `json:"success"`
}

func (t *DeleteTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	if t.Client == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "nil vectordb client"}
	}
	var req *DeleteRequest
	if ctx != nil && ctx.Request != nil && ctx.Request.ToolReq != nil {
		if input, ok := ctx.Request.ToolReq.Input.(*DeleteRequest); ok {
			req = input
		}
	}
	if req == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "expected DeleteRequest input"}
	}

	// Validate required fields
	if req.ID == "" {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "document ID is required"}
	}

	resp, err := t.Client.Delete(ctx.Ctx, *req)
	if err != nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: fmt.Sprintf("vectordb delete error: %v", err)}
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &DeleteOutput{
			DeletedID: resp.Deleted,
			Success:   resp.Deleted != "",
		},
	}
}
