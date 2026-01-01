package client

import (
	"fmt"

	"github.com/phenomenon0/Agent-GO/core"
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

// FileSearchTool searches the local file index (localvault collection) for documents.
// This is the primary tool for the file locator / RAG system, with a simpler
// agent-friendly interface than the raw QueryTool.
type FileSearchTool struct {
	Client *Client
}

func (t *FileSearchTool) Name() string { return "file_search" }

// FileSearchInput is the agent-facing input payload.
type FileSearchInput struct {
	Query      string            `json:"query"`                 // required: natural language search query
	Collection string            `json:"collection,omitempty"`  // defaults to "localvault"
	TopK       int               `json:"top_k,omitempty"`       // defaults to 10
	PathPrefix string            `json:"path_prefix,omitempty"` // filter by path prefix (e.g., "/home/user/Documents/")
	Meta       map[string]string `json:"meta,omitempty"`        // exact match metadata filters
}

// FileSearchOutput is the agent-facing response.
type FileSearchOutput struct {
	Matches []FileMatch `json:"matches"`
	Total   int         `json:"total"`
	Stats   string      `json:"stats,omitempty"`
}

// FileMatch represents a single search result with file location info.
type FileMatch struct {
	Score float32        `json:"score"`
	Text  string         `json:"text"` // the matched chunk content
	ID    string         `json:"id"`   // vectordb document ID
	Meta  *FileMatchMeta `json:"meta,omitempty"`
}

// FileMatchMeta contains file location metadata for opening/navigating to results.
type FileMatchMeta struct {
	Path        string            `json:"path,omitempty"`         // absolute file path
	Filename    string            `json:"filename,omitempty"`     // just the filename
	AnchorType  string            `json:"anchor_type,omitempty"`  // "line" or "page"
	AnchorStart int               `json:"anchor_start,omitempty"` // start line/page number
	AnchorEnd   int               `json:"anchor_end,omitempty"`   // end line/page number
	Mtime       string            `json:"mtime,omitempty"`        // file modification time (RFC3339)
	Extra       map[string]string `json:"extra,omitempty"`        // pass-through for other metadata
}

func (t *FileSearchTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	if t.Client == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "nil vectordb client"}
	}

	var input *FileSearchInput
	if ctx != nil && ctx.Request != nil && ctx.Request.ToolReq != nil {
		if in, ok := ctx.Request.ToolReq.Input.(*FileSearchInput); ok {
			input = in
		}
	}
	if input == nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "expected FileSearchInput"}
	}

	// Validate required fields
	if input.Query == "" {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: "query text is required"}
	}

	// Build request with defaults
	collection := input.Collection
	if collection == "" {
		collection = "localvault"
	}
	topK := input.TopK
	if topK <= 0 {
		topK = 10
	}

	// Build metadata filter
	meta := input.Meta
	if meta == nil {
		meta = make(map[string]string)
	}
	// Path prefix filter is passed as metadata (server-side prefix matching)
	if input.PathPrefix != "" {
		meta["path_prefix"] = input.PathPrefix
	}

	req := QueryRequest{
		Query:       input.Query,
		TopK:        topK,
		Collection:  collection,
		Meta:        meta,
		IncludeMeta: true, // always include for file locator use case
	}

	resp, err := t.Client.Query(ctx.Ctx, req)
	if err != nil {
		return &core.ToolExecResult{
			Status: core.ToolFailed,
			Error:  fmt.Sprintf("file search error: %v", err),
		}
	}

	// Transform response into agent-friendly format
	matches := make([]FileMatch, 0, len(resp.IDs))
	for i, id := range resp.IDs {
		match := FileMatch{
			ID: id,
		}
		if i < len(resp.Scores) {
			match.Score = resp.Scores[i]
		}
		if i < len(resp.Docs) {
			match.Text = resp.Docs[i]
		}

		// Extract structured metadata
		if i < len(resp.Meta) && resp.Meta[i] != nil {
			rawMeta := resp.Meta[i]
			match.Meta = &FileMatchMeta{
				Path:     rawMeta["path"],
				Filename: rawMeta["filename"],
				Mtime:    rawMeta["mtime"],
				Extra:    make(map[string]string),
			}
			// Parse anchor fields
			if at := rawMeta["anchor_type"]; at != "" {
				match.Meta.AnchorType = at
			}
			if start := rawMeta["anchor_start"]; start != "" {
				if v, parseErr := parseAnchorInt(start); parseErr == nil {
					match.Meta.AnchorStart = v
				}
			}
			if end := rawMeta["anchor_end"]; end != "" {
				if v, parseErr := parseAnchorInt(end); parseErr == nil {
					match.Meta.AnchorEnd = v
				}
			}
			// Pass through other metadata
			for k, v := range rawMeta {
				switch k {
				case "path", "filename", "mtime", "anchor_type", "anchor_start", "anchor_end":
					// already handled above
				default:
					match.Meta.Extra[k] = v
				}
			}
		}

		matches = append(matches, match)
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &FileSearchOutput{
			Matches: matches,
			Total:   len(matches),
			Stats:   resp.Stats,
		},
	}
}

// parseAnchorInt parses an anchor value (line/page number) from string.
func parseAnchorInt(s string) (int, error) {
	var v int
	_, err := fmt.Sscanf(s, "%d", &v)
	return v, err
}
