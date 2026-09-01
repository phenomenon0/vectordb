// Package contract embeds the v3 agent contract: one JSON Schema file per MCP
// tool (each {"input": ..., "output": ...}) plus the error envelope and the
// agent-facing CONTRACT.md. The MCP server serves these bytes verbatim as
// inputSchema/outputSchema, so the files are the single source of truth.
package contract

import (
	"embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

//go:embed v3/schemas/*.json v3/operations.json v3/CONTRACT.md
var files embed.FS

// Operation is one V3 HTTP operation and its projections: the gRPC method
// that answers the same call (empty when there is none) and the MCP tool
// that reaches it (empty when no tool does). This list is the source the
// routes subcommand, GET /v3/status and the drift tests all read.
type Operation struct {
	Name       string `json:"name"`
	Method     string `json:"method"`
	Path       string `json:"path"`
	Permission string `json:"permission"`
	GRPCRPC    string `json:"grpc_rpc"`
	MCPTool    string `json:"mcp_tool"`
}

// Operations returns the V3 operation list in file order.
func Operations() ([]Operation, error) {
	b, err := files.ReadFile("v3/operations.json")
	if err != nil {
		return nil, fmt.Errorf("contract operations: %w", err)
	}
	var ops []Operation
	if err := json.Unmarshal(b, &ops); err != nil {
		return nil, fmt.Errorf("contract operations: %w", err)
	}
	return ops, nil
}

// Markdown is the agent-facing contract document (deepdata://contract).
var Markdown = func() string {
	b, err := files.ReadFile("v3/CONTRACT.md")
	if err != nil {
		panic(err)
	}
	return string(b)
}()

// Schema returns the raw contents of v3/schemas/<name>.json.
func Schema(name string) (json.RawMessage, error) {
	b, err := files.ReadFile("v3/schemas/" + name + ".json")
	if err != nil {
		return nil, fmt.Errorf("contract schema %q: %w", name, err)
	}
	return b, nil
}

// Names returns the tool schema names (every schema file except error), sorted.
func Names() []string {
	entries, err := files.ReadDir("v3/schemas")
	if err != nil {
		panic(err)
	}
	var names []string
	for _, e := range entries {
		name := strings.TrimSuffix(e.Name(), ".json")
		if name != "error" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}
