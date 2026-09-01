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

//go:embed v3/schemas/*.json v3/CONTRACT.md
var files embed.FS

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
