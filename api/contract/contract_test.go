package contract

import (
	"encoding/json"
	"strings"
	"testing"
)

// The embedded schema files are the MCP contract verbatim; a malformed or
// incomplete file would ship broken inputSchemas to every agent.
func TestSchemasParseWithInputAndOutput(t *testing.T) {
	names := Names()
	if len(names) != 6 {
		t.Fatalf("expected 6 tool schemas, got %d: %v", len(names), names)
	}
	for _, name := range names {
		if !strings.HasPrefix(name, "deepdata_") {
			t.Errorf("tool schema %q must be namespaced deepdata_", name)
		}
		raw, err := Schema(name)
		if err != nil {
			t.Fatalf("Schema(%q): %v", name, err)
		}
		var pair struct {
			Input  map[string]any `json:"input"`
			Output map[string]any `json:"output"`
		}
		if err := json.Unmarshal(raw, &pair); err != nil {
			t.Fatalf("%s.json is not valid JSON: %v", name, err)
		}
		if pair.Input["type"] != "object" || pair.Output["type"] != "object" {
			t.Errorf("%s.json: input and output must both be object schemas", name)
		}
	}
}

func TestErrorEnvelopeSchema(t *testing.T) {
	raw, err := Schema("error")
	if err != nil {
		t.Fatal(err)
	}
	var env struct {
		Properties map[string]any `json:"properties"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatalf("error.json: %v", err)
	}
	for _, field := range []string{"code", "message", "hint", "retryable"} {
		if _, ok := env.Properties[field]; !ok {
			t.Errorf("error envelope missing %q", field)
		}
	}
}

// CONTRACT.md is deepdata://contract — the one place an agent learns the
// verbs. A verb missing from it is undiscoverable.
func TestMarkdownNamesEveryTool(t *testing.T) {
	for _, name := range Names() {
		if !strings.Contains(Markdown, name) {
			t.Errorf("CONTRACT.md does not mention %s", name)
		}
	}
}

// operations.json is the single list of what the server serves; the routes
// subcommand, GET /v3/status and the docs linter all render it, so a
// malformed or ambiguous entry would go out on three surfaces at once.
func TestOperationsParseWithUniqueNames(t *testing.T) {
	ops, err := Operations()
	if err != nil {
		t.Fatal(err)
	}
	if len(ops) == 0 {
		t.Fatal("operations.json lists no operations")
	}
	names := map[string]bool{}
	routes := map[string]bool{}
	for _, op := range ops {
		if op.Name == "" || op.Method == "" || op.Path == "" || op.Permission == "" {
			t.Errorf("operation %+v is missing a required field", op)
		}
		if names[op.Name] {
			t.Errorf("duplicate operation name %q", op.Name)
		}
		names[op.Name] = true
		route := op.Method + " " + op.Path
		if routes[route] {
			t.Errorf("duplicate route %q", route)
		}
		routes[route] = true
	}
}
