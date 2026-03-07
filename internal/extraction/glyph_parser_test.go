package extraction

import (
	"testing"
)

func TestParseGlyphKnowledgeGraph_WellFormed(t *testing.T) {
	input := `
@tab Node [id name type desc]
golang "Go Language" TECHNOLOGY "Systems programming language"
kubernetes Kubernetes TECHNOLOGY "Container orchestration"
google Google ORGANIZATION "Tech company"
@end

@tab Edge [src tgt rel w]
google golang CREATED_BY 1.0
kubernetes golang DEPENDS_ON 0.9
@end
`

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(kg.Nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(kg.Nodes))
	}

	// Check first node has quoted name
	if kg.Nodes[0].ID != "golang" {
		t.Errorf("expected node ID 'golang', got '%s'", kg.Nodes[0].ID)
	}
	if kg.Nodes[0].Name != "Go Language" {
		t.Errorf("expected node name 'Go Language', got '%s'", kg.Nodes[0].Name)
	}
	if kg.Nodes[0].Type != "TECHNOLOGY" {
		t.Errorf("expected node type 'TECHNOLOGY', got '%s'", kg.Nodes[0].Type)
	}
	if kg.Nodes[0].Description != "Systems programming language" {
		t.Errorf("expected description 'Systems programming language', got '%s'", kg.Nodes[0].Description)
	}

	// Check unquoted name
	if kg.Nodes[1].Name != "Kubernetes" {
		t.Errorf("expected node name 'Kubernetes', got '%s'", kg.Nodes[1].Name)
	}

	if len(kg.Edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(kg.Edges))
	}

	if kg.Edges[0].SourceNodeID != "google" || kg.Edges[0].TargetNodeID != "golang" {
		t.Errorf("unexpected edge: %+v", kg.Edges[0])
	}
	if kg.Edges[0].RelationshipName != "CREATED_BY" {
		t.Errorf("expected CREATED_BY, got '%s'", kg.Edges[0].RelationshipName)
	}
	if kg.Edges[0].Weight != 1.0 {
		t.Errorf("expected weight 1.0, got %f", kg.Edges[0].Weight)
	}

	// Check edge with non-default weight
	if kg.Edges[1].Weight != 0.9 {
		t.Errorf("expected weight 0.9, got %f", kg.Edges[1].Weight)
	}
}

func TestParseGlyphKnowledgeGraph_QuotedStrings(t *testing.T) {
	input := `
@tab Node [id name type desc]
john_doe "John Doe" PERSON "Software engineer at Acme Corp"
acme_corp "Acme Corporation" ORGANIZATION "Global tech company"
@end

@tab Edge [src tgt rel w]
john_doe acme_corp WORKS_FOR 1.0
@end
`

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(kg.Nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(kg.Nodes))
	}

	if kg.Nodes[0].Name != "John Doe" {
		t.Errorf("expected 'John Doe', got '%s'", kg.Nodes[0].Name)
	}
	if kg.Nodes[0].Description != "Software engineer at Acme Corp" {
		t.Errorf("expected 'Software engineer at Acme Corp', got '%s'", kg.Nodes[0].Description)
	}
	if kg.Nodes[1].Name != "Acme Corporation" {
		t.Errorf("expected 'Acme Corporation', got '%s'", kg.Nodes[1].Name)
	}
}

func TestParseGlyphKnowledgeGraph_FallbackToJSON(t *testing.T) {
	// LLM ignored Glyph instruction and returned JSON
	input := `{
  "nodes": [
    {"id": "rust", "name": "Rust", "type": "TECHNOLOGY", "description": "Systems language"}
  ],
  "edges": []
}`

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("JSON fallback should work, got error: %v", err)
	}

	if len(kg.Nodes) != 1 {
		t.Fatalf("expected 1 node from JSON fallback, got %d", len(kg.Nodes))
	}
	if kg.Nodes[0].ID != "rust" {
		t.Errorf("expected node ID 'rust', got '%s'", kg.Nodes[0].ID)
	}
}

func TestParseGlyphKnowledgeGraph_FallbackToJSONWithMarkdown(t *testing.T) {
	input := "```json\n{\"nodes\": [{\"id\": \"x\", \"name\": \"X\", \"type\": \"CONCEPT\"}], \"edges\": []}\n```"

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("JSON fallback with markdown should work, got error: %v", err)
	}

	if len(kg.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(kg.Nodes))
	}
}

func TestParseGlyphKnowledgeGraph_EmptyTables(t *testing.T) {
	input := `
@tab Node [id name type desc]
@end

@tab Edge [src tgt rel w]
@end
`

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("empty tables should not error: %v", err)
	}

	if len(kg.Nodes) != 0 {
		t.Errorf("expected 0 nodes, got %d", len(kg.Nodes))
	}
	if len(kg.Edges) != 0 {
		t.Errorf("expected 0 edges, got %d", len(kg.Edges))
	}
}

func TestParseGlyphKnowledgeGraph_EdgeWeightDefault(t *testing.T) {
	input := `
@tab Node [id name type desc]
a A CONCEPT "Concept A"
b B CONCEPT "Concept B"
@end

@tab Edge [src tgt rel]
a b RELATED_TO
@end
`

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(kg.Edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(kg.Edges))
	}
	// Weight should default to 1.0 when not provided
	if kg.Edges[1-1].Weight != 1.0 {
		t.Errorf("expected default weight 1.0, got %f", kg.Edges[0].Weight)
	}
}

func TestParseGlyphKnowledgeGraph_FiltersInvalidEdges(t *testing.T) {
	input := `
@tab Node [id name type desc]
a A CONCEPT "Concept A"
@end

@tab Edge [src tgt rel w]
a nonexistent RELATED_TO 1.0
a a RELATED_TO 1.0
@end
`

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Only self-referencing edge should survive (a->a), not a->nonexistent
	if len(kg.Edges) != 1 {
		t.Fatalf("expected 1 valid edge after filtering, got %d", len(kg.Edges))
	}
	if kg.Edges[0].SourceNodeID != "a" || kg.Edges[0].TargetNodeID != "a" {
		t.Errorf("expected self-referencing edge a->a, got %+v", kg.Edges[0])
	}
}

func TestParseGlyphTemporalKnowledgeGraph(t *testing.T) {
	input := `
@tab Node [id name type desc]
rust "Rust Language" TECHNOLOGY "Systems programming language"
mozilla Mozilla ORGANIZATION "Open source org"
@end

@tab Edge [src tgt rel w]
mozilla rust CREATED_BY 1.0
@end

@tab Event [id desc year date_text entities]
rust_release "Rust 1.0 released" 2015 "May 2015" [rust mozilla]
@end
`

	tkg, err := parseGlyphTemporalKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(tkg.Nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(tkg.Nodes))
	}

	if len(tkg.Edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(tkg.Edges))
	}

	if len(tkg.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(tkg.Events))
	}

	ev := tkg.Events[0]
	if ev.ID != "rust_release" {
		t.Errorf("expected event ID 'rust_release', got '%s'", ev.ID)
	}
	if ev.Description != "Rust 1.0 released" {
		t.Errorf("expected description 'Rust 1.0 released', got '%s'", ev.Description)
	}
	if ev.Year != 2015 {
		t.Errorf("expected year 2015, got %d", ev.Year)
	}
	if ev.DateText != "May 2015" {
		t.Errorf("expected date_text 'May 2015', got '%s'", ev.DateText)
	}
	if len(ev.Entities) != 2 {
		t.Fatalf("expected 2 entities in event, got %d", len(ev.Entities))
	}
	if ev.Entities[0] != "rust" || ev.Entities[1] != "mozilla" {
		t.Errorf("expected entities [rust mozilla], got %v", ev.Entities)
	}
}

func TestParseGlyphTemporalKnowledgeGraph_FallbackJSON(t *testing.T) {
	input := `{
  "nodes": [{"id": "a", "name": "A", "type": "CONCEPT"}],
  "edges": [],
  "events": [{"id": "ev1", "description": "Something happened", "year": 2024, "entities": ["a"]}]
}`

	tkg, err := parseGlyphTemporalKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("JSON fallback should work for temporal: %v", err)
	}

	if len(tkg.Nodes) != 1 {
		t.Errorf("expected 1 node, got %d", len(tkg.Nodes))
	}
	if len(tkg.Events) != 1 {
		t.Errorf("expected 1 event, got %d", len(tkg.Events))
	}
}

func TestSplitGlyphFields(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "simple unquoted",
			input:    "foo bar BAZ",
			expected: []string{"foo", "bar", "BAZ"},
		},
		{
			name:     "quoted strings",
			input:    `entity_id "Entity Name" TYPE "Brief description"`,
			expected: []string{"entity_id", "Entity Name", "TYPE", "Brief description"},
		},
		{
			name:     "bracket list",
			input:    `ev1 "Something" 2024 "in 2024" [a b c]`,
			expected: []string{"ev1", "Something", "2024", "in 2024", "[a b c]"},
		},
		{
			name:     "extra whitespace",
			input:    "  foo   bar  ",
			expected: []string{"foo", "bar"},
		},
		{
			name:     "empty bracket list",
			input:    `ev1 "test" 2024 "now" []`,
			expected: []string{"ev1", "test", "2024", "now", "[]"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := splitGlyphFields(tt.input)
			if len(got) != len(tt.expected) {
				t.Fatalf("expected %d fields, got %d: %v", len(tt.expected), len(got), got)
			}
			for i := range got {
				if got[i] != tt.expected[i] {
					t.Errorf("field %d: expected '%s', got '%s'", i, tt.expected[i], got[i])
				}
			}
		})
	}
}

func TestParseBracketList(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{"[a b c]", []string{"a", "b", "c"}},
		{"[single]", []string{"single"}},
		{"[]", nil},
		{"[  spaced  items  ]", []string{"spaced", "items"}},
	}

	for _, tt := range tests {
		got := parseBracketList(tt.input)
		if len(got) != len(tt.expected) {
			t.Errorf("parseBracketList(%q): expected %v, got %v", tt.input, tt.expected, got)
			continue
		}
		for i := range got {
			if got[i] != tt.expected[i] {
				t.Errorf("parseBracketList(%q)[%d]: expected '%s', got '%s'", tt.input, i, tt.expected[i], got[i])
			}
		}
	}
}

func TestParseGlyphKnowledgeGraph_MarkdownWrapped(t *testing.T) {
	input := "```glyph\n@tab Node [id name type desc]\nfoo Foo CONCEPT \"A concept\"\n@end\n\n@tab Edge [src tgt rel w]\n@end\n```"

	kg, err := parseGlyphKnowledgeGraph(input)
	if err != nil {
		t.Fatalf("markdown-wrapped glyph should parse: %v", err)
	}

	if len(kg.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(kg.Nodes))
	}
	if kg.Nodes[0].ID != "foo" {
		t.Errorf("expected ID 'foo', got '%s'", kg.Nodes[0].ID)
	}
}
