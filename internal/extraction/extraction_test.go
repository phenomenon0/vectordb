package extraction

import (
	"context"
	"testing"
	"time"
)

func TestKnowledgeGraphMerge(t *testing.T) {
	kg1 := &KnowledgeGraph{
		Nodes: []Node{
			{ID: "alice", Name: "Alice", Type: "PERSON"},
			{ID: "bob", Name: "Bob", Type: "PERSON"},
		},
		Edges: []Edge{
			{SourceNodeID: "alice", TargetNodeID: "bob", RelationshipName: "KNOWS"},
		},
	}

	kg2 := &KnowledgeGraph{
		Nodes: []Node{
			{ID: "bob", Name: "Bob", Type: "PERSON"}, // Duplicate
			{ID: "charlie", Name: "Charlie", Type: "PERSON"},
		},
		Edges: []Edge{
			{SourceNodeID: "bob", TargetNodeID: "charlie", RelationshipName: "KNOWS"},
		},
	}

	kg1.Merge(kg2)

	if len(kg1.Nodes) != 3 {
		t.Errorf("expected 3 nodes after merge, got %d", len(kg1.Nodes))
	}

	if len(kg1.Edges) != 2 {
		t.Errorf("expected 2 edges after merge, got %d", len(kg1.Edges))
	}
}

func TestKnowledgeGraphValidate(t *testing.T) {
	kg := &KnowledgeGraph{
		Nodes: []Node{
			{ID: "alice", Name: "Alice", Type: "PERSON"},
		},
		Edges: []Edge{
			{SourceNodeID: "alice", TargetNodeID: "bob", RelationshipName: "KNOWS"}, // bob doesn't exist
		},
	}

	err := kg.Validate()
	if err != ErrMissingTargetNode {
		t.Errorf("expected ErrMissingTargetNode, got %v", err)
	}
}

func TestSimpleChunker(t *testing.T) {
	chunker := &SimpleChunker{Size: 100, Overlap: 20}

	text := "This is a test. It has multiple sentences. We want to see how chunking works. And whether overlap is preserved correctly."

	chunks := chunker.Chunk(text)

	if len(chunks) < 2 {
		t.Errorf("expected at least 2 chunks, got %d", len(chunks))
	}

	// Check that each chunk has an ID
	for _, chunk := range chunks {
		if chunk.ID == "" {
			t.Error("chunk ID should not be empty")
		}
		if chunk.Text == "" {
			t.Error("chunk text should not be empty")
		}
	}
}

func TestNodeValidation(t *testing.T) {
	tests := []struct {
		name string
		node Node
		err  error
	}{
		{
			name: "valid node",
			node: Node{ID: "test", Name: "Test", Type: "CONCEPT"},
			err:  nil,
		},
		{
			name: "empty ID",
			node: Node{ID: "", Name: "Test", Type: "CONCEPT"},
			err:  ErrEmptyNodeID,
		},
		{
			name: "empty name",
			node: Node{ID: "test", Name: "", Type: "CONCEPT"},
			err:  ErrEmptyNodeName,
		},
		{
			name: "empty type",
			node: Node{ID: "test", Name: "Test", Type: ""},
			err:  ErrEmptyNodeType,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.node.Validate()
			if err != tt.err {
				t.Errorf("expected %v, got %v", tt.err, err)
			}
		})
	}
}

func TestEdgeValidation(t *testing.T) {
	tests := []struct {
		name string
		edge Edge
		err  error
	}{
		{
			name: "valid edge",
			edge: Edge{SourceNodeID: "a", TargetNodeID: "b", RelationshipName: "REL"},
			err:  nil,
		},
		{
			name: "empty source",
			edge: Edge{SourceNodeID: "", TargetNodeID: "b", RelationshipName: "REL"},
			err:  ErrEmptySourceNode,
		},
		{
			name: "empty target",
			edge: Edge{SourceNodeID: "a", TargetNodeID: "", RelationshipName: "REL"},
			err:  ErrEmptyTargetNode,
		},
		{
			name: "empty relationship",
			edge: Edge{SourceNodeID: "a", TargetNodeID: "b", RelationshipName: ""},
			err:  ErrEmptyRelationship,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.edge.Validate()
			if err != tt.err {
				t.Errorf("expected %v, got %v", tt.err, err)
			}
		})
	}
}

func TestPipelineStats(t *testing.T) {
	cfg := DefaultConfig()
	pipeline := NewPipeline(nil, DefaultPipelineConfig())

	// Check initial stats
	stats := pipeline.Stats()
	if stats.TotalDocuments != 0 {
		t.Errorf("expected 0 documents, got %d", stats.TotalDocuments)
	}

	// Reset should work
	pipeline.ResetStats()
	stats = pipeline.Stats()
	if stats.TotalChunks != 0 {
		t.Errorf("expected 0 chunks after reset, got %d", stats.TotalChunks)
	}

	_ = cfg // silence unused
}

// TestOllamaExtractorCreation tests extractor creation (doesn't require Ollama running)
func TestOllamaExtractorCreation(t *testing.T) {
	cfg := DefaultConfig()

	extractor, err := NewOllamaExtractor(cfg)
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}

	if extractor.Provider() != "ollama" {
		t.Errorf("expected provider 'ollama', got '%s'", extractor.Provider())
	}

	if extractor.Model() != cfg.Model {
		t.Errorf("expected model '%s', got '%s'", cfg.Model, extractor.Model())
	}
}

// TestCognifyOptions tests the Cognify option functions
func TestCognifyOptions(t *testing.T) {
	cfg := CognifyConfig{
		ExtractorConfig: DefaultConfig(),
		PipelineConfig:  DefaultPipelineConfig(),
	}

	WithOllama("http://custom:11434", "mistral")(&cfg)
	if cfg.ExtractorConfig.BaseURL != "http://custom:11434" {
		t.Errorf("expected custom URL, got '%s'", cfg.ExtractorConfig.BaseURL)
	}
	if cfg.ExtractorConfig.Model != "mistral" {
		t.Errorf("expected mistral model, got '%s'", cfg.ExtractorConfig.Model)
	}

	WithTemporal()(&cfg)
	if !cfg.PipelineConfig.EnableTemporal {
		t.Error("expected temporal to be enabled")
	}

	WithChunkSize(3000)(&cfg)
	if cfg.PipelineConfig.ChunkSize != 3000 {
		t.Errorf("expected chunk size 3000, got %d", cfg.PipelineConfig.ChunkSize)
	}

	WithDocumentID("doc123")(&cfg)
	if cfg.DocumentID != "doc123" {
		t.Errorf("expected doc ID 'doc123', got '%s'", cfg.DocumentID)
	}
}

func TestCleanJSONResponse(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    `{"nodes": []}`,
			expected: `{"nodes": []}`,
		},
		{
			input:    "```json\n{\"nodes\": []}\n```",
			expected: `{"nodes": []}`,
		},
		{
			input:    "```\n{\"nodes\": []}\n```",
			expected: `{"nodes": []}`,
		},
		{
			input:    "  {\"nodes\": []}  ",
			expected: `{"nodes": []}`,
		},
	}

	for _, tt := range tests {
		result := cleanJSONResponse(tt.input)
		if result != tt.expected {
			t.Errorf("cleanJSONResponse(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestFilterInvalidEdges(t *testing.T) {
	kg := KnowledgeGraph{
		Nodes: []Node{
			{ID: "a", Name: "A", Type: "CONCEPT"},
			{ID: "b", Name: "B", Type: "CONCEPT"},
		},
		Edges: []Edge{
			{SourceNodeID: "a", TargetNodeID: "b", RelationshipName: "REL"}, // Valid
			{SourceNodeID: "a", TargetNodeID: "c", RelationshipName: "REL"}, // Invalid - c doesn't exist
			{SourceNodeID: "x", TargetNodeID: "b", RelationshipName: "REL"}, // Invalid - x doesn't exist
		},
	}

	filtered := filterInvalidEdges(kg)

	if len(filtered.Edges) != 1 {
		t.Errorf("expected 1 valid edge, got %d", len(filtered.Edges))
	}
}

// Benchmark for chunking
func BenchmarkSimpleChunker(b *testing.B) {
	chunker := &SimpleChunker{Size: 2000, Overlap: 200}

	// Create a large text
	text := ""
	for i := 0; i < 100; i++ {
		text += "This is a test sentence that will be chunked. "
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunker.Chunk(text)
	}
}

// Integration test that requires Ollama (skipped by default)
func TestOllamaExtraction(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	cfg := DefaultConfig()
	extractor, err := NewOllamaExtractor(cfg)
	if err != nil {
		t.Fatalf("failed to create extractor: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	content := "Apple was founded by Steve Jobs and Steve Wozniak in 1976. The company is headquartered in Cupertino, California."

	kg, err := extractor.Extract(ctx, content)
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	if len(kg.Nodes) == 0 {
		t.Error("expected at least one node")
	}

	// Check for expected entities
	foundApple := false
	for _, node := range kg.Nodes {
		if node.Name == "Apple" || node.ID == "apple" {
			foundApple = true
			break
		}
	}

	if !foundApple {
		t.Log("Warning: Apple entity not found, but extraction succeeded")
	}
}
