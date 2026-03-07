package graph

import (
	"testing"

	"github.com/phenomenon0/vectordb/internal/extraction"
	"github.com/phenomenon0/vectordb/internal/feedback"
)

func TestApplyEdgeBoosts_Empty(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())
	applied := g.ApplyEdgeBoosts(nil)
	if applied != 0 {
		t.Errorf("expected 0 applied, got %d", applied)
	}
}

func TestApplyEdgeBoosts_ValidBoosts(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	// Add a knowledge graph with two entities
	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "entity_a", Name: "Entity A"},
			{ID: "entity_b", Name: "Entity B"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "entity_a", TargetNodeID: "entity_b"},
		},
	}
	g.AddKnowledgeGraph(1, kg)

	initialEdges := g.EdgeCount()

	// Apply boosts for the known entities
	boosts := []feedback.EdgeBoost{
		{SourceID: "entity_a", TargetID: "entity_b", Boost: 1.5},
	}
	applied := g.ApplyEdgeBoosts(boosts)

	if applied != 1 {
		t.Errorf("expected 1 applied, got %d", applied)
	}

	// Should have added edges
	if g.EdgeCount() <= initialEdges {
		t.Errorf("expected more edges after boost: initial=%d, now=%d", initialEdges, g.EdgeCount())
	}
}

func TestApplyEdgeBoosts_UnknownNodes(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	// Add a knowledge graph
	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "known", Name: "Known"},
		},
	}
	g.AddKnowledgeGraph(1, kg)

	// Boost references unknown nodes — should be skipped
	boosts := []feedback.EdgeBoost{
		{SourceID: "unknown1", TargetID: "unknown2", Boost: 1.5},
	}
	applied := g.ApplyEdgeBoosts(boosts)

	if applied != 0 {
		t.Errorf("expected 0 applied for unknown nodes, got %d", applied)
	}
}

func TestApplyEdgeBoosts_DirtiesGraph(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "x", Name: "X"},
			{ID: "y", Name: "Y"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "x", TargetNodeID: "y"},
		},
	}
	g.AddKnowledgeGraph(1, kg)

	// Force computation to clear dirty flag
	g.Search([]string{"x"}, 1)

	// Apply boost should make graph dirty again (PageRank needs recompute)
	boosts := []feedback.EdgeBoost{
		{SourceID: "x", TargetID: "y", Boost: 2.0},
	}
	g.ApplyEdgeBoosts(boosts)

	// Verify we can still search (recomputes PageRank with boosted edges)
	results := g.Search([]string{"x"}, 10)
	if len(results) == 0 {
		t.Error("expected search results after boost")
	}
}
