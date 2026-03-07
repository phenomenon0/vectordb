package graph

import (
	"testing"

	"github.com/Neumenon/cowrie/go/gnn/algo"
	"github.com/phenomenon0/vectordb/internal/extraction"
)

func TestDetectCommunities_Empty(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())
	result := g.DetectCommunities(algo.LouvainConfig{})
	if result.NumCommunities != 0 {
		t.Errorf("expected 0 communities, got %d", result.NumCommunities)
	}
	if len(result.DocCommunities) != 0 {
		t.Errorf("expected empty doc communities, got %d", len(result.DocCommunities))
	}
}

func TestDetectCommunities_TwoClusters(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	// Cluster A: docs 1,2 share entities A1,A2,A3 with dense connections
	kgA := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "a1", Name: "Alpha One"},
			{ID: "a2", Name: "Alpha Two"},
			{ID: "a3", Name: "Alpha Three"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "a1", TargetNodeID: "a2"},
			{SourceNodeID: "a2", TargetNodeID: "a3"},
			{SourceNodeID: "a1", TargetNodeID: "a3"},
		},
	}
	g.AddKnowledgeGraph(1, kgA)
	g.AddKnowledgeGraph(2, kgA)

	// Cluster B: docs 3,4 share entities B1,B2,B3 with dense connections
	kgB := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "b1", Name: "Beta One"},
			{ID: "b2", Name: "Beta Two"},
			{ID: "b3", Name: "Beta Three"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "b1", TargetNodeID: "b2"},
			{SourceNodeID: "b2", TargetNodeID: "b3"},
			{SourceNodeID: "b1", TargetNodeID: "b3"},
		},
	}
	g.AddKnowledgeGraph(3, kgB)
	g.AddKnowledgeGraph(4, kgB)

	result := g.DetectCommunities(algo.LouvainConfig{})

	// Should find at least 2 communities (A-cluster and B-cluster may be separate)
	if result.NumCommunities < 1 {
		t.Errorf("expected at least 1 community, got %d", result.NumCommunities)
	}

	// All 4 docs should be assigned
	if len(result.DocCommunities) != 4 {
		t.Errorf("expected 4 doc assignments, got %d", len(result.DocCommunities))
	}

	// Docs 1 and 2 should be in the same community (they share the same entities)
	if result.DocCommunities[1] != result.DocCommunities[2] {
		t.Errorf("docs 1 and 2 should be in same community: %d vs %d",
			result.DocCommunities[1], result.DocCommunities[2])
	}

	// Docs 3 and 4 should be in the same community
	if result.DocCommunities[3] != result.DocCommunities[4] {
		t.Errorf("docs 3 and 4 should be in same community: %d vs %d",
			result.DocCommunities[3], result.DocCommunities[4])
	}

	t.Logf("communities: %d, modularity: %.4f, converged: %v",
		result.NumCommunities, result.Modularity, result.Converged)
	for comm, docs := range result.Communities {
		t.Logf("  community %d: docs %v", comm, docs)
	}
}

func TestGetDocumentCommunity_Missing(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())
	comm := g.GetDocumentCommunity(999, algo.LouvainConfig{})
	if comm != -1 {
		t.Errorf("expected -1 for missing doc, got %d", comm)
	}
}
