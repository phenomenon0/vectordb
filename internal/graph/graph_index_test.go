package graph

import (
	"testing"

	"github.com/phenomenon0/vectordb/internal/extraction"
)

func TestGraphIndexBasic(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "alice", Name: "Alice", Type: "PERSON"},
			{ID: "bob", Name: "Bob", Type: "PERSON"},
			{ID: "acme", Name: "Acme Corp", Type: "ORGANIZATION"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "alice", TargetNodeID: "acme", RelationshipName: "WORKS_FOR"},
			{SourceNodeID: "bob", TargetNodeID: "acme", RelationshipName: "WORKS_FOR"},
			{SourceNodeID: "alice", TargetNodeID: "bob", RelationshipName: "COLLABORATES_WITH"},
		},
	}

	g.AddKnowledgeGraph(1, kg)

	if g.NodeCount() != 3 {
		t.Errorf("expected 3 nodes, got %d", g.NodeCount())
	}
	if g.EdgeCount() != 3 {
		t.Errorf("expected 3 edges, got %d", g.EdgeCount())
	}

	stats := g.Stats()
	if stats.NumDocuments != 1 {
		t.Errorf("expected 1 document, got %d", stats.NumDocuments)
	}
}

func TestGraphIndexSearch(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	// Doc 1: Alice works at Acme
	kg1 := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "alice", Name: "Alice Smith", Type: "PERSON"},
			{ID: "acme", Name: "Acme Corp", Type: "ORGANIZATION"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "alice", TargetNodeID: "acme", RelationshipName: "WORKS_FOR"},
		},
	}
	g.AddKnowledgeGraph(1, kg1)

	// Doc 2: Bob works at Acme, knows Alice
	kg2 := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "bob", Name: "Bob Jones", Type: "PERSON"},
			{ID: "acme", Name: "Acme Corp", Type: "ORGANIZATION"},
			{ID: "alice", Name: "Alice Smith", Type: "PERSON"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "bob", TargetNodeID: "acme", RelationshipName: "WORKS_FOR"},
			{SourceNodeID: "bob", TargetNodeID: "alice", RelationshipName: "COLLABORATES_WITH"},
		},
	}
	g.AddKnowledgeGraph(2, kg2)

	// Doc 3: Unrelated - Charlie at BigCo
	kg3 := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "charlie", Name: "Charlie Brown", Type: "PERSON"},
			{ID: "bigco", Name: "BigCo Inc", Type: "ORGANIZATION"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "charlie", TargetNodeID: "bigco", RelationshipName: "WORKS_FOR"},
		},
	}
	g.AddKnowledgeGraph(3, kg3)

	// Search for "alice" — should rank docs 1 and 2 higher than 3
	results := g.Search([]string{"alice"}, 10)
	if len(results) == 0 {
		t.Fatal("expected results from graph search")
	}

	// Doc 1 and 2 should appear (they contain alice)
	foundDoc1, foundDoc2, foundDoc3 := false, false, false
	for _, r := range results {
		switch r.DocID {
		case 1:
			foundDoc1 = true
		case 2:
			foundDoc2 = true
		case 3:
			foundDoc3 = true
		}
	}

	if !foundDoc1 || !foundDoc2 {
		t.Error("expected docs 1 and 2 in results for 'alice' query")
	}

	// Doc 3 may appear with low score from PageRank, that's fine
	// But docs 1 and 2 should have higher scores
	if foundDoc3 && len(results) >= 3 {
		doc3Score := float32(0)
		maxOtherScore := float32(0)
		for _, r := range results {
			if r.DocID == 3 {
				doc3Score = r.Score
			} else if r.Score > maxOtherScore {
				maxOtherScore = r.Score
			}
		}
		if doc3Score > maxOtherScore {
			t.Error("doc 3 should not score higher than docs containing 'alice'")
		}
	}
}

func TestGraphIndexEmpty(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())
	results := g.Search([]string{"anything"}, 10)
	if len(results) != 0 {
		t.Errorf("expected no results from empty graph, got %d", len(results))
	}
}

func TestGraphIndexRemoveDocument(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "alice", Name: "Alice", Type: "PERSON"},
		},
		Edges: nil,
	}

	g.AddKnowledgeGraph(1, kg)
	if g.Stats().NumDocuments != 1 {
		t.Fatal("expected 1 document")
	}

	g.RemoveDocument(1)
	if g.Stats().NumDocuments != 0 {
		t.Error("expected 0 documents after removal")
	}
}

func TestGraphIndexReindexClearsStaleMembership(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	g.AddKnowledgeGraph(1, &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "alice", Name: "Alice", Type: "PERSON"},
		},
	})

	g.AddKnowledgeGraph(1, &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "bob", Name: "Bob", Type: "PERSON"},
		},
	})

	if docs := g.nodeDocMap["alice"]; docs != nil && docs[1] {
		t.Fatal("re-indexed document should be removed from stale node membership")
	}

	if nodes := g.docNodes[1]; nodes["alice"] {
		t.Fatal("re-indexed document should not retain stale node membership")
	}

	if nodes := g.docNodes[1]; !nodes["bob"] {
		t.Fatal("re-indexed document should retain new node membership")
	}

	foundBob := false
	for _, result := range g.Search([]string{"bob"}, 10) {
		if result.DocID == 1 {
			foundBob = true
			break
		}
	}
	if !foundBob {
		t.Fatal("re-indexed document should remain discoverable for new node membership")
	}
}

func TestGraphIndexNilKnowledgeGraph(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())
	g.AddKnowledgeGraph(1, nil) // Should not panic
	if g.NodeCount() != 0 {
		t.Error("expected 0 nodes after adding nil KG")
	}
}

func TestGraphIndexGlobalPageRank(t *testing.T) {
	g := NewGraphIndex(DefaultConfig())

	// Create a hub-and-spoke graph where "hub" should have highest PageRank
	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "hub", Name: "Central Hub", Type: "CONCEPT"},
			{ID: "spoke1", Name: "Spoke 1", Type: "CONCEPT"},
			{ID: "spoke2", Name: "Spoke 2", Type: "CONCEPT"},
			{ID: "spoke3", Name: "Spoke 3", Type: "CONCEPT"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "spoke1", TargetNodeID: "hub", RelationshipName: "RELATED_TO"},
			{SourceNodeID: "spoke2", TargetNodeID: "hub", RelationshipName: "RELATED_TO"},
			{SourceNodeID: "spoke3", TargetNodeID: "hub", RelationshipName: "RELATED_TO"},
		},
	}

	g.AddKnowledgeGraph(1, kg)

	// Search with no matching terms — falls back to global PageRank
	results := g.Search([]string{"nonexistent_term"}, 10)
	if len(results) == 0 {
		t.Fatal("expected results from global PageRank fallback")
	}
}
