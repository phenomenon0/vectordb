package graph

import (
	"testing"

	"github.com/phenomenon0/vectordb/internal/extraction"
)

func TestKGToGNNContainerBasic(t *testing.T) {
	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "alice", Name: "Alice", Type: "PERSON", Description: "Engineer"},
			{ID: "bob", Name: "Bob", Type: "PERSON", Description: "Manager"},
			{ID: "acme", Name: "Acme Corp", Type: "ORGANIZATION"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "alice", TargetNodeID: "acme", RelationshipName: "WORKS_FOR"},
			{SourceNodeID: "bob", TargetNodeID: "acme", RelationshipName: "WORKS_FOR"},
			{SourceNodeID: "alice", TargetNodeID: "bob", RelationshipName: "COLLABORATES_WITH"},
		},
	}

	c := KGToGNNContainer(kg, "test_graph")
	if c == nil {
		t.Fatal("expected non-nil container")
	}

	meta := c.Meta()
	if meta.DatasetName != "test_graph" {
		t.Errorf("expected dataset name 'test_graph', got %s", meta.DatasetName)
	}
	if !meta.Directed {
		t.Error("expected directed graph")
	}
	if !meta.Heterogeneous {
		t.Error("expected heterogeneous graph")
	}

	// Should have CSR flag set
	if !c.HasCSR() {
		t.Error("expected CSR flag to be set")
	}

	// Check sections exist
	nodeSections := c.GetSectionsByKind(1) // SectionNodeTable
	if len(nodeSections) == 0 {
		t.Error("expected node table section")
	}

	edgeSections := c.GetSectionsByKind(2) // SectionEdgeTable
	if len(edgeSections) == 0 {
		t.Error("expected edge table section")
	}

	auxSections := c.GetSectionsByKind(5) // SectionAux
	if len(auxSections) == 0 {
		t.Error("expected aux (CSR) section")
	}
}

func TestKGToGNNRoundtrip(t *testing.T) {
	original := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{
			{ID: "n1", Name: "Node 1", Type: "CONCEPT"},
			{ID: "n2", Name: "Node 2", Type: "CONCEPT"},
		},
		Edges: []extraction.Edge{
			{SourceNodeID: "n1", TargetNodeID: "n2", RelationshipName: "RELATED_TO"},
		},
	}

	c := KGToGNNContainer(original, "roundtrip_test")
	recovered, err := GNNContainerToKG(c)
	if err != nil {
		t.Fatalf("roundtrip failed: %v", err)
	}

	if len(recovered.Nodes) != len(original.Nodes) {
		t.Errorf("expected %d nodes, got %d", len(original.Nodes), len(recovered.Nodes))
	}
	if len(recovered.Edges) != len(original.Edges) {
		t.Errorf("expected %d edges, got %d", len(original.Edges), len(recovered.Edges))
	}

	// Verify node data preserved
	for i, node := range recovered.Nodes {
		if node.ID != original.Nodes[i].ID {
			t.Errorf("node %d ID mismatch: %s vs %s", i, node.ID, original.Nodes[i].ID)
		}
		if node.Name != original.Nodes[i].Name {
			t.Errorf("node %d Name mismatch: %s vs %s", i, node.Name, original.Nodes[i].Name)
		}
	}
}

func TestKGToGNNNil(t *testing.T) {
	c := KGToGNNContainer(nil, "empty")
	if c == nil {
		t.Fatal("expected non-nil container for nil KG")
	}
	if c.Meta().DatasetName != "empty" {
		t.Error("expected 'empty' dataset name")
	}
}

func TestKGToGNNEmptyGraph(t *testing.T) {
	kg := &extraction.KnowledgeGraph{
		Nodes: []extraction.Node{},
		Edges: []extraction.Edge{},
	}

	c := KGToGNNContainer(kg, "empty_graph")
	if c.HasCSR() {
		t.Error("empty graph should not have CSR")
	}
}
