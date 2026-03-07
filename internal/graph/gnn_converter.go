package graph

import (
	"encoding/binary"
	"encoding/json"

	"github.com/Neumenon/cowrie/go/gnn"
	"github.com/phenomenon0/vectordb/internal/extraction"
)

// KGToGNNContainer converts a DeepData KnowledgeGraph into a Cowrie-GNN Container.
// This enables PageRank, Louvain, and other graph algorithms from cowrie/go/gnn/algo
// to operate directly on extraction output.
func KGToGNNContainer(kg *extraction.KnowledgeGraph, name string) *gnn.Container {
	if kg == nil {
		return gnn.NewContainer(name)
	}

	c := gnn.NewContainer(name)
	c.SetDirected(true)
	c.SetHeterogeneous(true)

	// Collect unique node types
	nodeTypes := make(map[string]bool)
	for _, node := range kg.Nodes {
		nodeTypes[node.Type] = true
	}
	// Collect unique edge types
	edgeTypes := make(map[string]bool)
	for _, edge := range kg.Edges {
		edgeTypes[edge.RelationshipName] = true
	}

	// Register node types
	nodesByType := make(map[string][]extraction.Node)
	for _, node := range kg.Nodes {
		nodesByType[node.Type] = append(nodesByType[node.Type], node)
	}
	for nt, nodes := range nodesByType {
		c.AddNodeType(nt, int64(len(nodes)))
	}

	// Register edge types
	for et := range edgeTypes {
		// For simplicity, register as entity->entity for each relationship
		c.AddEdgeType("entity", et, "entity")
	}

	// Build node table body (JSON for compatibility)
	nodeTableBody, _ := json.Marshal(kg.Nodes)
	c.AddSectionWithEncoding(gnn.SectionNodeTable, "nodes", gnn.SectionEncodingJSON, nodeTableBody)

	// Build edge table body
	edgeTableBody, _ := json.Marshal(kg.Edges)
	c.AddSectionWithEncoding(gnn.SectionEdgeTable, "edges", gnn.SectionEncodingJSON, edgeTableBody)

	// Build CSR auxiliary section
	csrData := buildCSRAux(kg)
	if csrData != nil {
		c.Flags |= gnn.FlagHasCSR
		c.AddSectionWithEncoding(gnn.SectionAux, "csr", gnn.SectionEncodingRawTensor, csrData)
	}

	return c
}

// buildCSRAux builds a CSR binary section from a KnowledgeGraph.
// Returns nil if the graph is empty.
func buildCSRAux(kg *extraction.KnowledgeGraph) []byte {
	if len(kg.Nodes) == 0 {
		return nil
	}

	// Map node IDs to indices
	nodeIndex := make(map[string]int64, len(kg.Nodes))
	for i, node := range kg.Nodes {
		nodeIndex[node.ID] = int64(i)
	}

	n := int64(len(kg.Nodes))

	// Count outgoing edges per node
	outCount := make([]int64, n)
	for _, e := range kg.Edges {
		if src, ok := nodeIndex[e.SourceNodeID]; ok {
			outCount[src]++
		}
	}

	// Build indptr (prefix sum)
	indPtr := make([]int64, n+1)
	for i := int64(0); i < n; i++ {
		indPtr[i+1] = indPtr[i] + outCount[i]
	}

	// Build indices
	numEdges := indPtr[n]
	indices := make([]int64, numEdges)
	pos := make([]int64, n)
	for _, e := range kg.Edges {
		src, srcOK := nodeIndex[e.SourceNodeID]
		dst, dstOK := nodeIndex[e.TargetNodeID]
		if srcOK && dstOK {
			offset := indPtr[src] + pos[src]
			if offset < numEdges {
				indices[offset] = dst
				pos[src]++
			}
		}
	}

	// Serialize: [numNodes int64][numEdges int64][indptr bytes][indices bytes]
	buf := make([]byte, 16+(n+1)*8+numEdges*8)
	binary.LittleEndian.PutUint64(buf[0:], uint64(n))
	binary.LittleEndian.PutUint64(buf[8:], uint64(numEdges))

	off := 16
	for _, v := range indPtr {
		binary.LittleEndian.PutUint64(buf[off:], uint64(v))
		off += 8
	}
	for _, v := range indices {
		binary.LittleEndian.PutUint64(buf[off:], uint64(v))
		off += 8
	}

	return buf
}

// GNNContainerToKG converts a Cowrie-GNN Container back to a DeepData KnowledgeGraph.
// This is useful for importing GNN datasets into the extraction pipeline.
func GNNContainerToKG(c *gnn.Container) (*extraction.KnowledgeGraph, error) {
	kg := &extraction.KnowledgeGraph{}

	// Find node table section
	for _, s := range c.GetSectionsByKind(gnn.SectionNodeTable) {
		var nodes []extraction.Node
		if err := json.Unmarshal(s.Body, &nodes); err != nil {
			return nil, err
		}
		kg.Nodes = append(kg.Nodes, nodes...)
	}

	// Find edge table section
	for _, s := range c.GetSectionsByKind(gnn.SectionEdgeTable) {
		var edges []extraction.Edge
		if err := json.Unmarshal(s.Body, &edges); err != nil {
			return nil, err
		}
		kg.Edges = append(kg.Edges, edges...)
	}

	return kg, nil
}
