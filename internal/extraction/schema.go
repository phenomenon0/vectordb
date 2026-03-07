// Package extraction provides LLM-based entity and relationship extraction
// for building knowledge graphs from unstructured text.
//
// This package implements "cognify" functionality similar to Cognee,
// automatically extracting entities and relationships from text chunks
// and storing them in the existing EntityGraph infrastructure.
package extraction

import (
	"encoding/json"
	"time"
)

// Node represents an entity in the knowledge graph.
type Node struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// Edge represents a relationship between two nodes.
type Edge struct {
	SourceNodeID     string  `json:"source_node_id"`
	TargetNodeID     string  `json:"target_node_id"`
	RelationshipName string  `json:"relationship_name"`
	Weight           float32 `json:"weight,omitempty"`
}

// KnowledgeGraph represents extracted entities and relationships from a chunk.
type KnowledgeGraph struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// TemporalEvent represents an event with temporal information.
type TemporalEvent struct {
	ID          string     `json:"id"`
	Description string     `json:"description"`
	Date        *time.Time `json:"date,omitempty"`
	Year        int        `json:"year,omitempty"`
	DateText    string     `json:"date_text,omitempty"` // Original text representation
	Entities    []string   `json:"entities,omitempty"`  // Related entity IDs
}

// TemporalKnowledgeGraph extends KnowledgeGraph with temporal events.
type TemporalKnowledgeGraph struct {
	KnowledgeGraph
	Events []TemporalEvent `json:"events,omitempty"`
}

// ChunkResult holds extraction results for a single text chunk.
type ChunkResult struct {
	ChunkID   string                  `json:"chunk_id"`
	ChunkText string                  `json:"chunk_text,omitempty"`
	Graph     KnowledgeGraph          `json:"graph"`
	Temporal  *TemporalKnowledgeGraph `json:"temporal,omitempty"`
	Error     string                  `json:"error,omitempty"`
}

// ExtractionResult holds results for an entire document.
type ExtractionResult struct {
	DocumentID string          `json:"document_id"`
	Chunks     []ChunkResult   `json:"chunks"`
	Merged     KnowledgeGraph  `json:"merged"` // Deduplicated merged graph
	Stats      ExtractionStats `json:"stats"`
}

// ExtractionStats tracks extraction metrics.
type ExtractionStats struct {
	TotalChunks     int           `json:"total_chunks"`
	ProcessedChunks int           `json:"processed_chunks"`
	FailedChunks    int           `json:"failed_chunks"`
	TotalNodes      int           `json:"total_nodes"`
	TotalEdges      int           `json:"total_edges"`
	UniqueNodes     int           `json:"unique_nodes"`
	UniqueEdges     int           `json:"unique_edges"`
	TotalEvents     int           `json:"total_events,omitempty"`
	Duration        time.Duration `json:"duration"`
}

// EntityType constants for common entity types.
const (
	EntityTypePerson       = "PERSON"
	EntityTypeOrganization = "ORGANIZATION"
	EntityTypeLocation     = "LOCATION"
	EntityTypeConcept      = "CONCEPT"
	EntityTypeTechnology   = "TECHNOLOGY"
	EntityTypeEvent        = "EVENT"
	EntityTypeDate         = "DATE"
	EntityTypeProduct      = "PRODUCT"
	EntityTypeCode         = "CODE"
	EntityTypeFunction     = "FUNCTION"
	EntityTypeClass        = "CLASS"
	EntityTypeModule       = "MODULE"
)

// RelationType constants for common relationship types.
const (
	RelationTypeRelatedTo    = "RELATED_TO"
	RelationTypePartOf       = "PART_OF"
	RelationTypeCreatedBy    = "CREATED_BY"
	RelationTypeUsedBy       = "USED_BY"
	RelationTypeDependsOn    = "DEPENDS_ON"
	RelationTypeImplements   = "IMPLEMENTS"
	RelationTypeExtends      = "EXTENDS"
	RelationTypeCalls        = "CALLS"
	RelationTypeContains     = "CONTAINS"
	RelationTypeLocatedIn    = "LOCATED_IN"
	RelationTypeOccurredAt   = "OCCURRED_AT"
	RelationTypeWorksFor     = "WORKS_FOR"
	RelationTypeCollaborates = "COLLABORATES_WITH"
)

// Validate checks if a Node is valid.
func (n *Node) Validate() error {
	if n.ID == "" {
		return ErrEmptyNodeID
	}
	if n.Name == "" {
		return ErrEmptyNodeName
	}
	if n.Type == "" {
		return ErrEmptyNodeType
	}
	return nil
}

// Validate checks if an Edge is valid.
func (e *Edge) Validate() error {
	if e.SourceNodeID == "" {
		return ErrEmptySourceNode
	}
	if e.TargetNodeID == "" {
		return ErrEmptyTargetNode
	}
	if e.RelationshipName == "" {
		return ErrEmptyRelationship
	}
	return nil
}

// Validate checks if a KnowledgeGraph is valid.
func (kg *KnowledgeGraph) Validate() error {
	nodeIDs := make(map[string]bool)
	for _, node := range kg.Nodes {
		if err := node.Validate(); err != nil {
			return err
		}
		nodeIDs[node.ID] = true
	}

	for _, edge := range kg.Edges {
		if err := edge.Validate(); err != nil {
			return err
		}
		// Check that source and target exist
		if !nodeIDs[edge.SourceNodeID] {
			return ErrMissingSourceNode
		}
		if !nodeIDs[edge.TargetNodeID] {
			return ErrMissingTargetNode
		}
	}

	return nil
}

// Merge combines two knowledge graphs, deduplicating nodes by ID.
func (kg *KnowledgeGraph) Merge(other *KnowledgeGraph) {
	if other == nil {
		return
	}

	nodeMap := make(map[string]*Node)
	for i := range kg.Nodes {
		nodeMap[kg.Nodes[i].ID] = &kg.Nodes[i]
	}

	// Add new nodes or update existing
	for _, node := range other.Nodes {
		if existing, ok := nodeMap[node.ID]; ok {
			// Keep the one with more description
			if len(node.Description) > len(existing.Description) {
				existing.Description = node.Description
			}
		} else {
			kg.Nodes = append(kg.Nodes, node)
			nodeMap[node.ID] = &kg.Nodes[len(kg.Nodes)-1]
		}
	}

	// Track existing edges
	edgeKey := func(e Edge) string {
		return e.SourceNodeID + "|" + e.TargetNodeID + "|" + e.RelationshipName
	}
	edgeMap := make(map[string]bool)
	for _, edge := range kg.Edges {
		edgeMap[edgeKey(edge)] = true
	}

	// Add new edges
	for _, edge := range other.Edges {
		key := edgeKey(edge)
		if !edgeMap[key] {
			kg.Edges = append(kg.Edges, edge)
			edgeMap[key] = true
		}
	}
}

// ToJSON serializes the KnowledgeGraph to JSON.
func (kg *KnowledgeGraph) ToJSON() ([]byte, error) {
	return json.Marshal(kg)
}

// FromJSON deserializes a KnowledgeGraph from JSON.
func (kg *KnowledgeGraph) FromJSON(data []byte) error {
	return json.Unmarshal(data, kg)
}

// NodesByType returns all nodes of a specific type.
func (kg *KnowledgeGraph) NodesByType(nodeType string) []Node {
	var result []Node
	for _, node := range kg.Nodes {
		if node.Type == nodeType {
			result = append(result, node)
		}
	}
	return result
}

// EdgesFrom returns all edges originating from a node.
func (kg *KnowledgeGraph) EdgesFrom(nodeID string) []Edge {
	var result []Edge
	for _, edge := range kg.Edges {
		if edge.SourceNodeID == nodeID {
			result = append(result, edge)
		}
	}
	return result
}

// EdgesTo returns all edges pointing to a node.
func (kg *KnowledgeGraph) EdgesTo(nodeID string) []Edge {
	var result []Edge
	for _, edge := range kg.Edges {
		if edge.TargetNodeID == nodeID {
			result = append(result, edge)
		}
	}
	return result
}
