// Package graph provides GraphRAG integration for DeepData.
//
// It maintains a knowledge graph built from extraction pipeline output,
// stored as CSR adjacency for efficient PageRank computation.
// Graph importance scores are used as a third signal in hybrid fusion
// alongside dense (vector) and sparse (BM25) results.
package graph

import (
	"strings"
	"sync"

	"github.com/Neumenon/cowrie/go/gnn/algo"
	"github.com/phenomenon0/vectordb/internal/extraction"
	"github.com/phenomenon0/vectordb/internal/hybrid"
)

// GraphIndex maintains a knowledge graph and provides PageRank-based
// document importance scores for use in hybrid search fusion.
type GraphIndex struct {
	mu sync.RWMutex

	// Node ID (string) -> internal index (int64)
	nodeIndex map[string]int64
	// Internal index -> Node ID
	nodeIDs []string
	// Internal index -> node metadata
	nodes []extraction.Node

	// Edge list for CSR construction
	edges []edge

	// Document ID -> set of node IDs contained in that document
	docNodes map[uint64]map[string]bool
	// Node ID -> set of document IDs containing this node
	nodeDocMap map[string]map[uint64]bool

	// Cached CSR and PageRank (invalidated on mutation)
	csr       *algo.CSR
	pagerank  *algo.PageRankResult
	dirty     bool
	prConfig  algo.PageRankConfig
}

type edge struct {
	src, dst int64
}

// Config configures the GraphIndex.
type Config struct {
	// PageRank damping factor (default 0.85)
	Damping float32
	// Max PageRank iterations (default 20)
	Iterations int
	// Convergence tolerance (default 1e-6)
	Tolerance float32
}

// DefaultConfig returns default GraphIndex configuration.
func DefaultConfig() Config {
	return Config{
		Damping:    0.85,
		Iterations: 20,
		Tolerance:  1e-6,
	}
}

// NewGraphIndex creates a new graph index.
func NewGraphIndex(cfg Config) *GraphIndex {
	if cfg.Damping <= 0 || cfg.Damping >= 1 {
		cfg.Damping = 0.85
	}
	if cfg.Iterations <= 0 {
		cfg.Iterations = 20
	}
	if cfg.Tolerance <= 0 {
		cfg.Tolerance = 1e-6
	}

	return &GraphIndex{
		nodeIndex:  make(map[string]int64),
		docNodes:   make(map[uint64]map[string]bool),
		nodeDocMap: make(map[string]map[uint64]bool),
		dirty:      true,
		prConfig: algo.PageRankConfig{
			Damping:    cfg.Damping,
			Iterations: cfg.Iterations,
			Tolerance:  cfg.Tolerance,
		},
	}
}

// AddKnowledgeGraph ingests a KnowledgeGraph from extraction, associating
// its entities with a document ID.
func (g *GraphIndex) AddKnowledgeGraph(docID uint64, kg *extraction.KnowledgeGraph) {
	if kg == nil {
		return
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Ensure doc entry exists
	if g.docNodes[docID] == nil {
		g.docNodes[docID] = make(map[string]bool)
	}

	// Add nodes
	for _, node := range kg.Nodes {
		idx := g.getOrCreateNode(node)
		g.docNodes[docID][node.ID] = true

		if g.nodeDocMap[node.ID] == nil {
			g.nodeDocMap[node.ID] = make(map[uint64]bool)
		}
		g.nodeDocMap[node.ID][docID] = true
		_ = idx
	}

	// Add edges
	for _, e := range kg.Edges {
		srcIdx, srcOK := g.nodeIndex[e.SourceNodeID]
		dstIdx, dstOK := g.nodeIndex[e.TargetNodeID]
		if srcOK && dstOK {
			g.edges = append(g.edges, edge{src: srcIdx, dst: dstIdx})
		}
	}

	g.dirty = true
}

// RemoveDocument removes a document's association from the graph.
// Nodes shared with other documents are kept.
func (g *GraphIndex) RemoveDocument(docID uint64) {
	g.mu.Lock()
	defer g.mu.Unlock()

	nodeIDs, ok := g.docNodes[docID]
	if !ok {
		return
	}

	for nodeID := range nodeIDs {
		if docs, ok := g.nodeDocMap[nodeID]; ok {
			delete(docs, docID)
		}
	}
	delete(g.docNodes, docID)
	g.dirty = true
}

// getOrCreateNode returns the internal index for a node, creating it if needed.
// Caller must hold the write lock.
func (g *GraphIndex) getOrCreateNode(node extraction.Node) int64 {
	if idx, ok := g.nodeIndex[node.ID]; ok {
		// Update node metadata if newer description is longer
		if len(node.Description) > len(g.nodes[idx].Description) {
			g.nodes[idx] = node
		}
		return idx
	}

	idx := int64(len(g.nodeIDs))
	g.nodeIndex[node.ID] = idx
	g.nodeIDs = append(g.nodeIDs, node.ID)
	g.nodes = append(g.nodes, node)
	return idx
}

// ensureComputed rebuilds CSR and PageRank if the graph is dirty.
// Caller must hold at least a read lock; upgrades to write if needed.
func (g *GraphIndex) ensureComputed() {
	if !g.dirty {
		return
	}

	// Build CSR from edge list
	n := int64(len(g.nodeIDs))
	if n == 0 {
		g.csr = nil
		g.pagerank = nil
		g.dirty = false
		return
	}

	// Count outdegrees for indptr
	indPtr := make([]int64, n+1)
	for _, e := range g.edges {
		if e.src >= 0 && e.src < n {
			indPtr[e.src+1]++
		}
	}
	// Prefix sum
	for i := int64(1); i <= n; i++ {
		indPtr[i] += indPtr[i-1]
	}

	// Fill indices
	indices := make([]int64, len(g.edges))
	pos := make([]int64, n)
	for _, e := range g.edges {
		if e.src >= 0 && e.src < n {
			offset := indPtr[e.src] + pos[e.src]
			if offset < int64(len(indices)) {
				indices[offset] = e.dst
				pos[e.src]++
			}
		}
	}

	g.csr = algo.NewCSR(n, indPtr, indices)
	g.pagerank = algo.PageRank(g.csr, g.prConfig)
	g.dirty = false
}

// Search returns graph-importance scores for documents, given query-matched entity names.
// It runs Personalized PageRank seeded from entities matching the query terms,
// then aggregates node scores to document scores.
func (g *GraphIndex) Search(queryTerms []string, topK int) []hybrid.SearchResult {
	g.mu.Lock()
	g.ensureComputed()
	g.mu.Unlock()

	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.csr == nil || g.pagerank == nil || len(g.nodeIDs) == 0 {
		return nil
	}

	// Find seed nodes matching query terms
	seeds := g.matchQueryToNodes(queryTerms)

	var scores []float32
	if len(seeds) > 0 {
		// Personalized PageRank from matched entities
		ppr := algo.PersonalizedPageRank(g.csr, g.prConfig, seeds)
		scores = ppr.Scores
	} else {
		// Fall back to global PageRank
		scores = g.pagerank.Scores
	}

	// Aggregate node scores to document scores
	docScores := make(map[uint64]float32)
	for nodeIdx, score := range scores {
		if nodeIdx >= len(g.nodeIDs) {
			break
		}
		nodeID := g.nodeIDs[nodeIdx]
		if docs, ok := g.nodeDocMap[nodeID]; ok {
			for docID := range docs {
				docScores[docID] += score
			}
		}
	}

	// Convert to results
	results := make([]hybrid.SearchResult, 0, len(docScores))
	for docID, score := range docScores {
		results = append(results, hybrid.SearchResult{
			DocID: docID,
			Score: score,
		})
	}

	// Sort by score descending (selection sort for small K)
	for i := 0; i < len(results) && i < topK; i++ {
		maxIdx := i
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[maxIdx].Score {
				maxIdx = j
			}
		}
		results[i], results[maxIdx] = results[maxIdx], results[i]
	}

	if topK < len(results) {
		results = results[:topK]
	}

	return results
}

// matchQueryToNodes finds internal node indices that match query terms.
// Matches on node Name or ID (case-insensitive substring).
func (g *GraphIndex) matchQueryToNodes(queryTerms []string) []int64 {
	var matched []int64
	seen := make(map[int64]bool)

	for _, term := range queryTerms {
		termLower := strings.ToLower(term)
		if len(termLower) < 2 {
			continue
		}

		for nodeID, idx := range g.nodeIndex {
			if seen[idx] {
				continue
			}
			node := g.nodes[idx]
			if strings.Contains(strings.ToLower(node.Name), termLower) ||
				strings.Contains(strings.ToLower(nodeID), termLower) {
				matched = append(matched, idx)
				seen[idx] = true
			}
		}
	}

	return matched
}

// Stats returns graph statistics.
type Stats struct {
	NumNodes     int
	NumEdges     int
	NumDocuments int
	PRConverged  bool
	PRIterations int
}

// Stats returns current graph index statistics.
func (g *GraphIndex) Stats() Stats {
	g.mu.RLock()
	defer g.mu.RUnlock()

	s := Stats{
		NumNodes:     len(g.nodeIDs),
		NumEdges:     len(g.edges),
		NumDocuments: len(g.docNodes),
	}

	if g.pagerank != nil {
		s.PRConverged = g.pagerank.Converged
		s.PRIterations = g.pagerank.Iterations
	}

	return s
}

// NodeCount returns the number of nodes in the graph.
func (g *GraphIndex) NodeCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.nodeIDs)
}

// EdgeCount returns the number of edges in the graph.
func (g *GraphIndex) EdgeCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.edges)
}
