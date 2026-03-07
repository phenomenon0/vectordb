package graph

import (
	"github.com/Neumenon/cowrie/go/gnn/algo"
)

// CommunityResult holds the result of community detection on the graph index.
type CommunityResult struct {
	// DocCommunities maps document ID to community ID.
	DocCommunities map[uint64]int64
	// Communities maps community ID to list of document IDs.
	Communities map[int64][]uint64
	// NumCommunities is the total number of communities found.
	NumCommunities int
	// Modularity is the quality score of the partitioning (0-1).
	Modularity float64
	// Converged indicates whether the algorithm converged.
	Converged bool
}

// DetectCommunities runs Louvain community detection on the knowledge graph
// and maps node communities to document communities.
//
// A document inherits the community of its most common node community
// (majority vote across its entities).
func (g *GraphIndex) DetectCommunities(cfg algo.LouvainConfig) *CommunityResult {
	g.mu.Lock()
	g.ensureComputed()
	g.mu.Unlock()

	g.mu.RLock()
	defer g.mu.RUnlock()

	result := &CommunityResult{
		DocCommunities: make(map[uint64]int64),
		Communities:    make(map[int64][]uint64),
	}

	if g.csr == nil || len(g.nodeIDs) == 0 {
		return result
	}

	// Run Louvain on the CSR graph
	lr := algo.LouvainUnweighted(g.csr, cfg)
	if lr == nil {
		return result
	}

	result.NumCommunities = lr.NumComms
	result.Modularity = lr.Modularity
	result.Converged = lr.Converged

	// Map each document to its dominant community via majority vote
	for docID, nodeIDs := range g.docNodes {
		if len(nodeIDs) == 0 {
			continue
		}

		// Count community votes from this document's nodes
		votes := make(map[int64]int)
		for nodeID := range nodeIDs {
			idx, ok := g.nodeIndex[nodeID]
			if !ok || int(idx) >= len(lr.Communities) {
				continue
			}
			comm := lr.Communities[idx]
			votes[comm]++
		}

		if len(votes) == 0 {
			continue
		}

		// Pick the community with the most votes
		var bestComm int64
		bestCount := 0
		for comm, count := range votes {
			if count > bestCount {
				bestCount = count
				bestComm = comm
			}
		}

		result.DocCommunities[docID] = bestComm
		result.Communities[bestComm] = append(result.Communities[bestComm], docID)
	}

	return result
}

// GetDocumentCommunity returns the community ID for a single document,
// or -1 if the document is not in the graph.
func (g *GraphIndex) GetDocumentCommunity(docID uint64, cfg algo.LouvainConfig) int64 {
	result := g.DetectCommunities(cfg)
	if comm, ok := result.DocCommunities[docID]; ok {
		return comm
	}
	return -1
}
