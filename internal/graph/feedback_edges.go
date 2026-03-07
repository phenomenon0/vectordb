package graph

import (
	"github.com/phenomenon0/vectordb/internal/feedback"
)

// ApplyEdgeBoosts applies feedback-derived edge weight adjustments to the graph.
// Edges between co-clicked entities get boosted, which improves their PageRank
// on subsequent searches — closing the reinforcement learning loop.
//
// Call this periodically (e.g., after processing a batch of feedback) to let
// user signals flow into graph-boosted search results.
func (g *GraphIndex) ApplyEdgeBoosts(boosts []feedback.EdgeBoost) int {
	if len(boosts) == 0 {
		return 0
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	applied := 0
	for _, eb := range boosts {
		srcIdx, srcOK := g.nodeIndex[eb.SourceID]
		dstIdx, dstOK := g.nodeIndex[eb.TargetID]
		if !srcOK || !dstOK {
			continue
		}

		// Add reinforcement edges (duplicates increase effective weight in PageRank)
		// Each boost > 1.0 adds extra edge copies proportional to the boost magnitude.
		extraEdges := int(eb.Boost) // e.g., boost 1.3 → 1 edge, boost 2.0 → 2 edges
		if extraEdges < 1 {
			extraEdges = 1
		}
		for i := 0; i < extraEdges; i++ {
			g.edges = append(g.edges, edge{src: srcIdx, dst: dstIdx})
		}
		applied++
	}

	if applied > 0 {
		g.dirty = true
	}
	return applied
}
