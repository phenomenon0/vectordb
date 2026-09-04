package graph

// Local CSR + PageRank implementation replacing the external cowrie/gnn/algo
// dependency (retired private module). Semantics follow the standard power
// iteration with uniform teleport; PersonalizedPageRank teleports uniformly
// over the provided seed nodes instead of all nodes.

// CSR is a compressed sparse row adjacency matrix over int64 node indices.
type CSR struct {
	n       int
	indPtr  []int64 // length n+1
	indices []int64 // length indPtr[n]
}

// NewCSR builds a CSR matrix from row pointers and packed column indices.
func NewCSR(n int, indPtr, indices []int64) *CSR {
	return &CSR{n: n, indPtr: indPtr, indices: indices}
}

// PageRankConfig configures the power iteration.
type PageRankConfig struct {
	// Damping is the teleport probability complement (default 0.85).
	Damping float32
	// Iterations bounds the power iteration (default 20).
	Iterations int
	// Tolerance stops iteration when the L1 score delta falls below it.
	Tolerance float32
}

// PageRankResult holds one score per node, summing to ~1.
type PageRankResult struct {
	Scores []float32
	// Converged reports whether the L1 delta fell below tolerance before
	// the iteration budget was exhausted.
	Converged bool
	// Iterations is the number of power-iteration sweeps performed.
	Iterations int
}

// PageRank runs power iteration on the CSR graph with uniform teleport.
func PageRank(csr *CSR, cfg PageRankConfig) *PageRankResult {
	return pageRank(csr, cfg, nil)
}

// NewCSR is kept for symmetry; n equals len(indPtr)-1.

// PersonalizedPageRank runs power iteration teleporting uniformly over seeds
// instead of all nodes. Empty or out-of-range seeds fall back to global
// PageRank semantics for the affected entries.
func PersonalizedPageRank(csr *CSR, cfg PageRankConfig, seeds []int64) *PageRankResult {
	if len(seeds) == 0 || csr == nil || csr.n == 0 {
		return PageRank(csr, cfg)
	}
	preferred := make([]float32, csr.n)
	for _, s := range seeds {
		if s >= 0 && int(s) < csr.n {
			preferred[s]++
		}
	}
	total := float32(0)
	for _, v := range preferred {
		total += v
	}
	if total == 0 {
		return PageRank(csr, cfg)
	}
	for i := range preferred {
		preferred[i] /= total
	}
	return pageRank(csr, cfg, preferred)
}

func pageRank(csr *CSR, cfg PageRankConfig, teleport []float32) *PageRankResult {
	n := csr.n
	if n == 0 {
		return &PageRankResult{Scores: nil}
	}
	damping := cfg.Damping
	if damping <= 0 || damping >= 1 {
		damping = 0.85
	}
	iterations := cfg.Iterations
	if iterations <= 0 {
		iterations = 20
	}
	tolerance := cfg.Tolerance
	if tolerance <= 0 {
		tolerance = 1e-6
	}

	// Out-degree per node and dangling-node list.
	outDeg := make([]int64, n)
	for i := 0; i < n; i++ {
		outDeg[i] = csr.indPtr[i+1] - csr.indPtr[i]
	}

	scores := make([]float32, n)
	next := make([]float32, n)
	base := float32(1) / float32(n)
	for i := range scores {
		scores[i] = base
	}

	iter := 0
	for ; iter < iterations; iter++ {
		for i := range next {
			next[i] = 0
		}
		danglingMass := float32(0)
		for i := 0; i < n; i++ {
			if scores[i] == 0 {
				continue
			}
			if outDeg[i] == 0 {
				danglingMass += scores[i]
				continue
			}
			share := scores[i] / float32(outDeg[i])
			for k := csr.indPtr[i]; k < csr.indPtr[i+1]; k++ {
				dst := csr.indices[k]
				if dst >= 0 && int(dst) < n {
					next[dst] += share
				}
			}
		}

		delta := float32(0)
		for i := 0; i < n; i++ {
			var teleportWeight float32
			if teleport != nil {
				teleportWeight = teleport[i]
			} else {
				teleportWeight = base
			}
			v := (1-damping)*teleportWeight + damping*(next[i]+danglingMass*base)
			delta += abs32(v - scores[i])
			scores[i] = v
		}
		if delta < tolerance*float32(n) {
			return &PageRankResult{Scores: scores, Converged: true, Iterations: iter + 1}
		}
	}
	return &PageRankResult{Scores: scores, Converged: false, Iterations: iterations}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
