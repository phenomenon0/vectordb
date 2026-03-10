package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// TestRecallAtScale verifies recall doesn't degrade catastrophically as the graph grows.
// Baseline: recall@10 should stay above 0.85 at 50k with ef_search=128.
func TestRecallAtScale(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping scale recall test in short mode")
	}

	scales := []int{1000, 10000, 50000}
	dim := 128
	k := 10
	numQueries := 100

	for _, n := range scales {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			// Generate random vectors
			vectors := make([][]float32, n)
			for i := range vectors {
				v := make([]float32, dim)
				for j := range v {
					v[j] = rng.Float32()*2 - 1
				}
				// Normalize
				var norm float32
				for _, x := range v {
					norm += x * x
				}
				norm = float32(math.Sqrt(float64(norm)))
				for j := range v {
					v[j] /= norm
				}
				vectors[i] = v
			}

			// Build HNSW graph
			g := &Graph[uint64]{
				M:        16,
				Ml:       1.0 / math.Log(16),
				Distance: CosineDistance,
				EfSearch:  300, // ef_construction during build
				Rng:      rand.New(rand.NewSource(42)),
			}

			nodes := make([]Node[uint64], n)
			for i := 0; i < n; i++ {
				nodes[i] = MakeNode(uint64(i), vectors[i])
			}
			g.Add(nodes...)

			// Generate queries and compute recall
			var totalRecall float64
			for q := 0; q < numQueries; q++ {
				query := vectors[rng.Intn(n)]

				// Brute force ground truth
				type idDist struct {
					id   uint64
					dist float32
				}
				dists := make([]idDist, n)
				for i := 0; i < n; i++ {
					dists[i] = idDist{uint64(i), CosineDistance(query, vectors[i])}
				}
				sort.Slice(dists, func(i, j int) bool { return dists[i].dist < dists[j].dist })
				truth := make(map[uint64]bool, k)
				for i := 0; i < k; i++ {
					truth[dists[i].id] = true
				}

				// HNSW search with ef=128
				results := g.SearchWithEf(query, k, 200, nil)
				hits := 0
				for _, r := range results {
					if truth[r.Key] {
						hits++
					}
				}
				totalRecall += float64(hits) / float64(k)
			}

			recall := totalRecall / float64(numQueries)
			t.Logf("n=%d recall@%d = %.4f", n, k, recall)

			if recall < 0.95 {
				t.Errorf("recall@%d = %.4f, want >= 0.95", k, recall)
			}
		})
	}
}
