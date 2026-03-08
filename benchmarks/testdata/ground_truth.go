package testdata

import (
	"context"
	"fmt"

	"github.com/phenomenon0/vectordb/internal/index"
)

// GroundTruth holds exact nearest-neighbor results computed via brute-force FLAT index.
type GroundTruth struct {
	// Neighbors[i] contains the k nearest neighbor IDs for query i, ordered by distance.
	Neighbors [][]uint64
	// Distances[i] contains the corresponding distances.
	Distances [][]float32
}

// ComputeGroundTruth computes exact nearest neighbors using a FLAT index (brute-force).
// This is the reference for measuring recall of approximate indexes.
func ComputeGroundTruth(vectors, queries [][]float32, k int) (*GroundTruth, error) {
	if len(vectors) == 0 || len(queries) == 0 {
		return nil, fmt.Errorf("vectors and queries must be non-empty")
	}
	dim := len(vectors[0])

	flat, err := index.NewFLATIndex(dim, map[string]interface{}{
		"metric": "cosine",
	})
	if err != nil {
		return nil, fmt.Errorf("creating FLAT index: %w", err)
	}

	ctx := context.Background()
	for i, v := range vectors {
		if err := flat.Add(ctx, uint64(i), v); err != nil {
			return nil, fmt.Errorf("adding vector %d: %w", i, err)
		}
	}

	gt := &GroundTruth{
		Neighbors: make([][]uint64, len(queries)),
		Distances: make([][]float32, len(queries)),
	}

	for i, q := range queries {
		results, err := flat.Search(ctx, q, k, &index.DefaultSearchParams{})
		if err != nil {
			return nil, fmt.Errorf("searching query %d: %w", i, err)
		}
		ids := make([]uint64, len(results))
		dists := make([]float32, len(results))
		for j, r := range results {
			ids[j] = r.ID
			dists[j] = r.Distance
		}
		gt.Neighbors[i] = ids
		gt.Distances[i] = dists
	}
	return gt, nil
}

// RecallAt computes recall@k: fraction of true top-k neighbors found in the result set.
func RecallAt(gt *GroundTruth, results [][]uint64, k int) float64 {
	if len(gt.Neighbors) == 0 || len(results) == 0 {
		return 0
	}
	totalRecall := 0.0
	n := len(gt.Neighbors)
	if len(results) < n {
		n = len(results)
	}

	for i := 0; i < n; i++ {
		trueSet := make(map[uint64]bool)
		limit := k
		if limit > len(gt.Neighbors[i]) {
			limit = len(gt.Neighbors[i])
		}
		for _, id := range gt.Neighbors[i][:limit] {
			trueSet[id] = true
		}

		found := 0
		resultLimit := k
		if resultLimit > len(results[i]) {
			resultLimit = len(results[i])
		}
		for _, id := range results[i][:resultLimit] {
			if trueSet[id] {
				found++
			}
		}
		if limit > 0 {
			totalRecall += float64(found) / float64(limit)
		}
	}
	return totalRecall / float64(n)
}
