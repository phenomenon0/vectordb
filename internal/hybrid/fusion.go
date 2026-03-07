package hybrid

import (
	"fmt"
	"sort"
)

// FusionStrategy defines the method used to combine results from multiple indexes.
type FusionStrategy int

const (
	// FusionRRF uses Reciprocal Rank Fusion (default, parameter-free).
	// Formula: score(doc) = Σ[1 / (k + rank_i)] where k=60, rank_i = position in result list i
	FusionRRF FusionStrategy = iota

	// FusionWeighted uses weighted sum of normalized scores.
	// Requires specifying weights for each result list.
	FusionWeighted

	// FusionLinear uses linear combination of raw scores.
	// Assumes scores are already normalized to same scale.
	FusionLinear
)

// SearchResult represents a single search result with document ID and score.
type SearchResult struct {
	DocID uint64
	Score float32
}

// ResultSet represents results from a single index (dense or sparse).
type ResultSet struct {
	Results []SearchResult
	Weight  float32 // Weight for this result set (used in weighted fusion)
}

// FusionParams configures the fusion strategy.
type FusionParams struct {
	Strategy FusionStrategy

	// RRF parameters
	K float32 // Constant for RRF (default: 60)

	// Weighted fusion parameters
	DenseWeight  float32 // Weight for dense results (default: 0.7)
	SparseWeight float32 // Weight for sparse results (default: 0.3)
}

// DefaultFusionParams returns recommended fusion parameters.
func DefaultFusionParams() FusionParams {
	return FusionParams{
		Strategy:     FusionRRF,
		K:            60.0,
		DenseWeight:  0.7,
		SparseWeight: 0.3,
	}
}

// Fuse combines multiple result sets using the specified fusion strategy.
func Fuse(resultSets []ResultSet, params FusionParams, topK int) []SearchResult {
	switch params.Strategy {
	case FusionRRF:
		return fuseRRF(resultSets, params.K, topK)
	case FusionWeighted:
		return fuseWeighted(resultSets, topK)
	case FusionLinear:
		return fuseLinear(resultSets, topK)
	default:
		return fuseRRF(resultSets, params.K, topK)
	}
}

// fuseRRF implements Reciprocal Rank Fusion.
//
// RRF is a parameter-free fusion method that works well for combining
// results from heterogeneous sources (dense + sparse).
//
// Formula: score(doc) = Σ[1 / (k + rank_i)]
// where k=60 (constant), rank_i = position (0-indexed) in result list i
//
// References:
//   - Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
func fuseRRF(resultSets []ResultSet, k float32, topK int) []SearchResult {
	if len(resultSets) == 0 {
		return []SearchResult{}
	}

	// Accumulate RRF scores for each document
	scores := make(map[uint64]float32)

	for _, rs := range resultSets {
		// Process each result in this set
		for rank, result := range rs.Results {
			// RRF score: 1 / (k + rank)
			// rank is 0-indexed
			rrfScore := 1.0 / (k + float32(rank))

			scores[result.DocID] += rrfScore
		}
	}

	// Convert to result array
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{
			DocID: docID,
			Score: score,
		})
	}

	// Sort by RRF score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top-k
	if topK < len(results) {
		results = results[:topK]
	}

	return results
}

// fuseWeighted implements weighted fusion of normalized scores.
//
// This method:
// 1. Normalizes scores within each result set to [0, 1]
// 2. Applies weights to each normalized score
// 3. Sums weighted scores per document
func fuseWeighted(resultSets []ResultSet, topK int) []SearchResult {
	if len(resultSets) == 0 {
		return []SearchResult{}
	}

	// Accumulate weighted scores
	scores := make(map[uint64]float32)

	for _, rs := range resultSets {
		if len(rs.Results) == 0 {
			continue
		}

		// Normalize scores in this result set to [0, 1]
		minScore := rs.Results[len(rs.Results)-1].Score
		maxScore := rs.Results[0].Score
		scoreRange := maxScore - minScore

		// Handle edge case: all scores are the same
		if scoreRange == 0 {
			scoreRange = 1.0
		}

		// Add weighted normalized scores
		for _, result := range rs.Results {
			normalizedScore := (result.Score - minScore) / scoreRange
			weightedScore := normalizedScore * rs.Weight

			scores[result.DocID] += weightedScore
		}
	}

	// Convert to result array
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{
			DocID: docID,
			Score: score,
		})
	}

	// Sort by weighted score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top-k
	if topK < len(results) {
		results = results[:topK]
	}

	return results
}

// fuseLinear implements linear combination of raw scores.
//
// Assumes scores are already on comparable scales.
// Formula: score(doc) = Σ[weight_i * score_i]
func fuseLinear(resultSets []ResultSet, topK int) []SearchResult {
	if len(resultSets) == 0 {
		return []SearchResult{}
	}

	// Accumulate linear scores
	scores := make(map[uint64]float32)

	for _, rs := range resultSets {
		for _, result := range rs.Results {
			weightedScore := result.Score * rs.Weight
			scores[result.DocID] += weightedScore
		}
	}

	// Convert to result array
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{
			DocID: docID,
			Score: score,
		})
	}

	// Sort by linear score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top-k
	if topK < len(results) {
		results = results[:topK]
	}

	return results
}

// HybridSearch performs hybrid search combining dense and sparse results.
//
// This is a convenience function that wraps Fuse for the common case
// of combining exactly two result sets (dense + sparse).
func HybridSearch(
	denseResults []SearchResult,
	sparseResults []SearchResult,
	params FusionParams,
	topK int,
) ([]SearchResult, error) {
	if len(denseResults) == 0 && len(sparseResults) == 0 {
		return []SearchResult{}, nil
	}

	resultSets := []ResultSet{
		{
			Results: denseResults,
			Weight:  params.DenseWeight,
		},
		{
			Results: sparseResults,
			Weight:  params.SparseWeight,
		},
	}

	return Fuse(resultSets, params, topK), nil
}

// ValidateFusionParams checks if fusion parameters are valid.
func ValidateFusionParams(params FusionParams) error {
	if params.K <= 0 {
		return fmt.Errorf("RRF constant K must be positive, got %f", params.K)
	}

	if params.Strategy == FusionWeighted {
		if params.DenseWeight < 0 || params.SparseWeight < 0 {
			return fmt.Errorf("weights must be non-negative")
		}

		total := params.DenseWeight + params.SparseWeight
		if total == 0 {
			return fmt.Errorf("at least one weight must be positive")
		}
	}

	return nil
}

// NormalizeWeights normalizes fusion weights to sum to 1.0.
func NormalizeWeights(params *FusionParams) {
	if params.Strategy == FusionWeighted {
		total := params.DenseWeight + params.SparseWeight
		if total > 0 {
			params.DenseWeight /= total
			params.SparseWeight /= total
		}
	}
}
