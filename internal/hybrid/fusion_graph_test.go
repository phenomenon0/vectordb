package hybrid

import (
	"testing"
)

func TestHybridSearchWithGraph(t *testing.T) {
	dense := []SearchResult{
		{DocID: 1, Score: 0.9},
		{DocID: 2, Score: 0.7},
		{DocID: 3, Score: 0.5},
	}

	sparse := []SearchResult{
		{DocID: 2, Score: 0.8},
		{DocID: 1, Score: 0.6},
		{DocID: 4, Score: 0.4},
	}

	graph := []SearchResult{
		{DocID: 3, Score: 0.95}, // Graph says doc 3 is important
		{DocID: 1, Score: 0.5},
		{DocID: 2, Score: 0.3},
	}

	params := FusionParams{
		Strategy:     FusionRRF,
		K:            60,
		DenseWeight:  0.5,
		SparseWeight: 0.3,
		GraphWeight:  0.2,
	}

	results, err := HybridSearchWithGraph(dense, sparse, graph, params, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected results")
	}

	// With RRF, all 4 docs should appear
	docIDs := make(map[uint64]bool)
	for _, r := range results {
		docIDs[r.DocID] = true
	}

	for _, id := range []uint64{1, 2, 3, 4} {
		if !docIDs[id] {
			t.Errorf("expected doc %d in results", id)
		}
	}
}

func TestHybridSearchWithGraphDisabled(t *testing.T) {
	dense := []SearchResult{
		{DocID: 1, Score: 0.9},
	}
	sparse := []SearchResult{
		{DocID: 1, Score: 0.8},
	}

	params := FusionParams{
		Strategy:     FusionRRF,
		K:            60,
		DenseWeight:  0.7,
		SparseWeight: 0.3,
		GraphWeight:  0.0, // Disabled
	}

	// Empty graph results, weight 0 — should work like normal HybridSearch
	results, err := HybridSearchWithGraph(dense, sparse, nil, params, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}

func TestGraphWeightNormalization(t *testing.T) {
	params := &FusionParams{
		Strategy:     FusionWeighted,
		DenseWeight:  0.5,
		SparseWeight: 0.3,
		GraphWeight:  0.2,
	}

	NormalizeWeights(params)

	total := params.DenseWeight + params.SparseWeight + params.GraphWeight
	if total < 0.999 || total > 1.001 {
		t.Errorf("weights should sum to 1.0, got %f", total)
	}
}

func TestValidateFusionParamsWithGraph(t *testing.T) {
	params := FusionParams{
		Strategy:     FusionWeighted,
		K:            60,
		DenseWeight:  0.5,
		SparseWeight: 0.3,
		GraphWeight:  -0.1, // Invalid
	}

	if err := ValidateFusionParams(params); err == nil {
		t.Error("expected error for negative graph weight")
	}
}
