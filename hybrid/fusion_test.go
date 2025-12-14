package hybrid

import (
	"math"
	"testing"
)

func TestFuseRRF_SingleResultSet(t *testing.T) {
	results := []SearchResult{
		{DocID: 1, Score: 0.9},
		{DocID: 2, Score: 0.8},
		{DocID: 3, Score: 0.7},
	}

	resultSets := []ResultSet{
		{Results: results, Weight: 1.0},
	}

	fused := fuseRRF(resultSets, 60.0, 10)

	// Should preserve order
	if len(fused) != 3 {
		t.Fatalf("expected 3 results, got %d", len(fused))
	}

	expectedOrder := []uint64{1, 2, 3}
	for i, docID := range expectedOrder {
		if fused[i].DocID != docID {
			t.Errorf("position %d: expected doc%d, got doc%d", i, docID, fused[i].DocID)
		}
	}
}

func TestFuseRRF_TwoResultSets(t *testing.T) {
	// Dense results: doc1 > doc2 > doc3
	dense := []SearchResult{
		{DocID: 1, Score: 0.9},
		{DocID: 2, Score: 0.8},
		{DocID: 3, Score: 0.7},
	}

	// Sparse results: doc3 > doc1 > doc4
	sparse := []SearchResult{
		{DocID: 3, Score: 10.0},
		{DocID: 1, Score: 8.0},
		{DocID: 4, Score: 6.0},
	}

	resultSets := []ResultSet{
		{Results: dense, Weight: 0.7},
		{Results: sparse, Weight: 0.3},
	}

	fused := fuseRRF(resultSets, 60.0, 10)

	// Verify doc1 and doc3 both appear (they're in both lists)
	foundDocs := make(map[uint64]bool)
	for _, result := range fused {
		foundDocs[result.DocID] = true
	}

	if !foundDocs[1] {
		t.Error("doc1 should be in fused results")
	}
	if !foundDocs[3] {
		t.Error("doc3 should be in fused results")
	}

	// doc3 should rank high because it's first in sparse and third in dense
	// doc1 should rank high because it's first in dense and second in sparse
	if len(fused) < 2 {
		t.Fatal("expected at least 2 results")
	}

	// Both doc1 and doc3 should be in top 2
	top2 := map[uint64]bool{fused[0].DocID: true, fused[1].DocID: true}
	if !top2[1] || !top2[3] {
		t.Error("doc1 and doc3 should both be in top 2")
	}
}

func TestFuseRRF_Formula(t *testing.T) {
	// Test RRF formula directly
	// score(doc) = Σ[1 / (k + rank_i)]

	// Doc1 appears at rank 0 in one list
	// Expected score: 1 / (60 + 0) = 1/60 ≈ 0.01667

	results := []SearchResult{
		{DocID: 1, Score: 1.0},
	}

	resultSets := []ResultSet{
		{Results: results, Weight: 1.0},
	}

	fused := fuseRRF(resultSets, 60.0, 10)

	expectedScore := float32(1.0 / 60.0)
	actualScore := fused[0].Score

	if math.Abs(float64(actualScore-expectedScore)) > 0.0001 {
		t.Errorf("RRF score mismatch: got %f, want %f", actualScore, expectedScore)
	}
}

func TestFuseWeighted_Normalization(t *testing.T) {
	// Results with different score ranges
	results1 := []SearchResult{
		{DocID: 1, Score: 100.0},
		{DocID: 2, Score: 50.0},
	}

	results2 := []SearchResult{
		{DocID: 2, Score: 1.0},
		{DocID: 3, Score: 0.5},
	}

	resultSets := []ResultSet{
		{Results: results1, Weight: 0.7},
		{Results: results2, Weight: 0.3},
	}

	fused := fuseWeighted(resultSets, 10)

	// Doc2 appears in both lists, should have higher score
	foundDoc2 := false
	for _, result := range fused {
		if result.DocID == 2 {
			foundDoc2 = true
			// Doc2 should have combined score from both lists
			if result.Score <= 0 {
				t.Error("doc2 score should be positive")
			}
		}
	}

	if !foundDoc2 {
		t.Error("doc2 should be in results")
	}
}

func TestFuseLinear(t *testing.T) {
	results1 := []SearchResult{
		{DocID: 1, Score: 10.0},
		{DocID: 2, Score: 5.0},
	}

	results2 := []SearchResult{
		{DocID: 2, Score: 8.0},
		{DocID: 3, Score: 3.0},
	}

	resultSets := []ResultSet{
		{Results: results1, Weight: 0.5},
		{Results: results2, Weight: 0.5},
	}

	fused := fuseLinear(resultSets, 10)

	// Doc2: 5.0 * 0.5 + 8.0 * 0.5 = 6.5
	// Doc1: 10.0 * 0.5 = 5.0
	// Doc3: 3.0 * 0.5 = 1.5

	if len(fused) != 3 {
		t.Fatalf("expected 3 results, got %d", len(fused))
	}

	// Doc2 should be first (highest combined score)
	if fused[0].DocID != 2 {
		t.Errorf("doc2 should be first, got doc%d", fused[0].DocID)
	}

	expectedDoc2Score := float32(6.5)
	if math.Abs(float64(fused[0].Score-expectedDoc2Score)) > 0.01 {
		t.Errorf("doc2 score mismatch: got %f, want %f", fused[0].Score, expectedDoc2Score)
	}
}

func TestHybridSearch(t *testing.T) {
	dense := []SearchResult{
		{DocID: 1, Score: 0.95},
		{DocID: 2, Score: 0.90},
		{DocID: 3, Score: 0.85},
	}

	sparse := []SearchResult{
		{DocID: 2, Score: 15.0},
		{DocID: 4, Score: 12.0},
		{DocID: 1, Score: 10.0},
	}

	params := DefaultFusionParams()
	results, err := HybridSearch(dense, sparse, params, 5)

	if err != nil {
		t.Fatalf("HybridSearch failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected results from HybridSearch")
	}

	// Verify doc1 and doc2 are in results (both appear in both lists)
	foundDocs := make(map[uint64]bool)
	for _, r := range results {
		foundDocs[r.DocID] = true
	}

	if !foundDocs[1] || !foundDocs[2] {
		t.Error("doc1 and doc2 should be in hybrid results")
	}
}

func TestHybridSearch_EmptyResults(t *testing.T) {
	params := DefaultFusionParams()

	// Both empty
	results, err := HybridSearch([]SearchResult{}, []SearchResult{}, params, 5)
	if err != nil {
		t.Errorf("should not error on empty results")
	}
	if len(results) != 0 {
		t.Error("expected empty results")
	}

	// Only dense
	dense := []SearchResult{{DocID: 1, Score: 0.9}}
	results, err = HybridSearch(dense, []SearchResult{}, params, 5)
	if err != nil {
		t.Errorf("should not error on empty sparse")
	}
	if len(results) == 0 {
		t.Error("expected results from dense only")
	}

	// Only sparse
	sparse := []SearchResult{{DocID: 2, Score: 10.0}}
	results, err = HybridSearch([]SearchResult{}, sparse, params, 5)
	if err != nil {
		t.Errorf("should not error on empty dense")
	}
	if len(results) == 0 {
		t.Error("expected results from sparse only")
	}
}

func TestTopK_Limiting(t *testing.T) {
	results := []SearchResult{
		{DocID: 1, Score: 10.0},
		{DocID: 2, Score: 9.0},
		{DocID: 3, Score: 8.0},
		{DocID: 4, Score: 7.0},
		{DocID: 5, Score: 6.0},
	}

	resultSets := []ResultSet{
		{Results: results, Weight: 1.0},
	}

	// Request top-3
	fused := fuseRRF(resultSets, 60.0, 3)

	if len(fused) != 3 {
		t.Errorf("expected 3 results, got %d", len(fused))
	}

	// Verify order preserved in top-3
	expectedIDs := []uint64{1, 2, 3}
	for i, expectedID := range expectedIDs {
		if fused[i].DocID != expectedID {
			t.Errorf("position %d: expected doc%d, got doc%d", i, expectedID, fused[i].DocID)
		}
	}
}

func TestValidateFusionParams(t *testing.T) {
	// Valid params
	params := DefaultFusionParams()
	if err := ValidateFusionParams(params); err != nil {
		t.Errorf("default params should be valid: %v", err)
	}

	// Invalid K
	badParams := params
	badParams.K = 0
	if err := ValidateFusionParams(badParams); err == nil {
		t.Error("should error on K = 0")
	}

	// Negative weights
	badParams = params
	badParams.Strategy = FusionWeighted
	badParams.DenseWeight = -0.1
	if err := ValidateFusionParams(badParams); err == nil {
		t.Error("should error on negative weights")
	}

	// Both weights zero
	badParams = params
	badParams.Strategy = FusionWeighted
	badParams.DenseWeight = 0
	badParams.SparseWeight = 0
	if err := ValidateFusionParams(badParams); err == nil {
		t.Error("should error when all weights are zero")
	}
}

func TestNormalizeWeights(t *testing.T) {
	params := FusionParams{
		Strategy:     FusionWeighted,
		DenseWeight:  0.7,
		SparseWeight: 0.3,
	}

	// Already normalized (sum = 1.0)
	NormalizeWeights(&params)

	sum := params.DenseWeight + params.SparseWeight
	if math.Abs(float64(sum-1.0)) > 0.0001 {
		t.Errorf("normalized weights should sum to 1.0, got %f", sum)
	}

	// Unnormalized weights
	params.DenseWeight = 2.0
	params.SparseWeight = 1.0

	NormalizeWeights(&params)

	sum = params.DenseWeight + params.SparseWeight
	if math.Abs(float64(sum-1.0)) > 0.0001 {
		t.Errorf("normalized weights should sum to 1.0, got %f", sum)
	}

	// Should be 2/3 and 1/3
	expectedDense := float32(2.0 / 3.0)
	if math.Abs(float64(params.DenseWeight-expectedDense)) > 0.0001 {
		t.Errorf("dense weight should be 2/3, got %f", params.DenseWeight)
	}
}

func BenchmarkFuseRRF(b *testing.B) {
	// Simulate realistic hybrid search: 100 dense + 100 sparse results
	dense := make([]SearchResult, 100)
	for i := range dense {
		dense[i] = SearchResult{DocID: uint64(i), Score: float32(100 - i)}
	}

	sparse := make([]SearchResult, 100)
	for i := range sparse {
		sparse[i] = SearchResult{DocID: uint64(i + 50), Score: float32(100 - i)}
	}

	resultSets := []ResultSet{
		{Results: dense, Weight: 0.7},
		{Results: sparse, Weight: 0.3},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fuseRRF(resultSets, 60.0, 10)
	}
}

func BenchmarkFuseWeighted(b *testing.B) {
	dense := make([]SearchResult, 100)
	for i := range dense {
		dense[i] = SearchResult{DocID: uint64(i), Score: float32(100 - i)}
	}

	sparse := make([]SearchResult, 100)
	for i := range sparse {
		sparse[i] = SearchResult{DocID: uint64(i + 50), Score: float32(100 - i)}
	}

	resultSets := []ResultSet{
		{Results: dense, Weight: 0.7},
		{Results: sparse, Weight: 0.3},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fuseWeighted(resultSets, 10)
	}
}

func BenchmarkHybridSearch(b *testing.B) {
	dense := make([]SearchResult, 100)
	for i := range dense {
		dense[i] = SearchResult{DocID: uint64(i), Score: float32(100 - i)}
	}

	sparse := make([]SearchResult, 100)
	for i := range sparse {
		sparse[i] = SearchResult{DocID: uint64(i + 50), Score: float32(100 - i)}
	}

	params := DefaultFusionParams()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = HybridSearch(dense, sparse, params, 10)
	}
}
