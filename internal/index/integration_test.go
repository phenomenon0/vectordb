package index

import (
	"context"
	"testing"
)

func TestFLATIndexWithFloat16Quantization(t *testing.T) {
	dim := 128
	ctx := context.Background()

	// Create FLAT index with Float16 quantization
	config := map[string]interface{}{
		"metric": "cosine",
		"quantization": map[string]interface{}{
			"type": "float16",
		},
	}

	idx, err := NewFLATIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create FLAT index: %v", err)
	}

	// Add test vectors
	numVectors := 100
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = float32(i+j) / float32(numVectors+dim)
		}

		err := idx.Add(ctx, uint64(i), vectors[i])
		if err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Search for nearest neighbors
	query := vectors[0]
	k := 5

	results, err := idx.Search(ctx, query, k, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Verify results
	if len(results) != k {
		t.Fatalf("expected %d results, got %d", k, len(results))
	}

	// The query vector itself should be the top result
	if results[0].ID != 0 {
		t.Errorf("expected top result ID to be 0, got %d", results[0].ID)
	}

	// Distance to self should be very small
	if results[0].Distance > 0.01 {
		t.Errorf("distance to self too large: %f", results[0].Distance)
	}

	// Check stats
	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Errorf("expected count %d, got %d", numVectors, stats.Count)
	}

	// Verify quantization info in stats
	if stats.Extra["quantization"] != QuantizationFloat16 {
		t.Errorf("expected quantization type %v, got %v", QuantizationFloat16, stats.Extra["quantization"])
	}

	compressionRatio := stats.Extra["compression_ratio"].(float64)
	if compressionRatio < 1.9 || compressionRatio > 2.1 {
		t.Errorf("expected compression ratio ~2.0, got %f", compressionRatio)
	}

	t.Logf("FLAT index with Float16: %d vectors, compression ratio: %.2fx", numVectors, compressionRatio)
	t.Logf("Memory used: %d bytes (%.2f MB)", stats.MemoryUsed, float64(stats.MemoryUsed)/(1024*1024))
}

func TestFLATIndexWithUint8Quantization(t *testing.T) {
	dim := 128
	ctx := context.Background()

	// Create FLAT index with Uint8 quantization
	config := map[string]interface{}{
		"metric": "cosine",
		"quantization": map[string]interface{}{
			"type": "uint8",
		},
	}

	idx, err := NewFLATIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create FLAT index: %v", err)
	}

	flat := idx.(*FLATIndex)

	// Train the quantizer first
	numTraining := 500
	trainingVectors := make([]float32, numTraining*dim)
	for i := range trainingVectors {
		trainingVectors[i] = float32(i%256) / 255.0 // Range [0, 1]
	}

	uint8Quantizer := flat.quantizer.(*Uint8Quantizer)
	err = uint8Quantizer.Train(trainingVectors)
	if err != nil {
		t.Fatalf("failed to train quantizer: %v", err)
	}

	// Add test vectors
	numVectors := 100
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = float32((i+j)%256) / 255.0
		}

		err := idx.Add(ctx, uint64(i), vectors[i])
		if err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Search for nearest neighbors
	query := vectors[0]
	k := 5

	results, err := idx.Search(ctx, query, k, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Verify results
	if len(results) != k {
		t.Fatalf("expected %d results, got %d", k, len(results))
	}

	// Check stats
	stats := idx.Stats()
	compressionRatio := stats.Extra["compression_ratio"].(float64)
	if compressionRatio < 3.9 || compressionRatio > 4.1 {
		t.Errorf("expected compression ratio ~4.0, got %f", compressionRatio)
	}

	t.Logf("FLAT index with Uint8: %d vectors, compression ratio: %.2fx", numVectors, compressionRatio)
	t.Logf("Memory used: %d bytes (%.2f MB)", stats.MemoryUsed, float64(stats.MemoryUsed)/(1024*1024))
}

func TestFLATIndexWithoutQuantization(t *testing.T) {
	dim := 128
	ctx := context.Background()

	// Create FLAT index without quantization
	config := map[string]interface{}{
		"metric": "cosine",
	}

	idx, err := NewFLATIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create FLAT index: %v", err)
	}

	// Add test vectors
	numVectors := 100
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vector[j] = float32(i+j) / float32(numVectors+dim)
		}

		err := idx.Add(ctx, uint64(i), vector)
		if err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Search
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = float32(j) / float32(dim)
	}

	results, err := idx.Search(ctx, query, 5, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}

	// Verify no quantization in stats
	stats := idx.Stats()
	if _, hasQuant := stats.Extra["quantization"]; hasQuant {
		t.Error("expected no quantization in stats")
	}

	t.Logf("FLAT index without quantization: %d vectors", numVectors)
	t.Logf("Memory used: %d bytes (%.2f MB)", stats.MemoryUsed, float64(stats.MemoryUsed)/(1024*1024))
}
func TestHNSWIndexWithFloat16Quantization(t *testing.T) {
	dim := 128
	ctx := context.Background()

	// Create HNSW index with Float16 quantization
	config := map[string]interface{}{
		"m":         16,
		"ef_search": 64,
		"quantization": map[string]interface{}{
			"type": "float16",
		},
	}

	idx, err := NewHNSWIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create HNSW index: %v", err)
	}

	hnsw := idx.(*HNSWIndex)

	// Add test vectors
	numVectors := 100
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = float32(i+j) / float32(numVectors+dim)
		}

		err := idx.Add(ctx, uint64(i), vectors[i])
		if err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Verify vectors were quantized
	if len(hnsw.quantizedData) != numVectors {
		t.Errorf("expected %d quantized vectors, got %d", numVectors, len(hnsw.quantizedData))
	}

	// Verify unquantized vectors were not stored
	if len(hnsw.vectors) != 0 {
		t.Errorf("expected no unquantized vectors, but found %d", len(hnsw.vectors))
	}

	// Search for nearest neighbors
	query := vectors[0]
	k := 5

	searchParams := HNSWSearchParams{
		EfSearch: 64,
	}

	results, err := idx.Search(ctx, query, k, searchParams)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Verify results
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}

	// The query vector itself should be in top results
	foundSelf := false
	for _, res := range results {
		if res.ID == 0 {
			foundSelf = true
			if res.Distance > 0.01 {
				t.Errorf("distance to self too large: %f", res.Distance)
			}
			break
		}
	}
	if !foundSelf {
		t.Error("expected to find query vector in results")
	}

	// Check stats
	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Errorf("expected count %d, got %d", numVectors, stats.Count)
	}

	// Verify quantization info in stats
	if stats.Extra["quantization"] != QuantizationFloat16 {
		t.Errorf("expected quantization type %v, got %v", QuantizationFloat16, stats.Extra["quantization"])
	}

	compressionRatio := stats.Extra["compression_ratio"].(float64)
	if compressionRatio < 1.9 || compressionRatio > 2.1 {
		t.Errorf("expected compression ratio ~2.0, got %f", compressionRatio)
	}

	t.Logf("HNSW index with Float16: %d vectors, compression ratio: %.2fx", numVectors, compressionRatio)
	t.Logf("Memory used: %d bytes (%.2f MB)", stats.MemoryUsed, float64(stats.MemoryUsed)/(1024*1024))
	t.Logf("Note: %s", stats.Extra["note"].(string))
}

func TestHNSWIndexWithUint8Quantization(t *testing.T) {
	dim := 128
	ctx := context.Background()

	// Create HNSW index with Uint8 quantization
	config := map[string]interface{}{
		"m":         16,
		"ef_search": 64,
		"quantization": map[string]interface{}{
			"type": "uint8",
		},
	}

	idx, err := NewHNSWIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create HNSW index: %v", err)
	}

	hnsw := idx.(*HNSWIndex)

	// Train the quantizer first
	numTraining := 500
	trainingVectors := make([]float32, numTraining*dim)
	for i := range trainingVectors {
		trainingVectors[i] = float32(i%256) / 255.0
	}

	uint8Quantizer := hnsw.quantizer.(*Uint8Quantizer)
	err = uint8Quantizer.Train(trainingVectors)
	if err != nil {
		t.Fatalf("failed to train quantizer: %v", err)
	}

	// Add test vectors
	numVectors := 100
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = float32((i+j)%256) / 255.0
		}

		err := idx.Add(ctx, uint64(i), vectors[i])
		if err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Verify vectors were quantized
	if len(hnsw.quantizedData) != numVectors {
		t.Errorf("expected %d quantized vectors, got %d", numVectors, len(hnsw.quantizedData))
	}

	// Search for nearest neighbors
	query := vectors[0]
	k := 5

	searchParams := HNSWSearchParams{
		EfSearch: 64,
	}

	results, err := idx.Search(ctx, query, k, searchParams)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Verify results
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}

	// Check stats
	stats := idx.Stats()
	compressionRatio := stats.Extra["compression_ratio"].(float64)
	if compressionRatio < 3.9 || compressionRatio > 4.1 {
		t.Errorf("expected compression ratio ~4.0, got %f", compressionRatio)
	}

	t.Logf("HNSW index with Uint8: %d vectors, compression ratio: %.2fx", numVectors, compressionRatio)
	t.Logf("Memory used: %d bytes (%.2f MB)", stats.MemoryUsed, float64(stats.MemoryUsed)/(1024*1024))
}

func TestHNSWIndexWithoutQuantization(t *testing.T) {
	dim := 128
	ctx := context.Background()

	// Create HNSW index without quantization
	config := map[string]interface{}{
		"m":         16,
		"ef_search": 64,
	}

	idx, err := NewHNSWIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create HNSW index: %v", err)
	}

	hnsw := idx.(*HNSWIndex)

	// Add vectors
	numVectors := 100
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = float32(i+j) / float32(numVectors+dim)
		}

		err := idx.Add(ctx, uint64(i), vectors[i])
		if err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Verify vectors are stored unquantized
	if len(hnsw.vectors) != numVectors {
		t.Errorf("expected %d unquantized vectors, got %d", numVectors, len(hnsw.vectors))
	}

	// Verify no quantized data
	if len(hnsw.quantizedData) != 0 {
		t.Errorf("expected no quantized data, but found %d vectors", len(hnsw.quantizedData))
	}

	// Search
	query := vectors[0]
	searchParams := HNSWSearchParams{
		EfSearch: 64,
	}

	results, err := idx.Search(ctx, query, 5, searchParams)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}

	// Verify no quantization in stats
	stats := idx.Stats()
	if _, hasQuant := stats.Extra["quantization"]; hasQuant {
		t.Error("expected no quantization in stats")
	}

	t.Logf("HNSW index without quantization: %d vectors", numVectors)
	t.Logf("Memory used: %d bytes (%.2f MB)", stats.MemoryUsed, float64(stats.MemoryUsed)/(1024*1024))
}
