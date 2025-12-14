package index

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
)

// Benchmark quantization overhead for different index types at various scales

// Helper function to generate random vectors
func generateRandomVectors(count, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rand.Float32()
		}
	}
	return vectors
}

// Benchmark FLAT index memory and search performance with quantization

func BenchmarkFLATNoQuantization_1K(b *testing.B)       { benchmarkFLAT(b, 1000, "none") }
func BenchmarkFLATFloat16_1K(b *testing.B)              { benchmarkFLAT(b, 1000, "float16") }
func BenchmarkFLATUint8_1K(b *testing.B)                { benchmarkFLAT(b, 1000, "uint8") }
func BenchmarkFLATNoQuantization_10K(b *testing.B)      { benchmarkFLAT(b, 10000, "none") }
func BenchmarkFLATFloat16_10K(b *testing.B)             { benchmarkFLAT(b, 10000, "float16") }
func BenchmarkFLATUint8_10K(b *testing.B)               { benchmarkFLAT(b, 10000, "uint8") }
func BenchmarkFLATNoQuantization_100K(b *testing.B)     { benchmarkFLAT(b, 100000, "none") }
func BenchmarkFLATFloat16_100K(b *testing.B)            { benchmarkFLAT(b, 100000, "float16") }
func BenchmarkFLATUint8_100K(b *testing.B)              { benchmarkFLAT(b, 100000, "uint8") }

func benchmarkFLAT(b *testing.B, vectorCount int, quantType string) {
	dim := 384 // Common embedding dimension
	ctx := context.Background()

	// Create index
	config := map[string]interface{}{
		"metric": "cosine",
	}

	if quantType != "none" {
		config["quantization"] = map[string]interface{}{
			"type": quantType,
		}
	}

	idx, err := NewFLATIndex(dim, config)
	if err != nil {
		b.Fatal(err)
	}

	// Train uint8 quantizer if needed
	if quantType == "uint8" {
		flat := idx.(*FLATIndex)
		trainingData := make([]float32, 1000*dim)
		for i := range trainingData {
			trainingData[i] = rand.Float32()
		}
		uint8Quantizer := flat.quantizer.(*Uint8Quantizer)
		if err := uint8Quantizer.Train(trainingData); err != nil {
			b.Fatal(err)
		}
	}

	// Add vectors
	vectors := generateRandomVectors(vectorCount, dim)
	for i, vec := range vectors {
		if err := idx.Add(ctx, uint64(i), vec); err != nil {
			b.Fatal(err)
		}
	}

	// Get memory stats
	stats := idx.Stats()
	memoryMB := float64(stats.MemoryUsed) / (1024 * 1024)

	b.ReportMetric(memoryMB, "MB")
	b.ReportMetric(float64(stats.MemoryUsed)/float64(vectorCount), "bytes/vector")

	// Benchmark search
	query := vectors[0]
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.Search(ctx, query, 10, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark IVF index with quantization

func BenchmarkIVFNoQuantization_10K(b *testing.B)  { benchmarkIVF(b, 10000, "none") }
func BenchmarkIVFFloat16_10K(b *testing.B)         { benchmarkIVF(b, 10000, "float16") }
func BenchmarkIVFUint8_10K(b *testing.B)           { benchmarkIVF(b, 10000, "uint8") }
func BenchmarkIVFNoQuantization_100K(b *testing.B) { benchmarkIVF(b, 100000, "none") }
func BenchmarkIVFFloat16_100K(b *testing.B)        { benchmarkIVF(b, 100000, "float16") }
func BenchmarkIVFUint8_100K(b *testing.B)          { benchmarkIVF(b, 100000, "uint8") }

func benchmarkIVF(b *testing.B, vectorCount int, quantType string) {
	dim := 384
	ctx := context.Background()

	// Configure IVF
	nlist := vectorCount / 100 // 1% of vectors as clusters
	if nlist < 10 {
		nlist = 10
	}

	config := map[string]interface{}{
		"metric": "cosine",
		"nlist":  nlist,
		"nprobe": 10,
	}

	if quantType != "none" {
		config["quantization"] = map[string]interface{}{
			"type": quantType,
		}
	}

	idx, err := NewIVFIndex(dim, config)
	if err != nil {
		b.Fatal(err)
	}

	// Train uint8 quantizer if needed
	if quantType == "uint8" {
		ivf := idx.(*IVFIndex)
		trainingData := make([]float32, 1000*dim)
		for i := range trainingData {
			trainingData[i] = rand.Float32()
		}
		uint8Quantizer := ivf.quantizer.(*Uint8Quantizer)
		if err := uint8Quantizer.Train(trainingData); err != nil {
			b.Fatal(err)
		}
	}

	// Add vectors
	vectors := generateRandomVectors(vectorCount, dim)
	for i, vec := range vectors {
		if err := idx.Add(ctx, uint64(i), vec); err != nil {
			b.Fatal(err)
		}
	}

	// Get memory stats
	stats := idx.Stats()
	memoryMB := float64(stats.MemoryUsed) / (1024 * 1024)

	b.ReportMetric(memoryMB, "MB")
	b.ReportMetric(float64(stats.MemoryUsed)/float64(vectorCount), "bytes/vector")

	// Benchmark search
	query := vectors[0]
	searchParams := &IVFSearchParams{NProbe: 10}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.Search(ctx, query, 10, searchParams)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark HNSW index with quantization

func BenchmarkHNSWNoQuantization_10K(b *testing.B)  { benchmarkHNSW(b, 10000, "none") }
func BenchmarkHNSWFloat16_10K(b *testing.B)         { benchmarkHNSW(b, 10000, "float16") }
func BenchmarkHNSWUint8_10K(b *testing.B)           { benchmarkHNSW(b, 10000, "uint8") }
func BenchmarkHNSWNoQuantization_100K(b *testing.B) { benchmarkHNSW(b, 100000, "none") }
func BenchmarkHNSWFloat16_100K(b *testing.B)        { benchmarkHNSW(b, 100000, "float16") }
func BenchmarkHNSWUint8_100K(b *testing.B)          { benchmarkHNSW(b, 100000, "uint8") }

func benchmarkHNSW(b *testing.B, vectorCount int, quantType string) {
	dim := 384
	ctx := context.Background()

	config := map[string]interface{}{
		"m":         16,
		"ef_search": 64,
	}

	if quantType != "none" {
		config["quantization"] = map[string]interface{}{
			"type": quantType,
		}
	}

	idx, err := NewHNSWIndex(dim, config)
	if err != nil {
		b.Fatal(err)
	}

	// Train uint8 quantizer if needed
	if quantType == "uint8" {
		hnsw := idx.(*HNSWIndex)
		trainingData := make([]float32, 1000*dim)
		for i := range trainingData {
			trainingData[i] = rand.Float32()
		}
		uint8Quantizer := hnsw.quantizer.(*Uint8Quantizer)
		if err := uint8Quantizer.Train(trainingData); err != nil {
			b.Fatal(err)
		}
	}

	// Add vectors
	vectors := generateRandomVectors(vectorCount, dim)
	for i, vec := range vectors {
		if err := idx.Add(ctx, uint64(i), vec); err != nil {
			b.Fatal(err)
		}
	}

	// Get memory stats
	stats := idx.Stats()
	memoryMB := float64(stats.MemoryUsed) / (1024 * 1024)

	b.ReportMetric(memoryMB, "MB")
	b.ReportMetric(float64(stats.MemoryUsed)/float64(vectorCount), "bytes/vector")

	// Benchmark search
	query := vectors[0]
	searchParams := HNSWSearchParams{EfSearch: 64}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.Search(ctx, query, 10, searchParams)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark DiskANN index with quantization

func BenchmarkDiskANNNoQuantization_10K(b *testing.B)  { benchmarkDiskANN(b, 10000, "none") }
func BenchmarkDiskANNFloat16_10K(b *testing.B)         { benchmarkDiskANN(b, 10000, "float16") }
func BenchmarkDiskANNUint8_10K(b *testing.B)           { benchmarkDiskANN(b, 10000, "uint8") }
func BenchmarkDiskANNNoQuantization_100K(b *testing.B) { benchmarkDiskANN(b, 100000, "none") }
func BenchmarkDiskANNFloat16_100K(b *testing.B)        { benchmarkDiskANN(b, 100000, "float16") }
func BenchmarkDiskANNUint8_100K(b *testing.B)          { benchmarkDiskANN(b, 100000, "uint8") }

func benchmarkDiskANN(b *testing.B, vectorCount int, quantType string) {
	dim := 384
	ctx := context.Background()

	indexPath := fmt.Sprintf("/tmp/diskann_bench_%s_%d.idx", quantType, vectorCount)
	defer func() {
		// Cleanup
		// os.Remove(indexPath) // Keep commented for now to avoid file issues
	}()

	config := map[string]interface{}{
		"memory_limit": vectorCount / 10, // 10% in memory
		"max_degree":   32,
		"ef_search":    50,
		"index_path":   indexPath,
	}

	if quantType != "none" {
		config["quantization"] = map[string]interface{}{
			"type": quantType,
		}
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		b.Fatal(err)
	}

	// Train uint8 quantizer if needed
	if quantType == "uint8" {
		diskann := idx.(*DiskANNIndex)
		trainingData := make([]float32, 1000*dim)
		for i := range trainingData {
			trainingData[i] = rand.Float32()
		}
		uint8Quantizer := diskann.quantizer.(*Uint8Quantizer)
		if err := uint8Quantizer.Train(trainingData); err != nil {
			b.Fatal(err)
		}
	}

	// Add vectors
	vectors := generateRandomVectors(vectorCount, dim)
	for i, vec := range vectors {
		if err := idx.Add(ctx, uint64(i), vec); err != nil {
			b.Fatal(err)
		}
	}

	// Get memory and disk stats
	stats := idx.Stats()
	memoryMB := float64(stats.MemoryUsed) / (1024 * 1024)
	diskMB := float64(stats.DiskUsed) / (1024 * 1024)
	totalMB := memoryMB + diskMB

	b.ReportMetric(memoryMB, "memory_MB")
	b.ReportMetric(diskMB, "disk_MB")
	b.ReportMetric(totalMB, "total_MB")
	b.ReportMetric(float64(stats.MemoryUsed)/float64(vectorCount), "bytes/vector")

	// Benchmark search
	query := vectors[0]

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := idx.Search(ctx, query, 10, nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Benchmark quantization/dequantization overhead

func BenchmarkFloat16Quantize(b *testing.B) {
	dim := 384
	quantizer := NewFloat16Quantizer(dim)
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := quantizer.Quantize(vector)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFloat16Dequantize(b *testing.B) {
	dim := 384
	quantizer := NewFloat16Quantizer(dim)
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()
	}

	quantized, _ := quantizer.Quantize(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := quantizer.Dequantize(quantized)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUint8Quantize(b *testing.B) {
	dim := 384
	quantizer := NewUint8Quantizer(dim)

	// Train
	trainingData := make([]float32, 1000*dim)
	for i := range trainingData {
		trainingData[i] = rand.Float32()
	}
	quantizer.Train(trainingData)

	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := quantizer.Quantize(vector)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUint8Dequantize(b *testing.B) {
	dim := 384
	quantizer := NewUint8Quantizer(dim)

	// Train
	trainingData := make([]float32, 1000*dim)
	for i := range trainingData {
		trainingData[i] = rand.Float32()
	}
	quantizer.Train(trainingData)

	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()
	}

	quantized, _ := quantizer.Quantize(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := quantizer.Dequantize(quantized)
		if err != nil {
			b.Fatal(err)
		}
	}
}
