package index

import (
	"math"
	"testing"
)

func TestFloat16Quantization(t *testing.T) {
	dim := 128
	numVectors := 100

	// Create test vectors
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = float32(i%256) / 255.0 // Range [0, 1]
	}

	// Create quantizer
	quantizer := NewFloat16Quantizer(dim)

	// Quantize
	data, err := quantizer.Quantize(vectors)
	if err != nil {
		t.Fatalf("quantize failed: %v", err)
	}

	// Check size
	expectedSize := numVectors * dim * 2 // 2 bytes per float16
	if len(data) != expectedSize {
		t.Fatalf("expected %d bytes, got %d", expectedSize, len(data))
	}

	// Dequantize
	recovered, err := quantizer.Dequantize(data)
	if err != nil {
		t.Fatalf("dequantize failed: %v", err)
	}

	// Check length
	if len(recovered) != len(vectors) {
		t.Fatalf("expected %d vectors, got %d", len(vectors), len(recovered))
	}

	// Check accuracy (float16 has limited precision)
	maxError := float32(0.0)
	for i := range vectors {
		error := float32(math.Abs(float64(vectors[i] - recovered[i])))
		if error > maxError {
			maxError = error
		}
	}

	// Float16 precision is ~3 decimal digits
	if maxError > 0.01 {
		t.Fatalf("max error too high: %f", maxError)
	}

	t.Logf("Float16 max error: %f", maxError)
	t.Logf("Compression ratio: %.2fx", float64(len(vectors)*4)/float64(len(data)))
}

func TestUint8Quantization(t *testing.T) {
	dim := 128
	numVectors := 100

	// Create test vectors with known range
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = float32(i%256)/255.0*2.0 - 1.0 // Range [-1, 1]
	}

	// Create and train quantizer
	quantizer := NewUint8Quantizer(dim)

	err := quantizer.Train(vectors)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	// Quantize
	data, err := quantizer.Quantize(vectors)
	if err != nil {
		t.Fatalf("quantize failed: %v", err)
	}

	// Check size
	expectedSize := numVectors * dim // 1 byte per value
	if len(data) != expectedSize {
		t.Fatalf("expected %d bytes, got %d", expectedSize, len(data))
	}

	// Dequantize
	recovered, err := quantizer.Dequantize(data)
	if err != nil {
		t.Fatalf("dequantize failed: %v", err)
	}

	// Check length
	if len(recovered) != len(vectors) {
		t.Fatalf("expected %d vectors, got %d", len(vectors), len(recovered))
	}

	// Check accuracy
	maxError := float32(0.0)
	for i := range vectors {
		error := float32(math.Abs(float64(vectors[i] - recovered[i])))
		if error > maxError {
			maxError = error
		}
	}

	// Uint8 quantization should have ~1/255 precision
	if maxError > 0.02 {
		t.Fatalf("max error too high: %f", maxError)
	}

	t.Logf("Uint8 max error: %f", maxError)
	t.Logf("Compression ratio: %.2fx", float64(len(vectors)*4)/float64(len(data)))
}

func TestProductQuantization(t *testing.T) {
	dim := 128
	m := 8 // 8 subvectors
	ksub := 256
	numVectors := 1000
	if testing.Short() {
		numVectors = 300 // Minimum for k-means with ksub=256
	}

	// Create test vectors
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = float32(i%100) / 100.0
	}

	// Create quantizer
	quantizer, err := NewProductQuantizer(dim, m, ksub)
	if err != nil {
		t.Fatalf("failed to create quantizer: %v", err)
	}

	// Train
	err = quantizer.Train(vectors, 10)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	// Quantize
	codes, err := quantizer.Quantize(vectors)
	if err != nil {
		t.Fatalf("quantize failed: %v", err)
	}

	// Check size
	expectedSize := numVectors * m // 1 byte per subvector
	if len(codes) != expectedSize {
		t.Fatalf("expected %d bytes, got %d", expectedSize, len(codes))
	}

	// Dequantize
	recovered, err := quantizer.Dequantize(codes)
	if err != nil {
		t.Fatalf("dequantize failed: %v", err)
	}

	// Check length
	if len(recovered) != len(vectors) {
		t.Fatalf("expected %d values, got %d", len(vectors), len(recovered))
	}

	// Check accuracy (PQ is lossy)
	maxError := float32(0.0)
	avgError := float64(0.0)
	for i := range vectors {
		error := float32(math.Abs(float64(vectors[i] - recovered[i])))
		if error > maxError {
			maxError = error
		}
		avgError += float64(error)
	}
	avgError /= float64(len(vectors))

	t.Logf("Product Quantization max error: %f", maxError)
	t.Logf("Product Quantization avg error: %f", avgError)
	t.Logf("Compression ratio: %.2fx", float64(len(vectors)*4)/float64(len(codes)))

	// PQ is quite lossy, but should still be reasonable
	if avgError > 0.1 {
		t.Fatalf("average error too high: %f", avgError)
	}
}

func TestQuantizationInfo(t *testing.T) {
	dim := 384
	numVectors := 10000

	tests := []struct {
		name                string
		quantizer           Quantizer
		minCompressionRatio float64
	}{
		{
			name:                "Float16",
			quantizer:           NewFloat16Quantizer(dim),
			minCompressionRatio: 1.9, // ~2x
		},
		{
			name:                "Uint8",
			quantizer:           NewUint8Quantizer(dim),
			minCompressionRatio: 3.9, // ~4x
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := GetQuantizationInfo(tt.quantizer, numVectors)

			t.Logf("%s Info:", tt.name)
			t.Logf("  Original size: %d bytes (%.2f MB)", info.OriginalSize, float64(info.OriginalSize)/1024/1024)
			t.Logf("  Compressed size: %d bytes (%.2f MB)", info.CompressedSize, float64(info.CompressedSize)/1024/1024)
			t.Logf("  Compression ratio: %.2fx", info.CompressionRatio)
			t.Logf("  Memory savings: %.1f%%", (1.0-1.0/info.CompressionRatio)*100)

			if info.CompressionRatio < tt.minCompressionRatio {
				t.Fatalf("compression ratio too low: got %.2f, want >= %.2f", info.CompressionRatio, tt.minCompressionRatio)
			}
		})
	}
}

func TestFloat16EdgeCases(t *testing.T) {
	dim := 10
	quantizer := NewFloat16Quantizer(dim)

	// Test special values
	specialValues := []float32{
		0.0,
		-0.0,
		1.0,
		-1.0,
		0.5,
		-0.5,
		float32(math.MaxFloat32),
		float32(-math.MaxFloat32),
		float32(math.SmallestNonzeroFloat32),
	}

	vectors := make([]float32, dim)
	for i := 0; i < dim && i < len(specialValues); i++ {
		vectors[i] = specialValues[i]
	}

	// Quantize and dequantize
	data, err := quantizer.Quantize(vectors)
	if err != nil {
		t.Fatalf("quantize failed: %v", err)
	}

	recovered, err := quantizer.Dequantize(data)
	if err != nil {
		t.Fatalf("dequantize failed: %v", err)
	}

	// Check finite values
	for i := 0; i < len(specialValues) && i < dim; i++ {
		orig := specialValues[i]
		rec := recovered[i]

		// Skip inf/nan checks as float16 handles them differently
		if math.IsInf(float64(orig), 0) || math.IsNaN(float64(orig)) {
			continue
		}

		if math.IsNaN(float64(rec)) {
			t.Errorf("value %d became NaN: orig=%f", i, orig)
		}
	}
}

func TestUint8Codebook(t *testing.T) {
	dim := 128
	quantizer := NewUint8Quantizer(dim)

	// Create training vectors
	vectors := make([]float32, 1000*dim)
	for i := range vectors {
		vectors[i] = float32(i%256) / 128.0 // Range [0, 2]
	}

	// Train
	err := quantizer.Train(vectors)
	if err != nil {
		t.Fatalf("train failed: %v", err)
	}

	// Get codebook
	min, max := quantizer.GetCodebook()

	if len(min) != dim {
		t.Fatalf("min codebook size wrong: got %d, want %d", len(min), dim)
	}
	if len(max) != dim {
		t.Fatalf("max codebook size wrong: got %d, want %d", len(max), dim)
	}

	// Check min < max for all dimensions
	for i := 0; i < dim; i++ {
		if min[i] > max[i] {
			t.Errorf("dimension %d: min > max (%f > %f)", i, min[i], max[i])
		}
	}

	// Create new quantizer and set codebook
	newQuantizer := NewUint8Quantizer(dim)
	err = newQuantizer.SetCodebook(min, max)
	if err != nil {
		t.Fatalf("set codebook failed: %v", err)
	}

	// Should be able to quantize now
	testVec := vectors[:dim]
	_, err = newQuantizer.Quantize(testVec)
	if err != nil {
		t.Fatalf("quantize with loaded codebook failed: %v", err)
	}
}

func BenchmarkFloat16Quantization(b *testing.B) {
	dim := 384
	numVectors := 1000
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = float32(i) / float32(len(vectors))
	}

	quantizer := NewFloat16Quantizer(dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = quantizer.Quantize(vectors)
	}
}

func BenchmarkUint8Quantization(b *testing.B) {
	dim := 384
	numVectors := 1000
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = float32(i) / float32(len(vectors))
	}

	quantizer := NewUint8Quantizer(dim)
	_ = quantizer.Train(vectors)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = quantizer.Quantize(vectors)
	}
}

func BenchmarkProductQuantization(b *testing.B) {
	dim := 128
	m := 8
	ksub := 256
	numVectors := 1000

	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = float32(i) / float32(len(vectors))
	}

	quantizer, _ := NewProductQuantizer(dim, m, ksub)
	_ = quantizer.Train(vectors, 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = quantizer.Quantize(vectors)
	}
}
