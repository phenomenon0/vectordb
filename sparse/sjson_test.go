package sparse

import (
	"encoding/json"
	"testing"
)

func TestSparseVector_MarshalJSON(t *testing.T) {
	// Create sparse vector: [1.0, 0, 0, 2.0, 0, 3.0]
	sv, err := NewSparseVector([]uint32{0, 3, 5}, []float32{1.0, 2.0, 3.0}, 10)
	if err != nil {
		t.Fatalf("failed to create sparse vector: %v", err)
	}

	// Marshal to JSON
	data, err := json.Marshal(sv)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	// Verify JSON structure
	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	// Check fields exist
	if _, ok := result["indices"]; !ok {
		t.Error("missing 'indices' field")
	}
	if _, ok := result["values"]; !ok {
		t.Error("missing 'values' field")
	}
	if _, ok := result["dim"]; !ok {
		t.Error("missing 'dim' field")
	}
}

func TestSparseVector_UnmarshalJSON(t *testing.T) {
	jsonData := `{
		"indices": [0, 3, 5],
		"values": [1.0, 2.0, 3.0],
		"dim": 10
	}`

	var sv SparseVector
	if err := json.Unmarshal([]byte(jsonData), &sv); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	// Verify fields
	if sv.Dim != 10 {
		t.Errorf("dim mismatch: got %d, want 10", sv.Dim)
	}

	expectedIndices := []uint32{0, 3, 5}
	for i, idx := range sv.Indices {
		if idx != expectedIndices[i] {
			t.Errorf("indices[%d] mismatch: got %d, want %d", i, idx, expectedIndices[i])
		}
	}

	expectedValues := []float32{1.0, 2.0, 3.0}
	for i, val := range sv.Values {
		if val != expectedValues[i] {
			t.Errorf("values[%d] mismatch: got %f, want %f", i, val, expectedValues[i])
		}
	}
}

func TestSparseVector_RoundTrip(t *testing.T) {
	// Original vector
	original, err := NewSparseVector([]uint32{0, 5, 10, 15}, []float32{1.0, 2.0, 3.0, 4.0}, 100)
	if err != nil {
		t.Fatalf("failed to create vector: %v", err)
	}

	// Marshal
	data, err := original.MarshalJSON()
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	// Unmarshal
	var decoded SparseVector
	if err := decoded.UnmarshalJSON(data); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	// Compare
	if decoded.Dim != original.Dim {
		t.Errorf("dim mismatch after round-trip: got %d, want %d", decoded.Dim, original.Dim)
	}

	if len(decoded.Indices) != len(original.Indices) {
		t.Errorf("indices length mismatch: got %d, want %d", len(decoded.Indices), len(original.Indices))
	}

	for i := range original.Indices {
		if decoded.Indices[i] != original.Indices[i] {
			t.Errorf("indices[%d] mismatch: got %d, want %d", i, decoded.Indices[i], original.Indices[i])
		}
		if decoded.Values[i] != original.Values[i] {
			t.Errorf("values[%d] mismatch: got %f, want %f", i, decoded.Values[i], original.Values[i])
		}
	}
}

func TestSparseVector_ToBytes(t *testing.T) {
	sv, _ := NewSparseVector([]uint32{0, 5}, []float32{1.0, 2.0}, 10)

	data, err := sv.ToBytes()
	if err != nil {
		t.Fatalf("ToBytes failed: %v", err)
	}

	if len(data) == 0 {
		t.Error("ToBytes returned empty data")
	}
}

func TestFromBytes(t *testing.T) {
	original, _ := NewSparseVector([]uint32{0, 5}, []float32{1.0, 2.0}, 10)

	// Serialize
	data, err := original.ToBytes()
	if err != nil {
		t.Fatalf("ToBytes failed: %v", err)
	}

	// Deserialize
	decoded, err := FromBytes(data)
	if err != nil {
		t.Fatalf("FromBytes failed: %v", err)
	}

	// Compare
	if decoded.Dim != original.Dim {
		t.Errorf("dim mismatch: got %d, want %d", decoded.Dim, original.Dim)
	}

	if len(decoded.Indices) != len(original.Indices) {
		t.Fatalf("indices length mismatch")
	}

	for i := range original.Indices {
		if decoded.Indices[i] != original.Indices[i] || decoded.Values[i] != original.Values[i] {
			t.Error("data mismatch after FromBytes")
		}
	}
}

func TestCompressionRatio(t *testing.T) {
	tests := []struct {
		name            string
		indices         []uint32
		values          []float32
		dim             int
		expectedMaxRatio float64 // Max expected ratio
	}{
		{
			name:            "3% sparsity (300 nnz in 10K)",
			indices:         make([]uint32, 300),
			values:          make([]float32, 300),
			dim:             10000,
			expectedMaxRatio: 0.07, // ~6% actual
		},
		{
			name:            "50% sparsity",
			indices:         make([]uint32, 50),
			values:          make([]float32, 50),
			dim:             100,
			expectedMaxRatio: 1.1, // ~103% actual (sparse not beneficial at 50%)
		},
		{
			name:            "1% sparsity",
			indices:         make([]uint32, 10),
			values:          make([]float32, 10),
			dim:             1000,
			expectedMaxRatio: 0.03, // ~2.3% actual
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Fill indices and values
			for i := range tt.indices {
				tt.indices[i] = uint32(i * (tt.dim / len(tt.indices)))
				tt.values[i] = float32(i + 1)
			}

			sv, err := NewSparseVector(tt.indices, tt.values, tt.dim)
			if err != nil {
				t.Fatalf("failed to create vector: %v", err)
			}

			ratio := sv.CompressionRatio()

			if ratio > tt.expectedMaxRatio {
				t.Errorf("compression ratio too high: got %f, want <= %f", ratio, tt.expectedMaxRatio)
			}

			if ratio <= 0 {
				t.Error("compression ratio should be positive")
			}

			t.Logf("Compression: sparse=%d bytes, dense=%d bytes, ratio=%.2f%%, sparsity=%.1f%%",
				len(tt.indices)*8+12,
				tt.dim*4,
				ratio*100,
				sv.Sparsity()*100)
		})
	}
}

func TestEncodingStats(t *testing.T) {
	// 300 non-zeros in 10K dimension (typical BM25 scenario)
	indices := make([]uint32, 300)
	values := make([]float32, 300)
	for i := range indices {
		indices[i] = uint32(i * 33) // Spread out
		values[i] = float32(i + 1)
	}

	sv, _ := NewSparseVector(indices, values, 10000)
	stats := sv.EncodingStats()

	// Verify stats
	if stats.SparseBytes <= 0 {
		t.Error("sparse bytes should be positive")
	}

	if stats.DenseBytes != 40000 { // 10K * 4 bytes
		t.Errorf("dense bytes wrong: got %d, want 40000", stats.DenseBytes)
	}

	if stats.CompressionRatio >= 1.0 {
		t.Error("compression ratio should be < 1.0 (sparse should be smaller)")
	}

	if stats.Sparsity < 0.95 || stats.Sparsity > 1.0 {
		t.Errorf("sparsity out of range: got %f", stats.Sparsity)
	}

	if stats.SavingsPercent <= 0 {
		t.Error("savings percent should be positive")
	}

	t.Logf("Encoding Stats: sparse=%d bytes, dense=%d bytes, ratio=%.2f%%, savings=%.1f%%",
		stats.SparseBytes,
		stats.DenseBytes,
		stats.CompressionRatio*100,
		stats.SavingsPercent)
}

func TestMarshalJSON_EmptyVector(t *testing.T) {
	sv, _ := NewSparseVector([]uint32{}, []float32{}, 10)

	data, err := json.Marshal(sv)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded SparseVector
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if len(decoded.Indices) != 0 || len(decoded.Values) != 0 {
		t.Error("empty vector should have no indices or values")
	}
}

func BenchmarkMarshalJSON(b *testing.B) {
	// Typical BM25 vector: 300 non-zeros in 10K dimension
	indices := make([]uint32, 300)
	values := make([]float32, 300)
	for i := range indices {
		indices[i] = uint32(i * 33)
		values[i] = float32(i + 1)
	}

	sv, _ := NewSparseVector(indices, values, 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = json.Marshal(sv)
	}
}

func BenchmarkUnmarshalJSON(b *testing.B) {
	indices := make([]uint32, 300)
	values := make([]float32, 300)
	for i := range indices {
		indices[i] = uint32(i * 33)
		values[i] = float32(i + 1)
	}

	sv, _ := NewSparseVector(indices, values, 10000)
	data, _ := json.Marshal(sv)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var decoded SparseVector
		_ = json.Unmarshal(data, &decoded)
	}
}

func BenchmarkRoundTrip(b *testing.B) {
	indices := make([]uint32, 300)
	values := make([]float32, 300)
	for i := range indices {
		indices[i] = uint32(i * 33)
		values[i] = float32(i + 1)
	}

	sv, _ := NewSparseVector(indices, values, 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, _ := json.Marshal(sv)
		var decoded SparseVector
		_ = json.Unmarshal(data, &decoded)
	}
}

// Benchmark comparing sparse vs dense JSON encoding size
func BenchmarkEncodingComparison(b *testing.B) {
	indices := make([]uint32, 300)
	values := make([]float32, 300)
	for i := range indices {
		indices[i] = uint32(i * 33)
		values[i] = float32(i + 1)
	}

	sv, _ := NewSparseVector(indices, values, 10000)

	b.Run("Sparse", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(sv)
		}
	})

	b.Run("Dense", func(b *testing.B) {
		dense := sv.ToDense()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(dense)
		}
	})
}
