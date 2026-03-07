package main

import (
	"math"
	"testing"
)

func TestNewFloat32Vector(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	vec := NewFloat32Vector(data)

	if vec.Type != VectorTypeFloat32 {
		t.Errorf("expected type Float32, got %v", vec.Type)
	}

	if vec.Dimension != 4 {
		t.Errorf("expected dimension 4, got %d", vec.Dimension)
	}

	if len(vec.F32Data) != 4 {
		t.Errorf("expected 4 elements, got %d", len(vec.F32Data))
	}
}

func TestNewFloat16Vector(t *testing.T) {
	data := []float32{1.5, -2.5, 0.0, 3.25}
	vec := NewFloat16Vector(data)

	if vec.Type != VectorTypeFloat16 {
		t.Errorf("expected type Float16, got %v", vec.Type)
	}

	// Convert back and check approximate equality
	f32Data := vec.ToFloat32()
	for i, v := range data {
		diff := math.Abs(float64(v - f32Data[i]))
		if diff > 0.01 { // Float16 has reduced precision
			t.Errorf("conversion error at index %d: expected ~%f, got %f", i, v, f32Data[i])
		}
	}
}

func TestNewUint8Vector(t *testing.T) {
	data := []float32{0.0, 0.5, 1.0}
	vec := NewUint8Vector(data, 0.0, 1.0)

	if vec.Type != VectorTypeUint8 {
		t.Errorf("expected type Uint8, got %v", vec.Type)
	}

	expected := []uint8{0, 127, 255}
	for i, e := range expected {
		// Allow small tolerance for rounding
		if math.Abs(float64(vec.U8Data[i]-e)) > 1 {
			t.Errorf("quantization error at index %d: expected ~%d, got %d", i, e, vec.U8Data[i])
		}
	}
}

func TestNewSparseCoOVector(t *testing.T) {
	indices := []uint32{0, 5, 10}
	values := []float32{1.0, 2.0, 3.0}
	dim := 100

	vec, err := NewSparseCoOVector(indices, values, dim)
	if err != nil {
		t.Fatalf("failed to create sparse vector: %v", err)
	}

	if vec.Type != VectorTypeSparseCoO {
		t.Errorf("expected type SparseCoO, got %v", vec.Type)
	}

	if vec.Dimension != dim {
		t.Errorf("expected dimension %d, got %d", dim, vec.Dimension)
	}

	// Check sparsity
	sparsity := vec.Sparsity()
	expectedSparsity := 1.0 - (3.0 / 100.0)
	if math.Abs(sparsity-expectedSparsity) > 0.001 {
		t.Errorf("expected sparsity ~%f, got %f", expectedSparsity, sparsity)
	}
}

func TestNewSparseCoOVectorErrors(t *testing.T) {
	// Mismatched lengths
	_, err := NewSparseCoOVector([]uint32{0, 1}, []float32{1.0}, 10)
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}

	// Index out of bounds
	_, err = NewSparseCoOVector([]uint32{0, 100}, []float32{1.0, 2.0}, 10)
	if err == nil {
		t.Error("expected error for index out of bounds")
	}
}

func TestVectorDataToFloat32(t *testing.T) {
	// Test Float32 -> Float32
	f32Vec := NewFloat32Vector([]float32{1.0, 2.0, 3.0})
	result := f32Vec.ToFloat32()
	if len(result) != 3 || result[0] != 1.0 || result[1] != 2.0 || result[2] != 3.0 {
		t.Error("Float32 conversion failed")
	}

	// Test Sparse -> Dense
	sparseVec, _ := NewSparseCoOVector([]uint32{0, 2, 4}, []float32{1.0, 2.0, 3.0}, 5)
	densified := sparseVec.ToFloat32()
	expected := []float32{1.0, 0.0, 2.0, 0.0, 3.0}
	for i, e := range expected {
		if densified[i] != e {
			t.Errorf("sparse to dense conversion error at index %d: expected %f, got %f", i, e, densified[i])
		}
	}
}

func TestMemoryBytes(t *testing.T) {
	// Float32: 4 bytes per element
	f32Vec := NewFloat32Vector([]float32{1.0, 2.0, 3.0, 4.0})
	if f32Vec.MemoryBytes() != 16 {
		t.Errorf("expected 16 bytes, got %d", f32Vec.MemoryBytes())
	}

	// Float16: 2 bytes per element
	f16Vec := NewFloat16Vector([]float32{1.0, 2.0, 3.0, 4.0})
	if f16Vec.MemoryBytes() != 8 {
		t.Errorf("expected 8 bytes, got %d", f16Vec.MemoryBytes())
	}

	// Uint8: 1 byte per element
	u8Vec := NewUint8Vector([]float32{0.0, 0.5, 1.0}, 0.0, 1.0)
	if u8Vec.MemoryBytes() != 3 {
		t.Errorf("expected 3 bytes, got %d", u8Vec.MemoryBytes())
	}

	// Sparse: 4 bytes per index + 4 bytes per value
	sparseVec, _ := NewSparseCoOVector([]uint32{0, 5, 10}, []float32{1.0, 2.0, 3.0}, 100)
	expectedMem := 3*4 + 3*4 // 24 bytes
	if sparseVec.MemoryBytes() != expectedMem {
		t.Errorf("expected %d bytes, got %d", expectedMem, sparseVec.MemoryBytes())
	}
}

func TestSparseCoOToDense(t *testing.T) {
	sparse := &SparseCoO{
		Indices: []uint32{0, 2, 4},
		Values:  []float32{1.0, 2.0, 3.0},
		Dim:     5,
	}

	dense := sparse.ToDense()
	expected := []float32{1.0, 0.0, 2.0, 0.0, 3.0}

	for i, e := range expected {
		if dense[i] != e {
			t.Errorf("index %d: expected %f, got %f", i, e, dense[i])
		}
	}
}

func TestSparseCoONonZeroCount(t *testing.T) {
	sparse := &SparseCoO{
		Indices: []uint32{0, 2, 4},
		Values:  []float32{1.0, 2.0, 3.0},
		Dim:     100,
	}

	if sparse.NonZeroCount() != 3 {
		t.Errorf("expected 3 non-zero elements, got %d", sparse.NonZeroCount())
	}
}

func TestFloat16Conversion(t *testing.T) {
	tests := []float32{0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 100.0, -100.0}

	for _, v := range tests {
		f16 := float32ToFloat16(v)
		f32 := float16ToFloat32(f16)

		// Float16 has reduced precision, allow 1% error
		diff := math.Abs(float64(v - f32))
		maxError := math.Max(0.01, math.Abs(float64(v))*0.01)

		if diff > maxError {
			t.Errorf("float16 conversion error: %f -> %d -> %f (diff: %f)", v, f16, f32, diff)
		}
	}
}

func TestVectorDataMarshalBinary(t *testing.T) {
	// Test Float32
	f32Vec := NewFloat32Vector([]float32{1.0, 2.0, 3.0})
	data, err := f32Vec.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded VectorData
	err = decoded.UnmarshalBinary(data)
	if err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.Type != VectorTypeFloat32 {
		t.Errorf("expected type Float32, got %v", decoded.Type)
	}

	if len(decoded.F32Data) != 3 {
		t.Errorf("expected 3 elements, got %d", len(decoded.F32Data))
	}

	for i, v := range f32Vec.F32Data {
		if decoded.F32Data[i] != v {
			t.Errorf("data mismatch at index %d: expected %f, got %f", i, v, decoded.F32Data[i])
		}
	}
}

func TestSparseMarshalBinary(t *testing.T) {
	// Test Sparse CoO
	sparseVec, _ := NewSparseCoOVector([]uint32{0, 5, 10}, []float32{1.0, 2.0, 3.0}, 100)
	data, err := sparseVec.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded VectorData
	err = decoded.UnmarshalBinary(data)
	if err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.Type != VectorTypeSparseCoO {
		t.Errorf("expected type SparseCoO, got %v", decoded.Type)
	}

	if decoded.Dimension != 100 {
		t.Errorf("expected dimension 100, got %d", decoded.Dimension)
	}

	if len(decoded.SparseCoO.Indices) != 3 {
		t.Errorf("expected 3 indices, got %d", len(decoded.SparseCoO.Indices))
	}

	for i := range sparseVec.SparseCoO.Indices {
		if decoded.SparseCoO.Indices[i] != sparseVec.SparseCoO.Indices[i] {
			t.Errorf("index mismatch at %d", i)
		}
		if decoded.SparseCoO.Values[i] != sparseVec.SparseCoO.Values[i] {
			t.Errorf("value mismatch at %d", i)
		}
	}
}

func TestVectorTypeString(t *testing.T) {
	tests := []struct {
		vt       VectorType
		expected string
	}{
		{VectorTypeFloat32, "float32"},
		{VectorTypeFloat16, "float16"},
		{VectorTypeUint8, "uint8"},
		{VectorTypeSparseCoO, "sparse_coo"},
		{VectorTypeSparseCSR, "sparse_csr"},
	}

	for _, tt := range tests {
		if tt.vt.String() != tt.expected {
			t.Errorf("VectorType.String(): expected %s, got %s", tt.expected, tt.vt.String())
		}
	}
}

func BenchmarkFloat32ToFloat16(b *testing.B) {
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, v := range data {
			_ = float32ToFloat16(v)
		}
	}
}

func BenchmarkSparseToDense(b *testing.B) {
	indices := make([]uint32, 100)
	values := make([]float32, 100)
	for i := range indices {
		indices[i] = uint32(i * 10)
		values[i] = float32(i)
	}

	sparse := &SparseCoO{
		Indices: indices,
		Values:  values,
		Dim:     1000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sparse.ToDense()
	}
}

func BenchmarkDenseToSparse(b *testing.B) {
	dense := make([]float32, 1000)
	for i := 0; i < 100; i++ {
		dense[i*10] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DenseToSparseCoO(dense, 1e-6)
	}
}
