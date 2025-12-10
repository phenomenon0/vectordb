package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

// VectorType represents the encoding type of a vector
type VectorType int

const (
	// VectorTypeFloat32 is the default dense float32 encoding (4 bytes/dim)
	VectorTypeFloat32 VectorType = iota
	// VectorTypeFloat16 is dense float16 encoding (2 bytes/dim, 50% compression)
	VectorTypeFloat16
	// VectorTypeUint8 is scalar quantization encoding (1 byte/dim, 75% compression)
	VectorTypeUint8
	// VectorTypeSparseCoO is sparse Coordinate format (indices + values)
	VectorTypeSparseCoO
	// VectorTypeSparseCSR is sparse Compressed Sparse Row format
	VectorTypeSparseCSR
)

// String returns the string representation of VectorType
func (vt VectorType) String() string {
	switch vt {
	case VectorTypeFloat32:
		return "float32"
	case VectorTypeFloat16:
		return "float16"
	case VectorTypeUint8:
		return "uint8"
	case VectorTypeSparseCoO:
		return "sparse_coo"
	case VectorTypeSparseCSR:
		return "sparse_csr"
	default:
		return "unknown"
	}
}

// VectorData is a unified structure that can hold any vector type
type VectorData struct {
	Type      VectorType  // Type of encoding
	Dimension int         // Vector dimension
	F32Data   []float32   // Dense float32 data
	F16Data   []uint16    // Dense float16 data (as uint16 bits)
	U8Data    []uint8     // Dense uint8 quantized data
	SparseCoO *SparseCoO  // Sparse CoO format
	SparseCSR *SparseCSR  // Sparse CSR format
}

// SparseCoO represents a sparse vector in Coordinate format
// Stores only non-zero elements as (index, value) pairs
type SparseCoO struct {
	Indices []uint32  // Indices of non-zero elements (sorted)
	Values  []float32 // Values at those indices
	Dim     int       // Total dimension of the vector
}

// SparseCSR represents a sparse vector in Compressed Sparse Row format
// More efficient for operations but slightly more complex
type SparseCSR struct {
	IndPtr  []uint32  // Index pointer array (length 2 for single vector)
	Indices []uint32  // Column indices of non-zero elements
	Values  []float32 // Values at those indices
	Dim     int       // Total dimension of the vector
}

// NewFloat32Vector creates a VectorData from a float32 slice
func NewFloat32Vector(data []float32) *VectorData {
	return &VectorData{
		Type:      VectorTypeFloat32,
		Dimension: len(data),
		F32Data:   data,
	}
}

// NewFloat16Vector creates a VectorData from float32 slice, converting to float16
func NewFloat16Vector(data []float32) *VectorData {
	f16Data := make([]uint16, len(data))
	for i, v := range data {
		f16Data[i] = float32ToFloat16(v)
	}
	return &VectorData{
		Type:      VectorTypeFloat16,
		Dimension: len(data),
		F16Data:   f16Data,
	}
}

// NewUint8Vector creates a VectorData from float32 slice, quantizing to uint8
// Assumes input is normalized to [-1, 1] or [0, 1] range
func NewUint8Vector(data []float32, min, max float32) *VectorData {
	u8Data := make([]uint8, len(data))
	scale := 255.0 / (max - min)
	for i, v := range data {
		normalized := (v - min) * scale
		u8Data[i] = uint8(math.Max(0, math.Min(255, normalized)))
	}
	return &VectorData{
		Type:      VectorTypeUint8,
		Dimension: len(data),
		U8Data:    u8Data,
	}
}

// NewSparseCoOVector creates a sparse vector in CoO format
func NewSparseCoOVector(indices []uint32, values []float32, dim int) (*VectorData, error) {
	if len(indices) != len(values) {
		return nil, fmt.Errorf("indices and values length mismatch: %d vs %d", len(indices), len(values))
	}

	// Validate indices are within dimension
	for i, idx := range indices {
		if int(idx) >= dim {
			return nil, fmt.Errorf("index %d at position %d exceeds dimension %d", idx, i, dim)
		}
	}

	return &VectorData{
		Type:      VectorTypeSparseCoO,
		Dimension: dim,
		SparseCoO: &SparseCoO{
			Indices: indices,
			Values:  values,
			Dim:     dim,
		},
	}, nil
}

// ToFloat32 converts any VectorData type to dense float32 representation
func (vd *VectorData) ToFloat32() []float32 {
	switch vd.Type {
	case VectorTypeFloat32:
		return vd.F32Data

	case VectorTypeFloat16:
		result := make([]float32, len(vd.F16Data))
		for i, f16 := range vd.F16Data {
			result[i] = float16ToFloat32(f16)
		}
		return result

	case VectorTypeUint8:
		// Simple dequantization: map [0, 255] to [0, 1]
		result := make([]float32, len(vd.U8Data))
		for i, u8 := range vd.U8Data {
			result[i] = float32(u8) / 255.0
		}
		return result

	case VectorTypeSparseCoO:
		return vd.SparseCoO.ToDense()

	case VectorTypeSparseCSR:
		return vd.SparseCSR.ToDense()

	default:
		return nil
	}
}

// MemoryBytes returns the approximate memory usage in bytes
func (vd *VectorData) MemoryBytes() int {
	switch vd.Type {
	case VectorTypeFloat32:
		return len(vd.F32Data) * 4
	case VectorTypeFloat16:
		return len(vd.F16Data) * 2
	case VectorTypeUint8:
		return len(vd.U8Data)
	case VectorTypeSparseCoO:
		return len(vd.SparseCoO.Indices)*4 + len(vd.SparseCoO.Values)*4
	case VectorTypeSparseCSR:
		return len(vd.SparseCSR.IndPtr)*4 + len(vd.SparseCSR.Indices)*4 + len(vd.SparseCSR.Values)*4
	default:
		return 0
	}
}

// Sparsity returns the sparsity ratio (0 = dense, 1 = all zeros)
func (vd *VectorData) Sparsity() float64 {
	switch vd.Type {
	case VectorTypeSparseCoO:
		if vd.Dimension == 0 {
			return 0
		}
		return 1.0 - (float64(len(vd.SparseCoO.Indices)) / float64(vd.Dimension))
	case VectorTypeSparseCSR:
		if vd.Dimension == 0 {
			return 0
		}
		return 1.0 - (float64(len(vd.SparseCSR.Indices)) / float64(vd.Dimension))
	default:
		// Dense vectors have zero sparsity
		return 0
	}
}

// SparseCoO methods

// ToDense converts sparse CoO to dense float32 vector
func (sc *SparseCoO) ToDense() []float32 {
	dense := make([]float32, sc.Dim)
	for i, idx := range sc.Indices {
		if int(idx) < sc.Dim {
			dense[idx] = sc.Values[i]
		}
	}
	return dense
}

// NonZeroCount returns the number of non-zero elements
func (sc *SparseCoO) NonZeroCount() int {
	return len(sc.Indices)
}

// Validate checks if the sparse vector is valid
func (sc *SparseCoO) Validate() error {
	if len(sc.Indices) != len(sc.Values) {
		return fmt.Errorf("indices and values length mismatch: %d vs %d", len(sc.Indices), len(sc.Values))
	}

	if sc.Dim <= 0 {
		return fmt.Errorf("dimension must be positive, got %d", sc.Dim)
	}

	// Check indices are sorted and within bounds
	for i, idx := range sc.Indices {
		if int(idx) >= sc.Dim {
			return fmt.Errorf("index %d at position %d exceeds dimension %d", idx, i, sc.Dim)
		}
		if i > 0 && sc.Indices[i] <= sc.Indices[i-1] {
			return fmt.Errorf("indices must be sorted and unique, found %d after %d at position %d", idx, sc.Indices[i-1], i)
		}
	}

	return nil
}

// SparseCSR methods

// ToDense converts sparse CSR to dense float32 vector
func (sr *SparseCSR) ToDense() []float32 {
	dense := make([]float32, sr.Dim)
	// For single vector, IndPtr[0] to IndPtr[1] gives the range
	if len(sr.IndPtr) >= 2 {
		start := sr.IndPtr[0]
		end := sr.IndPtr[1]
		for i := start; i < end; i++ {
			if int(sr.Indices[i]) < sr.Dim {
				dense[sr.Indices[i]] = sr.Values[i]
			}
		}
	}
	return dense
}

// NonZeroCount returns the number of non-zero elements
func (sr *SparseCSR) NonZeroCount() int {
	if len(sr.IndPtr) >= 2 {
		return int(sr.IndPtr[1] - sr.IndPtr[0])
	}
	return 0
}

// Float16 conversion utilities

// float32ToFloat16 converts float32 to float16 (IEEE 754 half precision)
// Stored as uint16 bits
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)

	// Extract sign, exponent, mantissa
	sign := (bits >> 31) & 0x1
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF

	// Handle special cases
	if exp == 0xFF {
		// Infinity or NaN
		if mant != 0 {
			return uint16((sign << 15) | 0x7C00 | 0x200) // NaN
		}
		return uint16((sign << 15) | 0x7C00) // Infinity
	}

	if exp == 0 {
		// Zero or subnormal
		return uint16(sign << 15)
	}

	// Convert exponent (bias 127 -> bias 15)
	newExp := int(exp) - 127 + 15

	if newExp >= 31 {
		// Overflow to infinity
		return uint16((sign << 15) | 0x7C00)
	}

	if newExp <= 0 {
		// Underflow to zero
		return uint16(sign << 15)
	}

	// Convert mantissa (23 bits -> 10 bits)
	newMant := mant >> 13

	return uint16((sign << 15) | (uint16(newExp) << 10) | uint16(newMant))
}

// float16ToFloat32 converts float16 (stored as uint16) to float32
func float16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 0x1)
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	// Handle special cases
	if exp == 0x1F {
		// Infinity or NaN
		if mant != 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000 | (mant << 13))
		}
		return math.Float32frombits((sign << 31) | 0x7F800000)
	}

	if exp == 0 {
		if mant == 0 {
			// Zero
			return math.Float32frombits(sign << 31)
		}
		// Subnormal - not handling for simplicity
		return 0
	}

	// Convert exponent (bias 15 -> bias 127)
	newExp := exp - 15 + 127

	// Convert mantissa (10 bits -> 23 bits)
	newMant := mant << 13

	bits := (sign << 31) | (newExp << 23) | newMant
	return math.Float32frombits(bits)
}

// Serialization helpers for SJSON integration

// MarshalBinary encodes VectorData to binary format for storage
func (vd *VectorData) MarshalBinary() ([]byte, error) {
	// Simple binary format:
	// [1 byte: type] [4 bytes: dimension] [type-specific data]

	buf := make([]byte, 0, 5+vd.MemoryBytes())

	// Type
	buf = append(buf, byte(vd.Type))

	// Dimension (4 bytes, little-endian)
	dimBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(dimBytes, uint32(vd.Dimension))
	buf = append(buf, dimBytes...)

	// Type-specific data
	switch vd.Type {
	case VectorTypeFloat32:
		for _, v := range vd.F32Data {
			vBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(vBytes, math.Float32bits(v))
			buf = append(buf, vBytes...)
		}

	case VectorTypeFloat16:
		for _, v := range vd.F16Data {
			vBytes := make([]byte, 2)
			binary.LittleEndian.PutUint16(vBytes, v)
			buf = append(buf, vBytes...)
		}

	case VectorTypeUint8:
		buf = append(buf, vd.U8Data...)

	case VectorTypeSparseCoO:
		// [4 bytes: nnz] [indices...] [values...]
		nnz := len(vd.SparseCoO.Indices)
		nnzBytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(nnzBytes, uint32(nnz))
		buf = append(buf, nnzBytes...)

		for _, idx := range vd.SparseCoO.Indices {
			idxBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(idxBytes, idx)
			buf = append(buf, idxBytes...)
		}

		for _, val := range vd.SparseCoO.Values {
			valBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(valBytes, math.Float32bits(val))
			buf = append(buf, valBytes...)
		}
	}

	return buf, nil
}

// UnmarshalBinary decodes VectorData from binary format
func (vd *VectorData) UnmarshalBinary(data []byte) error {
	if len(data) < 5 {
		return fmt.Errorf("invalid binary data: too short")
	}

	// Read type
	vd.Type = VectorType(data[0])

	// Read dimension
	vd.Dimension = int(binary.LittleEndian.Uint32(data[1:5]))

	offset := 5

	switch vd.Type {
	case VectorTypeFloat32:
		count := vd.Dimension
		if len(data) < offset+count*4 {
			return fmt.Errorf("invalid float32 data length")
		}
		vd.F32Data = make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			vd.F32Data[i] = math.Float32frombits(bits)
			offset += 4
		}

	case VectorTypeFloat16:
		count := vd.Dimension
		if len(data) < offset+count*2 {
			return fmt.Errorf("invalid float16 data length")
		}
		vd.F16Data = make([]uint16, count)
		for i := 0; i < count; i++ {
			vd.F16Data[i] = binary.LittleEndian.Uint16(data[offset : offset+2])
			offset += 2
		}

	case VectorTypeUint8:
		count := vd.Dimension
		if len(data) < offset+count {
			return fmt.Errorf("invalid uint8 data length")
		}
		vd.U8Data = make([]uint8, count)
		copy(vd.U8Data, data[offset:offset+count])

	case VectorTypeSparseCoO:
		if len(data) < offset+4 {
			return fmt.Errorf("invalid sparse CoO data")
		}
		nnz := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		if len(data) < offset+nnz*8 {
			return fmt.Errorf("invalid sparse CoO data length")
		}

		indices := make([]uint32, nnz)
		for i := 0; i < nnz; i++ {
			indices[i] = binary.LittleEndian.Uint32(data[offset : offset+4])
			offset += 4
		}

		values := make([]float32, nnz)
		for i := 0; i < nnz; i++ {
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			values[i] = math.Float32frombits(bits)
			offset += 4
		}

		vd.SparseCoO = &SparseCoO{
			Indices: indices,
			Values:  values,
			Dim:     vd.Dimension,
		}
	}

	return nil
}
