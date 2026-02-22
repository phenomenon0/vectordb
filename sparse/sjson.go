package sparse

import (
	"encoding/json"
	"fmt"
)

// SparseVectorJSON represents the JSON encoding format for sparse vectors.
// Optimized for Cowrie with separate arrays for indices and values.
type SparseVectorJSON struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
	Dim     int       `json:"dim"`
}

// MarshalJSON implements json.Marshaler for efficient Cowrie encoding.
func (sv *SparseVector) MarshalJSON() ([]byte, error) {
	return json.Marshal(SparseVectorJSON{
		Indices: sv.Indices,
		Values:  sv.Values,
		Dim:     sv.Dim,
	})
}

// UnmarshalJSON implements json.Unmarshaler for Cowrie decoding.
func (sv *SparseVector) UnmarshalJSON(data []byte) error {
	var svj SparseVectorJSON
	if err := json.Unmarshal(data, &svj); err != nil {
		return err
	}

	if len(svj.Indices) != len(svj.Values) {
		return fmt.Errorf("indices and values length mismatch: %d vs %d", len(svj.Indices), len(svj.Values))
	}

	sv.Indices = svj.Indices
	sv.Values = svj.Values
	sv.Dim = svj.Dim

	return nil
}

// ToBytes serializes the sparse vector to bytes (for storage).
// Uses JSON encoding which will be compressed by Cowrie codec.
func (sv *SparseVector) ToBytes() ([]byte, error) {
	return json.Marshal(sv)
}

// FromBytes deserializes a sparse vector from bytes.
func FromBytes(data []byte) (*SparseVector, error) {
	var sv SparseVector
	if err := json.Unmarshal(data, &sv); err != nil {
		return nil, err
	}
	return &sv, nil
}

// CompressionRatio estimates the compression ratio vs dense representation.
// Returns the ratio: sparse_size / dense_size
func (sv *SparseVector) CompressionRatio() float64 {
	// Sparse size: indices (4 bytes each) + values (4 bytes each) + overhead
	sparseSize := len(sv.Indices)*4 + len(sv.Values)*4 + 12 // 12 bytes overhead (dim + length fields)

	// Dense size: dimension * 4 bytes per float32
	denseSize := sv.Dim * 4

	if denseSize == 0 {
		return 0
	}

	return float64(sparseSize) / float64(denseSize)
}

// CowrieStats provides statistics about Cowrie encoding efficiency.
type CowrieStats struct {
	SparseBytes      int     // Bytes used by sparse encoding
	DenseBytes       int     // Bytes that would be used by dense
	CompressionRatio float64 // sparse / dense
	Sparsity         float64 // Percentage of zeros
	SavingsPercent   float64 // Percentage saved
}

// EncodingStats returns statistics about the encoding efficiency.
func (sv *SparseVector) EncodingStats() CowrieStats {
	sparseBytes := len(sv.Indices)*4 + len(sv.Values)*4 + 12
	denseBytes := sv.Dim * 4
	ratio := sv.CompressionRatio()

	return CowrieStats{
		SparseBytes:      sparseBytes,
		DenseBytes:       denseBytes,
		CompressionRatio: ratio,
		Sparsity:         sv.Sparsity(),
		SavingsPercent:   (1 - ratio) * 100,
	}
}
