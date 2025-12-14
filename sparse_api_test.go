package main

import (
	"testing"
)

func TestSparseCoOValidate(t *testing.T) {
	tests := []struct {
		name      string
		sparse    *SparseCoO
		wantError bool
	}{
		{
			name: "valid sparse vector",
			sparse: &SparseCoO{
				Indices: []uint32{0, 5, 10},
				Values:  []float32{1.0, 2.0, 3.0},
				Dim:     100,
			},
			wantError: false,
		},
		{
			name: "length mismatch",
			sparse: &SparseCoO{
				Indices: []uint32{0, 5},
				Values:  []float32{1.0, 2.0, 3.0},
				Dim:     100,
			},
			wantError: true,
		},
		{
			name: "index out of bounds",
			sparse: &SparseCoO{
				Indices: []uint32{0, 5, 150},
				Values:  []float32{1.0, 2.0, 3.0},
				Dim:     100,
			},
			wantError: true,
		},
		{
			name: "unsorted indices",
			sparse: &SparseCoO{
				Indices: []uint32{10, 5, 15},
				Values:  []float32{1.0, 2.0, 3.0},
				Dim:     100,
			},
			wantError: true,
		},
		{
			name: "negative dimension",
			sparse: &SparseCoO{
				Indices: []uint32{0, 5},
				Values:  []float32{1.0, 2.0},
				Dim:     -1,
			},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sparse.Validate()
			if (err != nil) != tt.wantError {
				t.Errorf("Validate() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestSparseCoOToDense and TestSparseCoONonZeroCount are in vector_types_test.go
