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

func TestSparseCoOToDense(t *testing.T) {
	sparse := &SparseCoO{
		Indices: []uint32{0, 5, 10},
		Values:  []float32{1.0, 2.0, 3.0},
		Dim:     12,
	}

	dense := sparse.ToDense()

	if len(dense) != 12 {
		t.Errorf("expected length 12, got %d", len(dense))
	}

	// Check non-zero values
	if dense[0] != 1.0 {
		t.Errorf("expected dense[0] = 1.0, got %f", dense[0])
	}
	if dense[5] != 2.0 {
		t.Errorf("expected dense[5] = 2.0, got %f", dense[5])
	}
	if dense[10] != 3.0 {
		t.Errorf("expected dense[10] = 3.0, got %f", dense[10])
	}

	// Check zero values
	if dense[1] != 0.0 {
		t.Errorf("expected dense[1] = 0.0, got %f", dense[1])
	}
	if dense[11] != 0.0 {
		t.Errorf("expected dense[11] = 0.0, got %f", dense[11])
	}
}

func TestSparseCoONonZeroCount(t *testing.T) {
	sparse := &SparseCoO{
		Indices: []uint32{0, 5, 10},
		Values:  []float32{1.0, 2.0, 3.0},
		Dim:     100,
	}

	count := sparse.NonZeroCount()
	if count != 3 {
		t.Errorf("expected 3 non-zero elements, got %d", count)
	}
}
