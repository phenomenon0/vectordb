package collection

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

// Vector is one named vector field of a Document. Exactly one of Dense or
// Sparse is set. The JSON form is unchanged from the interface{} era: dense
// encodes as a JSON array of numbers, sparse as {"indices":[...],"values":[...],"dim":N}.
type Vector struct {
	Dense  []float32
	Sparse *sparse.SparseVector
}

// IsZero reports whether the vector has neither a dense nor a sparse value.
func (v Vector) IsZero() bool {
	return v.Dense == nil && v.Sparse == nil
}

// Clone returns a deep copy of the vector. Sparse cloning preserves nil vs.
// empty Indices/Values (sparse.SparseVector.Clone uses make(), which turns a
// nil slice into a non-nil empty one and would flip the journaled JSON for a
// termless sparse vector from "indices":null to "indices":[]).
func (v Vector) Clone() Vector {
	if v.Sparse != nil {
		return Vector{Sparse: &sparse.SparseVector{
			Indices: append([]uint32(nil), v.Sparse.Indices...),
			Values:  append([]float32(nil), v.Sparse.Values...),
			Dim:     v.Sparse.Dim,
		}}
	}
	if v.Dense != nil {
		return Vector{Dense: append([]float32(nil), v.Dense...)}
	}
	return Vector{}
}

// MarshalJSON implements json.Marshaler.
func (v Vector) MarshalJSON() ([]byte, error) {
	if v.Sparse != nil {
		return json.Marshal(v.Sparse)
	}
	if v.Dense != nil {
		return json.Marshal(v.Dense)
	}
	return []byte("null"), nil
}

// UnmarshalJSON implements json.Unmarshaler.
func (v *Vector) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || bytes.Equal(b, []byte("null")) {
		*v = Vector{}
		return nil
	}
	switch b[0] {
	case '[':
		var dense []float32
		if err := json.Unmarshal(b, &dense); err != nil {
			return err
		}
		*v = Vector{Dense: dense}
		return nil
	case '{':
		var sv sparse.SparseVector
		if err := json.Unmarshal(b, &sv); err != nil {
			return err
		}
		*v = Vector{Sparse: &sv}
		return nil
	default:
		return fmt.Errorf("collection: invalid vector JSON: %s", b)
	}
}
