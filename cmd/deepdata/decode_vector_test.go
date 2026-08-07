package main

import (
	"encoding/json"
	"testing"
)

func TestDecodeDenseVectorFast(t *testing.T) {
	vec, err := decodeDenseVectorFast(json.RawMessage(`[1.5, -2.25, 3]`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(vec) != 3 || vec[0] != 1.5 || vec[1] != -2.25 || vec[2] != 3 {
		t.Errorf("decoded vector = %v, want [1.5 -2.25 3]", vec)
	}
}

func TestDecodeDenseVectorFastRejectsTrailingGarbage(t *testing.T) {
	for _, raw := range []string{
		`[1,2,3]]`,     // a second closing bracket
		`[1,2,3][4,5]`, // a second array after the vector
		`[1,2,3,"x"]`,  // unsupported element type
		`[1,2,3`,       // unclosed array
		`[1,2,3] xyz`,  // trailing non-JSON text
		`not-an-array`, // invalid shape
	} {
		if vec, err := decodeDenseVectorFast(json.RawMessage(raw)); err == nil {
			t.Errorf("decodeDenseVectorFast(%q) = %v, want error for trailing/malformed input", raw, vec)
		}
	}
}

func TestDecodeDenseVectorFastRejectsNonFiniteNumbers(t *testing.T) {
	for _, raw := range []string{
		`[1,NaN,3]`,
		`[1,Infinity,3]`,
	} {
		if _, err := decodeDenseVectorFast(json.RawMessage(raw)); err == nil {
			t.Errorf("decodeDenseVectorFast(%q) accepted non-finite element", raw)
		}
	}
}

func TestDecodeDenseVectorFastRejectsOverflow(t *testing.T) {
	// A value beyond float32's finite range must be rejected, not silently
	// rounded to +Inf and stored.
	if _, err := decodeDenseVectorFast(json.RawMessage(`[3.6e39, 1]`)); err == nil {
		t.Error("decodeDenseVectorFast accepted a value overflowing float32")
	}
}
