//go:build amd64

package simd

import (
	"math"
	"unsafe"
)

// dotProductAVX2 processes n elements (must be multiple of 8) using AVX2+FMA.
//
//go:noescape
func dotProductAVX2(a, b unsafe.Pointer, n int) float32

// l2DistanceSquaredAVX2 processes n elements (must be multiple of 8) using AVX2+FMA.
//
//go:noescape
func l2DistanceSquaredAVX2(a, b unsafe.Pointer, n int) float32

// cosineComponentsAVX2 computes dot(a,b), dot(a,a), dot(b,b) in a single pass.
// n MUST be a multiple of 8. Caller handles the tail.
//
//go:noescape
func cosineComponentsAVX2(a, b unsafe.Pointer, n int) (dot, normA, normB float32)

func dotProductF32(a, b []float32) float32 {
	n := len(a)
	bulk := n &^ 7 // round down to multiple of 8
	var sum float32
	if bulk > 0 {
		sum = dotProductAVX2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), bulk)
	}
	// scalar tail
	for i := bulk; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func l2DistanceSquaredF32(a, b []float32) float32 {
	n := len(a)
	bulk := n &^ 7
	var sum float32
	if bulk > 0 {
		sum = l2DistanceSquaredAVX2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), bulk)
	}
	for i := bulk; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func cosineDistanceF32(a, b []float32) float32 {
	n := len(a)
	var dot, normA, normB float32

	bulk := n &^ 7
	if bulk > 0 {
		dot, normA, normB = cosineComponentsAVX2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), bulk)
	}

	// scalar tail
	for i := bulk; i < n; i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := float64(dot) / (math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	return 1.0 - float32(sim)
}
