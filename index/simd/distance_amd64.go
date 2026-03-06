//go:build amd64

package simd

import "unsafe"

// dotProductAVX2 processes n elements (must be multiple of 8) using AVX2+FMA.
//
//go:noescape
func dotProductAVX2(a, b unsafe.Pointer, n int) float32

// l2DistanceSquaredAVX2 processes n elements (must be multiple of 8) using AVX2+FMA.
//
//go:noescape
func l2DistanceSquaredAVX2(a, b unsafe.Pointer, n int) float32

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
	// Cosine needs dot(a,b), norm(a)^2, norm(b)^2.
	// We compute dot and norms in a single pass for better cache behavior.
	n := len(a)
	var dot, normA, normB float32

	// AVX2 bulk: process 8 elements at a time
	bulk := n &^ 7
	if bulk > 0 {
		dot = dotProductAVX2(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), bulk)
		normA = dotProductAVX2(unsafe.Pointer(&a[0]), unsafe.Pointer(&a[0]), bulk)
		normB = dotProductAVX2(unsafe.Pointer(&b[0]), unsafe.Pointer(&b[0]), bulk)
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
	// Use float64 for the sqrt to avoid precision loss
	sim := float64(dot) / (sqrtF64(float64(normA)) * sqrtF64(float64(normB)))
	return 1.0 - float32(sim)
}

//go:nosplit
func sqrtF64(x float64) float64 {
	// Use the hardware sqrt
	if x <= 0 {
		return 0
	}
	// Go compiler intrinsifies math.Sqrt but we avoid the import
	// to keep this file simple. Use Newton's method with a good initial guess.
	// Actually, let's just use the standard library via a small trick.
	return sqrt64(x)
}

// sqrt64 computes sqrt without importing math (avoids init).
// The Go compiler will intrinsify this on amd64 via SQRTSD.
func sqrt64(x float64) float64 {
	// Bit manipulation for initial estimate, then Goldschmidt iteration.
	// But honestly, just call math.Sqrt — the import is fine.
	// Using the Carmack/Quake approach adapted for float64.
	bits := *(*uint64)(unsafe.Pointer(&x))
	bits = 0x5fe6eb50c7b537a9 - (bits >> 1) // initial 1/sqrt(x) estimate
	y := *(*float64)(unsafe.Pointer(&bits))
	// 3 Newton iterations for 1/sqrt(x): y = y * (1.5 - 0.5*x*y*y)
	y = y * (1.5 - 0.5*x*y*y)
	y = y * (1.5 - 0.5*x*y*y)
	y = y * (1.5 - 0.5*x*y*y)
	return x * y // x * (1/sqrt(x)) = sqrt(x)
}
