package simd

import (
	"math"
	"math/rand"
	"testing"
)

func randVec(n int, rng *rand.Rand) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1 // [-1, 1]
	}
	return v
}

func TestDotProduct(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	sizes := []int{1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 128, 256, 768, 1024}

	for _, n := range sizes {
		a := randVec(n, rng)
		b := randVec(n, rng)

		got := DotProductF32(a, b)
		want := dotProductScalar(a, b)

		relErr := relError(got, want)
		if relErr > 1e-5 {
			t.Errorf("DotProduct n=%d: got %v, want %v (relErr=%e)", n, got, want, relErr)
		}
	}
}

func TestL2DistanceSquared(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	sizes := []int{1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 128, 256, 768, 1024}

	for _, n := range sizes {
		a := randVec(n, rng)
		b := randVec(n, rng)

		got := L2DistanceSquaredF32(a, b)
		want := l2DistanceSquaredScalar(a, b)

		relErr := relError(got, want)
		if relErr > 1e-5 {
			t.Errorf("L2Squared n=%d: got %v, want %v (relErr=%e)", n, got, want, relErr)
		}
	}
}

func TestL2Distance(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	got := L2DistanceF32(a, b)
	// sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
	want := float32(math.Sqrt(27))
	if relError(got, want) > 1e-5 {
		t.Errorf("L2Distance: got %v, want %v", got, want)
	}
}

func TestCosineDistance(t *testing.T) {
	rng := rand.New(rand.NewSource(999))
	sizes := []int{1, 7, 8, 15, 16, 31, 32, 64, 128, 256, 768, 1024}

	for _, n := range sizes {
		a := randVec(n, rng)
		b := randVec(n, rng)

		got := CosineDistanceF32(a, b)
		want := cosineDistanceScalar(a, b)

		if math.Abs(float64(got-want)) > 1e-4 {
			t.Errorf("CosineDistance n=%d: got %v, want %v (diff=%e)", n, got, want, got-want)
		}
	}
}

func TestCosineDistance_Identical(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	got := CosineDistanceF32(a, a)
	if got > 1e-6 {
		t.Errorf("CosineDistance(a, a) should be ~0, got %v", got)
	}
}

func TestCosineDistance_Orthogonal(t *testing.T) {
	a := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	b := []float32{0, 1, 0, 0, 0, 0, 0, 0}
	got := CosineDistanceF32(a, b)
	if math.Abs(float64(got)-1.0) > 1e-6 {
		t.Errorf("CosineDistance(orthogonal) should be ~1, got %v", got)
	}
}

func TestCosineDistance_Zero(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	got := CosineDistanceF32(a, b)
	if got != 1.0 {
		t.Errorf("CosineDistance(zero, b) should be 1.0, got %v", got)
	}
}

func TestPanic_Mismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on mismatched lengths")
		}
	}()
	DotProductF32([]float32{1, 2}, []float32{1})
}

func TestEmpty(t *testing.T) {
	if DotProductF32(nil, nil) != 0 {
		t.Error("DotProduct(nil, nil) should be 0")
	}
	if L2DistanceSquaredF32(nil, nil) != 0 {
		t.Error("L2Squared(nil, nil) should be 0")
	}
	if CosineDistanceF32(nil, nil) != 1.0 {
		t.Error("CosineDistance(nil, nil) should be 1.0")
	}
}

// Benchmarks at dim=768 (common embedding dimension)

func BenchmarkDotProduct768(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randVec(768, rng)
	v := randVec(768, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductF32(a, v)
	}
}

func BenchmarkDotProduct768_Scalar(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randVec(768, rng)
	v := randVec(768, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotProductScalar(a, v)
	}
}

func BenchmarkL2Distance768(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randVec(768, rng)
	v := randVec(768, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2DistanceSquaredF32(a, v)
	}
}

func BenchmarkL2Distance768_Scalar(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randVec(768, rng)
	v := randVec(768, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l2DistanceSquaredScalar(a, v)
	}
}

func BenchmarkCosineDistance768(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randVec(768, rng)
	v := randVec(768, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineDistanceF32(a, v)
	}
}

func BenchmarkCosineDistance768_Scalar(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	a := randVec(768, rng)
	v := randVec(768, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cosineDistanceScalar(a, v)
	}
}

func relError(got, want float32) float64 {
	if want == 0 {
		return float64(math.Abs(float64(got)))
	}
	return math.Abs(float64(got-want)) / math.Abs(float64(want))
}
