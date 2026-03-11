package simd

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/index/simd/cref"
)

// ─── Correctness Tests (CGo C vs Go SIMD) ────────────────────────────────────

func TestCRef_Correctness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dims := []int{8, 64, 128, 256, 768, 1024}

	for _, dim := range dims {
		a := randVec(dim, rng)
		b := randVec(dim, rng)

		cDot := cref.AVX2Dot(a, b)
		goDot := DotProductF32(a, b)
		if relError(cDot, goDot) > 1e-5 {
			t.Errorf("dot dim=%d: C=%v Go=%v", dim, cDot, goDot)
		}

		cL2 := cref.AVX2L2(a, b)
		goL2 := L2DistanceSquaredF32(a, b)
		if relError(cL2, goL2) > 1e-5 {
			t.Errorf("l2 dim=%d: C=%v Go=%v", dim, cL2, goL2)
		}

		cCos := cref.AVX2Cosine(a, b)
		goCos := CosineDistanceF32(a, b)
		if relError(cCos, goCos) > 1e-4 {
			t.Errorf("cosine dim=%d: C=%v Go=%v", dim, cCos, goCos)
		}
	}
}

func TestCRef_AVX512_Correctness(t *testing.T) {
	if !cref.HasAVX512() {
		t.Skip("AVX-512 not supported on this CPU")
	}

	rng := rand.New(rand.NewSource(42))
	dims := []int{16, 64, 128, 256, 768, 1024}

	for _, dim := range dims {
		a := randVec(dim, rng)
		b := randVec(dim, rng)

		c512Dot := cref.BatchAVX512Dot(a, b, 1)
		goDot := DotProductF32(a, b)
		if relError(c512Dot, goDot) > 1e-5 {
			t.Errorf("avx512 dot dim=%d: C=%v Go=%v", dim, c512Dot, goDot)
		}

		c512L2 := cref.BatchAVX512L2(a, b, 1)
		goL2 := L2DistanceSquaredF32(a, b)
		if relError(c512L2, goL2) > 1e-5 {
			t.Errorf("avx512 l2 dim=%d: C=%v Go=%v", dim, c512L2, goL2)
		}

		c512Cos := cref.BatchAVX512Cosine(a, b, 1)
		goCos := CosineDistanceF32(a, b)
		if relError(c512Cos, goCos) > 1e-4 {
			t.Errorf("avx512 cosine dim=%d: C=%v Go=%v", dim, c512Cos, goCos)
		}
	}
}

// ─── C Reference Benchmarks ──────────────────────────────────────────────────

const cBatchIters = 1000

func BenchmarkCRef(b *testing.B) {
	type cFunc struct {
		name   string
		fnName string
		batch  func(a, b []float32, iters int) float32
	}

	funcs := []cFunc{
		{"C_AVX2_dot", "dot", cref.BatchAVX2Dot},
		{"C_AVX2_l2", "l2", cref.BatchAVX2L2},
		{"C_AVX2_cosine", "cosine", cref.BatchAVX2Cosine},
	}

	for _, fn := range funcs {
		b.Run(fn.name, func(b *testing.B) {
			for _, dim := range benchDims {
				rng := rand.New(rand.NewSource(42))
				a := randVec(dim, rng)
				v := randVec(dim, rng)
				bytesPerCall := float64(2 * dim * 4)
				flopsPerCall := flopsPerElement(fn.fnName) * float64(dim)

				b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
					b.SetBytes(int64(bytesPerCall) * cBatchIters)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						fn.batch(a, v, cBatchIters)
					}
					nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / cBatchIters
					reportMetrics(b, nsPerOp, flopsPerCall, bytesPerCall)
				})
			}
		})
	}
}

func BenchmarkCRef_AVX512(b *testing.B) {
	if !cref.HasAVX512() {
		b.Skip("AVX-512 not supported on this CPU")
	}

	type cFunc struct {
		name   string
		fnName string
		batch  func(a, b []float32, iters int) float32
	}

	funcs := []cFunc{
		{"C_AVX512_dot", "dot", cref.BatchAVX512Dot},
		{"C_AVX512_l2", "l2", cref.BatchAVX512L2},
		{"C_AVX512_cosine", "cosine", cref.BatchAVX512Cosine},
	}

	for _, fn := range funcs {
		b.Run(fn.name, func(b *testing.B) {
			for _, dim := range benchDims {
				rng := rand.New(rand.NewSource(42))
				a := randVec(dim, rng)
				v := randVec(dim, rng)
				bytesPerCall := float64(2 * dim * 4)
				flopsPerCall := flopsPerElement(fn.fnName) * float64(dim)

				b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
					b.SetBytes(int64(bytesPerCall) * cBatchIters)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						fn.batch(a, v, cBatchIters)
					}
					nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / cBatchIters
					reportMetrics(b, nsPerOp, flopsPerCall, bytesPerCall)
				})
			}
		})
	}
}

// ─── C Reference Report ──────────────────────────────────────────────────────

func TestCRefReport(t *testing.T) {
	dims := []int{128, 384, 768, 1536}
	if testing.Short() {
		dims = []int{768}
	}

	type benchResult struct {
		function string
		dim      int
		nsPerOp  float64
		gflops   float64
		gbps     float64
		pctPeak  float64
	}

	iters := 100000
	if testing.Short() {
		iters = 10000
	}

	type distSpec struct {
		name   string
		fnName string
		cBatch func(a, b []float32, iters int) float32
		goCall func([]float32, []float32) float32
	}

	specs := []distSpec{
		{"DotProduct", "dot", cref.BatchAVX2Dot, DotProductF32},
		{"L2Squared", "l2", cref.BatchAVX2L2, L2DistanceSquaredF32},
		{"CosineDistance", "cosine", cref.BatchAVX2Cosine, CosineDistanceF32},
	}

	var results []benchResult

	for _, spec := range specs {
		for _, dim := range dims {
			rng := rand.New(rand.NewSource(42))
			a := randVec(dim, rng)
			v := randVec(dim, rng)
			bytesPerCall := float64(2 * dim * 4)
			flopsPerCall := flopsPerElement(spec.fnName) * float64(dim)

			// Go ASM timing
			goNs := benchmarkNs(func() { spec.goCall(a, v) }, iters)
			goGflops := flopsPerCall / goNs
			goGbps := bytesPerCall / goNs
			goPct := (goGflops / zen4PeakGFLOPS) * 100

			results = append(results, benchResult{
				function: spec.name + " (Go ASM)", dim: dim,
				nsPerOp: goNs, gflops: goGflops, gbps: goGbps, pctPeak: goPct,
			})

			// C AVX2 timing (batched, then divide)
			outerIters := iters / cBatchIters
			if outerIters < 1 {
				outerIters = 1
			}
			cNs := benchmarkCBatchNs(func() { spec.cBatch(a, v, cBatchIters) }, outerIters, cBatchIters)
			cGflops := flopsPerCall / cNs
			cGbps := bytesPerCall / cNs
			cPct := (cGflops / zen4PeakGFLOPS) * 100

			results = append(results, benchResult{
				function: spec.name + " (C AVX2)", dim: dim,
				nsPerOp: cNs, gflops: cGflops, gbps: cGbps, pctPeak: cPct,
			})
		}
	}

	// AVX-512 results if available
	hasAVX512 := cref.HasAVX512()
	if hasAVX512 {
		avx512Specs := []struct {
			name   string
			fnName string
			cBatch func(a, b []float32, iters int) float32
		}{
			{"DotProduct", "dot", cref.BatchAVX512Dot},
			{"L2Squared", "l2", cref.BatchAVX512L2},
			{"CosineDistance", "cosine", cref.BatchAVX512Cosine},
		}

		for _, spec := range avx512Specs {
			for _, dim := range dims {
				rng := rand.New(rand.NewSource(42))
				a := randVec(dim, rng)
				v := randVec(dim, rng)
				bytesPerCall := float64(2 * dim * 4)
				flopsPerCall := flopsPerElement(spec.fnName) * float64(dim)

				outerIters := iters / cBatchIters
				if outerIters < 1 {
					outerIters = 1
				}
				cNs := benchmarkCBatchNs(func() { spec.cBatch(a, v, cBatchIters) }, outerIters, cBatchIters)
				cGflops := flopsPerCall / cNs
				cGbps := bytesPerCall / cNs
				cPct := (cGflops / zen4PeakGFLOPS) * 100

				results = append(results, benchResult{
					function: spec.name + " (C AVX512)", dim: dim,
					nsPerOp: cNs, gflops: cGflops, gbps: cGbps, pctPeak: cPct,
				})
			}
		}
	}

	// Print comparison report
	fmt.Println()
	fmt.Printf("Go ASM vs C Reference Comparison — %s\n", cpuName())
	fmt.Printf("Theoretical peak: %.1f GFLOP/s (FP32 AVX2 FMA, single core @ %.1f GHz)\n",
		zen4PeakGFLOPS, zen4BoostGHz)
	if hasAVX512 {
		fmt.Println("AVX-512: SUPPORTED")
	} else {
		fmt.Println("AVX-512: not available")
	}
	fmt.Println(strings.Repeat("═", 85))
	fmt.Printf("%-28s %5s %9s %9s %8s %7s\n",
		"Function", "Dim", "ns/op", "GFLOP/s", "GB/s", "%Peak")
	fmt.Println(strings.Repeat("─", 85))

	for _, r := range results {
		fmt.Printf("%-28s %5d %9.1f %9.2f %8.1f %6.1f%%\n",
			r.function, r.dim, r.nsPerOp, r.gflops, r.gbps, r.pctPeak)
	}
	fmt.Println(strings.Repeat("═", 85))
	fmt.Println()
}

// benchmarkCBatchNs runs a batched C function and returns per-iteration ns.
func benchmarkCBatchNs(f func(), outerIters, innerBatch int) float64 {
	if outerIters < 1 {
		outerIters = 1
	}
	for i := 0; i < outerIters/10+1; i++ {
		f()
	}
	start := nanotime()
	for i := 0; i < outerIters; i++ {
		f()
	}
	elapsed := nanotime() - start
	return float64(elapsed) / (float64(outerIters) * float64(innerBatch))
}
