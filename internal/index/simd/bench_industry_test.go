package simd

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"
	_ "unsafe"
)

// ─── Theoretical Peak Constants ───────────────────────────────────────────────
// Zen 4 (Ryzen 7 7700X): 2× 256-bit FMA units, 8 FP32/cycle each = 16 FP32 FMAs/cycle
// FMA = 2 FLOPs (multiply + add) → 32 FLOP/cycle
// At 5.4 GHz boost: 32 × 5.4 = 172.8 GFLOP/s theoretical peak (single core)
const (
	zen4BoostGHz    = 5.4
	zen4FMAsPerCycle = 32 // 2 FMA units × 8 floats × 2 ops
	zen4PeakGFLOPS  = zen4BoostGHz * float64(zen4FMAsPerCycle) // 172.8
)

// FLOPs per element for each distance function.
func flopsPerElement(fn string) float64 {
	switch fn {
	case "dot":
		return 2 // multiply + add
	case "l2":
		return 3 // sub + mul + add
	case "cosine":
		return 6 // dot(a,b): mul+add, dot(a,a): mul+add, dot(b,b): mul+add
	case "normdot":
		return 2 // same as dot, but on pre-normalized vectors
	default:
		return 2
	}
}

// ─── Dimension Sweep Benchmarks ───────────────────────────────────────────────

var benchDims = []int{64, 128, 256, 384, 768, 1024, 1536, 3072}

func BenchmarkSIMD(b *testing.B) {
	type distFunc struct {
		name   string
		fnName string // for FLOP counting
		simd   func([]float32, []float32) float32
		scalar func([]float32, []float32) float32
	}

	funcs := []distFunc{
		{"dot", "dot", DotProductF32, dotProductScalar},
		{"l2", "l2", L2DistanceSquaredF32, l2DistanceSquaredScalar},
		{"cosine", "cosine", CosineDistanceF32, cosineDistanceScalar},
	}

	for _, fn := range funcs {
		b.Run(fn.name, func(b *testing.B) {
			for _, dim := range benchDims {
				rng := rand.New(rand.NewSource(42))
				a := randVec(dim, rng)
				v := randVec(dim, rng)
				bytesPerCall := float64(2 * dim * 4)
				flopsPerCall := flopsPerElement(fn.fnName) * float64(dim)

				b.Run(fmt.Sprintf("dim=%d/avx2", dim), func(b *testing.B) {
					b.SetBytes(int64(bytesPerCall))
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						fn.simd(a, v)
					}
					nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
					reportMetrics(b, nsPerOp, flopsPerCall, bytesPerCall)
				})

				b.Run(fmt.Sprintf("dim=%d/scalar", dim), func(b *testing.B) {
					b.SetBytes(int64(bytesPerCall))
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						fn.scalar(a, v)
					}
					nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
					reportMetrics(b, nsPerOp, flopsPerCall, bytesPerCall)
				})
			}
		})
	}
}

// ─── Cold-Cache Benchmark ─────────────────────────────────────────────────────
// Simulates real HNSW neighbor access by cycling through 10K vector pairs randomly.

func BenchmarkSIMD_ColdCache(b *testing.B) {
	const (
		numPairs = 10000
		dim      = 768
	)

	type distFunc struct {
		name string
		fn   func([]float32, []float32) float32
	}

	funcs := []distFunc{
		{"dot/avx2", DotProductF32},
		{"l2/avx2", L2DistanceSquaredF32},
		{"cosine/avx2", CosineDistanceF32},
	}

	rng := rand.New(rand.NewSource(99))
	as := make([][]float32, numPairs)
	bs := make([][]float32, numPairs)
	for i := range as {
		as[i] = randVec(dim, rng)
		bs[i] = randVec(dim, rng)
	}

	// Shuffled access order to defeat prefetcher
	order := make([]int, numPairs)
	for i := range order {
		order[i] = i
	}
	rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })

	for _, fn := range funcs {
		b.Run(fmt.Sprintf("cold/%s/dim=%d", fn.name, dim), func(b *testing.B) {
			b.SetBytes(int64(2 * dim * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx := order[i%numPairs]
				fn.fn(as[idx], bs[idx])
			}
		})
	}
}

// ─── Normalized-Dot Benchmark ─────────────────────────────────────────────────
// Pre-normalized vectors use `1 - dot(a,b)` instead of full cosine.
// This is what Faiss/Qdrant use — eliminates norm computation.

func BenchmarkSIMD_NormalizedDot(b *testing.B) {
	for _, dim := range benchDims {
		rng := rand.New(rand.NewSource(42))
		a := normalizeVec(randVec(dim, rng))
		v := normalizeVec(randVec(dim, rng))
		bytesPerCall := float64(2 * dim * 4)
		flopsPerCall := flopsPerElement("normdot") * float64(dim)

		b.Run(fmt.Sprintf("normdot/dim=%d/avx2", dim), func(b *testing.B) {
			b.SetBytes(int64(bytesPerCall))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = 1.0 - DotProductF32(a, v)
			}
			nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
			reportMetrics(b, nsPerOp, flopsPerCall, bytesPerCall)
		})

		b.Run(fmt.Sprintf("cosine_full/dim=%d/avx2", dim), func(b *testing.B) {
			b.SetBytes(int64(bytesPerCall))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				CosineDistanceF32(a, v)
			}
			nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
			reportMetrics(b, nsPerOp, flopsPerElement("cosine")*float64(dim), bytesPerCall)
		})
	}
}

// ─── Industry Report ──────────────────────────────────────────────────────────

func TestSIMDIndustryReport(t *testing.T) {
	type benchResult struct {
		function string
		dim      int
		variant  string
		nsPerOp  float64
		gflops   float64
		gbps     float64
		pctPeak  float64
	}

	iters := 100000
	if testing.Short() {
		iters = 10000
	}

	type distFunc struct {
		name   string
		fnName string
		simd   func([]float32, []float32) float32
		scalar func([]float32, []float32) float32
	}

	funcs := []distFunc{
		{"DotProduct", "dot", DotProductF32, dotProductScalar},
		{"L2Squared", "l2", L2DistanceSquaredF32, l2DistanceSquaredScalar},
		{"CosineDistance", "cosine", CosineDistanceF32, cosineDistanceScalar},
	}

	dims := []int{128, 384, 768, 1536}
	if testing.Short() {
		dims = []int{768}
	}

	var results []benchResult

	for _, fn := range funcs {
		for _, dim := range dims {
			rng := rand.New(rand.NewSource(42))
			a := randVec(dim, rng)
			v := randVec(dim, rng)
			bytesPerCall := float64(2 * dim * 4)
			flopsPerCall := flopsPerElement(fn.fnName) * float64(dim)

			// Warmup
			for i := 0; i < 1000; i++ {
				fn.simd(a, v)
			}

			// SIMD timing
			simdNs := benchmarkNs(func() { fn.simd(a, v) }, iters)
			simdGflops := flopsPerCall / simdNs
			simdGbps := bytesPerCall / simdNs
			simdPct := (simdGflops / zen4PeakGFLOPS) * 100

			results = append(results, benchResult{
				function: fn.name, dim: dim, variant: "AVX2",
				nsPerOp: simdNs, gflops: simdGflops, gbps: simdGbps, pctPeak: simdPct,
			})

			// Scalar timing
			scalarNs := benchmarkNs(func() { fn.scalar(a, v) }, iters)
			scalarGflops := flopsPerCall / scalarNs
			scalarGbps := bytesPerCall / scalarNs
			scalarPct := (scalarGflops / zen4PeakGFLOPS) * 100

			results = append(results, benchResult{
				function: fn.name, dim: dim, variant: "scalar",
				nsPerOp: scalarNs, gflops: scalarGflops, gbps: scalarGbps, pctPeak: scalarPct,
			})
		}
	}

	// Normalized-dot for comparison
	for _, dim := range dims {
		rng := rand.New(rand.NewSource(42))
		a := normalizeVec(randVec(dim, rng))
		v := normalizeVec(randVec(dim, rng))
		bytesPerCall := float64(2 * dim * 4)
		flopsPerCall := flopsPerElement("normdot") * float64(dim)

		for i := 0; i < 1000; i++ {
			_ = 1.0 - DotProductF32(a, v)
		}

		ns := benchmarkNs(func() { _ = 1.0 - DotProductF32(a, v) }, iters)
		gflops := flopsPerCall / ns
		gbps := bytesPerCall / ns
		pct := (gflops / zen4PeakGFLOPS) * 100

		results = append(results, benchResult{
			function: "NormDot(1-dot)", dim: dim, variant: "AVX2",
			nsPerOp: ns, gflops: gflops, gbps: gbps, pctPeak: pct,
		})
	}

	// Print report
	fmt.Println()
	fmt.Printf("SIMD Distance Benchmark — %s\n", cpuName())
	fmt.Printf("Theoretical peak: %.1f GFLOP/s (FP32 AVX2 FMA, single core @ %.1f GHz)\n",
		zen4PeakGFLOPS, zen4BoostGHz)
	fmt.Println(strings.Repeat("═", 80))
	fmt.Printf("%-18s %5s %7s %9s %8s %7s %8s\n",
		"Function", "Dim", "Variant", "ns/op", "GFLOP/s", "GB/s", "%Peak")
	fmt.Println(strings.Repeat("─", 80))

	for _, r := range results {
		fmt.Printf("%-18s %5d %7s %9.1f %9.2f %8.1f %6.1f%%\n",
			r.function, r.dim, r.variant, r.nsPerOp, r.gflops, r.gbps, r.pctPeak)
	}
	fmt.Println(strings.Repeat("═", 80))

	// Print optimization opportunities
	fmt.Println()
	fmt.Println("Optimization Opportunities:")
	fmt.Println("──────────────────────────────────────────────────────────────")

	// Find cosine vs normdot at 768d
	var cosine768, normdot768 float64
	for _, r := range results {
		if r.dim == 768 && r.variant == "AVX2" {
			if r.function == "CosineDistance" {
				cosine768 = r.nsPerOp
			}
			if r.function == "NormDot(1-dot)" {
				normdot768 = r.nsPerOp
			}
		}
	}
	if cosine768 > 0 && normdot768 > 0 {
		fmt.Printf("  Pre-normalized dot: %.1fns vs cosine %.1fns = %.0f%% faster\n",
			normdot768, cosine768, (1-normdot768/cosine768)*100)
	}

	// Find SIMD vs scalar speedups at 768d
	for _, fn := range funcs {
		var simdNs, scalarNs float64
		for _, r := range results {
			if r.dim == 768 && r.function == fn.name {
				if r.variant == "AVX2" {
					simdNs = r.nsPerOp
				}
				if r.variant == "scalar" {
					scalarNs = r.nsPerOp
				}
			}
		}
		if simdNs > 0 && scalarNs > 0 {
			fmt.Printf("  %s: AVX2 %.1fx over scalar\n", fn.name, scalarNs/simdNs)
		}
	}
	fmt.Println()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func reportMetrics(b *testing.B, nsPerOp, flopsPerCall, bytesPerCall float64) {
	b.Helper()
	if nsPerOp <= 0 {
		return
	}
	gflops := flopsPerCall / nsPerOp // ns → GFLOP/s: flops / ns = Gflops/s
	gbps := bytesPerCall / nsPerOp
	pctPeak := (gflops / zen4PeakGFLOPS) * 100
	b.ReportMetric(gflops, "GFLOP/s")
	b.ReportMetric(gbps, "GB/s")
	b.ReportMetric(pctPeak, "%peak")
}

func normalizeVec(v []float32) []float32 {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}

// benchmarkNs runs f iters times and returns average ns/op.
func benchmarkNs(f func(), iters int) float64 {
	// Warmup
	for i := 0; i < iters/10; i++ {
		f()
	}
	runtime.GC()

	start := nanotime()
	for i := 0; i < iters; i++ {
		f()
	}
	elapsed := nanotime() - start
	return float64(elapsed) / float64(iters)
}

//go:noescape
//go:linkname nanotime runtime.nanotime
func nanotime() int64

func cpuName() string {
	// Best-effort CPU name; falls back to GOARCH
	return fmt.Sprintf("%s/%s (GOARCH=%s)", runtime.GOOS, runtime.GOARCH, runtime.GOARCH)
}
