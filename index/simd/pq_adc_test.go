package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantizePQ4Tables(t *testing.T) {
	// Create tables with known values
	m := 8
	tables := make([][]float32, m)
	for i := 0; i < m; i++ {
		tables[i] = make([]float32, 16)
		for k := 0; k < 16; k++ {
			tables[i][k] = float32(k) * float32(i+1) * 0.1
		}
	}

	ft := QuantizePQ4Tables(tables)
	if ft.M != m {
		t.Fatalf("M: got %d, want %d", ft.M, m)
	}
	if len(ft.Data) != m*16 {
		t.Fatalf("Data len: got %d, want %d", len(ft.Data), m*16)
	}

	// Check quantization preserves ordering within each subtable
	for i := 0; i < m; i++ {
		for k := 1; k < 16; k++ {
			if ft.Data[i*16+k] < ft.Data[i*16+k-1] {
				t.Errorf("sub %d: quantized[%d]=%d < quantized[%d]=%d (order violated)",
					i, k, ft.Data[i*16+k], k-1, ft.Data[i*16+k-1])
			}
		}
		// First should be 0 (min), last should be 255 (max)
		if ft.Data[i*16+0] != 0 {
			t.Errorf("sub %d: min should quantize to 0, got %d", i, ft.Data[i*16+0])
		}
		if ft.Data[i*16+15] != 255 {
			t.Errorf("sub %d: max should quantize to 255, got %d", i, ft.Data[i*16+15])
		}
	}
}

func TestTransposePQ4Codes(t *testing.T) {
	nVec := 5
	m := 4 // 2 pairs
	halfM := m / 2

	// Create packed codes
	codes := make([]byte, nVec*halfM)
	for v := 0; v < nVec; v++ {
		for p := 0; p < halfM; p++ {
			codes[v*halfM+p] = byte(v*halfM + p)
		}
	}

	transposed, nPad := TransposePQ4Codes(codes, nVec, m)
	if nPad != 32 { // padded to multiple of 32
		t.Fatalf("nVecPadded: got %d, want 32", nPad)
	}

	// Verify transposition
	for v := 0; v < nVec; v++ {
		for p := 0; p < halfM; p++ {
			orig := codes[v*halfM+p]
			got := transposed[p*nPad+v]
			if got != orig {
				t.Errorf("vec %d pair %d: transposed=%d, original=%d", v, p, got, orig)
			}
		}
	}

	// Verify padding is zero
	for v := nVec; v < nPad; v++ {
		for p := 0; p < halfM; p++ {
			if transposed[p*nPad+v] != 0 {
				t.Errorf("padding vec %d pair %d should be 0, got %d", v, p, transposed[p*nPad+v])
			}
		}
	}
}

func TestPQ4FastScan_MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	m := 16 // must be even
	nVec := 100
	halfM := m / 2

	// Create random float32 distance tables
	tables := make([][]float32, m)
	for i := 0; i < m; i++ {
		tables[i] = make([]float32, 16)
		for k := 0; k < 16; k++ {
			tables[i][k] = rng.Float32() * 10.0
		}
	}

	// Create random packed PQ4 codes
	codes := make([]byte, nVec*halfM)
	for i := range codes {
		codes[i] = byte(rng.Intn(256))
	}

	// Compute reference distances using scalar float32 lookup
	refDists := make([]float32, nVec)
	for v := 0; v < nVec; v++ {
		var d float32
		for p := 0; p < halfM; p++ {
			packed := codes[v*halfM+p]
			lo := packed & 0x0F
			hi := (packed >> 4) & 0x0F
			d += tables[p*2][lo]
			d += tables[p*2+1][hi]
		}
		refDists[v] = d
	}

	// Compute FastScan distances
	ft := QuantizePQ4Tables(tables)
	transposed, nPad := TransposePQ4Codes(codes, nVec, m)
	fastDists := PQ4LookupBatchFastScan(ft, transposed, nPad, nVec)

	// Unquantize for comparison
	approxDists := make([]float32, nVec)
	for i := 0; i < nVec; i++ {
		// Per-subtable unquantization for accuracy
		var d float32
		for p := 0; p < halfM; p++ {
			packed := codes[i*halfM+p]
			lo := packed & 0x0F
			hi := (packed >> 4) & 0x0F
			subM0 := p * 2
			subM1 := subM0 + 1
			d += ft.Scales[subM0]*float32(ft.Data[subM0*16+int(lo)]) + ft.Biases[subM0]
			d += ft.Scales[subM1]*float32(ft.Data[subM1*16+int(hi)]) + ft.Biases[subM1]
		}
		approxDists[i] = d
	}

	// Check that approximate distances are close to reference
	for i := 0; i < nVec; i++ {
		relErr := math.Abs(float64(approxDists[i]-refDists[i])) / (math.Abs(float64(refDists[i])) + 1e-10)
		if relErr > 0.05 { // 5% tolerance for quantization
			t.Errorf("vec %d: approx=%.4f, ref=%.4f, relErr=%.4f",
				i, approxDists[i], refDists[i], relErr)
		}
	}

	// Check that ranking is preserved (Spearman rank correlation)
	// Sort both by distance and verify top-10 overlap
	type idxDist struct {
		idx  int
		dist float32
	}
	refSorted := make([]idxDist, nVec)
	fastSorted := make([]idxDist, nVec)
	for i := 0; i < nVec; i++ {
		refSorted[i] = idxDist{i, refDists[i]}
		fastSorted[i] = idxDist{i, float32(fastDists[i])}
	}

	// Simple bubble sort for small N (test only)
	for i := 0; i < nVec-1; i++ {
		for j := i + 1; j < nVec; j++ {
			if refSorted[j].dist < refSorted[i].dist {
				refSorted[i], refSorted[j] = refSorted[j], refSorted[i]
			}
			if fastSorted[j].dist < fastSorted[i].dist {
				fastSorted[i], fastSorted[j] = fastSorted[j], fastSorted[i]
			}
		}
	}

	// Top-10 overlap should be high
	top := 10
	if nVec < top {
		top = nVec
	}
	refTop := make(map[int]bool)
	for i := 0; i < top; i++ {
		refTop[refSorted[i].idx] = true
	}
	overlap := 0
	for i := 0; i < top; i++ {
		if refTop[fastSorted[i].idx] {
			overlap++
		}
	}
	// With M=16 subquantizers and 8-bit quantization, ranking should be very close
	if overlap < top-3 {
		t.Errorf("top-%d overlap: %d (too low, quantization degraded ranking)", top, overlap)
	}
	t.Logf("top-%d overlap: %d/%d", top, overlap, top)
}

func TestPQLookupBatchFlat(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	m := 8
	k := 256
	nVec := 50

	// Create random tables
	flatTables := make([]float32, m*k)
	for i := range flatTables {
		flatTables[i] = rng.Float32() * 5.0
	}

	// Create random codes
	codes := make([]byte, nVec*m)
	for i := range codes {
		codes[i] = byte(rng.Intn(k))
	}

	// Reference scalar
	refDists := make([]float32, nVec)
	for v := 0; v < nVec; v++ {
		var d float32
		for j := 0; j < m; j++ {
			d += flatTables[j*k+int(codes[v*m+j])]
		}
		refDists[v] = d
	}

	// Optimized
	fastDists := PQLookupBatchFlat(flatTables, k, codes, m, nVec)

	for i := 0; i < nVec; i++ {
		if math.Abs(float64(fastDists[i]-refDists[i])) > 1e-4 {
			t.Errorf("vec %d: fast=%.6f, ref=%.6f", i, fastDists[i], refDists[i])
		}
	}
}

func BenchmarkPQ4FastScan_32Kvecs(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	m := 48 // 384d / 8 dsub
	nVec := 32768
	halfM := m / 2

	tables := make([][]float32, m)
	for i := 0; i < m; i++ {
		tables[i] = make([]float32, 16)
		for k := 0; k < 16; k++ {
			tables[i][k] = rng.Float32() * 10.0
		}
	}

	codes := make([]byte, nVec*halfM)
	for i := range codes {
		codes[i] = byte(rng.Intn(256))
	}

	ft := QuantizePQ4Tables(tables)
	transposed, nPad := TransposePQ4Codes(codes, nVec, m)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		PQ4LookupBatchFastScan(ft, transposed, nPad, nVec)
	}
	b.ReportMetric(float64(nVec)/float64(b.Elapsed().Seconds())*float64(b.N), "vec/s")
}

func BenchmarkPQ4Scalar_32Kvecs(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	m := 48
	nVec := 32768
	halfM := m / 2

	tables := make([][]float32, m)
	for i := 0; i < m; i++ {
		tables[i] = make([]float32, 16)
		for k := 0; k < 16; k++ {
			tables[i][k] = rng.Float32() * 10.0
		}
	}

	codes := make([]byte, nVec*halfM)
	for i := range codes {
		codes[i] = byte(rng.Intn(256))
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		distances := make([]float32, nVec)
		for v := 0; v < nVec; v++ {
			var d float32
			for p := 0; p < halfM; p++ {
				packed := codes[v*halfM+p]
				lo := packed & 0x0F
				hi := (packed >> 4) & 0x0F
				d += tables[p*2][lo]
				d += tables[p*2+1][hi]
			}
			distances[v] = d
		}
		_ = distances
	}
	b.ReportMetric(float64(nVec)/float64(b.Elapsed().Seconds())*float64(b.N), "vec/s")
}

func BenchmarkPQ8Flat_32Kvecs(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	m := 48
	k := 256
	nVec := 32768

	flatTables := make([]float32, m*k)
	for i := range flatTables {
		flatTables[i] = rng.Float32() * 10.0
	}
	codes := make([]byte, nVec*m)
	for i := range codes {
		codes[i] = byte(rng.Intn(k))
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		PQLookupBatchFlat(flatTables, k, codes, m, nVec)
	}
	b.ReportMetric(float64(nVec)/float64(b.Elapsed().Seconds())*float64(b.N), "vec/s")
}
