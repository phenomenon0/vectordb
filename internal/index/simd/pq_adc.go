// PQ4 FastScan: SIMD-accelerated PQ distance computation using VPSHUFB.
//
// Standard PQ4 lookup: scalar O(M) per vector (~2,500 QPS for 384d).
// FastScan: VPSHUFB processes 32 vectors in parallel per instruction (~25,000+ QPS).
//
// Requires two preprocessing steps (amortized over many queries):
// 1. QuantizePQ4Tables: float32 tables → uint8 (once per query)
// 2. TransposePQ4Codes: row-major → column-major codes (once at index build time)
package simd

import "math"

// PQ4FastScanTable holds quantized PQ4 distance tables for VPSHUFB lookup.
// Each subtable has 16 uint8 entries (4-bit codes → 16 centroids).
type PQ4FastScanTable struct {
	// Data holds M contiguous 16-byte quantized subtables.
	// Layout: [sub0_val0..sub0_val15, sub1_val0..sub1_val15, ...]
	Data []byte // len = M * 16

	// Per-subtable dequantization: true_dist ≈ scale[m] * quantized + bias[m]
	Scales []float32 // [M]
	Biases []float32 // [M]

	// BiasSum = sum(biases). Constant for all vectors — can be ignored for ranking.
	BiasSum float32

	// GlobalScale for approximate unquantization of summed uint16.
	// Used when all subtables have similar ranges.
	GlobalScale float32

	M int // number of subquantizers
}

// QuantizePQ4Tables converts float32 PQ4 distance tables to uint8 for VPSHUFB.
// Input: tables[m] has exactly 16 float32 entries (K=16 for PQ4).
// Quantization is per-subtable: each subtable gets its own scale/bias.
func QuantizePQ4Tables(tables [][]float32) *PQ4FastScanTable {
	m := len(tables)
	ft := &PQ4FastScanTable{
		Data:   make([]byte, m*16),
		Scales: make([]float32, m),
		Biases: make([]float32, m),
		M:      m,
	}

	var biasSum float64
	var scaleSum float64

	for i := 0; i < m; i++ {
		tab := tables[i]
		if len(tab) < 16 {
			continue
		}

		// Find range
		minVal := float32(math.MaxFloat32)
		maxVal := float32(-math.MaxFloat32)
		for _, v := range tab[:16] {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}

		rng := maxVal - minVal
		if rng < 1e-10 {
			// All values identical — quantize to zero
			ft.Biases[i] = minVal
			ft.Scales[i] = 0
			continue
		}

		scale := rng / 255.0
		ft.Scales[i] = scale
		ft.Biases[i] = minVal
		biasSum += float64(minVal)
		scaleSum += float64(scale)

		// Quantize
		invScale := 255.0 / rng
		for k := 0; k < 16; k++ {
			q := (tab[k] - minVal) * invScale
			if q < 0 {
				q = 0
			}
			if q > 255 {
				q = 255
			}
			ft.Data[i*16+k] = byte(q + 0.5)
		}
	}

	ft.BiasSum = float32(biasSum)
	if m > 0 {
		ft.GlobalScale = float32(scaleSum / float64(m))
	}
	return ft
}

// TransposePQ4Codes rearranges packed PQ4 codes for SIMD-friendly access.
//
// Input layout (row-major): codes[vec * bytesPerVec + pair] for vec in [0,nVec), pair in [0,M/2)
// Output layout (column-major): out[pair * nVecPadded + vec] — all vectors' codes for
// the same subquantizer pair are contiguous. nVecPadded is rounded up to multiple of 32.
//
// This enables loading 32 consecutive vectors' codes with a single VMOVDQU.
func TransposePQ4Codes(codes []byte, nVec, m int) (transposed []byte, nVecPadded int) {
	halfM := m / 2
	bytesPerVec := halfM

	// Pad nVec to multiple of 32 for SIMD alignment
	nVecPadded = (nVec + 31) &^ 31

	transposed = make([]byte, halfM*nVecPadded)

	for v := 0; v < nVec; v++ {
		for p := 0; p < halfM; p++ {
			transposed[p*nVecPadded+v] = codes[v*bytesPerVec+p]
		}
	}
	// Padding vectors get zero codes → zero distances (correct behavior)
	return transposed, nVecPadded
}

// PQ4LookupBatchFastScan computes approximate distances for nVec vectors
// using VPSHUFB-based FastScan on amd64, scalar fallback elsewhere.
//
// Input:
//   - table: quantized distance tables from QuantizePQ4Tables
//   - transposedCodes: column-major codes from TransposePQ4Codes
//   - nVecPadded: padded vector count (multiple of 32)
//   - nVecActual: actual vector count (for output trimming)
//
// Output: uint16 approximate distances. For ranking (top-k), these are
// monotonically related to true distances. To get approximate float32:
//
//	approxDist ≈ table.GlobalScale * float32(result[i]) + table.BiasSum
func PQ4LookupBatchFastScan(table *PQ4FastScanTable, transposedCodes []byte, nVecPadded, nVecActual int) []uint16 {
	if nVecPadded == 0 || table.M == 0 {
		return make([]uint16, nVecActual)
	}
	results := make([]uint16, nVecPadded)
	pq4LookupBatch(table.Data, transposedCodes, table.M/2, nVecPadded/32, results)
	return results[:nVecActual]
}

// PQLookupBatchFlat is an optimized PQ8 batch lookup with flat table layout.
// flatTables: contiguous [M * K]float32 (row-major: table[m][k] = flatTables[m*K+k]).
// Eliminates slice header overhead and bounds checks vs [][]float32.
func PQLookupBatchFlat(flatTables []float32, k int, codes []byte, m, nVec int) []float32 {
	distances := make([]float32, nVec)
	pqLookupBatchFlat(flatTables, k, codes, m, nVec, distances)
	return distances
}
