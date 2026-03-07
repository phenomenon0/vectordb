//go:build !amd64

package simd

// Scalar fallback for PQ4 FastScan on non-amd64 architectures.

func pq4LookupBatch(tableData []byte, transposedCodes []byte, halfM, nVec32 int, out []uint16) {
	nVecPadded := nVec32 * 32
	codeStride := nVecPadded

	for batch := 0; batch < nVec32; batch++ {
		for v := 0; v < 32; v++ {
			vecIdx := batch*32 + v
			var acc uint16
			for p := 0; p < halfM; p++ {
				packed := transposedCodes[p*codeStride+vecIdx]
				loNib := packed & 0x0F
				hiNib := (packed >> 4) & 0x0F

				subM := p * 2
				acc += uint16(tableData[subM*16+int(loNib)])
				acc += uint16(tableData[(subM+1)*16+int(hiNib)])
			}
			out[vecIdx] = acc
		}
	}
}

func pqLookupBatchFlat(flatTables []float32, k int, codes []byte, m, nVec int, distances []float32) {
	for i := 0; i < nVec; i++ {
		var d float32
		off := i * m
		for j := 0; j < m; j++ {
			d += flatTables[j*k+int(codes[off+j])]
		}
		distances[i] = d
	}
}
