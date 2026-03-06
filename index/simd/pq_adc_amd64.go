//go:build amd64

package simd

import "unsafe"

// pq4FastScanAVX2 is the AVX2 VPSHUFB-based PQ4 FastScan kernel.
// Processes 32 vectors per iteration using parallel shuffle-based table lookups.
//
// Parameters:
//   - tableData: [M * 16]byte quantized subtables, contiguous
//   - transposedCodes: column-major [halfM][nVec32*32]byte
//   - halfM: M/2 (number of subquantizer pairs)
//   - nVec32: number of 32-vector batches
//   - codeStride: bytes between subquantizer columns (= nVecPadded)
//   - out: [nVec32 * 32]uint16 output
//
//go:noescape
func pq4FastScanAVX2(tableData, transposedCodes unsafe.Pointer, halfM, nVec32, codeStride int, out unsafe.Pointer)

func pq4LookupBatch(tableData []byte, transposedCodes []byte, halfM, nVec32 int, out []uint16) {
	if halfM == 0 || nVec32 == 0 {
		return
	}
	codeStride := nVec32 * 32
	pq4FastScanAVX2(
		unsafe.Pointer(&tableData[0]),
		unsafe.Pointer(&transposedCodes[0]),
		halfM,
		nVec32,
		codeStride,
		unsafe.Pointer(&out[0]),
	)
}

// pqLookupBatchFlat uses unsafe pointers to eliminate bounds checks for PQ8 lookup.
func pqLookupBatchFlat(flatTables []float32, k int, codes []byte, m, nVec int, distances []float32) {
	if nVec == 0 || m == 0 {
		return
	}
	tbl := unsafe.Pointer(&flatTables[0])
	codePtr := unsafe.Pointer(&codes[0])
	outPtr := unsafe.Pointer(&distances[0])

	// Process 4 vectors at a time (unrolled)
	i := 0
	for ; i+4 <= nVec; i += 4 {
		var d0, d1, d2, d3 float32
		off0 := uintptr(i) * uintptr(m)
		off1 := off0 + uintptr(m)
		off2 := off1 + uintptr(m)
		off3 := off2 + uintptr(m)

		for j := 0; j < m; j++ {
			tableOff := uintptr(j) * uintptr(k) * 4 // float32 = 4 bytes
			c0 := uintptr(*(*byte)(unsafe.Add(codePtr, off0+uintptr(j))))
			c1 := uintptr(*(*byte)(unsafe.Add(codePtr, off1+uintptr(j))))
			c2 := uintptr(*(*byte)(unsafe.Add(codePtr, off2+uintptr(j))))
			c3 := uintptr(*(*byte)(unsafe.Add(codePtr, off3+uintptr(j))))

			d0 += *(*float32)(unsafe.Add(tbl, tableOff+c0*4))
			d1 += *(*float32)(unsafe.Add(tbl, tableOff+c1*4))
			d2 += *(*float32)(unsafe.Add(tbl, tableOff+c2*4))
			d3 += *(*float32)(unsafe.Add(tbl, tableOff+c3*4))
		}
		*(*float32)(unsafe.Add(outPtr, uintptr(i)*4)) = d0
		*(*float32)(unsafe.Add(outPtr, uintptr(i+1)*4)) = d1
		*(*float32)(unsafe.Add(outPtr, uintptr(i+2)*4)) = d2
		*(*float32)(unsafe.Add(outPtr, uintptr(i+3)*4)) = d3
	}

	// Scalar tail
	for ; i < nVec; i++ {
		var d float32
		off := uintptr(i) * uintptr(m)
		for j := 0; j < m; j++ {
			tableOff := uintptr(j) * uintptr(k) * 4
			c := uintptr(*(*byte)(unsafe.Add(codePtr, off+uintptr(j))))
			d += *(*float32)(unsafe.Add(tbl, tableOff+c*4))
		}
		*(*float32)(unsafe.Add(outPtr, uintptr(i)*4)) = d
	}
}
