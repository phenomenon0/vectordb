#include "textflag.h"

// func pq4FastScanAVX2(tableData, transposedCodes unsafe.Pointer, halfM, nVec32, codeStride int, out unsafe.Pointer)
//
// PQ4 FastScan using AVX2 VPSHUFB for parallel 16-way byte lookup.
// Processes 32 vectors per iteration, 2 subquantizers per loop body.
//
// Arguments:
//   tableData      +0(FP)  [M*16]byte quantized subtables
//   transposedCodes+8(FP)  [halfM*codeStride]byte column-major packed codes
//   halfM          +16(FP) M/2 (subquantizer pairs)
//   nVec32         +24(FP) batches of 32 vectors
//   codeStride     +32(FP) bytes between subquantizer columns (= nVecPadded)
//   out            +40(FP) [nVec32*32]uint16 output
//
// Register allocation:
//   Y0, Y1   = uint16 accumulators (lo/hi 16 vectors each, 32 total)
//   Y2       = zero
//   Y3       = 0x0F nibble mask
//   Y4, Y5   = broadcast subtables
//   Y6       = loaded packed codes
//   Y7       = lo nibbles
//   Y8       = hi nibbles
//   Y9, Y10  = VPSHUFB lookup results
//   Y11, Y12 = temp for unpack/widen
//
//   SI = tableData base
//   DI = transposedCodes (advances per batch)
//   DX = output pointer
//   BX = halfM
//   CX = nVec32 (batch counter)
//   R8 = codeStride (bytes between sub columns)
//   R9 = pair counter per batch
//   R10 = table walker
//   R11 = code column walker
TEXT ·pq4FastScanAVX2(SB), NOSPLIT, $0-48
	MOVQ	tableData+0(FP), SI
	MOVQ	transposedCodes+8(FP), DI
	MOVQ	halfM+16(FP), BX
	MOVQ	nVec32+24(FP), CX
	MOVQ	codeStride+32(FP), R8
	MOVQ	out+40(FP), DX

	// Setup constants
	VPXOR	Y2, Y2, Y2                    // Y2 = zero

	// Load nibble mask: 32 bytes of 0x0F
	VMOVDQU	nibble_mask<>(SB), Y3

	TESTQ	CX, CX
	JZ		done

batch_loop:
	// Zero accumulators for this batch of 32 vectors
	VPXOR	Y0, Y0, Y0                    // accum_lo: 16 x uint16
	VPXOR	Y1, Y1, Y1                    // accum_hi: 16 x uint16

	MOVQ	BX, R9                        // R9 = pair counter
	MOVQ	SI, R10                       // R10 = table pointer (reset per batch)
	MOVQ	DI, R11                       // R11 = code column pointer (for this batch)

	TESTQ	R9, R9
	JZ		store_results

pair_loop:
	// Load subtable for subquantizer m: 16 bytes, broadcast to both 128-bit lanes
	VMOVDQU	(R10), X4                     // X4 = 16 bytes of subtable[m]
	VINSERTI128	$1, X4, Y4, Y4        // Y4 = [tab_m | tab_m] (broadcast)

	// Load subtable for subquantizer m+1
	VMOVDQU	16(R10), X5
	VINSERTI128	$1, X5, Y5, Y5        // Y5 = [tab_m+1 | tab_m+1]

	ADDQ	$32, R10                      // advance table by 2 subtables (32 bytes)

	// Load 32 packed code bytes for this subquantizer pair
	VMOVDQU	(R11), Y6
	ADDQ	R8, R11                       // advance to next sub pair's column

	// Extract lo nibbles (subquantizer m)
	VPAND	Y3, Y6, Y7                    // Y7 = codes & 0x0F

	// Extract hi nibbles (subquantizer m+1)
	VPSRLW	$4, Y6, Y8                    // shift uint16 words right by 4
	VPAND	Y3, Y8, Y8                    // mask to 0x0F (clean up cross-byte bits)

	// VPSHUFB: parallel 16-way byte lookup (32 lookups each)
	VPSHUFB	Y7, Y4, Y9                    // Y9 = 32 uint8 distances for sub m
	VPSHUFB	Y8, Y5, Y10                   // Y10 = 32 uint8 distances for sub m+1

	// Widen sub m results to uint16 and accumulate
	VPUNPCKLBW	Y2, Y9, Y11           // Y11 = lo 16 values zero-extended to uint16
	VPUNPCKHBW	Y2, Y9, Y12           // Y12 = hi 16 values zero-extended to uint16
	VPADDW	Y11, Y0, Y0                   // accum_lo += lo
	VPADDW	Y12, Y1, Y1                   // accum_hi += hi

	// Widen sub m+1 results to uint16 and accumulate
	VPUNPCKLBW	Y2, Y10, Y11
	VPUNPCKHBW	Y2, Y10, Y12
	VPADDW	Y11, Y0, Y0
	VPADDW	Y12, Y1, Y1

	DECQ	R9
	JNZ		pair_loop

store_results:
	// Store 32 uint16 results (2 x 32 bytes = 64 bytes)
	// Y0 has: [v0,v1,...v7 | v16,v17,...v23] as uint16 (from VPUNPCKLBW)
	// Y1 has: [v8,v9,...v15 | v24,v25,...v31] as uint16 (from VPUNPCKHBW)
	// We need to interleave them back to sequential order.
	//
	// Actually, the unpack interleaving means:
	//   Y0 low lane:  vectors 0,1,2,3,4,5,6,7 (from low 8 bytes of each VPSHUFB result)
	//   Y0 high lane: vectors 16,17,18,19,20,21,22,23
	//   Y1 low lane:  vectors 8,9,10,11,12,13,14,15
	//   Y1 high lane: vectors 24,25,26,27,28,29,30,31
	//
	// For correct sequential output [v0..v31]:
	//   out[0..15]  = Y0_lo_lane ++ Y1_lo_lane
	//   out[16..31] = Y0_hi_lane ++ Y1_hi_lane
	//
	// Use VPERM2I128 to rearrange lanes:
	VPERM2I128	$0x20, Y1, Y0, Y13    // Y13 = [Y0_lo | Y1_lo] = vectors 0-15
	VPERM2I128	$0x31, Y1, Y0, Y14    // Y14 = [Y0_hi | Y1_hi] = vectors 16-31

	VMOVDQU	Y13, (DX)                     // store vectors 0-15
	VMOVDQU	Y14, 32(DX)                   // store vectors 16-31

	ADDQ	$64, DX                        // advance output (32 x uint16 = 64 bytes)
	ADDQ	$32, DI                        // advance codes to next batch

	DECQ	CX
	JNZ		batch_loop

done:
	VZEROUPPER
	RET

// 32 bytes of 0x0F for nibble extraction
DATA nibble_mask<>+0(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA nibble_mask<>+8(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA nibble_mask<>+16(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA nibble_mask<>+24(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL nibble_mask<>(SB), RODATA|NOPTR, $32
