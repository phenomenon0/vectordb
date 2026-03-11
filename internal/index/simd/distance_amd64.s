#include "textflag.h"

// func dotProductAVX2(a, b unsafe.Pointer, n int) float32
//
// Computes dot product of n float32 elements using AVX2+FMA.
// n MUST be a multiple of 8. Caller handles the tail.
//
// Uses 4-way unrolling (32 elements per loop iteration) when n >= 32,
// falls back to 8-element loop otherwise.
TEXT ·dotProductAVX2(SB), NOSPLIT, $0-28
	MOVQ	a+0(FP), SI       // SI = &a[0]
	MOVQ	b+8(FP), DI       // DI = &b[0]
	MOVQ	n+16(FP), CX      // CX = n (multiple of 8)

	// Zero 4 accumulators for unrolled loop
	VXORPS	Y0, Y0, Y0        // acc0
	VXORPS	Y1, Y1, Y1        // acc1
	VXORPS	Y2, Y2, Y2        // acc2
	VXORPS	Y3, Y3, Y3        // acc3

	// Check if we can do 4-way unrolled (32 elements per iter)
	MOVQ	CX, DX
	SHRQ	$5, DX             // DX = n/32
	JZ		loop8_setup

	PCALIGN	$32
loop32:
	// Iteration: 32 elements = 4 x 8
	VMOVUPS	0(SI), Y4
	VMOVUPS	0(DI), Y8
	VFMADD231PS	Y4, Y8, Y0

	VMOVUPS	32(SI), Y5
	VMOVUPS	32(DI), Y9
	VFMADD231PS	Y5, Y9, Y1

	VMOVUPS	64(SI), Y6
	VMOVUPS	64(DI), Y10
	VFMADD231PS	Y6, Y10, Y2

	VMOVUPS	96(SI), Y7
	VMOVUPS	96(DI), Y11
	VFMADD231PS	Y7, Y11, Y3

	ADDQ	$128, SI
	ADDQ	$128, DI
	DECQ	DX
	JNZ		loop32

	// Merge 4 accumulators -> 1
	VADDPS	Y1, Y0, Y0
	VADDPS	Y3, Y2, Y2
	VADDPS	Y2, Y0, Y0

loop8_setup:
	// Remaining elements in groups of 8
	ANDQ	$31, CX            // CX = n % 32
	SHRQ	$3, CX             // CX = remaining / 8
	JZ		reduce

loop8:
	VMOVUPS	(SI), Y4
	VMOVUPS	(DI), Y5
	VFMADD231PS	Y4, Y5, Y0
	ADDQ	$32, SI
	ADDQ	$32, DI
	DECQ	CX
	JNZ		loop8

reduce:
	// Horizontal sum of Y0: 8 floats -> 1 float
	VEXTRACTF128	$1, Y0, X1    // X1 = high 128 bits
	VADDPS	X1, X0, X0            // X0 = low + high (4 floats)
	VMOVHLPS	X0, X1, X1        // X1 = [x2, x3, ?, ?]
	VADDPS	X1, X0, X0            // X0 = [x0+x2, x1+x3, ?, ?]
	VPSHUFD	$0x55, X0, X1         // X1 = [x1+x3, x1+x3, x1+x3, x1+x3]
	VADDSS	X1, X0, X0            // X0[0] = final sum
	VMOVSS	X0, ret+24(FP)
	VZEROUPPER
	RET

// func l2DistanceSquaredAVX2(a, b unsafe.Pointer, n int) float32
//
// Computes sum((a[i]-b[i])^2) for n float32 elements using AVX2+FMA.
// n MUST be a multiple of 8.
TEXT ·l2DistanceSquaredAVX2(SB), NOSPLIT, $0-28
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	n+16(FP), CX

	VXORPS	Y0, Y0, Y0        // acc0
	VXORPS	Y1, Y1, Y1        // acc1
	VXORPS	Y2, Y2, Y2        // acc2
	VXORPS	Y3, Y3, Y3        // acc3

	MOVQ	CX, DX
	SHRQ	$5, DX
	JZ		l2_loop8_setup

	PCALIGN	$32
l2_loop32:
	VMOVUPS	0(SI), Y4
	VMOVUPS	0(DI), Y8
	VSUBPS	Y8, Y4, Y4            // diff = a - b
	VFMADD231PS	Y4, Y4, Y0    // acc += diff * diff

	VMOVUPS	32(SI), Y5
	VMOVUPS	32(DI), Y9
	VSUBPS	Y9, Y5, Y5
	VFMADD231PS	Y5, Y5, Y1

	VMOVUPS	64(SI), Y6
	VMOVUPS	64(DI), Y10
	VSUBPS	Y10, Y6, Y6
	VFMADD231PS	Y6, Y6, Y2

	VMOVUPS	96(SI), Y7
	VMOVUPS	96(DI), Y11
	VSUBPS	Y11, Y7, Y7
	VFMADD231PS	Y7, Y7, Y3

	ADDQ	$128, SI
	ADDQ	$128, DI
	DECQ	DX
	JNZ		l2_loop32

	VADDPS	Y1, Y0, Y0
	VADDPS	Y3, Y2, Y2
	VADDPS	Y2, Y0, Y0

l2_loop8_setup:
	ANDQ	$31, CX
	SHRQ	$3, CX
	JZ		l2_reduce

l2_loop8:
	VMOVUPS	(SI), Y4
	VMOVUPS	(DI), Y5
	VSUBPS	Y5, Y4, Y4
	VFMADD231PS	Y4, Y4, Y0
	ADDQ	$32, SI
	ADDQ	$32, DI
	DECQ	CX
	JNZ		l2_loop8

l2_reduce:
	VEXTRACTF128	$1, Y0, X1
	VADDPS	X1, X0, X0
	VMOVHLPS	X0, X1, X1
	VADDPS	X1, X0, X0
	VPSHUFD	$0x55, X0, X1
	VADDSS	X1, X0, X0
	VMOVSS	X0, ret+24(FP)
	VZEROUPPER
	RET

// func cosineComponentsAVX2(a, b unsafe.Pointer, n int) (dot, normA, normB float32)
//
// Computes dot(a,b), dot(a,a), dot(b,b) in a single pass through memory.
// n MUST be a multiple of 8. Caller handles the tail.
//
// Uses 2-way unrolling (16 elements per loop iteration) when n >= 16,
// falls back to 8-element loop otherwise.
//
// Register allocation:
//   Y0, Y1   = dot(a,b) accumulators (2-way)
//   Y2, Y3   = dot(a,a) accumulators (2-way)
//   Y4, Y5   = dot(b,b) accumulators (2-way)
//   Y6, Y7   = a[i] loads
//   Y8, Y9   = b[i] loads
TEXT ·cosineComponentsAVX2(SB), NOSPLIT, $0-36
	MOVQ	a+0(FP), SI       // SI = &a[0]
	MOVQ	b+8(FP), DI       // DI = &b[0]
	MOVQ	n+16(FP), CX      // CX = n (multiple of 8)

	// Zero 6 accumulators
	VXORPS	Y0, Y0, Y0        // dot0
	VXORPS	Y1, Y1, Y1        // dot1
	VXORPS	Y2, Y2, Y2        // normA0
	VXORPS	Y3, Y3, Y3        // normA1
	VXORPS	Y4, Y4, Y4        // normB0
	VXORPS	Y5, Y5, Y5        // normB1

	// Check if we can do 2-way unrolled (16 elements per iter)
	MOVQ	CX, DX
	SHRQ	$4, DX             // DX = n/16
	JZ		cos_loop8_setup

	PCALIGN	$32
cos_loop16:
	// First 8 elements
	VMOVUPS	0(SI), Y6         // a[0..7]
	VMOVUPS	0(DI), Y8         // b[0..7]
	VFMADD231PS	Y6, Y8, Y0    // dot0 += a * b
	VFMADD231PS	Y6, Y6, Y2    // normA0 += a * a
	VFMADD231PS	Y8, Y8, Y4    // normB0 += b * b

	// Second 8 elements
	VMOVUPS	32(SI), Y7        // a[8..15]
	VMOVUPS	32(DI), Y9        // b[8..15]
	VFMADD231PS	Y7, Y9, Y1    // dot1 += a * b
	VFMADD231PS	Y7, Y7, Y3    // normA1 += a * a
	VFMADD231PS	Y9, Y9, Y5    // normB1 += b * b

	ADDQ	$64, SI
	ADDQ	$64, DI
	DECQ	DX
	JNZ		cos_loop16

	// Merge 2-way accumulators -> single set
	VADDPS	Y1, Y0, Y0        // dot
	VADDPS	Y3, Y2, Y2        // normA
	VADDPS	Y5, Y4, Y4        // normB

cos_loop8_setup:
	// Remaining elements in groups of 8
	ANDQ	$15, CX            // CX = n % 16
	SHRQ	$3, CX             // CX = remaining / 8
	JZ		cos_reduce

cos_loop8:
	VMOVUPS	(SI), Y6          // a[i..i+7]
	VMOVUPS	(DI), Y8          // b[i..i+7]
	VFMADD231PS	Y6, Y8, Y0    // dot += a * b
	VFMADD231PS	Y6, Y6, Y2    // normA += a * a
	VFMADD231PS	Y8, Y8, Y4    // normB += b * b
	ADDQ	$32, SI
	ADDQ	$32, DI
	DECQ	CX
	JNZ		cos_loop8

cos_reduce:
	// Horizontal sum of Y0 (dot) -> X0[0]
	VEXTRACTF128	$1, Y0, X1
	VADDPS	X1, X0, X0
	VMOVHLPS	X0, X1, X1
	VADDPS	X1, X0, X0
	VPSHUFD	$0x55, X0, X1
	VADDSS	X1, X0, X0
	VMOVSS	X0, dot+24(FP)

	// Horizontal sum of Y2 (normA) -> X2[0]
	VEXTRACTF128	$1, Y2, X3
	VADDPS	X3, X2, X2
	VMOVHLPS	X2, X3, X3
	VADDPS	X3, X2, X2
	VPSHUFD	$0x55, X2, X3
	VADDSS	X3, X2, X2
	VMOVSS	X2, normA+28(FP)

	// Horizontal sum of Y4 (normB) -> X4[0]
	VEXTRACTF128	$1, Y4, X5
	VADDPS	X5, X4, X4
	VMOVHLPS	X4, X5, X5
	VADDPS	X5, X4, X4
	VPSHUFD	$0x55, X4, X5
	VADDSS	X5, X4, X4
	VMOVSS	X4, normB+32(FP)

	VZEROUPPER
	RET
