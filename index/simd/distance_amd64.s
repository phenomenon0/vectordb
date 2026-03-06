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
