//go:build amd64

// Package cref provides C reference implementations of SIMD distance functions
// for benchmarking against the hand-written Go assembly in the simd package.
// Build with: go test -tags cref
package cref

/*
#cgo CFLAGS: -O3 -std=c11
#cgo LDFLAGS: -lm

#pragma GCC target("avx2,fma")

#include <immintrin.h>
#include <math.h>
#include <cpuid.h>

// ─── AVX2 Dot Product (4-way unrolled FMA) ────────────────────────────────────
__attribute__((target("avx2,fma")))
static float avx2_dot(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 31 < n; i += 32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), sum3);
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    for (; i + 7 < n; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), sum0);
    }

    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result;
    _mm_store_ss(&result, s);

    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// ─── AVX2 L2 Distance Squared (4-way unrolled) ───────────────────────────────
__attribute__((target("avx2,fma")))
static float avx2_l2sq(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24));
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    for (; i + 7 < n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i));
        sum0 = _mm256_fmadd_ps(d, d, sum0);
    }

    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result;
    _mm_store_ss(&result, s);

    for (; i < n; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

// ─── AVX2 Cosine Distance (2-way unrolled, 3 accumulators) ────────────────────
__attribute__((target("avx2,fma")))
static float avx2_cosine(const float* a, const float* b, int n) {
    __m256 dot0  = _mm256_setzero_ps();
    __m256 dot1  = _mm256_setzero_ps();
    __m256 na0   = _mm256_setzero_ps();
    __m256 na1   = _mm256_setzero_ps();
    __m256 nb0   = _mm256_setzero_ps();
    __m256 nb1   = _mm256_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a+i);
        __m256 vb0 = _mm256_loadu_ps(b+i);
        __m256 va1 = _mm256_loadu_ps(a+i+8);
        __m256 vb1 = _mm256_loadu_ps(b+i+8);
        dot0 = _mm256_fmadd_ps(va0, vb0, dot0);
        dot1 = _mm256_fmadd_ps(va1, vb1, dot1);
        na0  = _mm256_fmadd_ps(va0, va0, na0);
        na1  = _mm256_fmadd_ps(va1, va1, na1);
        nb0  = _mm256_fmadd_ps(vb0, vb0, nb0);
        nb1  = _mm256_fmadd_ps(vb1, vb1, nb1);
    }
    dot0 = _mm256_add_ps(dot0, dot1);
    na0  = _mm256_add_ps(na0, na1);
    nb0  = _mm256_add_ps(nb0, nb1);
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a+i);
        __m256 vb = _mm256_loadu_ps(b+i);
        dot0 = _mm256_fmadd_ps(va, vb, dot0);
        na0  = _mm256_fmadd_ps(va, va, na0);
        nb0  = _mm256_fmadd_ps(vb, vb, nb0);
    }

    __m128 hi, lo, s;

    hi = _mm256_extractf128_ps(dot0, 1);
    lo = _mm256_castps256_ps128(dot0);
    s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float dot_val; _mm_store_ss(&dot_val, s);

    hi = _mm256_extractf128_ps(na0, 1);
    lo = _mm256_castps256_ps128(na0);
    s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float na_val; _mm_store_ss(&na_val, s);

    hi = _mm256_extractf128_ps(nb0, 1);
    lo = _mm256_castps256_ps128(nb0);
    s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float nb_val; _mm_store_ss(&nb_val, s);

    for (; i < n; i++) {
        dot_val += a[i] * b[i];
        na_val  += a[i] * a[i];
        nb_val  += b[i] * b[i];
    }

    if (na_val == 0 || nb_val == 0) return 1.0f;
    return 1.0f - dot_val / (sqrtf(na_val) * sqrtf(nb_val));
}

// Compiler barrier: prevents GCC from optimizing away repeated calls.
// Using "memory" clobber + "+g" to prevent hoisting pure functions out of loops.
#define CLOBBER(x) __asm__ __volatile__("" : "+g"(x) : : "memory")

// ─── Batched wrappers (amortize CGo overhead ~100ns/call) ─────────────────────
__attribute__((target("avx2,fma")))
static float bench_avx2_dot(const float* a, const float* b, int n, int iters) {
    float result = 0;
    for (int i = 0; i < iters; i++) {
        result = avx2_dot(a, b, n);
        CLOBBER(result);
    }
    return result;
}

__attribute__((target("avx2,fma")))
static float bench_avx2_l2sq(const float* a, const float* b, int n, int iters) {
    float result = 0;
    for (int i = 0; i < iters; i++) {
        result = avx2_l2sq(a, b, n);
        CLOBBER(result);
    }
    return result;
}

__attribute__((target("avx2,fma")))
static float bench_avx2_cosine(const float* a, const float* b, int n, int iters) {
    float result = 0;
    for (int i = 0; i < iters; i++) {
        result = avx2_cosine(a, b, n);
        CLOBBER(result);
    }
    return result;
}

// ─── AVX-512 implementations (target attribute — no -mavx512f flag needed) ────

__attribute__((target("avx512f,fma")))
static float avx512_dot(const float* a, const float* b, int n) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    int i = 0;
    for (; i + 63 < n; i += 64) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i),    sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), sum1);
        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), sum2);
        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), sum3);
    }
    sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    for (; i + 15 < n; i += 16) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), sum0);
    }

    float result = _mm512_reduce_add_ps(sum0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

__attribute__((target("avx512f,fma")))
static float avx512_l2sq(const float* a, const float* b, int n) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    int i = 0;
    for (; i + 63 < n; i += 64) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48));
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
        sum2 = _mm512_fmadd_ps(d2, d2, sum2);
        sum3 = _mm512_fmadd_ps(d3, d3, sum3);
    }
    sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    for (; i + 15 < n; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i));
        sum0 = _mm512_fmadd_ps(d, d, sum0);
    }

    float result = _mm512_reduce_add_ps(sum0);
    for (; i < n; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

__attribute__((target("avx512f,fma")))
static float avx512_cosine(const float* a, const float* b, int n) {
    __m512 dot0 = _mm512_setzero_ps();
    __m512 dot1 = _mm512_setzero_ps();
    __m512 na0  = _mm512_setzero_ps();
    __m512 na1  = _mm512_setzero_ps();
    __m512 nb0  = _mm512_setzero_ps();
    __m512 nb1  = _mm512_setzero_ps();

    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a+i);
        __m512 vb0 = _mm512_loadu_ps(b+i);
        __m512 va1 = _mm512_loadu_ps(a+i+16);
        __m512 vb1 = _mm512_loadu_ps(b+i+16);
        dot0 = _mm512_fmadd_ps(va0, vb0, dot0);
        dot1 = _mm512_fmadd_ps(va1, vb1, dot1);
        na0  = _mm512_fmadd_ps(va0, va0, na0);
        na1  = _mm512_fmadd_ps(va1, va1, na1);
        nb0  = _mm512_fmadd_ps(vb0, vb0, nb0);
        nb1  = _mm512_fmadd_ps(vb1, vb1, nb1);
    }
    dot0 = _mm512_add_ps(dot0, dot1);
    na0  = _mm512_add_ps(na0, na1);
    nb0  = _mm512_add_ps(nb0, nb1);
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a+i);
        __m512 vb = _mm512_loadu_ps(b+i);
        dot0 = _mm512_fmadd_ps(va, vb, dot0);
        na0  = _mm512_fmadd_ps(va, va, na0);
        nb0  = _mm512_fmadd_ps(vb, vb, nb0);
    }

    float dot_val = _mm512_reduce_add_ps(dot0);
    float na_val  = _mm512_reduce_add_ps(na0);
    float nb_val  = _mm512_reduce_add_ps(nb0);

    for (; i < n; i++) {
        dot_val += a[i] * b[i];
        na_val  += a[i] * a[i];
        nb_val  += b[i] * b[i];
    }

    if (na_val == 0 || nb_val == 0) return 1.0f;
    return 1.0f - dot_val / (sqrtf(na_val) * sqrtf(nb_val));
}

__attribute__((target("avx512f,fma")))
static float bench_avx512_dot(const float* a, const float* b, int n, int iters) {
    float result = 0;
    for (int i = 0; i < iters; i++) {
        result = avx512_dot(a, b, n);
        CLOBBER(result);
    }
    return result;
}

__attribute__((target("avx512f,fma")))
static float bench_avx512_l2sq(const float* a, const float* b, int n, int iters) {
    float result = 0;
    for (int i = 0; i < iters; i++) {
        result = avx512_l2sq(a, b, n);
        CLOBBER(result);
    }
    return result;
}

__attribute__((target("avx512f,fma")))
static float bench_avx512_cosine(const float* a, const float* b, int n, int iters) {
    float result = 0;
    for (int i = 0; i < iters; i++) {
        result = avx512_cosine(a, b, n);
        CLOBBER(result);
    }
    return result;
}

// ─── CPU feature detection ────────────────────────────────────────────────────
static int check_avx512f(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    return (ebx >> 16) & 1;
}
*/
import "C"

import "unsafe"

// AVX2Dot computes dot product using C AVX2 intrinsics.
func AVX2Dot(a, b []float32) float32 {
	return float32(C.avx2_dot((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a))))
}

// AVX2L2 computes L2 squared distance using C AVX2 intrinsics.
func AVX2L2(a, b []float32) float32 {
	return float32(C.avx2_l2sq((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a))))
}

// AVX2Cosine computes cosine distance using C AVX2 intrinsics.
func AVX2Cosine(a, b []float32) float32 {
	return float32(C.avx2_cosine((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a))))
}

// BatchAVX2Dot runs AVX2 dot product iters times in C (amortizes CGo overhead).
func BatchAVX2Dot(a, b []float32, iters int) float32 {
	return float32(C.bench_avx2_dot((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a)), C.int(iters)))
}

// BatchAVX2L2 runs AVX2 L2 squared iters times in C.
func BatchAVX2L2(a, b []float32, iters int) float32 {
	return float32(C.bench_avx2_l2sq((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a)), C.int(iters)))
}

// BatchAVX2Cosine runs AVX2 cosine distance iters times in C.
func BatchAVX2Cosine(a, b []float32, iters int) float32 {
	return float32(C.bench_avx2_cosine((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a)), C.int(iters)))
}

// BatchAVX512Dot runs AVX-512 dot product iters times in C.
func BatchAVX512Dot(a, b []float32, iters int) float32 {
	return float32(C.bench_avx512_dot((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a)), C.int(iters)))
}

// BatchAVX512L2 runs AVX-512 L2 squared iters times in C.
func BatchAVX512L2(a, b []float32, iters int) float32 {
	return float32(C.bench_avx512_l2sq((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a)), C.int(iters)))
}

// BatchAVX512Cosine runs AVX-512 cosine distance iters times in C.
func BatchAVX512Cosine(a, b []float32, iters int) float32 {
	return float32(C.bench_avx512_cosine((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.int(len(a)), C.int(iters)))
}

// HasAVX512 reports whether the CPU supports AVX-512F.
func HasAVX512() bool {
	return C.check_avx512f() != 0
}
