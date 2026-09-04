// Package simd provides SIMD-accelerated distance functions for vector search.
//
// On amd64, uses AVX2+FMA instructions (8 float32s per cycle).
// Falls back to scalar Go loops on other architectures.
//
// Tower layer: L1 engine.
package simd
