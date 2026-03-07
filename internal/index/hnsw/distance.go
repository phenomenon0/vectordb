package hnsw

import (
	"reflect"

	"github.com/viterin/vek/vek32"
)

// DistanceFunc is a function that computes the distance between two vectors.
type DistanceFunc func(a, b []float32) float32

// CosineDistance computes the cosine distance between two vectors.
func CosineDistance(a, b []float32) float32 {
	return 1 - vek32.CosineSimilarity(a, b)
}

// EuclideanDistance computes the Euclidean distance between two vectors.
// Uses SIMD-accelerated operations via vek32 for performance.
func EuclideanDistance(a, b []float32) float32 {
	// Use vek32 SIMD operations: subtract vectors then compute norm
	// This is faster than scalar loop for vectors > ~8 elements
	diff := vek32.Sub(a, b)
	return vek32.Norm(diff)
}

var distanceFuncs = map[string]DistanceFunc{
	"euclidean": EuclideanDistance,
	"cosine":    CosineDistance,
}

func distanceFuncToName(fn DistanceFunc) (string, bool) {
	for name, f := range distanceFuncs {
		fnptr := reflect.ValueOf(fn).Pointer()
		fptr := reflect.ValueOf(f).Pointer()
		if fptr == fnptr {
			return name, true
		}
	}
	return "", false
}

// RegisterDistanceFunc registers a distance function with a name.
// A distance function must be registered here before a graph can be
// exported and imported.
func RegisterDistanceFunc(name string, fn DistanceFunc) {
	distanceFuncs[name] = fn
}
