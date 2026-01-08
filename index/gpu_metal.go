//go:build darwin && metal && vectordb_metal
// +build darwin,metal,vectordb_metal

package index

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

int metal_init();
int metal_is_available();

void metal_cosine_distances(
	const float* queries, int num_queries, int dim,
	const float* vectors, int num_vectors,
	float* distances
);

void metal_euclidean_distances(
	const float* queries, int num_queries, int dim,
	const float* vectors, int num_vectors,
	float* distances
);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// InitGPU initializes GPU acceleration for Metal
func InitGPU() error {
	if gpuInitialized {
		return nil
	}

	if C.metal_init() != 0 {
		currentBackend = GPUBackendMetal
		gpuInitialized = true
		return nil
	}

	return fmt.Errorf("Metal initialization failed")
}

func cudaBatchDistances(queries, vectors, distances []float32, numQueries, numVectors, dim int, metric string) error {
	return fmt.Errorf("CUDA not available on this platform")
}

func metalBatchDistances(queries, vectors, distances []float32, numQueries, numVectors, dim int, metric string) error {
	queryPtr := (*C.float)(unsafe.Pointer(&queries[0]))
	vectorPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	distPtr := (*C.float)(unsafe.Pointer(&distances[0]))

	switch metric {
	case "cosine":
		C.metal_cosine_distances(
			queryPtr, C.int(numQueries), C.int(dim),
			vectorPtr, C.int(numVectors),
			distPtr,
		)
	case "euclidean":
		C.metal_euclidean_distances(
			queryPtr, C.int(numQueries), C.int(dim),
			vectorPtr, C.int(numVectors),
			distPtr,
		)
	default:
		return fmt.Errorf("unsupported metric: %s", metric)
	}

	return nil
}

func getCUDAMemoryInfo() (total, free int64) {
	return -1, -1 // Not available on Metal
}
