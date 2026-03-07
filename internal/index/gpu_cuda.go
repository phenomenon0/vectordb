//go:build cuda && vectordb_cuda
// +build cuda,vectordb_cuda

// NOTE: This file requires vectordb-specific CUDA kernels that are not yet implemented.
// Build with -tags "cuda vectordb_cuda" only if you have the vectordb CUDA kernels compiled.
// For now, the vectordb GPU support is disabled even with -tags cuda.

package index

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcublas -lvectordb_kernels

#include <cuda_runtime.h>

// CUDA kernel for cosine distance batch computation
void cuda_cosine_distances(
	const float* queries, int num_queries, int dim,
	const float* vectors, int num_vectors,
	float* distances,
	int* status
);

// CUDA kernel for euclidean distance batch computation
void cuda_euclidean_distances(
	const float* queries, int num_queries, int dim,
	const float* vectors, int num_vectors,
	float* distances,
	int* status
);

// CUDA kernel for dot product batch computation
void cuda_dot_products(
	const float* queries, int num_queries, int dim,
	const float* vectors, int num_vectors,
	float* results,
	int* status
);

void cuda_init(int* status);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// InitGPU initializes GPU acceleration for CUDA
func InitGPU() error {
	if gpuInitialized {
		return nil
	}

	var status C.int
	C.cuda_init(&status)
	if status == 0 {
		currentBackend = GPUBackendCUDA
		gpuInitialized = true
		return nil
	}

	return fmt.Errorf("CUDA initialization failed: status=%d", status)
}

func cudaBatchDistances(queries, vectors, distances []float32, numQueries, numVectors, dim int, metric string) error {
	var status C.int

	queryPtr := (*C.float)(unsafe.Pointer(&queries[0]))
	vectorPtr := (*C.float)(unsafe.Pointer(&vectors[0]))
	distPtr := (*C.float)(unsafe.Pointer(&distances[0]))

	switch metric {
	case "cosine":
		C.cuda_cosine_distances(
			queryPtr, C.int(numQueries), C.int(dim),
			vectorPtr, C.int(numVectors),
			distPtr,
			&status,
		)
	case "euclidean":
		C.cuda_euclidean_distances(
			queryPtr, C.int(numQueries), C.int(dim),
			vectorPtr, C.int(numVectors),
			distPtr,
			&status,
		)
	default:
		return fmt.Errorf("unsupported metric: %s", metric)
	}

	if status != 0 {
		return fmt.Errorf("CUDA operation failed: status=%d", status)
	}

	return nil
}

func metalBatchDistances(queries, vectors, distances []float32, numQueries, numVectors, dim int, metric string) error {
	return fmt.Errorf("Metal not available on this platform")
}

func getCUDAMemoryInfo() (total, free int64) {
	var freeC, totalC C.size_t
	C.cudaMemGetInfo(&freeC, &totalC)
	return int64(totalC), int64(freeC)
}
