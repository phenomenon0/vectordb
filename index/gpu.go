//go:build cuda || metal
// +build cuda metal

package index

import (
	"fmt"
	"math"
	"unsafe"
)

/*
#cgo cuda CFLAGS: -DUSE_CUDA
#cgo cuda LDFLAGS: -lcudart -lcublas
#cgo darwin,metal CFLAGS: -x objective-c -DUSE_METAL
#cgo darwin,metal LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
#endif

#ifdef USE_METAL
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

// Metal device and queue (initialized once)
static id<MTLDevice> metalDevice = NULL;
static id<MTLCommandQueue> metalQueue = NULL;

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
#endif
*/
import "C"

// GPUBackend represents the type of GPU backend
type GPUBackend int

const (
	GPUBackendNone GPUBackend = iota
	GPUBackendCUDA
	GPUBackendMetal
)

var (
	currentBackend GPUBackend
	gpuInitialized bool
)

// InitGPU initializes GPU acceleration
func InitGPU() error {
	if gpuInitialized {
		return nil
	}

	#ifdef USE_CUDA
	// Try CUDA first
	var status C.int
	C.cuda_init(&status)
	if status == 0 {
		currentBackend = GPUBackendCUDA
		gpuInitialized = true
		return nil
	}
	#endif

	#ifdef USE_METAL
	// Fall back to Metal on macOS
	if C.metal_init() != 0 {
		currentBackend = GPUBackendMetal
		gpuInitialized = true
		return nil
	}
	#endif

	return fmt.Errorf("no GPU backend available")
}

// IsGPUAvailable returns true if GPU acceleration is available
func IsGPUAvailable() bool {
	if !gpuInitialized {
		InitGPU()
	}
	return currentBackend != GPUBackendNone
}

// GetGPUBackend returns the current GPU backend
func GetGPUBackend() GPUBackend {
	return currentBackend
}

// GPUBatchDistances computes distances between queries and vectors using GPU
// queries: [num_queries, dim]
// vectors: [num_vectors, dim]
// returns: [num_queries, num_vectors] distance matrix
func GPUBatchDistances(queries [][]float32, vectors [][]float32, metric string) ([][]float32, error) {
	if !IsGPUAvailable() {
		return nil, fmt.Errorf("GPU not available")
	}

	if len(queries) == 0 || len(vectors) == 0 {
		return nil, fmt.Errorf("empty input")
	}

	numQueries := len(queries)
	numVectors := len(vectors)
	dim := len(queries[0])

	// Flatten queries and vectors
	flatQueries := make([]float32, numQueries*dim)
	flatVectors := make([]float32, numVectors*dim)

	for i, q := range queries {
		copy(flatQueries[i*dim:(i+1)*dim], q)
	}
	for i, v := range vectors {
		copy(flatVectors[i*dim:(i+1)*dim], v)
	}

	// Allocate output
	distances := make([]float32, numQueries*numVectors)

	switch currentBackend {
	case GPUBackendCUDA:
		if err := cudaBatchDistances(flatQueries, flatVectors, distances, numQueries, numVectors, dim, metric); err != nil {
			return nil, err
		}

	case GPUBackendMetal:
		if err := metalBatchDistances(flatQueries, flatVectors, distances, numQueries, numVectors, dim, metric); err != nil {
			return nil, err
		}

	default:
		return nil, fmt.Errorf("no GPU backend")
	}

	// Unflatten results
	result := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		result[i] = distances[i*numVectors : (i+1)*numVectors]
	}

	return result, nil
}

#ifdef USE_CUDA
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
#else
func cudaBatchDistances(queries, vectors, distances []float32, numQueries, numVectors, dim int, metric string) error {
	return fmt.Errorf("CUDA not available")
}
#endif

#ifdef USE_METAL
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
#else
func metalBatchDistances(queries, vectors, distances []float32, numQueries, numVectors, dim int, metric string) error {
	return fmt.Errorf("Metal not available")
}
#endif

// GPUSingleDistance computes distance between single query and multiple vectors using GPU
func GPUSingleDistance(query []float32, vectors [][]float32, metric string) ([]float32, error) {
	if !IsGPUAvailable() {
		return nil, fmt.Errorf("GPU not available")
	}

	queries := [][]float32{query}
	distMatrix, err := GPUBatchDistances(queries, vectors, metric)
	if err != nil {
		return nil, err
	}

	return distMatrix[0], nil
}

// GPUBatchThreshold determines minimum batch size for GPU to be beneficial
// Below this threshold, CPU is faster due to transfer overhead
const GPUBatchThreshold = 1000

// ShouldUseGPU determines if GPU should be used based on batch size
func ShouldUseGPU(numQueries, numVectors int) bool {
	if !IsGPUAvailable() {
		return false
	}

	// GPU beneficial for large batches
	totalOps := numQueries * numVectors
	return totalOps >= GPUBatchThreshold
}

// GPUMemoryInfo returns GPU memory information
type GPUMemoryInfo struct {
	Backend      string
	TotalMemory  int64
	FreeMemory   int64
	UsedMemory   int64
	Available    bool
}

// GetGPUMemoryInfo returns current GPU memory statistics
func GetGPUMemoryInfo() GPUMemoryInfo {
	info := GPUMemoryInfo{
		Available: IsGPUAvailable(),
	}

	switch currentBackend {
	case GPUBackendCUDA:
		info.Backend = "CUDA"
		#ifdef USE_CUDA
		// Get CUDA memory info
		var free, total C.size_t
		C.cudaMemGetInfo(&free, &total)
		info.TotalMemory = int64(total)
		info.FreeMemory = int64(free)
		info.UsedMemory = info.TotalMemory - info.FreeMemory
		#endif

	case GPUBackendMetal:
		info.Backend = "Metal"
		// Metal doesn't expose memory info as directly
		info.TotalMemory = -1
		info.FreeMemory = -1
		info.UsedMemory = -1

	default:
		info.Backend = "None"
	}

	return info
}
