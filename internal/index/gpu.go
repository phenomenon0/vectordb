//go:build (cuda && vectordb_cuda) || (darwin && metal)
// +build cuda,vectordb_cuda darwin,metal

package index

import (
	"fmt"
)

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
	Backend     string
	TotalMemory int64
	FreeMemory  int64
	UsedMemory  int64
	Available   bool
}

// GetGPUMemoryInfo returns current GPU memory statistics
func GetGPUMemoryInfo() GPUMemoryInfo {
	info := GPUMemoryInfo{
		Available: IsGPUAvailable(),
	}

	switch currentBackend {
	case GPUBackendCUDA:
		info.Backend = "CUDA"
		info.TotalMemory, info.FreeMemory = getCUDAMemoryInfo()
		info.UsedMemory = info.TotalMemory - info.FreeMemory

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
