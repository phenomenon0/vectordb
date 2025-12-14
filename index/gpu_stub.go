//go:build !cuda && !metal
// +build !cuda,!metal

package index

import "fmt"

// GPUBackend represents the type of GPU backend
type GPUBackend int

const (
	GPUBackendNone GPUBackend = iota
	GPUBackendCUDA
	GPUBackendMetal
)

// InitGPU initializes GPU acceleration (stub)
func InitGPU() error {
	return fmt.Errorf("GPU support not compiled in (build with -tags cuda or -tags metal)")
}

// IsGPUAvailable returns true if GPU acceleration is available (always false for stub)
func IsGPUAvailable() bool {
	return false
}

// GetGPUBackend returns the current GPU backend (always None for stub)
func GetGPUBackend() GPUBackend {
	return GPUBackendNone
}

// GPUBatchDistances stub
func GPUBatchDistances(queries [][]float32, vectors [][]float32, metric string) ([][]float32, error) {
	return nil, fmt.Errorf("GPU support not available")
}

// GPUSingleDistance stub
func GPUSingleDistance(query []float32, vectors [][]float32, metric string) ([]float32, error) {
	return nil, fmt.Errorf("GPU support not available")
}

// GPUBatchThreshold for CPU-only builds
const GPUBatchThreshold = 0

// ShouldUseGPU always returns false for CPU-only builds
func ShouldUseGPU(numQueries, numVectors int) bool {
	return false
}

// GPUMemoryInfo returns GPU memory information
type GPUMemoryInfo struct {
	Backend     string
	TotalMemory int64
	FreeMemory  int64
	UsedMemory  int64
	Available   bool
}

// GetGPUMemoryInfo returns stub info
func GetGPUMemoryInfo() GPUMemoryInfo {
	return GPUMemoryInfo{
		Backend:     "None",
		TotalMemory: 0,
		FreeMemory:  0,
		UsedMemory:  0,
		Available:   false,
	}
}
