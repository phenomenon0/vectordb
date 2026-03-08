package index

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
)

// FLATIndex implements exact (brute-force) vector search
// Optimized for small datasets (<100K vectors) or when 100% recall is required
type FLATIndex struct {
	mu sync.RWMutex

	dim     int                  // Vector dimension
	vectors map[uint64][]float32 // ID -> vector storage (unquantized)
	deleted map[uint64]bool      // Tombstone deletions
	count   int                  // Total vectors added

	metric string // "cosine" or "euclidean"

	// Optional quantization
	quantizer     Quantizer         // Quantizer for compression
	quantizedData map[uint64][]byte // ID -> quantized vector storage
}

// NewFLATIndex creates a new FLAT index
func NewFLATIndex(dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dim)
	}

	metric := GetConfigString(config, "metric", "cosine")

	flat := &FLATIndex{
		dim:     dim,
		vectors: make(map[uint64][]float32),
		deleted: make(map[uint64]bool),
		metric:  metric,
	}

	// Check for quantization config
	if quantConfig, ok := config["quantization"].(map[string]interface{}); ok {
		quantType := GetConfigString(quantConfig, "type", "")

		switch quantType {
		case "float16":
			flat.quantizer = NewFloat16Quantizer(dim)
			flat.quantizedData = make(map[uint64][]byte)
		case "uint8":
			flat.quantizer = NewUint8Quantizer(dim)
			flat.quantizedData = make(map[uint64][]byte)
		case "pq":
			m := GetConfigInt(quantConfig, "m", 8)
			ksub := GetConfigInt(quantConfig, "ksub", 256)
			pq, err := NewProductQuantizer(dim, m, ksub)
			if err != nil {
				return nil, fmt.Errorf("failed to create product quantizer: %w", err)
			}
			flat.quantizer = pq
			flat.quantizedData = make(map[uint64][]byte)
		}
	}

	return flat, nil
}

func init() {
	// Register FLAT index type
	Register("flat", NewFLATIndex)
}

// Name returns the index type name
func (flat *FLATIndex) Name() string {
	return "FLAT"
}

// Add adds a vector to the FLAT index
func (flat *FLATIndex) Add(ctx context.Context, id uint64, vector []float32) error {
	if len(vector) != flat.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", flat.dim, len(vector))
	}

	flat.mu.Lock()
	defer flat.mu.Unlock()

	vecCopy := make([]float32, flat.dim)
	copy(vecCopy, vector)

	if err := flat.storeVectorLocked(id, vecCopy); err != nil {
		return err
	}

	delete(flat.deleted, id)
	flat.count++

	return nil
}

// Search performs exact brute-force search
func (flat *FLATIndex) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	if len(query) != flat.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", flat.dim, len(query))
	}

	flat.mu.RLock()
	defer flat.mu.RUnlock()

	// Collect non-deleted vectors
	activeVectors := make([][]float32, 0, len(flat.vectors)+len(flat.quantizedData))
	activeIDs := make([]uint64, 0, len(flat.vectors)+len(flat.quantizedData))

	for id, vec := range flat.vectors {
		if flat.deleted[id] {
			continue
		}
		activeVectors = append(activeVectors, vec)
		activeIDs = append(activeIDs, id)
	}

	if flat.quantizer != nil {
		for id, quantized := range flat.quantizedData {
			if flat.deleted[id] {
				continue
			}

			vec, err := flat.quantizer.Dequantize(quantized)
			if err != nil {
				return nil, fmt.Errorf("failed to dequantize vector %d: %w", id, err)
			}

			activeVectors = append(activeVectors, vec)
			activeIDs = append(activeIDs, id)
		}
	}

	if len(activeVectors) == 0 {
		return []Result{}, nil
	}

	var distances []float32
	var err error

	// Use GPU if beneficial (>1000 vectors)
	if ShouldUseGPU(1, len(activeVectors)) {
		distances, err = GPUSingleDistance(query, activeVectors, flat.metric)
		if err != nil {
			// Fall back to CPU if GPU fails
			distances = flat.computeDistancesCPU(query, activeVectors)
		}
	} else {
		// CPU computation for small batches
		distances = flat.computeDistancesCPU(query, activeVectors)
	}

	// Build results
	results := make([]Result, len(distances))
	for i, dist := range distances {
		results[i] = Result{
			ID:       activeIDs[i],
			Distance: dist,
			Score:    1.0 / (1.0 + dist),
		}
	}

	// Sort by distance (ascending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// Return top-k
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// computeDistancesCPU computes distances on CPU
func (flat *FLATIndex) computeDistancesCPU(query []float32, vectors [][]float32) []float32 {
	distances := make([]float32, len(vectors))
	for i, vec := range vectors {
		distances[i] = flat.computeDistance(query, vec)
	}
	return distances
}

// Delete marks a vector as deleted
func (flat *FLATIndex) Delete(ctx context.Context, id uint64) error {
	flat.mu.Lock()
	defer flat.mu.Unlock()

	// Check both storage types
	_, existsUnquantized := flat.vectors[id]
	_, existsQuantized := flat.quantizedData[id]

	if !existsUnquantized && !existsQuantized {
		return fmt.Errorf("vector %d not found", id)
	}

	flat.deleted[id] = true
	return nil
}

// Stats returns index statistics
func (flat *FLATIndex) Stats() IndexStats {
	flat.mu.RLock()
	defer flat.mu.RUnlock()

	deletedCount := len(flat.deleted)
	extra := map[string]interface{}{
		"metric": flat.metric,
		"type":   "exact_search",
	}

	// Add quantization info if enabled
	if flat.quantizer != nil {
		extra["quantization"] = flat.quantizer.Type()
		extra["quantized_bytes_per_vector"] = flat.quantizer.BytesPerVector()

		// Calculate compression ratio
		originalSize := flat.dim * 4 // float32 = 4 bytes/value
		compressedSize := flat.quantizer.BytesPerVector()
		extra["compression_ratio"] = float64(originalSize) / float64(compressedSize)
	}

	return IndexStats{
		Name:       "FLAT",
		Dim:        flat.dim,
		Count:      flat.count,
		Deleted:    deletedCount,
		Active:     flat.count - deletedCount,
		MemoryUsed: int64(flat.estimateMemory()),
		DiskUsed:   0,
		Extra:      extra,
	}
}

// Export serializes the FLAT index
func (flat *FLATIndex) Export() ([]byte, error) {
	flat.mu.RLock()
	defer flat.mu.RUnlock()

	type vectorEntry struct {
		ID        uint64    `json:"id"`
		Vector    []float32 `json:"vector,omitempty"`
		Quantized []byte    `json:"quantized,omitempty"`
		Deleted   bool      `json:"deleted,omitempty"`
	}

	type exportFormat struct {
		Version   int             `json:"version"`
		Dim       int             `json:"dim"`
		Count     int             `json:"count"`
		Metric    string          `json:"metric"`
		Quantizer *quantizerState `json:"quantizer,omitempty"`
		Vectors   []vectorEntry   `json:"vectors"`
	}

	quantizerState, err := exportQuantizerState(flat.quantizer)
	if err != nil {
		return nil, err
	}

	ids := flat.sortedIDsLocked()
	vectors := make([]vectorEntry, 0, len(ids))
	for _, id := range ids {
		entry := vectorEntry{
			ID:      id,
			Deleted: flat.deleted[id],
		}
		if vec, ok := flat.vectors[id]; ok {
			entry.Vector = append([]float32(nil), vec...)
		}
		if quantized, ok := flat.quantizedData[id]; ok {
			entry.Quantized = append([]byte(nil), quantized...)
		}
		vectors = append(vectors, entry)
	}

	return json.Marshal(exportFormat{
		Version:   2,
		Dim:       flat.dim,
		Count:     flat.count,
		Metric:    flat.metric,
		Quantizer: quantizerState,
		Vectors:   vectors,
	})
}

// Import deserializes the FLAT index
func (flat *FLATIndex) Import(data []byte) error {
	if len(data) > 0 && data[0] == '{' {
		return flat.importJSON(data)
	}
	return flat.importLegacyBinary(data)
}

func (flat *FLATIndex) importJSON(data []byte) error {
	type vectorEntry struct {
		ID        uint64    `json:"id"`
		Vector    []float32 `json:"vector,omitempty"`
		Quantized []byte    `json:"quantized,omitempty"`
		Deleted   bool      `json:"deleted,omitempty"`
	}

	type importFormat struct {
		Version   int             `json:"version"`
		Dim       int             `json:"dim"`
		Count     int             `json:"count"`
		Metric    string          `json:"metric"`
		Quantizer *quantizerState `json:"quantizer,omitempty"`
		Vectors   []vectorEntry   `json:"vectors"`
	}

	var imp importFormat
	if err := json.Unmarshal(data, &imp); err != nil {
		return fmt.Errorf("failed to unmarshal FLAT index: %w", err)
	}
	if imp.Version != 2 {
		return fmt.Errorf("unsupported FLAT export version: %d", imp.Version)
	}

	quantizer, err := importQuantizerState(imp.Dim, imp.Quantizer)
	if err != nil {
		return err
	}

	flat.mu.Lock()
	defer flat.mu.Unlock()

	flat.dim = imp.Dim
	flat.count = imp.Count
	flat.metric = imp.Metric
	flat.quantizer = quantizer
	flat.vectors = make(map[uint64][]float32, len(imp.Vectors))
	flat.deleted = make(map[uint64]bool, len(imp.Vectors))
	if quantizer != nil {
		flat.quantizedData = make(map[uint64][]byte, len(imp.Vectors))
	} else {
		flat.quantizedData = nil
	}

	for _, entry := range imp.Vectors {
		if len(entry.Vector) > 0 {
			flat.vectors[entry.ID] = append([]float32(nil), entry.Vector...)
		}
		if len(entry.Quantized) > 0 {
			if flat.quantizedData == nil {
				flat.quantizedData = make(map[uint64][]byte, len(imp.Vectors))
			}
			flat.quantizedData[entry.ID] = append([]byte(nil), entry.Quantized...)
		}
		if entry.Deleted {
			flat.deleted[entry.ID] = true
		}
	}

	return nil
}

func (flat *FLATIndex) importLegacyBinary(data []byte) error {
	if len(data) < 16 {
		return fmt.Errorf("invalid FLAT index data: too short")
	}

	flat.mu.Lock()
	defer flat.mu.Unlock()

	offset := 0

	readUint32 := func() uint32 {
		v := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4
		return v
	}

	// Header
	flat.dim = int(readUint32())
	flat.count = int(readUint32())

	// Metric
	metricLen := int(readUint32())
	flat.metric = string(data[offset : offset+metricLen])
	offset += metricLen

	// Initialize maps
	numVectors := int(readUint32())
	flat.vectors = make(map[uint64][]float32, numVectors)
	flat.deleted = make(map[uint64]bool)
	flat.quantizer = nil
	flat.quantizedData = nil

	// Vectors
	for i := 0; i < numVectors; i++ {
		// ID
		id := binary.LittleEndian.Uint64(data[offset : offset+8])
		offset += 8

		// Vector data
		vec := make([]float32, flat.dim)
		for j := 0; j < flat.dim; j++ {
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			vec[j] = math.Float32frombits(bits)
			offset += 4
		}
		flat.vectors[id] = vec

		// Deleted flag
		if data[offset] == 1 {
			flat.deleted[id] = true
		}
		offset++
	}

	return nil
}

// Helper methods

// computeDistance computes distance between two vectors based on metric
func (flat *FLATIndex) computeDistance(a, b []float32) float32 {
	if flat.metric == "euclidean" {
		return euclideanDistanceFlat(a, b)
	}
	// Default: cosine distance
	return cosineDistanceFlat(a, b)
}

// estimateMemory estimates memory usage in bytes
func (flat *FLATIndex) estimateMemory() int {
	vectorMem := len(flat.vectors) * (8 + flat.dim*4)
	for _, data := range flat.quantizedData {
		vectorMem += 8 + len(data)
	}

	// Deleted map: ID (8 bytes) + bool (1 byte) overhead
	deletedMem := len(flat.deleted) * 9

	return vectorMem + deletedMem
}

func (flat *FLATIndex) storeVectorLocked(id uint64, vec []float32) error {
	if flat.quantizer == nil {
		flat.vectors[id] = vec
		return nil
	}

	if quantizerNeedsTraining(flat.quantizer) {
		flat.vectors[id] = vec
		return flat.maybeTrainQuantizerLocked()
	}

	quantized, err := flat.quantizer.Quantize(vec)
	if err != nil {
		return fmt.Errorf("failed to quantize vector: %w", err)
	}
	if flat.quantizedData == nil {
		flat.quantizedData = make(map[uint64][]byte)
	}
	flat.quantizedData[id] = quantized
	delete(flat.vectors, id)
	return nil
}

func (flat *FLATIndex) maybeTrainQuantizerLocked() error {
	if flat.quantizer == nil || !quantizerNeedsTraining(flat.quantizer) {
		return nil
	}
	if len(flat.vectors) < quantizerMinTrainingVectors(flat.quantizer) {
		return nil
	}

	if err := trainQuantizerWithDefaults(flat.quantizer, flat.flattenVectorsLocked()); err != nil {
		return fmt.Errorf("failed to train quantizer: %w", err)
	}
	if flat.quantizedData == nil {
		flat.quantizedData = make(map[uint64][]byte, len(flat.vectors))
	}
	for _, id := range flat.sortedVectorIDsLocked() {
		quantized, err := flat.quantizer.Quantize(flat.vectors[id])
		if err != nil {
			return fmt.Errorf("failed to quantize vector %d: %w", id, err)
		}
		flat.quantizedData[id] = quantized
		delete(flat.vectors, id)
	}

	return nil
}

func (flat *FLATIndex) flattenVectorsLocked() []float32 {
	ids := flat.sortedVectorIDsLocked()
	flattened := make([]float32, 0, len(ids)*flat.dim)
	for _, id := range ids {
		flattened = append(flattened, flat.vectors[id]...)
	}
	return flattened
}

func (flat *FLATIndex) sortedVectorIDsLocked() []uint64 {
	ids := make([]uint64, 0, len(flat.vectors))
	for id := range flat.vectors {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

func (flat *FLATIndex) sortedIDsLocked() []uint64 {
	idSet := make(map[uint64]struct{}, len(flat.vectors)+len(flat.quantizedData))
	for id := range flat.vectors {
		idSet[id] = struct{}{}
	}
	for id := range flat.quantizedData {
		idSet[id] = struct{}{}
	}

	ids := make([]uint64, 0, len(idSet))
	for id := range idSet {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// ExportJSON exports index metadata as JSON (for debugging)
func (flat *FLATIndex) ExportJSON() ([]byte, error) {
	flat.mu.RLock()
	defer flat.mu.RUnlock()

	export := map[string]interface{}{
		"type":   "flat",
		"dim":    flat.dim,
		"count":  flat.count,
		"metric": flat.metric,
		"stats":  flat.Stats(),
	}

	return json.Marshal(export)
}

// Distance functions for FLAT index

func euclideanDistanceFlat(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func cosineDistanceFlat(a, b []float32) float32 {
	var dot, normA, normB float32

	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - similarity
}
