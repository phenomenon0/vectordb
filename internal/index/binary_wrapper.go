package index

import (
	"context"
	"encoding/gob"
	"fmt"
	"math"
	"sync"
)

// BinaryWrapper wraps BinaryIndex to implement the Index interface
// This provides binary quantization (32x compression) through the standard Index API.
//
// Features:
// - 32x memory compression (1 bit per dimension)
// - Fast Hamming distance using popcount
// - Good for large-scale initial candidate retrieval
// - Best used with rescoring for high recall
type BinaryWrapper struct {
	mu       sync.RWMutex
	index    *BinaryIndex
	dim      int
	vectors  map[uint64][]float32 // Store original vectors for rescoring (optional)
	keepFull bool                 // Whether to keep full vectors for rescoring
}

// BinarySearchParams are parameters specific to Binary index search
type BinarySearchParams struct {
	Rescore    bool // Whether to rescore with original vectors
	Candidates int  // Number of candidates for rescoring
}

func (BinarySearchParams) Type() string { return "binary" }

// NewBinaryWrapper creates a Binary quantized index wrapper
func NewBinaryWrapper(dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	keepFull := GetConfigBool(config, "keep_full_vectors", false)

	idx := NewBinaryIndex(dim)

	wrapper := &BinaryWrapper{
		index:    idx,
		dim:      dim,
		keepFull: keepFull,
	}

	if keepFull {
		wrapper.vectors = make(map[uint64][]float32)
	}

	return wrapper, nil
}

func (w *BinaryWrapper) Name() string {
	return "Binary"
}

// Train trains the quantizer on sample vectors
// This improves recall by learning per-dimension thresholds
func (w *BinaryWrapper) Train(vectors []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.index.Train(vectors)
}

func (w *BinaryWrapper) Add(ctx context.Context, id uint64, vector []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if len(vector) != w.dim {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", w.dim, len(vector))
	}

	// Store original vector if rescoring is enabled
	if w.keepFull {
		vecCopy := make([]float32, len(vector))
		copy(vecCopy, vector)
		w.vectors[id] = vecCopy
	}

	return w.index.Add(id, vector)
}

// AddBatch adds multiple vectors efficiently
func (w *BinaryWrapper) AddBatch(ids []uint64, vectors []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Store original vectors if rescoring is enabled
	if w.keepFull {
		numVecs := len(ids)
		for i := 0; i < numVecs; i++ {
			vecCopy := make([]float32, w.dim)
			copy(vecCopy, vectors[i*w.dim:(i+1)*w.dim])
			w.vectors[ids[i]] = vecCopy
		}
	}

	return w.index.AddBatch(ids, vectors)
}

func (w *BinaryWrapper) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	if len(query) != w.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", w.dim, len(query))
	}
	if k <= 0 {
		return []Result{}, nil
	}

	// Extract Binary specific params
	rescore := false
	candidates := k * 10

	if binParams, ok := params.(BinarySearchParams); ok {
		rescore = binParams.Rescore && w.keepFull
		if binParams.Candidates > 0 {
			candidates = binParams.Candidates
		}
	}

	// Get initial candidates
	searchK := k
	if rescore {
		searchK = candidates
	}

	results, err := w.index.Search(query, searchK)
	if err != nil {
		return nil, err
	}

	// Rescore if enabled
	if rescore && w.vectors != nil && len(results) > k {
		type scored struct {
			id    uint64
			score float32
		}
		rescored := make([]scored, 0, len(results))

		for _, r := range results {
			if vec, ok := w.vectors[r.ID]; ok {
				score := cosineSimFloat32(query, vec)
				rescored = append(rescored, scored{id: r.ID, score: score})
			}
		}

		// Sort by true cosine similarity
		for i := 0; i < len(rescored)-1; i++ {
			for j := i + 1; j < len(rescored); j++ {
				if rescored[j].score > rescored[i].score {
					rescored[i], rescored[j] = rescored[j], rescored[i]
				}
			}
		}

		// Return top-k rescored results
		if k > len(rescored) {
			k = len(rescored)
		}

		indexResults := make([]Result, k)
		for i := 0; i < k; i++ {
			indexResults[i] = Result{
				ID:       rescored[i].id,
				Distance: 1 - rescored[i].score,
				Score:    rescored[i].score,
			}
		}
		return indexResults, nil
	}

	// Convert to Index.Result format without rescoring
	if k > len(results) {
		k = len(results)
	}
	indexResults := make([]Result, k)
	for i := 0; i < k; i++ {
		indexResults[i] = Result{
			ID:       results[i].ID,
			Distance: 1 - results[i].ApproxScore,
			Score:    results[i].ApproxScore,
		}
	}

	return indexResults, nil
}

func (w *BinaryWrapper) Delete(ctx context.Context, id uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Binary index doesn't support delete yet
	return fmt.Errorf("delete not yet implemented for Binary index")
}

func (w *BinaryWrapper) Stats() IndexStats {
	w.mu.RLock()
	defer w.mu.RUnlock()

	memUsed := int64(w.index.MemoryUsage())

	// Add full vectors memory if stored
	if w.keepFull && w.vectors != nil {
		memUsed += int64(len(w.vectors) * w.dim * 4)
	}

	return IndexStats{
		Name:       "Binary",
		Dim:        w.dim,
		Count:      w.index.Size(),
		Deleted:    0,
		Active:     w.index.Size(),
		MemoryUsed: memUsed,
		DiskUsed:   0,
		Extra: map[string]interface{}{
			"compression_ratio": float64(w.index.Size()*w.dim*4) / float64(w.index.MemoryUsage()+1),
			"bytes_per_vector":  w.index.bytesPerVec,
		},
	}
}

// BinarySnapshot for persistence
type BinarySnapshot struct {
	Dim        int
	BinaryData []byte
	IDMap      []uint64
	Thresholds []float32
	Vectors    map[uint64][]float32
	KeepFull   bool
}

func (w *BinaryWrapper) Export() ([]byte, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	snap := BinarySnapshot{
		Dim:        w.dim,
		BinaryData: w.index.binaryData,
		IDMap:      w.index.idMap,
		Thresholds: w.index.quantizer.thresholds,
		Vectors:    w.vectors,
		KeepFull:   w.keepFull,
	}

	var buf []byte
	enc := gob.NewEncoder(&gobBuffer{buf: &buf})
	if err := enc.Encode(snap); err != nil {
		return nil, fmt.Errorf("failed to encode Binary snapshot: %w", err)
	}

	return buf, nil
}

func (w *BinaryWrapper) Import(data []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	var snap BinarySnapshot
	dec := gob.NewDecoder(&gobReader{data: data})
	if err := dec.Decode(&snap); err != nil {
		return fmt.Errorf("failed to decode Binary snapshot: %w", err)
	}

	// Recreate index
	idx := NewBinaryIndex(snap.Dim)
	idx.binaryData = snap.BinaryData
	idx.idMap = snap.IDMap
	idx.quantizer.thresholds = snap.Thresholds
	idx.quantizer.trained = true

	w.index = idx
	w.dim = snap.Dim
	w.keepFull = snap.KeepFull
	w.vectors = snap.Vectors

	return nil
}

// cosineSimFloat32 computes cosine similarity between two vectors
func cosineSimFloat32(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (sqrt32(normA) * sqrt32(normB))
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// Register Binary with the global factory
func init() {
	Register("binary", NewBinaryWrapper)
}
