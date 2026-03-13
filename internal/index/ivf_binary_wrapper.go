package index

import (
	"context"
	"encoding/gob"
	"fmt"
	"sync"
)

// IVFBinaryWrapper wraps IVFBinaryIndex to implement the Index interface
// This provides IVF + Binary quantization through the standard Index API.
//
// Features:
// - 30x memory compression (binary quantization)
// - 10-100x faster search (cluster pruning)
// - 95%+ recall with proper nprobe settings
// - Thread-safe for concurrent reads/writes
type IVFBinaryWrapper struct {
	mu       sync.RWMutex
	index    *IVFBinaryIndex
	dim      int
	trained  bool
	vectors  map[uint64][]float32 // Store original vectors for rescoring (optional)
	keepFull bool                 // Whether to keep full vectors for rescoring
}

// IVFBinarySearchParams are parameters specific to IVF-Binary search
type IVFBinarySearchParams struct {
	NProbe     int  // Number of clusters to search (higher = more accurate, slower)
	Rescore    bool // Whether to rescore with original vectors
	Candidates int  // Number of candidates for rescoring
}

func (IVFBinarySearchParams) Type() string { return "ivf_binary" }

// NewIVFBinaryWrapper creates an IVF-Binary index wrapper
func NewIVFBinaryWrapper(dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	nlist := GetConfigInt(config, "nlist", 100)
	nprobe := GetConfigInt(config, "nprobe", 10)
	keepFull := GetConfigBool(config, "keep_full_vectors", false)

	idx, err := NewIVFBinaryIndex(IVFBinaryConfig{
		Dim:    dim,
		Nlist:  nlist,
		Nprobe: nprobe,
	})
	if err != nil {
		return nil, err
	}

	wrapper := &IVFBinaryWrapper{
		index:    idx,
		dim:      dim,
		trained:  false,
		keepFull: keepFull,
	}

	if keepFull {
		wrapper.vectors = make(map[uint64][]float32)
	}

	return wrapper, nil
}

func (w *IVFBinaryWrapper) Name() string {
	return "IVF-Binary"
}

// Train trains the index on sample vectors
// Must be called before Add() if the index is not pre-trained
func (w *IVFBinaryWrapper) Train(vectors []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.index.Train(vectors); err != nil {
		return err
	}
	w.trained = true
	return nil
}

// TrainFromSamples trains using a map of ID -> vector
func (w *IVFBinaryWrapper) TrainFromSamples(samples map[uint64][]float32) error {
	// Flatten samples into a single slice
	flat := make([]float32, 0, len(samples)*w.dim)
	for _, vec := range samples {
		flat = append(flat, vec...)
	}
	return w.Train(flat)
}

func (w *IVFBinaryWrapper) Add(ctx context.Context, id uint64, vector []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

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
func (w *IVFBinaryWrapper) AddBatch(ids []uint64, vectors []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

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

func (w *IVFBinaryWrapper) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if len(query) != w.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", w.dim, len(query))
	}
	if k <= 0 {
		return []Result{}, nil
	}

	// Extract IVF-Binary specific params
	nprobe := w.index.nprobe
	rescore := false
	candidates := k * 10

	switch ivfParams := params.(type) {
	case IVFBinarySearchParams:
		if ivfParams.NProbe > 0 {
			nprobe = ivfParams.NProbe
		}
		rescore = ivfParams.Rescore && w.keepFull
		if ivfParams.Candidates > 0 {
			candidates = ivfParams.Candidates
		}
	case *IVFBinarySearchParams:
		if ivfParams != nil {
			if ivfParams.NProbe > 0 {
				nprobe = ivfParams.NProbe
			}
			rescore = ivfParams.Rescore && w.keepFull
			if ivfParams.Candidates > 0 {
				candidates = ivfParams.Candidates
			}
		}
	}

	// Temporarily set nprobe for this search
	origNprobe := w.index.nprobe
	w.index.nprobe = nprobe

	var results []SearchResult
	var err error

	if rescore && w.vectors != nil {
		// Use rescoring for better accuracy
		getVector := func(id uint64) []float32 {
			return w.vectors[id]
		}
		results, err = w.index.SearchWithRescore(query, k, candidates, getVector)
	} else {
		results, err = w.index.Search(query, k)
	}

	// Restore original nprobe
	w.index.nprobe = origNprobe

	if err != nil {
		return nil, err
	}

	// Convert to Index.Result format
	indexResults := make([]Result, len(results))
	for i, r := range results {
		indexResults[i] = Result{
			ID:       r.ID,
			Distance: 1 - r.Score, // Convert similarity to distance
			Score:    r.Score,
		}
	}

	return indexResults, nil
}

func (w *IVFBinaryWrapper) Delete(ctx context.Context, id uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	// IVF-Binary doesn't support delete yet, return error
	// TODO: Implement tombstone-based deletion
	return fmt.Errorf("delete not yet implemented for IVF-Binary index")
}

func (w *IVFBinaryWrapper) Stats() IndexStats {
	w.mu.RLock()
	defer w.mu.RUnlock()

	stats := w.index.Stats()
	memUsed := w.index.MemoryUsage()

	// Add full vectors memory if stored
	if w.keepFull && w.vectors != nil {
		memUsed += int64(len(w.vectors) * w.dim * 4)
	}

	return IndexStats{
		Name:       "IVF-Binary",
		Dim:        w.dim,
		Count:      w.index.Size(),
		Deleted:    0, // Not tracked yet
		Active:     w.index.Size(),
		MemoryUsed: memUsed,
		DiskUsed:   0, // In-memory only
		Extra:      stats,
	}
}

// IVFBinarySnapshot for persistence
type IVFBinarySnapshot struct {
	Dim       int
	Nlist     int
	Nprobe    int
	Trained   bool
	KeepFull  bool
	Centroids [][]float32
	Clusters  []clusterSnapshot
	Vectors   map[uint64][]float32 // Optional full vectors
}

type clusterSnapshot struct {
	BinaryData []byte
	IDs        []uint64
	Thresholds []float32
}

func (w *IVFBinaryWrapper) Export() ([]byte, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	// Create snapshot
	snap := IVFBinarySnapshot{
		Dim:       w.dim,
		Nlist:     w.index.nlist,
		Nprobe:    w.index.nprobe,
		Trained:   w.trained,
		KeepFull:  w.keepFull,
		Centroids: w.index.centroids,
		Clusters:  make([]clusterSnapshot, len(w.index.clusters)),
		Vectors:   w.vectors,
	}

	for i, c := range w.index.clusters {
		c.mu.RLock()
		snap.Clusters[i] = clusterSnapshot{
			BinaryData: c.binaryData,
			IDs:        c.ids,
			Thresholds: c.thresholds,
		}
		c.mu.RUnlock()
	}

	// Encode with gob
	var buf []byte
	enc := gob.NewEncoder(&gobBuffer{buf: &buf})
	if err := enc.Encode(snap); err != nil {
		return nil, fmt.Errorf("failed to encode IVF-Binary snapshot: %w", err)
	}

	return buf, nil
}

func (w *IVFBinaryWrapper) Import(data []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	var snap IVFBinarySnapshot
	dec := gob.NewDecoder(&gobReader{data: data})
	if err := dec.Decode(&snap); err != nil {
		return fmt.Errorf("failed to decode IVF-Binary snapshot: %w", err)
	}

	// Recreate index
	idx, err := NewIVFBinaryIndex(IVFBinaryConfig{
		Dim:    snap.Dim,
		Nlist:  snap.Nlist,
		Nprobe: snap.Nprobe,
	})
	if err != nil {
		return err
	}

	// Restore centroids and clusters
	idx.centroids = snap.Centroids
	idx.trained = snap.Trained

	// Rebuild centroid norms
	idx.centroidNorm = make([]float32, len(snap.Centroids))
	for i, c := range snap.Centroids {
		var norm float32
		for _, v := range c {
			norm += v * v
		}
		idx.centroidNorm[i] = norm
	}

	// Restore clusters
	idx.clusters = make([]*clusterData, len(snap.Clusters))
	for i, cs := range snap.Clusters {
		idx.clusters[i] = &clusterData{
			binaryData:  cs.BinaryData,
			ids:         cs.IDs,
			thresholds:  cs.Thresholds,
			bytesPerVec: (snap.Dim + 7) / 8,
		}
	}

	// Count total vectors
	total := 0
	idx.idMap = make(map[uint64]clusterLoc)
	for ci, c := range idx.clusters {
		for pos, id := range c.ids {
			idx.idMap[id] = clusterLoc{cluster: ci, pos: pos}
			total++
		}
	}
	idx.totalVectors = total

	w.index = idx
	w.dim = snap.Dim
	w.trained = snap.Trained
	w.keepFull = snap.KeepFull
	w.vectors = snap.Vectors

	return nil
}

// gobBuffer is a helper for gob encoding to []byte
type gobBuffer struct {
	buf *[]byte
}

func (b *gobBuffer) Write(p []byte) (n int, err error) {
	*b.buf = append(*b.buf, p...)
	return len(p), nil
}

// gobReader is a helper for gob decoding from []byte
type gobReader struct {
	data []byte
	pos  int
}

func (r *gobReader) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		return 0, fmt.Errorf("EOF")
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

// Register IVF-Binary with the global factory
func init() {
	Register("ivf_binary", NewIVFBinaryWrapper)
	Register("ivf-binary", NewIVFBinaryWrapper) // Alternative name
}
