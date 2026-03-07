package index

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// IncrementalUpdateConfig controls incremental update behavior
type IncrementalUpdateConfig struct {
	// BufferSize is the number of vectors to buffer before flushing
	BufferSize int
	// FlushInterval is the maximum time to wait before flushing
	FlushInterval time.Duration
	// MaxConcurrentFlushes limits parallel flush operations
	MaxConcurrentFlushes int
	// GraphUpdateStrategy controls how edges are updated
	GraphUpdateStrategy GraphUpdateStrategy
}

// GraphUpdateStrategy defines how new vectors are connected to the graph
type GraphUpdateStrategy int

const (
	// StrategyImmediate builds edges immediately (best accuracy, slowest)
	StrategyImmediate GraphUpdateStrategy = iota
	// StrategyLazy defers edge building to background (fast inserts, eventual consistency)
	StrategyLazy
	// StrategyBatched groups updates for efficient bulk processing
	StrategyBatched
)

// DefaultIncrementalConfig returns sensible defaults
func DefaultIncrementalConfig() IncrementalUpdateConfig {
	return IncrementalUpdateConfig{
		BufferSize:           1000,
		FlushInterval:        5 * time.Second,
		MaxConcurrentFlushes: 2,
		GraphUpdateStrategy:  StrategyBatched,
	}
}

// IncrementalUpdater manages online vector updates for DiskANN
type IncrementalUpdater struct {
	index  *DiskANNIndex
	config IncrementalUpdateConfig

	// Write buffer
	buffer         map[uint64][]float32
	bufferMu       sync.Mutex
	bufferMetadata map[uint64]map[string]interface{}

	// Pending edge builds (for lazy strategy)
	pendingEdges   []uint64
	pendingEdgesMu sync.Mutex

	// Background processing
	flushChan   chan struct{}
	stopChan    chan struct{}
	stoppedChan chan struct{}
	running     atomic.Bool
	flushSem    chan struct{} // Semaphore for concurrent flushes

	// Stats
	stats IncrementalStats
}

// IncrementalStats tracks update statistics
type IncrementalStats struct {
	VectorsBuffered    atomic.Int64
	VectorsFlushed     atomic.Int64
	EdgeBuildsQueued   atomic.Int64
	EdgeBuildsComplete atomic.Int64
	FlushCount         atomic.Int64
	LastFlushDuration  atomic.Int64 // nanoseconds
}

// NewIncrementalUpdater creates a new incremental updater
func NewIncrementalUpdater(index *DiskANNIndex, config IncrementalUpdateConfig) *IncrementalUpdater {
	if config.BufferSize <= 0 {
		config.BufferSize = 1000
	}
	if config.FlushInterval <= 0 {
		config.FlushInterval = 5 * time.Second
	}
	if config.MaxConcurrentFlushes <= 0 {
		config.MaxConcurrentFlushes = 2
	}

	return &IncrementalUpdater{
		index:          index,
		config:         config,
		buffer:         make(map[uint64][]float32),
		bufferMetadata: make(map[uint64]map[string]interface{}),
		pendingEdges:   make([]uint64, 0),
		flushChan:      make(chan struct{}, 1),
		stopChan:       make(chan struct{}),
		stoppedChan:    make(chan struct{}),
		flushSem:       make(chan struct{}, config.MaxConcurrentFlushes),
	}
}

// Start begins background processing
func (u *IncrementalUpdater) Start() {
	if u.running.Swap(true) {
		return // Already running
	}

	go u.backgroundLoop()
}

// Stop halts background processing and flushes remaining data
func (u *IncrementalUpdater) Stop(ctx context.Context) error {
	if !u.running.Swap(false) {
		return nil // Not running
	}

	close(u.stopChan)

	// Wait for background loop to exit
	select {
	case <-u.stoppedChan:
	case <-ctx.Done():
		return ctx.Err()
	}

	// Final flush
	return u.Flush(ctx)
}

// Add adds a vector to the incremental buffer
func (u *IncrementalUpdater) Add(id uint64, vector []float32, metadata map[string]interface{}) error {
	if len(vector) != u.index.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", u.index.dim, len(vector))
	}

	// Make copies
	vecCopy := make([]float32, len(vector))
	copy(vecCopy, vector)

	var metaCopy map[string]interface{}
	if metadata != nil {
		metaCopy = make(map[string]interface{}, len(metadata))
		for k, v := range metadata {
			metaCopy[k] = v
		}
	}

	u.bufferMu.Lock()
	u.buffer[id] = vecCopy
	if metaCopy != nil {
		u.bufferMetadata[id] = metaCopy
	}
	bufferLen := len(u.buffer)
	u.bufferMu.Unlock()

	u.stats.VectorsBuffered.Add(1)

	// Trigger flush if buffer is full
	if bufferLen >= u.config.BufferSize {
		u.triggerFlush()
	}

	return nil
}

// AddBatch adds multiple vectors to the buffer
func (u *IncrementalUpdater) AddBatch(vectors map[uint64][]float32) error {
	u.bufferMu.Lock()
	defer u.bufferMu.Unlock()

	for id, vec := range vectors {
		if len(vec) != u.index.dim {
			return fmt.Errorf("vector %d dimension mismatch: expected %d, got %d", id, u.index.dim, len(vec))
		}

		// Make copy
		vecCopy := make([]float32, len(vec))
		copy(vecCopy, vec)
		u.buffer[id] = vecCopy
	}

	u.stats.VectorsBuffered.Add(int64(len(vectors)))

	// Trigger flush if buffer exceeds threshold
	if len(u.buffer) >= u.config.BufferSize {
		u.triggerFlush()
	}

	return nil
}

// Delete marks a vector for deletion
func (u *IncrementalUpdater) Delete(id uint64) error {
	// Remove from buffer if present
	u.bufferMu.Lock()
	delete(u.buffer, id)
	delete(u.bufferMetadata, id)
	u.bufferMu.Unlock()

	// Mark as deleted in main index
	u.index.mu.Lock()
	u.index.deleted[id] = true
	u.index.mu.Unlock()

	// Remove from pending edges
	u.pendingEdgesMu.Lock()
	newPending := make([]uint64, 0, len(u.pendingEdges))
	for _, pid := range u.pendingEdges {
		if pid != id {
			newPending = append(newPending, pid)
		}
	}
	u.pendingEdges = newPending
	u.pendingEdgesMu.Unlock()

	return nil
}

// Flush writes buffered vectors to the index
func (u *IncrementalUpdater) Flush(ctx context.Context) error {
	// Acquire flush semaphore
	select {
	case u.flushSem <- struct{}{}:
		defer func() { <-u.flushSem }()
	case <-ctx.Done():
		return ctx.Err()
	}

	// Swap out buffer atomically
	u.bufferMu.Lock()
	if len(u.buffer) == 0 {
		u.bufferMu.Unlock()
		return nil
	}

	toFlush := u.buffer
	toFlushMeta := u.bufferMetadata
	u.buffer = make(map[uint64][]float32)
	u.bufferMetadata = make(map[uint64]map[string]interface{})
	u.bufferMu.Unlock()

	start := time.Now()

	// Phase 1: Write vectors to storage
	u.index.mu.Lock()
	ids := make([]uint64, 0, len(toFlush))

	for id, vec := range toFlush {
		// Decide: memory or disk storage
		if len(u.index.memoryVectors)+len(u.index.quantizedMemory) < u.index.memoryLimit {
			if u.index.quantizer != nil {
				quantized, err := u.index.quantizer.Quantize(vec)
				if err != nil {
					u.index.mu.Unlock()
					return fmt.Errorf("quantize failed for %d: %w", id, err)
				}
				u.index.quantizedMemory[id] = quantized
			} else {
				u.index.memoryVectors[id] = vec
			}
		} else {
			if err := u.index.writeToDisk(id, vec); err != nil {
				u.index.mu.Unlock()
				return fmt.Errorf("write to disk failed for %d: %w", id, err)
			}
		}

		delete(u.index.deleted, id)
		u.index.count++
		ids = append(ids, id)
	}
	u.index.mu.Unlock()

	// Set metadata if provided
	for id, meta := range toFlushMeta {
		if meta != nil {
			_ = u.index.SetMetadata(id, meta) // Ignore error, vector is already stored
		}
	}

	// Phase 2: Build edges based on strategy
	switch u.config.GraphUpdateStrategy {
	case StrategyImmediate:
		// Build edges immediately (blocking)
		if err := u.index.buildEdgesParallel(ctx, toFlush); err != nil {
			return fmt.Errorf("edge build failed: %w", err)
		}
		u.stats.EdgeBuildsComplete.Add(int64(len(ids)))

	case StrategyLazy:
		// Queue for background edge building
		u.pendingEdgesMu.Lock()
		u.pendingEdges = append(u.pendingEdges, ids...)
		u.pendingEdgesMu.Unlock()
		u.stats.EdgeBuildsQueued.Add(int64(len(ids)))

	case StrategyBatched:
		// Build edges in batch (non-blocking if small enough)
		if len(ids) < 100 {
			// Small batch: build immediately
			if err := u.index.buildEdgesParallel(ctx, toFlush); err != nil {
				return fmt.Errorf("edge build failed: %w", err)
			}
			u.stats.EdgeBuildsComplete.Add(int64(len(ids)))
		} else {
			// Large batch: queue for background
			u.pendingEdgesMu.Lock()
			u.pendingEdges = append(u.pendingEdges, ids...)
			u.pendingEdgesMu.Unlock()
			u.stats.EdgeBuildsQueued.Add(int64(len(ids)))
		}
	}

	u.stats.VectorsFlushed.Add(int64(len(toFlush)))
	u.stats.FlushCount.Add(1)
	u.stats.LastFlushDuration.Store(time.Since(start).Nanoseconds())

	return nil
}

// triggerFlush signals the background loop to flush
func (u *IncrementalUpdater) triggerFlush() {
	select {
	case u.flushChan <- struct{}{}:
	default:
		// Already triggered
	}
}

// backgroundLoop handles periodic flushing and lazy edge building
func (u *IncrementalUpdater) backgroundLoop() {
	defer close(u.stoppedChan)

	flushTicker := time.NewTicker(u.config.FlushInterval)
	defer flushTicker.Stop()

	edgeTicker := time.NewTicker(1 * time.Second) // Process pending edges every second
	defer edgeTicker.Stop()

	for {
		select {
		case <-u.stopChan:
			return

		case <-u.flushChan:
			_ = u.Flush(context.Background())

		case <-flushTicker.C:
			// Periodic flush
			u.bufferMu.Lock()
			hasData := len(u.buffer) > 0
			u.bufferMu.Unlock()
			if hasData {
				_ = u.Flush(context.Background())
			}

		case <-edgeTicker.C:
			// Process pending edge builds
			u.processPendingEdges(context.Background())
		}
	}
}

// processPendingEdges builds edges for queued vectors
func (u *IncrementalUpdater) processPendingEdges(ctx context.Context) {
	u.pendingEdgesMu.Lock()
	if len(u.pendingEdges) == 0 {
		u.pendingEdgesMu.Unlock()
		return
	}

	// Take a batch
	batchSize := 100
	if len(u.pendingEdges) < batchSize {
		batchSize = len(u.pendingEdges)
	}

	batch := u.pendingEdges[:batchSize]
	u.pendingEdges = u.pendingEdges[batchSize:]
	u.pendingEdgesMu.Unlock()

	// Build edges for batch
	vectors := make(map[uint64][]float32)
	u.index.mu.RLock()
	for _, id := range batch {
		vec, err := u.index.getVector(id)
		if err == nil {
			vectors[id] = vec
		}
	}
	u.index.mu.RUnlock()

	if len(vectors) > 0 {
		_ = u.index.buildEdgesParallel(ctx, vectors)
		u.stats.EdgeBuildsComplete.Add(int64(len(vectors)))
	}
}

// Stats returns current statistics
func (u *IncrementalUpdater) Stats() map[string]interface{} {
	return map[string]interface{}{
		"vectors_buffered":       u.stats.VectorsBuffered.Load(),
		"vectors_flushed":        u.stats.VectorsFlushed.Load(),
		"edge_builds_queued":     u.stats.EdgeBuildsQueued.Load(),
		"edge_builds_complete":   u.stats.EdgeBuildsComplete.Load(),
		"flush_count":            u.stats.FlushCount.Load(),
		"last_flush_duration_ms": u.stats.LastFlushDuration.Load() / 1e6,
		"buffer_size":            u.BufferSize(),
		"pending_edges":          u.PendingEdgeCount(),
	}
}

// BufferSize returns current buffer size
func (u *IncrementalUpdater) BufferSize() int {
	u.bufferMu.Lock()
	defer u.bufferMu.Unlock()
	return len(u.buffer)
}

// PendingEdgeCount returns number of pending edge builds
func (u *IncrementalUpdater) PendingEdgeCount() int {
	u.pendingEdgesMu.Lock()
	defer u.pendingEdgesMu.Unlock()
	return len(u.pendingEdges)
}

// WaitForEdges blocks until all pending edge builds complete
func (u *IncrementalUpdater) WaitForEdges(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if u.PendingEdgeCount() == 0 {
				return nil
			}
			// Trigger processing
			u.processPendingEdges(ctx)
		}
	}
}

// Update performs an in-place update of an existing vector
func (u *IncrementalUpdater) Update(id uint64, vector []float32, metadata map[string]interface{}) error {
	// Check if vector exists
	u.index.mu.RLock()
	_, existsMem := u.index.memoryVectors[id]
	_, existsQuant := u.index.quantizedMemory[id]
	_, existsDisk := u.index.diskOffsetIndex[id]
	_, existsUnquantDisk := u.index.unquantizedOffsetIndex[id]
	exists := existsMem || existsQuant || existsDisk || existsUnquantDisk
	u.index.mu.RUnlock()

	if !exists {
		// Check buffer
		u.bufferMu.Lock()
		_, existsBuffer := u.buffer[id]
		u.bufferMu.Unlock()
		if !existsBuffer {
			return fmt.Errorf("vector %d not found", id)
		}
	}

	// For updates, we delete the old and add the new
	// This is simpler than trying to update in place
	if exists {
		if err := u.Delete(id); err != nil {
			return err
		}
	}

	return u.Add(id, vector, metadata)
}

// Upsert adds or updates a vector
func (u *IncrementalUpdater) Upsert(id uint64, vector []float32, metadata map[string]interface{}) error {
	// Try to delete first (ignore error if not found)
	_ = u.Delete(id)

	// Add new vector
	return u.Add(id, vector, metadata)
}

// Rebuild triggers a full graph rebuild in the background
// This can improve search quality after many incremental updates
func (u *IncrementalUpdater) Rebuild(ctx context.Context) error {
	// First, flush any buffered vectors
	if err := u.Flush(ctx); err != nil {
		return fmt.Errorf("flush failed: %w", err)
	}

	// Wait for pending edge builds
	if err := u.WaitForEdges(ctx); err != nil {
		return fmt.Errorf("wait for edges failed: %w", err)
	}

	// Collect all vectors
	u.index.mu.RLock()
	vectors := make(map[uint64][]float32)

	for id := range u.index.graph {
		if u.index.deleted[id] {
			continue
		}
		vec, err := u.index.getVector(id)
		if err == nil {
			vectors[id] = vec
		}
	}
	u.index.mu.RUnlock()

	// Clear existing graph
	u.index.mu.Lock()
	u.index.graph = make(map[uint64][]uint64)
	u.index.mu.Unlock()

	// Rebuild graph
	return u.index.buildEdgesParallel(ctx, vectors)
}
