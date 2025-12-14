package index

import (
	"context"
	"fmt"
	"runtime"
	"sync"
)

// ParallelBuilder handles concurrent graph construction for batch insertions
type ParallelBuilder struct {
	maxWorkers int
	batchSize  int
	workerPool *WorkerPool
}

// WorkerPool manages a pool of worker goroutines
type WorkerPool struct {
	workers   int
	taskQueue chan func()
	wg        sync.WaitGroup
	stopOnce  sync.Once
	stopChan  chan struct{}
}

// buildResult holds the result of a parallel edge construction
type buildResult struct {
	id    uint64
	edges []uint64
	err   error
}

// NewParallelBuilder creates a new parallel builder
func NewParallelBuilder(maxWorkers, batchSize int) *ParallelBuilder {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	if batchSize <= 0 {
		batchSize = 100
	}

	pool := &WorkerPool{
		workers:   maxWorkers,
		taskQueue: make(chan func(), maxWorkers*2),
		stopChan:  make(chan struct{}),
	}

	// Start worker goroutines
	for i := 0; i < maxWorkers; i++ {
		pool.wg.Add(1)
		go pool.worker()
	}

	return &ParallelBuilder{
		maxWorkers: maxWorkers,
		batchSize:  batchSize,
		workerPool: pool,
	}
}

// worker processes tasks from the queue
func (wp *WorkerPool) worker() {
	defer wp.wg.Done()

	for {
		select {
		case task := <-wp.taskQueue:
			if task != nil {
				task()
			}
		case <-wp.stopChan:
			return
		}
	}
}

// Submit adds a task to the worker pool
func (wp *WorkerPool) Submit(task func()) {
	select {
	case wp.taskQueue <- task:
	case <-wp.stopChan:
		return
	}
}

// Wait waits for all tasks to complete
func (wp *WorkerPool) Wait() {
	close(wp.taskQueue)
	wp.wg.Wait()
}

// Stop stops the worker pool
func (wp *WorkerPool) Stop() {
	wp.stopOnce.Do(func() {
		close(wp.stopChan)
		wp.wg.Wait()
	})
}

// AddBatch adds multiple vectors in parallel with optimized graph construction
func (d *DiskANNIndex) AddBatch(ctx context.Context, vectors map[uint64][]float32) error {
	if len(vectors) == 0 {
		return nil
	}

	d.mu.Lock()

	// Phase 1: Add vectors to storage (fast, sequential)
	for id, vec := range vectors {
		if len(vec) != d.dim {
			d.mu.Unlock()
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", d.dim, len(vec))
		}

		// Make a copy
		vecCopy := make([]float32, d.dim)
		copy(vecCopy, vec)

		// Decide: memory or disk storage
		if len(d.memoryVectors)+len(d.quantizedMemory) < d.memoryLimit {
			// Store in memory (hot)
			if d.quantizer != nil {
				quantized, err := d.quantizer.Quantize(vecCopy)
				if err != nil {
					d.mu.Unlock()
					return err
				}
				d.quantizedMemory[id] = quantized
			} else {
				d.memoryVectors[id] = vecCopy
			}
		} else {
			// Store on disk (cold)
			if err := d.writeToDisk(id, vecCopy); err != nil {
				d.mu.Unlock()
				return err
			}
		}

		delete(d.deleted, id)
		d.count++
	}

	d.mu.Unlock()

	// Phase 2: Build edges in parallel (slow part)
	return d.buildEdgesParallel(ctx, vectors)
}

// bruteForceKNN finds k nearest neighbors using brute-force distance calculation
// Used for initial batch when graph is empty
func (d *DiskANNIndex) bruteForceKNN(query []float32, vectors map[uint64][]float32, k int) []Result {
	if len(vectors) == 0 {
		return []Result{}
	}

	// Calculate distances to all vectors
	results := make([]Result, 0, len(vectors))
	for id, vec := range vectors {
		dist := d.distance(query, vec)
		results = append(results, Result{ID: id, Distance: dist})
	}

	// Sort by distance
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Distance < results[i].Distance {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Return top k
	if len(results) > k {
		results = results[:k]
	}

	return results
}

// buildEdgesParallel constructs graph edges in parallel
func (d *DiskANNIndex) buildEdgesParallel(ctx context.Context, vectors map[uint64][]float32) error {
	builder := NewParallelBuilder(runtime.NumCPU(), 100)
	defer builder.workerPool.Stop()

	// Check if graph is empty (first batch)
	d.mu.RLock()
	isEmptyGraph := len(d.graph) == 0
	d.mu.RUnlock()

	// Collect IDs for batching
	ids := make([]uint64, 0, len(vectors))
	for id := range vectors {
		ids = append(ids, id)
	}

	// Partition into batches
	batches := partitionIDs(ids, builder.batchSize)

	// Result collection
	resultChan := make(chan buildResult, len(ids))
	errorChan := make(chan error, 1)

	var wg sync.WaitGroup

	// Process batches concurrently
	for _, batch := range batches {
		batch := batch // Capture for closure
		wg.Add(1)

		builder.workerPool.Submit(func() {
			defer wg.Done()

			// Check context cancellation
			select {
			case <-ctx.Done():
				select {
				case errorChan <- ctx.Err():
				default:
				}
				return
			default:
			}

			// Build edges for this batch
			for _, id := range batch {
				vec := vectors[id]

				var candidates []Result

				if isEmptyGraph {
					// Use brute-force KNN for initial batch
					candidates = d.bruteForceKNN(vec, vectors, d.efConstruction)
				} else {
					// Use greedy search for subsequent batches
				// DEADLOCK FIX: Use greedySearchSafe which copies graph snapshot
				// This prevents deadlock when workers hold locks
				candidates = d.greedySearchSafe(vec, d.efConstruction)
				}

				// Build edges
				edges := make([]uint64, 0, d.maxDegree)
				for _, c := range candidates {
					if len(edges) >= d.maxDegree {
						break
					}

					// Skip self-edges
					// Note: deleted check not needed during batch construction
					// since we're building edges for newly added vectors
					if c.ID != id {
						edges = append(edges, c.ID)
					}
				}

				// Send result
				select {
				case resultChan <- buildResult{id: id, edges: edges}:
				case <-ctx.Done():
					return
				}
			}
		})
	}

	// Wait for all workers to finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect all results first (without holding lock)
	var results []buildResult
	for result := range resultChan {
		if result.err != nil {
			return result.err
		}
		results = append(results, result)
	}

	// Check for errors
	select {
	case err := <-errorChan:
		return err
	default:
	}

	// Now update graph with write lock (all results collected)
	d.mu.Lock()
	defer d.mu.Unlock()

	for _, result := range results {
		// Update forward edges
		d.graph[result.id] = result.edges

		// Update backward edges (bidirectional)
		for _, neighborID := range result.edges {
			if neighbors, ok := d.graph[neighborID]; ok {
				// Add edge if not at max degree
				if len(neighbors) < d.maxDegree {
					d.graph[neighborID] = append(neighbors, result.id)
				} else {
					// Replace farthest neighbor
					neighborVec, err := d.getVector(neighborID)
					if err != nil {
						continue
					}
					farthestIdx := d.findFarthest(neighborVec, neighbors)
					d.graph[neighborID][farthestIdx] = result.id
				}
			}
		}
	}

	return nil
}

// partitionIDs splits IDs into batches
func partitionIDs(ids []uint64, batchSize int) [][]uint64 {
	var batches [][]uint64

	for i := 0; i < len(ids); i += batchSize {
		end := i + batchSize
		if end > len(ids) {
			end = len(ids)
		}
		batches = append(batches, ids[i:end])
	}

	return batches
}

// AddIncrementalBatch adds vectors incrementally with background processing
func (d *DiskANNIndex) AddIncrementalBatch(ctx context.Context, vectors map[uint64][]float32, bufferSize int) error {
	// If batch is small, use regular AddBatch
	if len(vectors) < bufferSize {
		return d.AddBatch(ctx, vectors)
	}

	// Split into smaller chunks for incremental processing
	ids := make([]uint64, 0, len(vectors))
	for id := range vectors {
		ids = append(ids, id)
	}

	chunks := partitionIDs(ids, bufferSize)

	for _, chunk := range chunks {
		// Check cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Build chunk map
		chunkVecs := make(map[uint64][]float32)
		for _, id := range chunk {
			chunkVecs[id] = vectors[id]
		}

		// Add chunk
		if err := d.AddBatch(ctx, chunkVecs); err != nil {
			return err
		}
	}

	return nil
}

// ParallelSearchBatch performs multiple searches in parallel
func (d *DiskANNIndex) ParallelSearchBatch(ctx context.Context, queries [][]float32, k int, params SearchParams) ([][]Result, error) {
	if len(queries) == 0 {
		return nil, nil
	}

	builder := NewParallelBuilder(runtime.NumCPU(), 10)
	defer builder.workerPool.Stop()

	results := make([][]Result, len(queries))
	errors := make([]error, len(queries))

	var wg sync.WaitGroup

	for i, query := range queries {
		i, query := i, query // Capture for closure
		wg.Add(1)

		builder.workerPool.Submit(func() {
			defer wg.Done()

			// Check cancellation
			select {
			case <-ctx.Done():
				errors[i] = ctx.Err()
				return
			default:
			}

			// Perform search
			res, err := d.Search(ctx, query, k, params)
			if err != nil {
				errors[i] = err
				return
			}

			results[i] = res
		})
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}
