package index

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

// CompactionStats tracks compaction metrics
type CompactionStats struct {
	StartTime        time.Time
	EndTime          time.Time
	Duration         time.Duration
	VectorsBefore    int
	VectorsAfter     int
	VectorsRemoved   int
	DiskBytesBefore  int64
	DiskBytesAfter   int64
	SpaceReclaimed   int64
	FragmentationPct float64
}

// CompactionConfig controls compaction behavior
type CompactionConfig struct {
	// MinFragmentation triggers compaction when fragmentation exceeds this percentage (default: 20%)
	MinFragmentation float64
	// MinDeletedVectors triggers compaction when deleted count exceeds this (default: 1000)
	MinDeletedVectors int
	// BackgroundMode runs compaction in background without blocking (default: true)
	BackgroundMode bool
}

// DefaultCompactionConfig returns sensible defaults
func DefaultCompactionConfig() CompactionConfig {
	return CompactionConfig{
		MinFragmentation:  20.0,
		MinDeletedVectors: 1000,
		BackgroundMode:    true,
	}
}

// NeedsCompaction checks if compaction should be triggered
func (d *DiskANNIndex) NeedsCompaction(config CompactionConfig) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()

	deletedCount := len(d.deleted)

	// Check deleted vector threshold
	if deletedCount >= config.MinDeletedVectors {
		return true
	}

	// Check fragmentation percentage
	if d.count > 0 {
		fragmentation := float64(deletedCount) / float64(d.count) * 100
		if fragmentation >= config.MinFragmentation {
			return true
		}
	}

	return false
}

// Compact rebuilds the index files without deleted entries
// This reclaims disk space and improves performance
func (d *DiskANNIndex) Compact(ctx context.Context) (*CompactionStats, error) {
	stats := &CompactionStats{
		StartTime: time.Now(),
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	stats.VectorsBefore = d.count
	stats.DiskBytesBefore = d.mmapOffset
	deletedCount := len(d.deleted)
	stats.VectorsRemoved = deletedCount

	// Calculate fragmentation
	if d.count > 0 {
		stats.FragmentationPct = float64(deletedCount) / float64(d.count) * 100
	}

	// Nothing to compact if no deleted vectors
	if deletedCount == 0 {
		stats.EndTime = time.Now()
		stats.Duration = stats.EndTime.Sub(stats.StartTime)
		stats.VectorsAfter = stats.VectorsBefore
		stats.DiskBytesAfter = stats.DiskBytesBefore
		return stats, nil
	}

	// Create temporary file for compacted data
	tempPath := d.indexPath + ".compact.tmp"
	tempFile, err := os.Create(tempPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() {
		tempFile.Close()
		if err != nil {
			os.Remove(tempPath)
		}
	}()

	// Initialize temp file with initial size
	initialSize := d.mmapOffset
	if err := tempFile.Truncate(initialSize); err != nil {
		return nil, fmt.Errorf("failed to resize temp file: %w", err)
	}

	// Memory-map the temp file
	tempData, err := mmapCreate(int(tempFile.Fd()), int(initialSize))
	if err != nil {
		return nil, fmt.Errorf("failed to mmap temp file: %w", err)
	}
	defer mmapUnmap(tempData)

	// Copy non-deleted vectors to temp file
	newOffset := int64(0)
	newOffsetIndex := make(map[uint64]int64)

	// Collect all vector IDs from all sources
	allIDs := make(map[uint64]bool)
	for id := range d.graph {
		allIDs[id] = true
	}
	for id := range d.memoryVectors {
		allIDs[id] = true
	}
	for id := range d.quantizedMemory {
		allIDs[id] = true
	}
	if d.quantizer != nil {
		for id := range d.diskOffsetIndex {
			allIDs[id] = true
		}
	} else if d.mmapData != nil && d.mmapOffset > 0 {
		// For unquantized disk vectors, scan mmap to find IDs
		recordSize := int64(8 + d.dim*4)
		for offset := int64(0); offset < d.mmapOffset; offset += recordSize {
			if offset+8 > int64(len(d.mmapData)) {
				break
			}
			id := binary.LittleEndian.Uint64(d.mmapData[offset:])
			allIDs[id] = true
		}
	}

	// Iterate through all vectors
	for id := range allIDs {
		// Skip deleted vectors
		if d.deleted[id] {
			continue
		}

		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Read vector from current mmap or memory
		var vec []float32
		var err error

		// Try memory first
		if v, ok := d.memoryVectors[id]; ok {
			vec = v
		} else if quantized, ok := d.quantizedMemory[id]; ok {
			vec, err = d.quantizer.Dequantize(quantized)
			if err != nil {
				return nil, fmt.Errorf("failed to dequantize vector %d: %w", id, err)
			}
		} else {
			// Read from disk
			vec, err = d.readFromDisk(id)
			if err != nil {
				return nil, fmt.Errorf("failed to read vector %d: %w", id, err)
			}
		}

		// Write to temp file
		if d.quantizer != nil {
			// Write quantized data
			quantized, err := d.quantizer.Quantize(vec)
			if err != nil {
				return nil, fmt.Errorf("failed to quantize vector %d: %w", id, err)
			}

			// Ensure space
			recordSize := int64(8 + 4 + len(quantized)) // ID + length + data
			if newOffset+recordSize > int64(len(tempData)) {
				// Need to grow temp mmap
				mmapUnmap(tempData)
				newSize := newOffset * 2
				if err := tempFile.Truncate(newSize); err != nil {
					return nil, fmt.Errorf("failed to grow temp file: %w", err)
				}
				tempData, err = mmapCreate(int(tempFile.Fd()), int(newSize))
				if err != nil {
					return nil, fmt.Errorf("failed to remap temp file: %w", err)
				}
			}

			// Write: ID + length + quantized data
			binary.LittleEndian.PutUint64(tempData[newOffset:], id)
			binary.LittleEndian.PutUint32(tempData[newOffset+8:], uint32(len(quantized)))
			copy(tempData[newOffset+12:], quantized)

			newOffsetIndex[id] = newOffset
			newOffset += recordSize
		} else {
			// Write unquantized data
			recordSize := int64(8 + d.dim*4) // ID + vector
			if newOffset+recordSize > int64(len(tempData)) {
				mmapUnmap(tempData)
				newSize := newOffset * 2
				if err := tempFile.Truncate(newSize); err != nil {
					return nil, fmt.Errorf("failed to grow temp file: %w", err)
				}
				tempData, err = mmapCreate(int(tempFile.Fd()), int(newSize))
				if err != nil {
					return nil, fmt.Errorf("failed to remap temp file: %w", err)
				}
			}

			// Write: ID + vector
			newOffsetIndex[id] = newOffset
			binary.LittleEndian.PutUint64(tempData[newOffset:], id)
			for i, val := range vec {
				bits := math.Float32bits(val)
				binary.LittleEndian.PutUint32(tempData[newOffset+8+int64(i*4):], bits)
			}

			newOffset += recordSize
		}
	}

	// Sync temp file
	if err := mmapUnmap(tempData); err != nil {
		return nil, fmt.Errorf("failed to unmap temp file: %w", err)
	}
	if err := tempFile.Sync(); err != nil {
		return nil, fmt.Errorf("failed to sync temp file: %w", err)
	}
	tempFile.Close()

	// Truncate to actual size
	if err := os.Truncate(tempPath, newOffset); err != nil {
		return nil, fmt.Errorf("failed to truncate temp file: %w", err)
	}

	// Unmap old file
	if d.mmapData != nil {
		if err := mmapUnmap(d.mmapData); err != nil {
			return nil, fmt.Errorf("failed to unmap old file: %w", err)
		}
		d.mmapData = nil
	}
	if d.mmapFile != nil {
		d.mmapFile.Close()
		d.mmapFile = nil
	}

	// Atomic rename (replace old file with compacted file)
	backupPath := d.indexPath + ".backup"
	if err := os.Rename(d.indexPath, backupPath); err != nil {
		return nil, fmt.Errorf("failed to backup old file: %w", err)
	}
	if err := os.Rename(tempPath, d.indexPath); err != nil {
		// Try to restore backup
		os.Rename(backupPath, d.indexPath)
		return nil, fmt.Errorf("failed to rename compacted file: %w", err)
	}
	os.Remove(backupPath) // Remove backup on success

	// Reinitialize mmap with compacted file
	file, err := os.OpenFile(d.indexPath, os.O_RDWR, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open compacted file: %w", err)
	}

	fileInfo, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to stat compacted file: %w", err)
	}

	if fileInfo.Size() > 0 {
		mmapData, err := mmapCreate(int(file.Fd()), int(fileInfo.Size()))
		if err != nil {
			file.Close()
			return nil, fmt.Errorf("failed to mmap compacted file: %w", err)
		}
		d.mmapData = mmapData
	}

	d.mmapFile = file
	d.mmapOffset = newOffset
	if d.quantizer != nil {
		d.diskOffsetIndex = newOffsetIndex
	} else {
		d.unquantizedOffsetIndex = newOffsetIndex
	}

	// Clear deleted vectors map and graph entries
	for id := range d.deleted {
		delete(d.memoryVectors, id)
		delete(d.quantizedMemory, id)
		delete(d.graph, id) // Also remove from graph
	}
	d.deleted = make(map[uint64]bool)
	d.count -= deletedCount

	// Update stats
	stats.VectorsAfter = d.count
	stats.DiskBytesAfter = newOffset
	stats.SpaceReclaimed = stats.DiskBytesBefore - stats.DiskBytesAfter
	stats.EndTime = time.Now()
	stats.Duration = stats.EndTime.Sub(stats.StartTime)

	return stats, nil
}

// BackgroundCompaction runs compaction in a background goroutine
// It periodically checks if compaction is needed and runs it if necessary
type BackgroundCompaction struct {
	index       *DiskANNIndex
	config      CompactionConfig
	interval    time.Duration
	stopChan    chan struct{}
	stoppedChan chan struct{}
	mu          sync.Mutex
	running     bool
	lastStats   *CompactionStats
}

// NewBackgroundCompaction creates a background compaction manager
func NewBackgroundCompaction(index *DiskANNIndex, config CompactionConfig, interval time.Duration) *BackgroundCompaction {
	if interval <= 0 {
		interval = 5 * time.Minute // Default: check every 5 minutes
	}

	return &BackgroundCompaction{
		index:       index,
		config:      config,
		interval:    interval,
		stopChan:    make(chan struct{}),
		stoppedChan: make(chan struct{}),
	}
}

// Start begins background compaction monitoring
func (bc *BackgroundCompaction) Start() {
	bc.mu.Lock()
	if bc.running {
		bc.mu.Unlock()
		return
	}
	bc.running = true
	bc.mu.Unlock()

	go bc.run()
}

// Stop halts background compaction
func (bc *BackgroundCompaction) Stop() {
	bc.mu.Lock()
	if !bc.running {
		bc.mu.Unlock()
		return
	}
	bc.running = false
	bc.mu.Unlock()

	close(bc.stopChan)
	<-bc.stoppedChan
}

// run is the background monitoring loop
func (bc *BackgroundCompaction) run() {
	defer close(bc.stoppedChan)

	ticker := time.NewTicker(bc.interval)
	defer ticker.Stop()

	for {
		select {
		case <-bc.stopChan:
			return
		case <-ticker.C:
			if bc.index.NeedsCompaction(bc.config) {
				stats, err := bc.index.Compact(context.Background())
				if err == nil {
					bc.mu.Lock()
					bc.lastStats = stats
					bc.mu.Unlock()
				}
			}
		}
	}
}

// LastStats returns the most recent compaction statistics
func (bc *BackgroundCompaction) LastStats() *CompactionStats {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	return bc.lastStats
}

// EstimateFragmentation estimates disk fragmentation percentage
func (d *DiskANNIndex) EstimateFragmentation() float64 {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if d.count == 0 {
		return 0
	}

	deletedCount := len(d.deleted)
	return float64(deletedCount) / float64(d.count) * 100
}

// DiskSpaceUsage returns current disk space usage and reclaimable space
func (d *DiskANNIndex) DiskSpaceUsage() (used int64, reclaimable int64) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	used = d.mmapOffset

	// Estimate reclaimable space (deleted vectors)
	if d.count > 0 {
		deletedPct := float64(len(d.deleted)) / float64(d.count)
		reclaimable = int64(float64(used) * deletedPct)
	}

	return used, reclaimable
}

// CompactIfNeeded triggers compaction if thresholds are met
func (d *DiskANNIndex) CompactIfNeeded(ctx context.Context, config CompactionConfig) (*CompactionStats, error) {
	if !d.NeedsCompaction(config) {
		return nil, nil // No compaction needed
	}
	return d.Compact(ctx)
}
