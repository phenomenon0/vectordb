package index

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// SnapshotMetadata contains information about a snapshot
type SnapshotMetadata struct {
	ID              string    `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	VectorCount     int       `json:"vector_count"`
	Dimension       int       `json:"dimension"`
	MemoryVectors   int       `json:"memory_vectors"`
	DiskVectors     int       `json:"disk_vectors"`
	GraphEdges      int       `json:"graph_edges"`
	DiskSizeBytes   int64     `json:"disk_size_bytes"`
	Quantization    string    `json:"quantization,omitempty"`
	Description     string    `json:"description,omitempty"`
	Checksum        string    `json:"checksum,omitempty"`
}

// SnapshotManager handles snapshot creation, restoration, and lifecycle
type SnapshotManager struct {
	index         *DiskANNIndex
	snapshotDir   string
	maxSnapshots  int  // Max snapshots to keep (0 = unlimited)
	mu            sync.Mutex
}

// SnapshotConfig configures snapshot behavior
type SnapshotConfig struct {
	SnapshotDir  string // Directory to store snapshots
	MaxSnapshots int    // Maximum snapshots to retain (0 = unlimited)
	AutoCleanup  bool   // Automatically delete old snapshots
}

// NewSnapshotManager creates a new snapshot manager
func NewSnapshotManager(index *DiskANNIndex, config SnapshotConfig) (*SnapshotManager, error) {
	if config.SnapshotDir == "" {
		config.SnapshotDir = filepath.Join(filepath.Dir(index.indexPath), "snapshots")
	}

	// Create snapshot directory
	if err := os.MkdirAll(config.SnapshotDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create snapshot directory: %w", err)
	}

	return &SnapshotManager{
		index:        index,
		snapshotDir:  config.SnapshotDir,
		maxSnapshots: config.MaxSnapshots,
	}, nil
}

// CreateSnapshot creates a new snapshot of the current index state
func (sm *SnapshotManager) CreateSnapshot(ctx context.Context, description string) (*SnapshotMetadata, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Generate snapshot ID (timestamp-based with milliseconds to avoid collisions)
	now := time.Now()
	snapshotID := fmt.Sprintf("snapshot_%d_%d", now.Unix(), now.UnixNano()/1000000%1000)
	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)

	// Create snapshot directory
	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create snapshot path: %w", err)
	}

	// Lock index for consistent snapshot
	sm.index.mu.RLock()
	defer sm.index.mu.RUnlock()

	// Collect metadata
	stats := sm.index.Stats()
	metadata := &SnapshotMetadata{
		ID:            snapshotID,
		Timestamp:     time.Now(),
		VectorCount:   stats.Count,
		Dimension:     sm.index.dim,
		MemoryVectors: len(sm.index.memoryVectors) + len(sm.index.quantizedMemory),
		DiskVectors:   stats.Count - (len(sm.index.memoryVectors) + len(sm.index.quantizedMemory)),
		GraphEdges:    sm.countGraphEdges(),
		DiskSizeBytes: sm.index.mmapOffset,
		Description:   description,
	}

	if sm.index.quantizer != nil {
		switch sm.index.quantizer.(type) {
		case *Float16Quantizer:
			metadata.Quantization = "float16"
		case *Uint8Quantizer:
			metadata.Quantization = "uint8"
		case *ProductQuantizer:
			metadata.Quantization = "pq"
		}
	}

	// Copy mmap file (disk vectors + graph)
	if sm.index.mmapFile != nil && sm.index.mmapOffset > 0 {
		destPath := filepath.Join(snapshotPath, "index.dat")
		if err := sm.copyFile(sm.index.indexPath, destPath); err != nil {
			os.RemoveAll(snapshotPath)
			return nil, fmt.Errorf("failed to copy index file: %w", err)
		}
	}

	// Save graph structure as JSON
	graphPath := filepath.Join(snapshotPath, "graph.json")
	if err := sm.saveGraph(graphPath); err != nil {
		os.RemoveAll(snapshotPath)
		return nil, fmt.Errorf("failed to save graph: %w", err)
	}

	// Save memory vectors (optional, for quick recovery)
	if len(sm.index.memoryVectors) > 0 {
		memPath := filepath.Join(snapshotPath, "memory_vectors.json")
		if err := sm.saveMemoryVectors(memPath); err != nil {
			os.RemoveAll(snapshotPath)
			return nil, fmt.Errorf("failed to save memory vectors: %w", err)
		}
	}

	// Save quantized memory (if exists)
	if len(sm.index.quantizedMemory) > 0 {
		quantPath := filepath.Join(snapshotPath, "quantized_memory.json")
		if err := sm.saveQuantizedMemory(quantPath); err != nil {
			os.RemoveAll(snapshotPath)
			return nil, fmt.Errorf("failed to save quantized memory: %w", err)
		}
	}

	// Save disk offset index (for quantized vectors)
	if sm.index.quantizer != nil && len(sm.index.diskOffsetIndex) > 0 {
		offsetPath := filepath.Join(snapshotPath, "offset_index.json")
		if err := sm.saveOffsetIndex(offsetPath); err != nil {
			os.RemoveAll(snapshotPath)
			return nil, fmt.Errorf("failed to save offset index: %w", err)
		}
	}

	// Save metadata
	metadataPath := filepath.Join(snapshotPath, "metadata.json")
	if err := sm.saveMetadata(metadataPath, metadata); err != nil {
		os.RemoveAll(snapshotPath)
		return nil, fmt.Errorf("failed to save metadata: %w", err)
	}

	// Auto-cleanup old snapshots if configured
	if sm.maxSnapshots > 0 {
		if err := sm.cleanupOldSnapshots(); err != nil {
			// Log error but don't fail snapshot creation
			fmt.Printf("Warning: failed to cleanup old snapshots: %v\n", err)
		}
	}

	return metadata, nil
}

// RestoreSnapshot restores the index from a snapshot
func (sm *SnapshotManager) RestoreSnapshot(ctx context.Context, snapshotID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)

	// Verify snapshot exists
	if _, err := os.Stat(snapshotPath); os.IsNotExist(err) {
		return fmt.Errorf("snapshot %s not found", snapshotID)
	}

	// Load metadata
	metadataPath := filepath.Join(snapshotPath, "metadata.json")
	metadata, err := sm.loadMetadata(metadataPath)
	if err != nil {
		return fmt.Errorf("failed to load metadata: %w", err)
	}

	// Close current mmap before locking (Close acquires its own lock)
	if sm.index.mmapData != nil {
		if err := sm.index.Close(); err != nil {
			return fmt.Errorf("failed to close current index: %w", err)
		}
	}

	// Lock index for restoration
	sm.index.mu.Lock()
	defer sm.index.mu.Unlock()

	// Restore index file
	indexSrc := filepath.Join(snapshotPath, "index.dat")
	if _, err := os.Stat(indexSrc); err == nil {
		if err := sm.copyFile(indexSrc, sm.index.indexPath); err != nil {
			return fmt.Errorf("failed to restore index file: %w", err)
		}
	}

	// Restore graph
	graphPath := filepath.Join(snapshotPath, "graph.json")
	if err := sm.loadGraph(graphPath); err != nil {
		return fmt.Errorf("failed to restore graph: %w", err)
	}

	// Restore memory vectors
	memPath := filepath.Join(snapshotPath, "memory_vectors.json")
	if _, err := os.Stat(memPath); err == nil {
		if err := sm.loadMemoryVectors(memPath); err != nil {
			return fmt.Errorf("failed to restore memory vectors: %w", err)
		}
	}

	// Restore quantized memory
	quantPath := filepath.Join(snapshotPath, "quantized_memory.json")
	if _, err := os.Stat(quantPath); err == nil {
		if err := sm.loadQuantizedMemory(quantPath); err != nil {
			return fmt.Errorf("failed to restore quantized memory: %w", err)
		}
	}

	// Restore offset index
	offsetPath := filepath.Join(snapshotPath, "offset_index.json")
	if _, err := os.Stat(offsetPath); err == nil {
		if err := sm.loadOffsetIndex(offsetPath); err != nil {
			return fmt.Errorf("failed to restore offset index: %w", err)
		}
	}

	// Reinitialize mmap
	if err := sm.index.initMmap(); err != nil {
		return fmt.Errorf("failed to reinitialize mmap: %w", err)
	}

	// Update index stats
	sm.index.count = metadata.VectorCount
	sm.index.deleted = make(map[uint64]bool)

	// Clear LRU cache to ensure fresh reads after restore
	sm.index.lruCache.Clear()

	return nil
}

// ListSnapshots returns all available snapshots sorted by timestamp (newest first)
func (sm *SnapshotManager) ListSnapshots() ([]*SnapshotMetadata, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	return sm.listSnapshotsUnlocked()
}

// DeleteSnapshot removes a snapshot
func (sm *SnapshotManager) DeleteSnapshot(snapshotID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)
	if err := os.RemoveAll(snapshotPath); err != nil {
		return fmt.Errorf("failed to delete snapshot: %w", err)
	}

	return nil
}

// cleanupOldSnapshots removes old snapshots beyond maxSnapshots limit
// Note: This assumes the caller already holds sm.mu lock
func (sm *SnapshotManager) cleanupOldSnapshots() error {
	snapshots, err := sm.listSnapshotsUnlocked()
	if err != nil {
		return err
	}

	if len(snapshots) <= sm.maxSnapshots {
		return nil // Nothing to cleanup
	}

	// Delete oldest snapshots
	for i := sm.maxSnapshots; i < len(snapshots); i++ {
		snapshotPath := filepath.Join(sm.snapshotDir, snapshots[i].ID)
		if err := os.RemoveAll(snapshotPath); err != nil {
			return fmt.Errorf("failed to delete snapshot %s: %w", snapshots[i].ID, err)
		}
	}

	return nil
}

// listSnapshotsUnlocked is the internal version without locking
func (sm *SnapshotManager) listSnapshotsUnlocked() ([]*SnapshotMetadata, error) {
	entries, err := os.ReadDir(sm.snapshotDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read snapshot directory: %w", err)
	}

	var snapshots []*SnapshotMetadata
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		metadataPath := filepath.Join(sm.snapshotDir, entry.Name(), "metadata.json")
		metadata, err := sm.loadMetadata(metadataPath)
		if err != nil {
			continue // Skip invalid snapshots
		}

		snapshots = append(snapshots, metadata)
	}

	// Sort by timestamp (newest first)
	sort.Slice(snapshots, func(i, j int) bool {
		return snapshots[i].Timestamp.After(snapshots[j].Timestamp)
	})

	return snapshots, nil
}

// Helper methods

func (sm *SnapshotManager) countGraphEdges() int {
	count := 0
	sm.index.graphStore.Range(func(_ uint64, neighbors []uint64) bool {
		count += len(neighbors)
		return true
	})
	return count
}

func (sm *SnapshotManager) copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	if _, err := io.Copy(dstFile, srcFile); err != nil {
		return err
	}

	return dstFile.Sync()
}

func (sm *SnapshotManager) saveGraph(path string) error {
	data, err := json.Marshal(sm.index.graphStore.Clone())
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (sm *SnapshotManager) loadGraph(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	graphMap := make(map[uint64][]uint64)
	if err := json.Unmarshal(data, &graphMap); err != nil {
		return err
	}
	sm.index.graphStore.ReplaceAll(graphMap)
	return nil
}

func (sm *SnapshotManager) saveMemoryVectors(path string) error {
	data, err := json.Marshal(sm.index.memoryVectors)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (sm *SnapshotManager) loadMemoryVectors(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	sm.index.memoryVectors = make(map[uint64][]float32)
	return json.Unmarshal(data, &sm.index.memoryVectors)
}

func (sm *SnapshotManager) saveQuantizedMemory(path string) error {
	data, err := json.Marshal(sm.index.quantizedMemory)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (sm *SnapshotManager) loadQuantizedMemory(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	sm.index.quantizedMemory = make(map[uint64][]byte)
	return json.Unmarshal(data, &sm.index.quantizedMemory)
}

func (sm *SnapshotManager) saveOffsetIndex(path string) error {
	data, err := json.Marshal(sm.index.diskOffsetIndex)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (sm *SnapshotManager) loadOffsetIndex(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	sm.index.diskOffsetIndex = make(map[uint64]int64)
	return json.Unmarshal(data, &sm.index.diskOffsetIndex)
}

func (sm *SnapshotManager) saveMetadata(path string, metadata *SnapshotMetadata) error {
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (sm *SnapshotManager) loadMetadata(path string) (*SnapshotMetadata, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var metadata SnapshotMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return nil, err
	}
	return &metadata, nil
}

// GetSnapshotSize returns the total size of a snapshot in bytes
func (sm *SnapshotManager) GetSnapshotSize(snapshotID string) (int64, error) {
	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)

	var totalSize int64
	err := filepath.Walk(snapshotPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			totalSize += info.Size()
		}
		return nil
	})

	return totalSize, err
}
