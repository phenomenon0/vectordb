package index

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

	// Component checksums for incremental snapshots.
	// If a component checksum matches the previous snapshot, it can be skipped.
	ComponentChecksums map[string]string `json:"component_checksums,omitempty"`
	// BaseSnapshotID is the snapshot this incremental is based on (empty = full).
	BaseSnapshotID string `json:"base_snapshot_id,omitempty"`
	// Incremental indicates this is a delta snapshot.
	Incremental bool `json:"incremental,omitempty"`
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

	// Compute component checksums for future incremental use
	checksums, csErr := computeComponentChecksums(snapshotPath)
	if csErr == nil {
		metadata.ComponentChecksums = checksums
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

// computeFileChecksum returns the SHA-256 hex digest of a file's contents.
func computeFileChecksum(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// computeComponentChecksums computes checksums for all component files in a snapshot directory.
func computeComponentChecksums(snapshotPath string) (map[string]string, error) {
	components := []string{"index.dat", "graph.json", "memory_vectors.json", "quantized_memory.json", "offset_index.json"}
	checksums := make(map[string]string)

	for _, name := range components {
		p := filepath.Join(snapshotPath, name)
		if _, err := os.Stat(p); os.IsNotExist(err) {
			continue
		}
		cs, err := computeFileChecksum(p)
		if err != nil {
			return nil, fmt.Errorf("checksum %s: %w", name, err)
		}
		checksums[name] = cs
	}
	return checksums, nil
}

// CreateIncrementalSnapshot creates a snapshot that only contains components
// that changed since the base snapshot. Unchanged components are referenced
// by their checksum in metadata — the restore path copies them from the base.
func (sm *SnapshotManager) CreateIncrementalSnapshot(ctx context.Context, baseSnapshotID, description string) (*SnapshotMetadata, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Load base snapshot metadata
	basePath := filepath.Join(sm.snapshotDir, baseSnapshotID)
	baseMetaPath := filepath.Join(basePath, "metadata.json")
	baseMeta, err := sm.loadMetadata(baseMetaPath)
	if err != nil {
		return nil, fmt.Errorf("base snapshot %s not found: %w", baseSnapshotID, err)
	}

	// Compute base checksums if not already stored
	baseChecksums := baseMeta.ComponentChecksums
	if len(baseChecksums) == 0 {
		baseChecksums, err = computeComponentChecksums(basePath)
		if err != nil {
			return nil, fmt.Errorf("compute base checksums: %w", err)
		}
		// Persist checksums back to base metadata for future use
		baseMeta.ComponentChecksums = baseChecksums
		if saveErr := sm.saveMetadata(baseMetaPath, baseMeta); saveErr != nil {
			return nil, fmt.Errorf("update base metadata: %w", saveErr)
		}
	}

	// Create a full snapshot first into a temp dir, then prune unchanged components
	now := time.Now()
	snapshotID := fmt.Sprintf("snapshot_%d_%d", now.Unix(), now.UnixNano()/1000000%1000)
	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)

	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		return nil, fmt.Errorf("create snapshot path: %w", err)
	}

	// Lock index for consistent snapshot
	sm.index.mu.RLock()
	defer sm.index.mu.RUnlock()

	stats := sm.index.Stats()
	metadata := &SnapshotMetadata{
		ID:             snapshotID,
		Timestamp:      now,
		VectorCount:    stats.Count,
		Dimension:      sm.index.dim,
		MemoryVectors:  len(sm.index.memoryVectors) + len(sm.index.quantizedMemory),
		DiskVectors:    stats.Count - (len(sm.index.memoryVectors) + len(sm.index.quantizedMemory)),
		GraphEdges:     sm.countGraphEdges(),
		DiskSizeBytes:  sm.index.mmapOffset,
		Description:    description,
		BaseSnapshotID: baseSnapshotID,
		Incremental:    true,
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

	// Write all components to snapshot dir
	type componentWriter struct {
		name    string
		write   func(string) error
		present bool
	}

	writers := []componentWriter{
		{"graph.json", sm.saveGraph, true},
	}
	if sm.index.mmapFile != nil && sm.index.mmapOffset > 0 {
		writers = append(writers, componentWriter{"index.dat", func(p string) error {
			return sm.copyFile(sm.index.indexPath, p)
		}, true})
	}
	if len(sm.index.memoryVectors) > 0 {
		writers = append(writers, componentWriter{"memory_vectors.json", sm.saveMemoryVectors, true})
	}
	if len(sm.index.quantizedMemory) > 0 {
		writers = append(writers, componentWriter{"quantized_memory.json", sm.saveQuantizedMemory, true})
	}
	if sm.index.quantizer != nil && len(sm.index.diskOffsetIndex) > 0 {
		writers = append(writers, componentWriter{"offset_index.json", sm.saveOffsetIndex, true})
	}

	for _, cw := range writers {
		p := filepath.Join(snapshotPath, cw.name)
		if err := cw.write(p); err != nil {
			os.RemoveAll(snapshotPath)
			return nil, fmt.Errorf("write %s: %w", cw.name, err)
		}
	}

	// Compute checksums for new components
	newChecksums, err := computeComponentChecksums(snapshotPath)
	if err != nil {
		os.RemoveAll(snapshotPath)
		return nil, fmt.Errorf("compute new checksums: %w", err)
	}
	metadata.ComponentChecksums = newChecksums

	// Remove unchanged components (same checksum as base)
	for name, newCS := range newChecksums {
		if baseCS, ok := baseChecksums[name]; ok && baseCS == newCS {
			os.Remove(filepath.Join(snapshotPath, name))
		}
	}

	// Save metadata
	metadataPath := filepath.Join(snapshotPath, "metadata.json")
	if err := sm.saveMetadata(metadataPath, metadata); err != nil {
		os.RemoveAll(snapshotPath)
		return nil, fmt.Errorf("save metadata: %w", err)
	}

	if sm.maxSnapshots > 0 {
		if err := sm.cleanupOldSnapshots(); err != nil {
			fmt.Printf("Warning: failed to cleanup old snapshots: %v\n", err)
		}
	}

	return metadata, nil
}

// RestoreIncrementalSnapshot restores from an incremental snapshot by
// merging unchanged components from the base snapshot with changed components
// from the incremental snapshot.
func (sm *SnapshotManager) RestoreIncrementalSnapshot(ctx context.Context, snapshotID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)
	metadataPath := filepath.Join(snapshotPath, "metadata.json")
	metadata, err := sm.loadMetadata(metadataPath)
	if err != nil {
		return fmt.Errorf("load incremental metadata: %w", err)
	}

	if !metadata.Incremental || metadata.BaseSnapshotID == "" {
		// Not incremental — delegate to normal restore
		sm.mu.Unlock()
		err := sm.RestoreSnapshot(ctx, snapshotID)
		sm.mu.Lock()
		return err
	}

	basePath := filepath.Join(sm.snapshotDir, metadata.BaseSnapshotID)
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		return fmt.Errorf("base snapshot %s not found", metadata.BaseSnapshotID)
	}

	// Build a merged view: for each component, use incremental if present, else base
	components := []string{"index.dat", "graph.json", "memory_vectors.json", "quantized_memory.json", "offset_index.json"}
	resolvedPaths := make(map[string]string) // component → actual file path

	for _, name := range components {
		incrFile := filepath.Join(snapshotPath, name)
		baseFile := filepath.Join(basePath, name)

		if _, err := os.Stat(incrFile); err == nil {
			resolvedPaths[name] = incrFile
		} else if _, err := os.Stat(baseFile); err == nil {
			resolvedPaths[name] = baseFile
		}
		// if neither exists, component is absent (normal for optional components)
	}

	// Close current mmap
	if sm.index.mmapData != nil {
		if err := sm.index.Close(); err != nil {
			return fmt.Errorf("close current index: %w", err)
		}
	}

	sm.index.mu.Lock()
	defer sm.index.mu.Unlock()

	// Restore index.dat
	if src, ok := resolvedPaths["index.dat"]; ok {
		if err := sm.copyFile(src, sm.index.indexPath); err != nil {
			return fmt.Errorf("restore index.dat: %w", err)
		}
	}

	// Restore graph
	if src, ok := resolvedPaths["graph.json"]; ok {
		if err := sm.loadGraph(src); err != nil {
			return fmt.Errorf("restore graph: %w", err)
		}
	}

	// Restore memory vectors
	if src, ok := resolvedPaths["memory_vectors.json"]; ok {
		if err := sm.loadMemoryVectors(src); err != nil {
			return fmt.Errorf("restore memory vectors: %w", err)
		}
	}

	// Restore quantized memory
	if src, ok := resolvedPaths["quantized_memory.json"]; ok {
		if err := sm.loadQuantizedMemory(src); err != nil {
			return fmt.Errorf("restore quantized memory: %w", err)
		}
	}

	// Restore offset index
	if src, ok := resolvedPaths["offset_index.json"]; ok {
		if err := sm.loadOffsetIndex(src); err != nil {
			return fmt.Errorf("restore offset index: %w", err)
		}
	}

	// Reinitialize mmap
	if err := sm.index.initMmap(); err != nil {
		return fmt.Errorf("reinitialize mmap: %w", err)
	}

	sm.index.count = metadata.VectorCount
	sm.index.deleted = make(map[uint64]bool)
	sm.index.lruCache.Clear()

	return nil
}
