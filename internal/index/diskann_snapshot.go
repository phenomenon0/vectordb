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

	now := time.Now()
	snapshotID := fmt.Sprintf("snapshot_%d_%d", now.Unix(), now.UnixNano()/1000000%1000)
	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)

	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create snapshot path: %w", err)
	}

	sm.index.mu.RLock()
	defer sm.index.mu.RUnlock()

	metadata := sm.buildSnapshotMetadata(snapshotID, now, description)

	if err := sm.writeSnapshotComponents(snapshotPath); err != nil {
		os.RemoveAll(snapshotPath)
		return nil, err
	}

	// Compute component checksums for future incremental use
	if checksums, err := computeComponentChecksums(snapshotPath); err == nil {
		metadata.ComponentChecksums = checksums
	}

	metadataPath := filepath.Join(snapshotPath, "metadata.json")
	if err := sm.saveMetadata(metadataPath, metadata); err != nil {
		os.RemoveAll(snapshotPath)
		return nil, fmt.Errorf("failed to save metadata: %w", err)
	}

	if sm.maxSnapshots > 0 {
		if err := sm.cleanupOldSnapshots(); err != nil {
			fmt.Printf("Warning: failed to cleanup old snapshots: %v\n", err)
		}
	}

	return metadata, nil
}

// RestoreSnapshot restores the index from a snapshot
func (sm *SnapshotManager) RestoreSnapshot(ctx context.Context, snapshotID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	return sm.restoreSnapshotLocked(snapshotID)
}

// restoreSnapshotLocked restores a snapshot. Caller must hold sm.mu.
func (sm *SnapshotManager) restoreSnapshotLocked(snapshotID string) error {
	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)

	metadataPath := filepath.Join(snapshotPath, "metadata.json")
	metadata, err := sm.loadMetadata(metadataPath)
	if err != nil {
		return fmt.Errorf("failed to load metadata: %w", err)
	}

	// Build resolved paths from the single snapshot directory
	resolvedPaths := make(map[string]string)
	for _, name := range snapshotComponents {
		p := filepath.Join(snapshotPath, name)
		if _, err := os.Stat(p); err == nil {
			resolvedPaths[name] = p
		}
	}

	return sm.restoreFromPaths(resolvedPaths, metadata)
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

// snapshotComponents lists the component file names used in snapshot creation and restoration.
var snapshotComponents = []string{"index.dat", "graph.json", "memory_vectors.json", "quantized_memory.json", "offset_index.json"}

// quantizerName returns a human-readable name for the index quantizer (empty if none).
func (sm *SnapshotManager) quantizerName() string {
	if sm.index.quantizer == nil {
		return ""
	}
	switch sm.index.quantizer.(type) {
	case *Float16Quantizer:
		return "float16"
	case *Uint8Quantizer:
		return "uint8"
	case *ProductQuantizer:
		return "pq"
	default:
		return ""
	}
}

// buildSnapshotMetadata constructs metadata from the current index state.
// Must be called with sm.index.mu held (at least RLock).
func (sm *SnapshotManager) buildSnapshotMetadata(snapshotID string, now time.Time, description string) *SnapshotMetadata {
	stats := sm.index.Stats()
	memCount := len(sm.index.memoryVectors) + len(sm.index.quantizedMemory)
	return &SnapshotMetadata{
		ID:            snapshotID,
		Timestamp:     now,
		VectorCount:   stats.Count,
		Dimension:     sm.index.dim,
		MemoryVectors: memCount,
		DiskVectors:   stats.Count - memCount,
		GraphEdges:    sm.countGraphEdges(),
		DiskSizeBytes: sm.index.mmapOffset,
		Quantization:  sm.quantizerName(),
		Description:   description,
	}
}

// writeSnapshotComponents writes all index components to snapshotPath.
// Must be called with sm.index.mu held (at least RLock).
func (sm *SnapshotManager) writeSnapshotComponents(snapshotPath string) error {
	// Graph (always present)
	if err := sm.saveGraph(filepath.Join(snapshotPath, "graph.json")); err != nil {
		return fmt.Errorf("save graph: %w", err)
	}

	// Index.dat (disk vectors)
	if sm.index.mmapFile != nil && sm.index.mmapOffset > 0 {
		if err := sm.copyFile(sm.index.indexPath, filepath.Join(snapshotPath, "index.dat")); err != nil {
			return fmt.Errorf("copy index file: %w", err)
		}
	}

	// Memory vectors
	if len(sm.index.memoryVectors) > 0 {
		if err := sm.saveMemoryVectors(filepath.Join(snapshotPath, "memory_vectors.json")); err != nil {
			return fmt.Errorf("save memory vectors: %w", err)
		}
	}

	// Quantized memory
	if len(sm.index.quantizedMemory) > 0 {
		if err := sm.saveQuantizedMemory(filepath.Join(snapshotPath, "quantized_memory.json")); err != nil {
			return fmt.Errorf("save quantized memory: %w", err)
		}
	}

	// Offset index
	if sm.index.quantizer != nil && len(sm.index.diskOffsetIndex) > 0 {
		if err := sm.saveOffsetIndex(filepath.Join(snapshotPath, "offset_index.json")); err != nil {
			return fmt.Errorf("save offset index: %w", err)
		}
	}

	return nil
}

// restoreFromPaths restores index state from resolved component file paths.
// Must be called with sm.mu held. Acquires sm.index.mu internally.
func (sm *SnapshotManager) restoreFromPaths(resolvedPaths map[string]string, metadata *SnapshotMetadata) error {
	// Close current mmap
	if sm.index.mmapData != nil {
		if err := sm.index.Close(); err != nil {
			return fmt.Errorf("close current index: %w", err)
		}
	}

	sm.index.mu.Lock()
	defer sm.index.mu.Unlock()

	if src, ok := resolvedPaths["index.dat"]; ok {
		if err := sm.copyFile(src, sm.index.indexPath); err != nil {
			return fmt.Errorf("restore index.dat: %w", err)
		}
	}
	if src, ok := resolvedPaths["graph.json"]; ok {
		if err := sm.loadGraph(src); err != nil {
			return fmt.Errorf("restore graph: %w", err)
		}
	}
	if src, ok := resolvedPaths["memory_vectors.json"]; ok {
		if err := sm.loadMemoryVectors(src); err != nil {
			return fmt.Errorf("restore memory vectors: %w", err)
		}
	}
	if src, ok := resolvedPaths["quantized_memory.json"]; ok {
		if err := sm.loadQuantizedMemory(src); err != nil {
			return fmt.Errorf("restore quantized memory: %w", err)
		}
	}
	if src, ok := resolvedPaths["offset_index.json"]; ok {
		if err := sm.loadOffsetIndex(src); err != nil {
			return fmt.Errorf("restore offset index: %w", err)
		}
	}

	if err := sm.index.initMmap(); err != nil {
		return fmt.Errorf("reinitialize mmap: %w", err)
	}

	sm.index.count = metadata.VectorCount
	sm.index.deleted = make(map[uint64]bool)
	sm.index.lruCache.Clear()
	return nil
}

// computeComponentChecksums computes checksums for all component files in a snapshot directory.
func computeComponentChecksums(snapshotPath string) (map[string]string, error) {
	components := snapshotComponents
	checksums := make(map[string]string)

	for _, name := range components {
		p := filepath.Join(snapshotPath, name)
		cs, err := computeFileChecksum(p)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
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

	// Load base snapshot checksums
	basePath := filepath.Join(sm.snapshotDir, baseSnapshotID)
	baseMetaPath := filepath.Join(basePath, "metadata.json")
	baseMeta, err := sm.loadMetadata(baseMetaPath)
	if err != nil {
		return nil, fmt.Errorf("base snapshot %s not found: %w", baseSnapshotID, err)
	}

	baseChecksums := baseMeta.ComponentChecksums
	if len(baseChecksums) == 0 {
		baseChecksums, err = computeComponentChecksums(basePath)
		if err != nil {
			return nil, fmt.Errorf("compute base checksums: %w", err)
		}
		baseMeta.ComponentChecksums = baseChecksums
		if saveErr := sm.saveMetadata(baseMetaPath, baseMeta); saveErr != nil {
			return nil, fmt.Errorf("update base metadata: %w", saveErr)
		}
	}

	now := time.Now()
	snapshotID := fmt.Sprintf("snapshot_%d_%d", now.Unix(), now.UnixNano()/1000000%1000)
	snapshotPath := filepath.Join(sm.snapshotDir, snapshotID)

	if err := os.MkdirAll(snapshotPath, 0755); err != nil {
		return nil, fmt.Errorf("create snapshot path: %w", err)
	}

	sm.index.mu.RLock()
	defer sm.index.mu.RUnlock()

	metadata := sm.buildSnapshotMetadata(snapshotID, now, description)
	metadata.BaseSnapshotID = baseSnapshotID
	metadata.Incremental = true

	if err := sm.writeSnapshotComponents(snapshotPath); err != nil {
		os.RemoveAll(snapshotPath)
		return nil, err
	}

	// Compute checksums and prune unchanged components
	newChecksums, err := computeComponentChecksums(snapshotPath)
	if err != nil {
		os.RemoveAll(snapshotPath)
		return nil, fmt.Errorf("compute new checksums: %w", err)
	}
	metadata.ComponentChecksums = newChecksums

	for name, newCS := range newChecksums {
		if baseCS, ok := baseChecksums[name]; ok && baseCS == newCS {
			os.Remove(filepath.Join(snapshotPath, name))
		}
	}

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
		return sm.restoreSnapshotLocked(snapshotID)
	}

	basePath := filepath.Join(sm.snapshotDir, metadata.BaseSnapshotID)

	// Build merged view: use incremental file if present, else fall back to base
	resolvedPaths := make(map[string]string)
	for _, name := range snapshotComponents {
		incrFile := filepath.Join(snapshotPath, name)
		baseFile := filepath.Join(basePath, name)

		if _, err := os.Stat(incrFile); err == nil {
			resolvedPaths[name] = incrFile
		} else if _, err := os.Stat(baseFile); err == nil {
			resolvedPaths[name] = baseFile
		}
	}

	return sm.restoreFromPaths(resolvedPaths, metadata)
}
