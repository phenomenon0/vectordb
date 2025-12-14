package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ===========================================================================================
// FULL SNAPSHOT SYNC
// Complete VectorStore snapshot for bootstrapping new replicas
// ===========================================================================================

// StoreSnapshot represents a complete VectorStore snapshot
type StoreSnapshot struct {
	// Metadata
	ID          string    `json:"id"`
	CreatedAt   time.Time `json:"created_at"`
	WALSequence uint64    `json:"wal_sequence"` // WAL sequence at snapshot time

	// Store state
	VectorCount int    `json:"vector_count"`
	Dimension   int    `json:"dimension"`
	Collections int    `json:"collections"`
	SizeBytes   int64  `json:"size_bytes"`
	Checksum    string `json:"checksum,omitempty"`
	Compressed  bool   `json:"compressed"`
	Version     int    `json:"version"` // Snapshot format version
}

// SnapshotSyncConfig configures snapshot sync behavior
type SnapshotSyncConfig struct {
	// SnapshotDir is where snapshots are stored
	SnapshotDir string

	// MaxSnapshots to retain (0 = unlimited)
	MaxSnapshots int

	// CompressionEnabled enables gzip compression
	CompressionEnabled bool

	// TransferTimeout for snapshot transfer
	TransferTimeout time.Duration

	// ChunkSize for streaming (default 1MB)
	ChunkSize int
}

// DefaultSnapshotSyncConfig returns sensible defaults
func DefaultSnapshotSyncConfig() SnapshotSyncConfig {
	return SnapshotSyncConfig{
		SnapshotDir:        "./snapshots",
		MaxSnapshots:       3,
		CompressionEnabled: true,
		TransferTimeout:    10 * time.Minute,
		ChunkSize:          1024 * 1024, // 1MB chunks
	}
}

// SnapshotSyncManager handles full store snapshots for replication
type SnapshotSyncManager struct {
	mu sync.RWMutex

	store     *VectorStore
	config    SnapshotSyncConfig
	walStream *WALStream

	// State
	latestSnapshot *StoreSnapshot
	inProgress     bool
}

// NewSnapshotSyncManager creates a new snapshot sync manager
func NewSnapshotSyncManager(store *VectorStore, walStream *WALStream, config SnapshotSyncConfig) (*SnapshotSyncManager, error) {
	// Create snapshot directory
	if err := os.MkdirAll(config.SnapshotDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create snapshot directory: %w", err)
	}

	return &SnapshotSyncManager{
		store:     store,
		config:    config,
		walStream: walStream,
	}, nil
}

// CreateSnapshot creates a new full store snapshot
func (ssm *SnapshotSyncManager) CreateSnapshot(ctx context.Context) (*StoreSnapshot, error) {
	ssm.mu.Lock()
	if ssm.inProgress {
		ssm.mu.Unlock()
		return nil, fmt.Errorf("snapshot already in progress")
	}
	ssm.inProgress = true
	ssm.mu.Unlock()

	defer func() {
		ssm.mu.Lock()
		ssm.inProgress = false
		ssm.mu.Unlock()
	}()

	// Generate snapshot ID
	now := time.Now()
	snapshotID := fmt.Sprintf("store_snapshot_%d", now.UnixNano())

	fmt.Printf("[SnapshotSync] Creating snapshot %s...\n", snapshotID)

	// Lock store for reading
	ssm.store.RLock()
	defer ssm.store.RUnlock()

	// Get current WAL sequence
	var walSeq uint64
	if ssm.walStream != nil {
		walSeq = ssm.walStream.GetLatestSeq()
	}

	// Create snapshot metadata
	snapshot := &StoreSnapshot{
		ID:          snapshotID,
		CreatedAt:   now,
		WALSequence: walSeq,
		VectorCount: ssm.store.Count,
		Dimension:   ssm.store.Dim,
		Collections: ssm.countCollections(),
		Compressed:  ssm.config.CompressionEnabled,
		Version:     1,
	}

	// Create snapshot file
	snapshotPath := ssm.getSnapshotPath(snapshotID)
	if err := ssm.writeSnapshot(snapshotPath, snapshot); err != nil {
		return nil, fmt.Errorf("failed to write snapshot: %w", err)
	}

	// Get file size
	if info, err := os.Stat(snapshotPath); err == nil {
		snapshot.SizeBytes = info.Size()
	}

	// Update latest snapshot
	ssm.mu.Lock()
	ssm.latestSnapshot = snapshot
	ssm.mu.Unlock()

	// Cleanup old snapshots
	if ssm.config.MaxSnapshots > 0 {
		ssm.cleanupOldSnapshots()
	}

	fmt.Printf("[SnapshotSync] Snapshot %s created (vectors=%d, size=%d bytes, wal_seq=%d)\n",
		snapshotID, snapshot.VectorCount, snapshot.SizeBytes, snapshot.WALSequence)

	return snapshot, nil
}

// writeSnapshot serializes store state to a file
func (ssm *SnapshotSyncManager) writeSnapshot(path string, metadata *StoreSnapshot) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var writer io.Writer = file
	var gzWriter *gzip.Writer

	if ssm.config.CompressionEnabled {
		gzWriter = gzip.NewWriter(file)
		writer = gzWriter
		defer gzWriter.Close()
	}

	// Write snapshot data structure
	data := &snapshotData{
		Metadata: metadata,
		Data:     ssm.store.Data,
		Docs:     ssm.store.Docs,
		IDs:      ssm.store.IDs,
		Seqs:     ssm.store.Seqs,
		Meta:     ssm.store.Meta,
		NumMeta:  ssm.store.NumMeta,
		TimeMeta: ssm.store.TimeMeta,
		Deleted:  ssm.store.Deleted,
		Coll:     ssm.store.Coll,
		TenantID: ssm.store.TenantID,
		LexTF:    ssm.store.lexTF,
		DocLen:   ssm.store.docLen,
		DF:       ssm.store.df,
		SumDocL:  ssm.store.sumDocL,
	}

	encoder := json.NewEncoder(writer)
	return encoder.Encode(data)
}

// snapshotData is the internal snapshot format
type snapshotData struct {
	Metadata *StoreSnapshot                  `json:"metadata"`
	Data     []float32                       `json:"data"`
	Docs     []string                        `json:"docs"`
	IDs      []string                        `json:"ids"`
	Seqs     []uint64                        `json:"seqs"`
	Meta     map[uint64]map[string]string    `json:"meta"`
	NumMeta  map[uint64]map[string]float64   `json:"num_meta"`
	TimeMeta map[uint64]map[string]time.Time `json:"time_meta"`
	Deleted  map[uint64]bool                 `json:"deleted"`
	Coll     map[uint64]string               `json:"coll"`
	TenantID map[uint64]string               `json:"tenant_id"`
	LexTF    map[uint64]map[string]int       `json:"lex_tf"`
	DocLen   map[uint64]int                  `json:"doc_len"`
	DF       map[string]int                  `json:"df"`
	SumDocL  int                             `json:"sum_doc_l"`
}

// LoadSnapshot loads a snapshot into the store
func (ssm *SnapshotSyncManager) LoadSnapshot(ctx context.Context, snapshotID string) error {
	snapshotPath := ssm.getSnapshotPath(snapshotID)

	fmt.Printf("[SnapshotSync] Loading snapshot %s...\n", snapshotID)

	file, err := os.Open(snapshotPath)
	if err != nil {
		return fmt.Errorf("failed to open snapshot: %w", err)
	}
	defer file.Close()

	var reader io.Reader = file

	// Check if compressed by reading magic bytes
	header := make([]byte, 2)
	if _, err := file.Read(header); err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}
	file.Seek(0, 0) // Reset to beginning

	// Gzip magic number: 0x1f 0x8b
	if header[0] == 0x1f && header[1] == 0x8b {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()
		reader = gzReader
	}

	// Decode snapshot data
	var data snapshotData
	decoder := json.NewDecoder(reader)
	if err := decoder.Decode(&data); err != nil {
		return fmt.Errorf("failed to decode snapshot: %w", err)
	}

	// Apply to store (requires exclusive lock)
	ssm.store.Lock()
	defer ssm.store.Unlock()

	// Clear current state
	ssm.store.Data = data.Data
	ssm.store.Docs = data.Docs
	ssm.store.IDs = data.IDs
	ssm.store.Seqs = data.Seqs
	ssm.store.Meta = data.Meta
	ssm.store.NumMeta = data.NumMeta
	ssm.store.TimeMeta = data.TimeMeta
	ssm.store.Deleted = data.Deleted
	ssm.store.Coll = data.Coll
	ssm.store.TenantID = data.TenantID
	ssm.store.lexTF = data.LexTF
	ssm.store.docLen = data.DocLen
	ssm.store.df = data.DF
	ssm.store.sumDocL = data.SumDocL
	ssm.store.Count = data.Metadata.VectorCount
	ssm.store.Dim = data.Metadata.Dimension

	// Rebuild idToIx mapping
	ssm.store.idToIx = make(map[uint64]int)
	for i, seq := range ssm.store.Seqs {
		ssm.store.idToIx[seq] = i
	}

	fmt.Printf("[SnapshotSync] Snapshot %s loaded (vectors=%d, wal_seq=%d)\n",
		snapshotID, data.Metadata.VectorCount, data.Metadata.WALSequence)

	return nil
}

// GetLatestSnapshot returns metadata of the latest snapshot
func (ssm *SnapshotSyncManager) GetLatestSnapshot() *StoreSnapshot {
	ssm.mu.RLock()
	defer ssm.mu.RUnlock()
	return ssm.latestSnapshot
}

// ListSnapshots returns all available snapshots
func (ssm *SnapshotSyncManager) ListSnapshots() ([]*StoreSnapshot, error) {
	entries, err := os.ReadDir(ssm.config.SnapshotDir)
	if err != nil {
		return nil, err
	}

	var snapshots []*StoreSnapshot
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		// Load metadata from snapshot file
		path := filepath.Join(ssm.config.SnapshotDir, entry.Name())
		metadata, err := ssm.loadSnapshotMetadata(path)
		if err != nil {
			continue // Skip invalid files
		}
		snapshots = append(snapshots, metadata)
	}

	return snapshots, nil
}

// loadSnapshotMetadata loads just the metadata from a snapshot file
func (ssm *SnapshotSyncManager) loadSnapshotMetadata(path string) (*StoreSnapshot, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var reader io.Reader = file

	// Check if compressed
	header := make([]byte, 2)
	if _, err := file.Read(header); err != nil {
		return nil, err
	}
	file.Seek(0, 0)

	if header[0] == 0x1f && header[1] == 0x8b {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return nil, err
		}
		defer gzReader.Close()
		reader = gzReader
	}

	// Just decode the beginning to get metadata
	var data snapshotData
	decoder := json.NewDecoder(reader)
	if err := decoder.Decode(&data); err != nil {
		return nil, err
	}

	return data.Metadata, nil
}

// getSnapshotPath returns the full path for a snapshot
func (ssm *SnapshotSyncManager) getSnapshotPath(snapshotID string) string {
	ext := ".snapshot"
	if ssm.config.CompressionEnabled {
		ext = ".snapshot.gz"
	}
	return filepath.Join(ssm.config.SnapshotDir, snapshotID+ext)
}

// countCollections counts unique collections
func (ssm *SnapshotSyncManager) countCollections() int {
	collections := make(map[string]bool)
	for _, coll := range ssm.store.Coll {
		collections[coll] = true
	}
	return len(collections)
}

// cleanupOldSnapshots removes old snapshots beyond limit
func (ssm *SnapshotSyncManager) cleanupOldSnapshots() {
	snapshots, err := ssm.ListSnapshots()
	if err != nil {
		return
	}

	if len(snapshots) <= ssm.config.MaxSnapshots {
		return
	}

	// Sort by timestamp (oldest first)
	// Note: snapshots are already in filesystem order, need to sort
	for i := ssm.config.MaxSnapshots; i < len(snapshots); i++ {
		path := ssm.getSnapshotPath(snapshots[i].ID)
		os.Remove(path)
		fmt.Printf("[SnapshotSync] Cleaned up old snapshot: %s\n", snapshots[i].ID)
	}
}

// ===========================================================================================
// HTTP HANDLERS FOR SNAPSHOT TRANSFER
// ===========================================================================================

// handleSnapshotList handles GET /snapshot/list - returns available snapshots
func (ssm *SnapshotSyncManager) handleSnapshotList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	snapshots, err := ssm.ListSnapshots()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to list snapshots: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"snapshots": snapshots,
		"count":     len(snapshots),
	})
}

// handleSnapshotCreate handles POST /snapshot/create - creates new snapshot
func (ssm *SnapshotSyncManager) handleSnapshotCreate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), ssm.config.TransferTimeout)
	defer cancel()

	snapshot, err := ssm.CreateSnapshot(ctx)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create snapshot: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(snapshot)
}

// handleSnapshotDownload handles GET /snapshot/download?id=xxx - streams snapshot
func (ssm *SnapshotSyncManager) handleSnapshotDownload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	snapshotID := r.URL.Query().Get("id")
	if snapshotID == "" {
		// Return latest snapshot
		latest := ssm.GetLatestSnapshot()
		if latest == nil {
			http.Error(w, "no snapshots available", http.StatusNotFound)
			return
		}
		snapshotID = latest.ID
	}

	snapshotPath := ssm.getSnapshotPath(snapshotID)
	file, err := os.Open(snapshotPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("snapshot not found: %v", err), http.StatusNotFound)
		return
	}
	defer file.Close()

	// Get file info
	info, err := file.Stat()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to stat snapshot: %v", err), http.StatusInternalServerError)
		return
	}

	// Set headers
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", info.Size()))
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%s", filepath.Base(snapshotPath)))
	w.Header().Set("X-Snapshot-ID", snapshotID)

	// Stream file
	io.Copy(w, file)
}

// handleSnapshotUpload handles POST /snapshot/upload - receives snapshot from primary
func (ssm *SnapshotSyncManager) handleSnapshotUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	snapshotID := r.Header.Get("X-Snapshot-ID")
	if snapshotID == "" {
		snapshotID = fmt.Sprintf("received_%d", time.Now().UnixNano())
	}

	// Create temp file
	tempPath := ssm.getSnapshotPath(snapshotID + ".tmp")
	file, err := os.Create(tempPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create temp file: %v", err), http.StatusInternalServerError)
		return
	}

	// Write uploaded data
	written, err := io.Copy(file, r.Body)
	file.Close()
	if err != nil {
		os.Remove(tempPath)
		http.Error(w, fmt.Sprintf("failed to write snapshot: %v", err), http.StatusInternalServerError)
		return
	}

	// Rename to final path
	finalPath := ssm.getSnapshotPath(snapshotID)
	if err := os.Rename(tempPath, finalPath); err != nil {
		os.Remove(tempPath)
		http.Error(w, fmt.Sprintf("failed to finalize snapshot: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"snapshot_id":   snapshotID,
		"bytes_written": written,
		"status":        "uploaded",
	})
}

// ===========================================================================================
// SNAPSHOT SYNC CLIENT
// ===========================================================================================

// SnapshotSyncClient pulls snapshots from primary for bootstrapping
type SnapshotSyncClient struct {
	primaryAddr string
	authToken   string
	httpClient  *http.Client
	config      SnapshotSyncConfig
}

// NewSnapshotSyncClient creates a new snapshot sync client
func NewSnapshotSyncClient(primaryAddr, authToken string, config SnapshotSyncConfig) *SnapshotSyncClient {
	return &SnapshotSyncClient{
		primaryAddr: primaryAddr,
		authToken:   authToken,
		httpClient:  &http.Client{Timeout: config.TransferTimeout},
		config:      config,
	}
}

// FetchLatestSnapshot downloads the latest snapshot from primary
func (ssc *SnapshotSyncClient) FetchLatestSnapshot(ctx context.Context) (*StoreSnapshot, []byte, error) {
	url := fmt.Sprintf("%s/snapshot/download", ssc.primaryAddr)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	if ssc.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+ssc.authToken)
	}

	resp, err := ssc.httpClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to fetch snapshot: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, nil, fmt.Errorf("snapshot download failed: %s - %s", resp.Status, string(body))
	}

	snapshotID := resp.Header.Get("X-Snapshot-ID")

	// Read snapshot data
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read snapshot data: %w", err)
	}

	// Parse metadata from data
	metadata := &StoreSnapshot{
		ID:        snapshotID,
		SizeBytes: int64(len(data)),
	}

	return metadata, data, nil
}

// FetchAndApplySnapshot fetches and applies snapshot to a store
func (ssc *SnapshotSyncClient) FetchAndApplySnapshot(ctx context.Context, store *VectorStore, walStream *WALStream) (*StoreSnapshot, error) {
	fmt.Printf("[SnapshotSyncClient] Fetching snapshot from %s...\n", ssc.primaryAddr)

	metadata, data, err := ssc.FetchLatestSnapshot(ctx)
	if err != nil {
		return nil, err
	}

	fmt.Printf("[SnapshotSyncClient] Downloaded snapshot %s (%d bytes)\n", metadata.ID, len(data))

	// Parse and apply snapshot
	var reader io.Reader = bytes.NewReader(data)

	// Check if compressed
	if len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b {
		gzReader, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()
		reader = gzReader
	}

	var snapshotData snapshotData
	if err := json.NewDecoder(reader).Decode(&snapshotData); err != nil {
		return nil, fmt.Errorf("failed to decode snapshot: %w", err)
	}

	// Apply to store
	store.Lock()
	defer store.Unlock()

	store.Data = snapshotData.Data
	store.Docs = snapshotData.Docs
	store.IDs = snapshotData.IDs
	store.Seqs = snapshotData.Seqs
	store.Meta = snapshotData.Meta
	store.NumMeta = snapshotData.NumMeta
	store.TimeMeta = snapshotData.TimeMeta
	store.Deleted = snapshotData.Deleted
	store.Coll = snapshotData.Coll
	store.TenantID = snapshotData.TenantID
	store.lexTF = snapshotData.LexTF
	store.docLen = snapshotData.DocLen
	store.df = snapshotData.DF
	store.sumDocL = snapshotData.SumDocL
	store.Count = snapshotData.Metadata.VectorCount
	store.Dim = snapshotData.Metadata.Dimension

	// Rebuild idToIx mapping
	store.idToIx = make(map[uint64]int)
	for i, seq := range store.Seqs {
		store.idToIx[seq] = i
	}

	fmt.Printf("[SnapshotSyncClient] Applied snapshot %s (vectors=%d, wal_seq=%d)\n",
		metadata.ID, snapshotData.Metadata.VectorCount, snapshotData.Metadata.WALSequence)

	return snapshotData.Metadata, nil
}

// ===========================================================================================
// INTEGRATION WITH FOLLOWER REPLICATOR
// ===========================================================================================

// BootstrapFromPrimary performs full bootstrap: snapshot + WAL catchup
func BootstrapFromPrimary(ctx context.Context, store *VectorStore, walStream *WALStream, primaryAddr, authToken string) error {
	config := DefaultSnapshotSyncConfig()

	// Step 1: Fetch and apply snapshot
	client := NewSnapshotSyncClient(primaryAddr, authToken, config)
	snapshot, err := client.FetchAndApplySnapshot(ctx, store, walStream)
	if err != nil {
		return fmt.Errorf("snapshot bootstrap failed: %w", err)
	}

	fmt.Printf("[Bootstrap] Snapshot applied, WAL sequence: %d\n", snapshot.WALSequence)

	// Step 2: Set up WAL streaming from snapshot sequence
	// The FollowerReplicator will catch up from here
	walClient := NewWALStreamClient(primaryAddr, authToken)
	walClient.lastSeq = snapshot.WALSequence

	fmt.Printf("[Bootstrap] Ready to stream WAL from sequence %d\n", snapshot.WALSequence)

	return nil
}
