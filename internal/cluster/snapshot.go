package cluster

import (
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
// Complete store snapshot for bootstrapping new replicas
// ===========================================================================================

// StoreSnapshot represents a complete store snapshot
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

	store     Store
	config    SnapshotSyncConfig
	walStream *WALStream

	// State
	latestSnapshot *StoreSnapshot
	inProgress     bool
}

// NewSnapshotSyncManager creates a new snapshot sync manager
func NewSnapshotSyncManager(store Store, walStream *WALStream, config SnapshotSyncConfig) (*SnapshotSyncManager, error) {
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
	ssm.store.StoreRLock()
	defer ssm.store.StoreRUnlock()

	// Get current WAL sequence
	var walSeq uint64
	if ssm.walStream != nil {
		walSeq = ssm.walStream.GetLatestSeq()
	}

	// Get snapshot data from store
	snapData := ssm.store.CreateSnapshotData()

	// Create snapshot metadata
	snapshot := &StoreSnapshot{
		ID:          snapshotID,
		CreatedAt:   now,
		WALSequence: walSeq,
		VectorCount: snapData.Count,
		Dimension:   snapData.Dim,
		Collections: countCollectionsFromData(snapData),
		Compressed:  ssm.config.CompressionEnabled,
		Version:     1,
	}

	// Create snapshot file
	snapshotPath := ssm.getSnapshotPath(snapshotID)
	if err := ssm.writeSnapshot(snapshotPath, snapshot, snapData); err != nil {
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

// countCollectionsFromData counts unique collections from snapshot data
func countCollectionsFromData(snapData *SnapshotData) int {
	collections := make(map[string]bool)
	for _, coll := range snapData.Coll {
		collections[coll] = true
	}
	return len(collections)
}

// snapshotFileData is the internal snapshot format
type snapshotFileData struct {
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

// writeSnapshot serializes store state to a file
func (ssm *SnapshotSyncManager) writeSnapshot(path string, metadata *StoreSnapshot, snapData *SnapshotData) error {
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

	// Filter out deleted vectors from the snapshot to prevent monotonic growth.
	// Build a set of live indices (not marked as deleted).
	liveIndices := make(map[int]bool, len(snapData.Seqs))
	for i, seq := range snapData.Seqs {
		if !snapData.Deleted[seq] {
			liveIndices[i] = true
		}
	}

	// Build filtered slices containing only live vectors
	filteredData := make([]float32, 0, len(liveIndices)*snapData.Dim)
	filteredDocs := make([]string, 0, len(liveIndices))
	filteredIDs := make([]string, 0, len(liveIndices))
	filteredSeqs := make([]uint64, 0, len(liveIndices))
	filteredMeta := make(map[uint64]map[string]string, len(liveIndices))
	filteredNumMeta := make(map[uint64]map[string]float64, len(liveIndices))
	filteredTimeMeta := make(map[uint64]map[string]time.Time, len(liveIndices))
	filteredColl := make(map[uint64]string, len(liveIndices))
	filteredTenantID := make(map[uint64]string, len(liveIndices))
	filteredLexTF := make(map[uint64]map[string]int, len(liveIndices))
	filteredDocLen := make(map[uint64]int, len(liveIndices))

	for i := 0; i < len(snapData.Seqs); i++ {
		if !liveIndices[i] {
			continue
		}
		seq := snapData.Seqs[i]

		// Copy vector data (dim floats per vector)
		start := i * snapData.Dim
		end := start + snapData.Dim
		if end <= len(snapData.Data) {
			filteredData = append(filteredData, snapData.Data[start:end]...)
		}
		if i < len(snapData.Docs) {
			filteredDocs = append(filteredDocs, snapData.Docs[i])
		}
		if i < len(snapData.IDs) {
			filteredIDs = append(filteredIDs, snapData.IDs[i])
		}
		filteredSeqs = append(filteredSeqs, seq)

		if m, ok := snapData.Meta[seq]; ok {
			filteredMeta[seq] = m
		}
		if m, ok := snapData.NumMeta[seq]; ok {
			filteredNumMeta[seq] = m
		}
		if m, ok := snapData.TimeMeta[seq]; ok {
			filteredTimeMeta[seq] = m
		}
		if c, ok := snapData.Coll[seq]; ok {
			filteredColl[seq] = c
		}
		if t, ok := snapData.TenantID[seq]; ok {
			filteredTenantID[seq] = t
		}
		if tf, ok := snapData.LexTF[seq]; ok {
			filteredLexTF[seq] = tf
		}
		if dl, ok := snapData.DocLen[seq]; ok {
			filteredDocLen[seq] = dl
		}
	}

	// Rebuild DF from only live documents' term frequencies to remove dead terms
	filteredDF := make(map[string]int, len(snapData.DF))
	filteredSumDocL := 0
	for seq, tf := range filteredLexTF {
		for term := range tf {
			filteredDF[term]++
		}
		if dl, ok := filteredDocLen[seq]; ok {
			filteredSumDocL += dl
		}
	}

	// Update metadata vector count to reflect filtered state
	metadata.VectorCount = len(filteredSeqs)

	// Write snapshot data structure (no Deleted map — deleted vectors are simply absent)
	data := &snapshotFileData{
		Metadata: metadata,
		Data:     filteredData,
		Docs:     filteredDocs,
		IDs:      filteredIDs,
		Seqs:     filteredSeqs,
		Meta:     filteredMeta,
		NumMeta:  filteredNumMeta,
		TimeMeta: filteredTimeMeta,
		Deleted:  nil, // Deleted vectors are excluded, not tombstoned
		Coll:     filteredColl,
		TenantID: filteredTenantID,
		LexTF:    filteredLexTF,
		DocLen:   filteredDocLen,
		DF:       filteredDF,
		SumDocL:  filteredSumDocL,
	}

	encoder := json.NewEncoder(writer)
	return encoder.Encode(data)
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
	var data snapshotFileData
	decoder := json.NewDecoder(reader)
	if err := decoder.Decode(&data); err != nil {
		return fmt.Errorf("failed to decode snapshot: %w", err)
	}

	// Build SnapshotData and load into store
	snapData := &SnapshotData{
		Count:    data.Metadata.VectorCount,
		Dim:      data.Metadata.Dimension,
		IDs:      data.IDs,
		Docs:     data.Docs,
		Data:     data.Data,
		Seqs:     data.Seqs,
		Meta:     data.Meta,
		NumMeta:  data.NumMeta,
		TimeMeta: data.TimeMeta,
		Deleted:  data.Deleted,
		Coll:     data.Coll,
		TenantID: data.TenantID,
		LexTF:    data.LexTF,
		DocLen:   data.DocLen,
		DF:       data.DF,
		SumDocL:  data.SumDocL,
	}

	// Rebuild idToIx mapping
	snapData.IDToIx = make(map[uint64]int)
	for i, seq := range data.Seqs {
		snapData.IDToIx[seq] = i
	}

	// Apply to store via interface
	ssm.store.StoreLock()
	defer ssm.store.StoreUnlock()

	if err := ssm.store.LoadSnapshotData(snapData); err != nil {
		return fmt.Errorf("failed to load snapshot data: %w", err)
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
	var data snapshotFileData
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
	for i := ssm.config.MaxSnapshots; i < len(snapshots); i++ {
		path := ssm.getSnapshotPath(snapshots[i].ID)
		os.Remove(path)
		fmt.Printf("[SnapshotSync] Cleaned up old snapshot: %s\n", snapshots[i].ID)
	}
}

// ===========================================================================================
// HTTP HANDLERS FOR SNAPSHOT TRANSFER
// ===========================================================================================

// HandleSnapshotList handles GET /snapshot/list - returns available snapshots
func (ssm *SnapshotSyncManager) HandleSnapshotList(w http.ResponseWriter, r *http.Request) {
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

// HandleSnapshotCreate handles POST /snapshot/create - creates new snapshot
func (ssm *SnapshotSyncManager) HandleSnapshotCreate(w http.ResponseWriter, r *http.Request) {
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

// HandleSnapshotDownload handles GET /snapshot/download?id=xxx - streams snapshot
func (ssm *SnapshotSyncManager) HandleSnapshotDownload(w http.ResponseWriter, r *http.Request) {
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

// HandleSnapshotUpload handles POST /snapshot/upload - receives snapshot from primary
func (ssm *SnapshotSyncManager) HandleSnapshotUpload(w http.ResponseWriter, r *http.Request) {
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

// FetchAndApplySnapshot fetches and applies snapshot to a store.
// Streams the response body to a temp file to avoid OOM on large snapshots.
func (ssc *SnapshotSyncClient) FetchAndApplySnapshot(ctx context.Context, store Store, walStream *WALStream) (*StoreSnapshot, error) {
	fmt.Printf("[SnapshotSyncClient] Fetching snapshot from %s...\n", ssc.primaryAddr)

	url := fmt.Sprintf("%s/snapshot/download", ssc.primaryAddr)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	if ssc.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+ssc.authToken)
	}

	resp, err := ssc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch snapshot: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("snapshot download failed: %s - %s", resp.Status, string(body))
	}

	// Stream to temp file instead of buffering in memory
	tmpFile, err := os.CreateTemp("", "snapshot-sync-*.tmp")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	written, err := io.Copy(tmpFile, resp.Body)
	if err != nil {
		tmpFile.Close()
		return nil, fmt.Errorf("failed to stream snapshot to file: %w", err)
	}
	tmpFile.Close()

	snapshotID := resp.Header.Get("X-Snapshot-ID")
	fmt.Printf("[SnapshotSyncClient] Downloaded snapshot %s (%d bytes) to temp file\n", snapshotID, written)

	// Open temp file for reading
	file, err := os.Open(tmpPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open temp snapshot: %w", err)
	}
	defer file.Close()

	var reader io.Reader = file

	// Check if compressed
	header := make([]byte, 2)
	if _, err := file.Read(header); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}
	file.Seek(0, 0)

	if header[0] == 0x1f && header[1] == 0x8b {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return nil, fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()
		reader = gzReader
	}

	var fileData snapshotFileData
	if err := json.NewDecoder(reader).Decode(&fileData); err != nil {
		return nil, fmt.Errorf("failed to decode snapshot: %w", err)
	}

	// Build SnapshotData
	snapData := &SnapshotData{
		Count:    fileData.Metadata.VectorCount,
		Dim:      fileData.Metadata.Dimension,
		IDs:      fileData.IDs,
		Docs:     fileData.Docs,
		Data:     fileData.Data,
		Seqs:     fileData.Seqs,
		Meta:     fileData.Meta,
		NumMeta:  fileData.NumMeta,
		TimeMeta: fileData.TimeMeta,
		Deleted:  fileData.Deleted,
		Coll:     fileData.Coll,
		TenantID: fileData.TenantID,
		LexTF:    fileData.LexTF,
		DocLen:   fileData.DocLen,
		DF:       fileData.DF,
		SumDocL:  fileData.SumDocL,
	}

	// Rebuild idToIx mapping
	snapData.IDToIx = make(map[uint64]int)
	for i, seq := range fileData.Seqs {
		snapData.IDToIx[seq] = i
	}

	// Apply to store via interface
	store.StoreLock()
	defer store.StoreUnlock()

	if err := store.LoadSnapshotData(snapData); err != nil {
		return nil, fmt.Errorf("failed to load snapshot data: %w", err)
	}

	fmt.Printf("[SnapshotSyncClient] Applied snapshot %s (vectors=%d, wal_seq=%d)\n",
		snapshotID, fileData.Metadata.VectorCount, fileData.Metadata.WALSequence)

	return fileData.Metadata, nil
}

// ===========================================================================================
// INTEGRATION WITH FOLLOWER REPLICATOR
// ===========================================================================================

// BootstrapFromPrimary performs full bootstrap: snapshot + WAL catchup
func BootstrapFromPrimary(ctx context.Context, store Store, walStream *WALStream, primaryAddr, authToken string) error {
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
