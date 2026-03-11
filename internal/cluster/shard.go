package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// ===========================================================================================
// VECTORDB SHARD SERVER
// Wraps Store with HTTP API and replication support
// ===========================================================================================

// ShardServer is a vectordb shard instance with replication support
type ShardServer struct {
	mu sync.RWMutex

	nodeID   string
	shardID  int
	role     ReplicaRole
	httpAddr string

	store     Store
	embedder  Embedder
	reranker  Reranker
	indexPath string

	// Dependencies from main package
	deps *Deps

	// Replication
	primaryAddr string       // For replicas: primary's HTTP address
	replicas    []string     // For primary: replica HTTP addresses
	walLog      []WalEntry   // Replication log (operations since last sync)
	walMu       sync.Mutex   // Protects walLog
	lastSyncSeq int          // Last synced sequence number
	syncTicker  *time.Ticker // Periodic sync for replicas

	// WAL Streaming (new)
	walStream       *WALStream       // WAL stream buffer (for primaries)
	walStreamClient *WALStreamClient // WAL stream client (for replicas)

	// Streaming snapshots (for primaries)
	snapshotManager *SnapshotSyncManager

	// HTTP server
	server *http.Server

	// Coordinator registration
	coordinatorAddr string
	heartbeatTicker *time.Ticker
}

// ShardServerConfig configures a shard server
type ShardServerConfig struct {
	NodeID          string
	ShardID         int
	Role            ReplicaRole
	HTTPAddr        string
	PrimaryAddr     string   // Required for replicas
	Replicas        []string // For primary: list of replica addresses
	CoordinatorAddr string   // Address to register with coordinator

	// VectorStore config
	Capacity  int
	Dimension int
	IndexPath string

	// Optional
	Embedder    Embedder
	Reranker    Reranker
	APIToken    string
	SnapshotDir string // Directory for streaming snapshots (primary only)

	// Dependencies from main package
	Deps *Deps
}

// NewShardServer creates a new shard server
func NewShardServer(cfg ShardServerConfig) (*ShardServer, error) {
	if cfg.Deps == nil {
		return nil, fmt.Errorf("Deps is required for ShardServer")
	}

	// Load or initialize store
	store, loaded := cfg.Deps.LoadOrInitStore(cfg.IndexPath, cfg.Capacity, cfg.Dimension)
	if loaded {
		fmt.Printf("Loaded shard %d (%s) from %s: %d vectors\n", cfg.ShardID, cfg.Role, cfg.IndexPath, store.StoreCount())
	} else {
		fmt.Printf("Initialized new shard %d (%s): capacity=%d, dim=%d\n", cfg.ShardID, cfg.Role, cfg.Capacity, cfg.Dimension)
	}

	// Set API token if provided
	if cfg.APIToken != "" {
		store.SetAPIToken(cfg.APIToken)
	}

	// Default embedder if not provided
	if cfg.Embedder == nil {
		cfg.Embedder = cfg.Deps.NewEmbedder(cfg.Dimension)
	}
	if cfg.Reranker == nil {
		cfg.Reranker = cfg.Deps.NewReranker(cfg.Embedder)
	}

	s := &ShardServer{
		nodeID:          cfg.NodeID,
		shardID:         cfg.ShardID,
		role:            cfg.Role,
		httpAddr:        cfg.HTTPAddr,
		store:           store,
		embedder:        cfg.Embedder,
		reranker:        cfg.Reranker,
		indexPath:       cfg.IndexPath,
		deps:            cfg.Deps,
		primaryAddr:     cfg.PrimaryAddr,
		replicas:        cfg.Replicas,
		walLog:          make([]WalEntry, 0),
		coordinatorAddr: cfg.CoordinatorAddr,
	}

	// Initialize WAL streaming
	if cfg.Role == RolePrimary {
		s.walStream = NewWALStream()
		store.SetWALHook(func(entry WalEntry) {
			if s.walStream != nil {
				s.walStream.Append(entry)
			}
		})
		fmt.Printf("WAL streaming enabled (primary mode)\n")

		// Initialize streaming snapshot manager if snapshot dir is configured
		if cfg.SnapshotDir != "" {
			snapCfg := DefaultSnapshotSyncConfig()
			snapCfg.SnapshotDir = cfg.SnapshotDir
			mgr, err := NewSnapshotSyncManager(store, s.walStream, snapCfg)
			if err != nil {
				return nil, fmt.Errorf("failed to create snapshot manager: %w", err)
			}
			s.snapshotManager = mgr
			fmt.Printf("Streaming snapshot manager enabled (dir=%s)\n", cfg.SnapshotDir)
		}
	} else if cfg.PrimaryAddr != "" {
		s.walStreamClient = NewWALStreamClient(cfg.PrimaryAddr, cfg.APIToken)
		fmt.Printf("WAL streaming client enabled (pulling from %s)\n", cfg.PrimaryAddr)
	}

	return s, nil
}

// Start starts the shard server
func (s *ShardServer) Start(ctx context.Context) error {
	// Setup HTTP handler
	handler := s.newHTTPHandler()

	s.server = &http.Server{
		Addr:         s.httpAddr,
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	// Start HTTP server
	go func() {
		fmt.Printf("Shard %d (%s) HTTP API listening on %s\n", s.shardID, s.role, s.httpAddr)
		if err := s.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("Shard server error: %v\n", err)
		}
	}()

	// If replica, start syncing from primary
	if s.role == RoleReplica && s.primaryAddr != "" {
		s.syncTicker = time.NewTicker(1 * time.Second)
		go s.syncLoop()
	}

	// Register with coordinator if configured
	if s.coordinatorAddr != "" {
		if err := s.registerWithCoordinator(); err != nil {
			fmt.Printf("Warning: Failed to register with coordinator: %v\n", err)
		}

		// Start heartbeat
		s.heartbeatTicker = time.NewTicker(5 * time.Second)
		go s.heartbeatLoop()
	}

	// Handle context cancellation
	go func() {
		<-ctx.Done()
		_ = s.Shutdown(context.Background())
	}()

	return nil
}

// Shutdown gracefully shuts down the shard server
func (s *ShardServer) Shutdown(ctx context.Context) error {
	fmt.Printf("Shutting down shard %d (%s)...\n", s.shardID, s.role)

	// Stop tickers
	if s.syncTicker != nil {
		s.syncTicker.Stop()
	}
	if s.heartbeatTicker != nil {
		s.heartbeatTicker.Stop()
	}

	// Shutdown HTTP server
	if s.server != nil {
		if err := s.server.Shutdown(ctx); err != nil {
			return err
		}
	}

	// Save final snapshot
	fmt.Println("Saving final snapshot...")
	if err := s.store.Save(s.indexPath); err != nil {
		return fmt.Errorf("failed to save final snapshot: %w", err)
	}

	fmt.Println("Shard shutdown complete")
	return nil
}

// newHTTPHandler creates the HTTP handler for this shard
func (s *ShardServer) newHTTPHandler() http.Handler {
	mux := http.NewServeMux()

	// Use existing HTTP handler from server.go as base
	baseHandler := s.deps.NewHTTPHandler(s.store, s.embedder, s.reranker, s.indexPath)

	// Add WAL streaming endpoint for primaries
	if s.role == RolePrimary && s.walStream != nil {
		mux.HandleFunc("/wal/stream", s.HandleWALStream)
	}

	// Add snapshot endpoint for full sync (primaries only)
	if s.role == RolePrimary {
		mux.HandleFunc("/internal/snapshot", s.handleSnapshot)
	}

	// Add streaming snapshot endpoints (primaries with snapshot manager)
	if s.snapshotManager != nil {
		mux.HandleFunc("/internal/snapshot/stream/download", func(w http.ResponseWriter, r *http.Request) {
			if err := AuthorizeWALStream(s.store, r); err != nil {
				http.Error(w, err.Error(), http.StatusUnauthorized)
				return
			}
			s.snapshotManager.HandleSnapshotDownload(w, r)
		})
		mux.HandleFunc("/internal/snapshot/stream/upload", func(w http.ResponseWriter, r *http.Request) {
			if err := AuthorizeWALStream(s.store, r); err != nil {
				http.Error(w, err.Error(), http.StatusUnauthorized)
				return
			}
			s.snapshotManager.HandleSnapshotUpload(w, r)
		})
	}

	// Register migration handlers for shard rebalancing
	s.deps.RegisterMigrationHandlers(mux, s.store)

	// Mount base handler on root
	mux.Handle("/", baseHandler)

	return mux
}

// walStreamMiddleware wraps handlers to capture writes in WAL stream
func (s *ShardServer) walStreamMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Capture write operations in WAL stream
		if r.Method == http.MethodPost && (r.URL.Path == "/insert" || r.URL.Path == "/batch_insert" || r.URL.Path == "/delete") {
			// Read and parse body
			var buf bytes.Buffer
			body := io.TeeReader(r.Body, &buf)
			bodyBytes, _ := ioutil.ReadAll(body)
			r.Body = ioutil.NopCloser(bytes.NewReader(bodyBytes))

			// Parse request to extract WAL info
			var req map[string]interface{}
			json.Unmarshal(bodyBytes, &req)

			// Serve the request first
			next.ServeHTTP(w, r)

			// After successful processing, append to WAL stream
			go s.appendToWAL(r.URL.Path, req)
		} else {
			next.ServeHTTP(w, r)
		}
	})
}

// appendToWAL adds a write operation to the WAL stream
func (s *ShardServer) appendToWAL(path string, req map[string]interface{}) {
	if s.walStream == nil {
		return
	}

	op := "insert"
	if path == "/delete" {
		op = "delete"
	}

	// Extract fields from request
	id, _ := req["id"].(string)
	doc, _ := req["doc"].(string)
	coll, _ := req["collection"].(string)
	if coll == "" {
		coll = "default"
	}

	// For insert/batch operations, we need to embed and get vectors
	var vec []float32
	if op == "insert" && doc != "" {
		v, err := s.embedder.Embed(doc)
		if err == nil {
			vec = v
		}
	}

	// Extract metadata
	meta := make(map[string]string)
	if m, ok := req["meta"].(map[string]interface{}); ok {
		for k, v := range m {
			if str, ok := v.(string); ok {
				meta[k] = str
			}
		}
	}

	// Append to WAL
	entry := WalEntry{
		Op:     op,
		ID:     id,
		Doc:    doc,
		Coll:   coll,
		Vec:    vec,
		Meta:   meta,
		Tenant: "",
	}
	seq := s.walStream.Append(entry)
	if seq%100 == 0 {
		fmt.Printf("WAL: Appended operation (seq=%d, op=%s, id=%s)\n", seq, op, id)
	}
}

// syncLoop periodically syncs from primary (for replicas)
func (s *ShardServer) syncLoop() {
	for range s.syncTicker.C {
		if err := s.syncFromPrimary(); err != nil {
			fmt.Printf("Sync from primary failed: %v\n", err)
		}
	}
}

// syncFromPrimary pulls latest WAL from primary and applies it
func (s *ShardServer) syncFromPrimary() error {
	if s.walStreamClient == nil {
		return nil
	}

	// Pull latest WAL entries from primary
	entries, err := s.walStreamClient.PullLatest()
	if err != nil {
		return fmt.Errorf("failed to pull WAL: %w", err)
	}

	// No new entries
	if len(entries) == 0 {
		return nil
	}

	// Apply entries to local store
	lastApplied, err := s.ApplyEntries(entries)
	if lastApplied > 0 {
		s.walStreamClient.Advance(lastApplied)
	}
	if err != nil {
		return fmt.Errorf("failed to apply WAL entries: %w", err)
	}

	fmt.Printf("Replica sync: Applied %d WAL entries (latest seq: %d)\n",
		len(entries), entries[len(entries)-1].Seq)

	return nil
}

// registerWithCoordinator registers this shard with the coordinator
func (s *ShardServer) registerWithCoordinator() error {
	node := ShardNode{
		NodeID:   s.nodeID,
		ShardID:  s.shardID,
		Role:     s.role,
		HTTPAddr: "http://" + s.httpAddr,
		Healthy:  true,
	}

	body, _ := json.Marshal(node)
	resp, err := http.Post(s.coordinatorAddr+"/admin/register_shard", "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("registration failed with status %d", resp.StatusCode)
	}

	fmt.Printf("Registered shard %d (%s) with coordinator\n", s.shardID, s.role)
	return nil
}

// heartbeatLoop sends periodic heartbeats to coordinator
func (s *ShardServer) heartbeatLoop() {
	for range s.heartbeatTicker.C {
		s.sendHeartbeat()
	}
}

// sendHeartbeat sends health status to coordinator
func (s *ShardServer) sendHeartbeat() {
	s.store.StoreRLock()
	vectorCount := s.store.StoreCount()
	deletedCount := len(s.store.StoreDeleted())
	s.store.StoreRUnlock()

	heartbeat := map[string]any{
		"node_id":       s.nodeID,
		"shard_id":      s.shardID,
		"healthy":       true,
		"vector_count":  vectorCount,
		"deleted_count": deletedCount,
		"timestamp":     time.Now().Unix(),
	}

	body, _ := json.Marshal(heartbeat)
	client := &http.Client{Timeout: 3 * time.Second}
	_, err := client.Post(s.coordinatorAddr+"/admin/heartbeat", "application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Printf("Heartbeat failed: %v\n", err)
	}
}

// LoadSnapshot loads a full snapshot into the shard store
func (s *ShardServer) LoadSnapshot(snapshot *Snapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Clear existing data via store interface
	s.store.StoreLock()
	clearSnap := &SnapshotData{
		Count:   0,
		IDs:     make([]string, 0),
		Docs:    make([]string, 0),
		Data:    make([]float32, 0),
		Seqs:    make([]uint64, 0),
		Meta:    make(map[uint64]map[string]string),
		Deleted: make(map[uint64]bool),
		Coll:    make(map[uint64]string),
		IDToIx:  make(map[uint64]int),
	}
	s.store.LoadSnapshotData(clearSnap)
	s.store.StoreUnlock()

	// Load snapshot data using store.Add() for each vector
	for _, entry := range snapshot.Vectors {
		meta := map[string]string{}
		if entry.Metadata != "" {
			meta["raw"] = entry.Metadata
		}
		_, err := s.store.Add(entry.Vector, "", entry.ID, meta, "default", "")
		if err != nil {
			return fmt.Errorf("failed to add vector %s from snapshot: %w", entry.ID, err)
		}
	}

	fmt.Printf("[ShardServer] Loaded snapshot: %d vectors at seq=%d\n",
		len(snapshot.Vectors), snapshot.Sequence)

	return nil
}

// CreateSnapshot creates a snapshot of the current shard state
func (s *ShardServer) CreateSnapshot() *Snapshot {
	hashID := s.deps.HashID

	s.store.StoreRLock()
	defer s.store.StoreRUnlock()

	snapshot := &Snapshot{
		ShardID:   s.shardID,
		Timestamp: time.Now(),
		Vectors:   make([]SnapshotEntry, 0, s.store.StoreCount()),
	}

	// Get current WAL sequence if available
	if s.walStream != nil {
		snapshot.Sequence = s.walStream.GetLatestSeq()
	}

	ids := s.store.StoreIDs()
	data := s.store.StoreData()
	dim := s.store.StoreDim()
	deleted := s.store.StoreDeleted()
	meta := s.store.StoreMeta()

	// Export all vectors
	for i, id := range ids {
		hash := hashID(id)
		if deleted[hash] {
			continue
		}

		// Extract vector slice from flat Data array
		startIdx := i * dim
		endIdx := startIdx + dim
		var vec []float32
		if endIdx <= len(data) {
			vec = make([]float32, dim)
			copy(vec, data[startIdx:endIdx])
		}

		entry := SnapshotEntry{
			ID:     id,
			Vector: vec,
		}
		if m, ok := meta[hash]; ok {
			if raw, ok := m["raw"]; ok {
				entry.Metadata = raw
			}
		}
		snapshot.Vectors = append(snapshot.Vectors, entry)
	}

	// Compute checksum
	snapshot.Checksum = snapshot.computeChecksum()

	return snapshot
}

// handleSnapshot handles snapshot requests from replicas
func (s *ShardServer) handleSnapshot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := AuthorizeWALStream(s.store, r); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	snapshot := s.CreateSnapshot()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(snapshot)

	fmt.Printf("[ShardServer] Served snapshot: %d vectors at seq=%d\n",
		len(snapshot.Vectors), snapshot.Sequence)
}

// RunStandaloneShard runs a shard server as a standalone process
func RunStandaloneShard(cfg ShardServerConfig) {
	// Create shard server
	shard, err := NewShardServer(cfg)
	if err != nil {
		fmt.Printf("Failed to create shard server: %v\n", err)
		os.Exit(1)
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	// Start server
	if err := shard.Start(ctx); err != nil {
		fmt.Printf("Failed to start shard server: %v\n", err)
		os.Exit(1)
	}

	// Wait for shutdown signal
	sig := <-sigCh
	fmt.Printf("\nReceived signal %v, shutting down...\n", sig)

	cancel()

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := shard.Shutdown(shutdownCtx); err != nil {
		fmt.Printf("Shutdown error: %v\n", err)
		os.Exit(1)
	}
}
