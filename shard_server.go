package main

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
// Wraps VectorStore with HTTP API and replication support
// ===========================================================================================

// ShardServer is a vectordb shard instance with replication support
type ShardServer struct {
	mu sync.RWMutex

	nodeID   string
	shardID  int
	role     ReplicaRole
	httpAddr string

	store     *VectorStore
	embedder  Embedder
	reranker  Reranker
	indexPath string

	// Replication
	primaryAddr string       // For replicas: primary's HTTP address
	replicas    []string     // For primary: replica HTTP addresses
	walLog      []walEntry   // Replication log (operations since last sync)
	walMu       sync.Mutex   // Protects walLog
	lastSyncSeq int          // Last synced sequence number
	syncTicker  *time.Ticker // Periodic sync for replicas

	// WAL Streaming (new)
	walStream       *WALStream       // WAL stream buffer (for primaries)
	walStreamClient *WALStreamClient // WAL stream client (for replicas)

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
	Embedder Embedder
	Reranker Reranker
	APIToken string
}

// NewShardServer creates a new shard server
func NewShardServer(cfg ShardServerConfig) (*ShardServer, error) {
	// Load or initialize store
	store, loaded := loadOrInitStore(cfg.IndexPath, cfg.Capacity, cfg.Dimension)
	if loaded {
		fmt.Printf("Loaded shard %d (%s) from %s: %d vectors\n", cfg.ShardID, cfg.Role, cfg.IndexPath, store.Count)
	} else {
		fmt.Printf("Initialized new shard %d (%s): capacity=%d, dim=%d\n", cfg.ShardID, cfg.Role, cfg.Capacity, cfg.Dimension)
	}

	// Set API token if provided
	if cfg.APIToken != "" {
		store.apiToken = cfg.APIToken
	}

	// Default embedder if not provided
	if cfg.Embedder == nil {
		cfg.Embedder = NewHashEmbedder(cfg.Dimension)
	}
	if cfg.Reranker == nil {
		cfg.Reranker = &SimpleReranker{Embedder: cfg.Embedder}
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
		primaryAddr:     cfg.PrimaryAddr,
		replicas:        cfg.Replicas,
		walLog:          make([]walEntry, 0),
		coordinatorAddr: cfg.CoordinatorAddr,
	}

	// Initialize WAL streaming
	if cfg.Role == RolePrimary {
		s.walStream = NewWALStream()
		fmt.Printf("✅ WAL streaming enabled (primary mode)\n")
	} else if cfg.PrimaryAddr != "" {
		s.walStreamClient = NewWALStreamClient(cfg.PrimaryAddr)
		fmt.Printf("✅ WAL streaming client enabled (pulling from %s)\n", cfg.PrimaryAddr)
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
	baseHandler := newHTTPHandler(s.store, s.embedder, s.reranker, s.indexPath)

	// Add WAL streaming endpoint for primaries
	if s.role == RolePrimary && s.walStream != nil {
		mux.HandleFunc("/wal/stream", s.handleWALStream)
	}

	// Register migration handlers for shard rebalancing
	registerMigrationHandlers(mux, s.store)

	// Mount base handler on root
	mux.Handle("/", baseHandler)

	// Wrap with replication middleware if primary
	if s.role == RolePrimary && s.walStream != nil {
		return s.walStreamMiddleware(mux)
	}

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
	seq := s.walStream.Append(op, id, doc, coll, vec, meta)
	if seq%100 == 0 {
		fmt.Printf("📝 WAL: Appended operation (seq=%d, op=%s, id=%s)\n", seq, op, id)
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
	if err := s.ApplyEntries(entries); err != nil {
		return fmt.Errorf("failed to apply WAL entries: %w", err)
	}

	fmt.Printf("✅ Replica sync: Applied %d WAL entries (latest seq: %d)\n",
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
	s.store.RLock()
	vectorCount := s.store.Count
	deletedCount := len(s.store.Deleted)
	s.store.RUnlock()

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

// RunStandalone runs a shard server as a standalone process
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
