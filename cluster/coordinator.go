package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// ===========================================================================================
// COORDINATOR SERVER
// HTTP API server that wraps DistributedVectorDB
// ===========================================================================================

// CoordinatorServer provides HTTP API for the distributed vectordb
type CoordinatorServer struct {
	distributed *DistributedVectorDB
	embedder    Embedder
	server      *http.Server
	addr        string

	// Production features
	failoverMgr *FailoverManager
	metrics     MetricsProvider
	authMw      AuthProvider
}

// CoordinatorServerConfig configures the coordinator server
type CoordinatorServerConfig struct {
	ListenAddr string
	Embedder   Embedder

	// Distributed config
	NumShards         int
	ReplicationFactor int
	ReadStrategy      ReadStrategy

	// Quorum config
	CoordinatorID    string   // This coordinator's unique ID
	PeerCoordinators []string // Peer coordinator addresses for voting

	// Production features
	EnableFailover bool
	FailoverConfig FailoverConfig
	EnableMetrics  bool
	EnableAuth     bool
	APIKeyMgr      interface{} // Opaque; passed to Deps.NewAuth
	JWTMgr         interface{} // Opaque; passed to Deps.NewAuth

	// Dependencies from main package
	Deps *Deps
}

// NewCoordinatorServer creates a new coordinator server
func NewCoordinatorServer(cfg CoordinatorServerConfig) *CoordinatorServer {
	// Default embedder
	if cfg.Embedder == nil && cfg.Deps != nil && cfg.Deps.NewEmbedder != nil {
		cfg.Embedder = cfg.Deps.NewEmbedder(384) // Default dimension
	}

	// Create distributed vectordb
	distributed := NewDistributedVectorDB(DistributedConfig{
		NumShards:         cfg.NumShards,
		ReplicationFactor: cfg.ReplicationFactor,
		ReadStrategy:      cfg.ReadStrategy,
		CoordinatorID:     cfg.CoordinatorID,
		PeerCoordinators:  cfg.PeerCoordinators,
	})

	c := &CoordinatorServer{
		distributed: distributed,
		embedder:    cfg.Embedder,
		addr:        cfg.ListenAddr,
	}

	// Initialize production features
	if cfg.EnableMetrics && cfg.Deps != nil && cfg.Deps.NewMetrics != nil {
		c.metrics = cfg.Deps.NewMetrics()
		fmt.Println("Prometheus metrics enabled at /metrics")
	}

	if cfg.EnableAuth && cfg.Deps != nil && cfg.Deps.NewAuth != nil {
		c.authMw = cfg.Deps.NewAuth(cfg.APIKeyMgr, cfg.JWTMgr)
		fmt.Println("Authentication enabled (API keys + JWT)")
	}

	if cfg.EnableFailover {
		c.failoverMgr = NewFailoverManager(distributed, cfg.FailoverConfig)
		fmt.Printf("Automatic failover configured (threshold: %v)\n", cfg.FailoverConfig.UnhealthyThreshold)
	}

	// Setup HTTP routes
	mux := http.NewServeMux()

	// Metrics endpoint (no auth required for monitoring)
	if c.metrics != nil {
		mux.Handle("/metrics", c.metrics.Handler())
	}

	// Helper to wrap handlers with auth if enabled
	wrapAuth := func(handler http.HandlerFunc, permissions ...string) http.HandlerFunc {
		if c.authMw != nil {
			return c.authMw.Middleware(permissions...)(handler)
		}
		return handler
	}

	// Client API endpoints (require "read" or "write" permissions)
	mux.HandleFunc("/insert", wrapAuth(c.handleInsert, "write"))
	mux.HandleFunc("/batch_insert", wrapAuth(c.handleBatchInsert, "write"))
	mux.HandleFunc("/query", wrapAuth(c.handleQuery, "read"))
	mux.HandleFunc("/delete", wrapAuth(c.handleDelete, "write"))
	mux.HandleFunc("/health", c.handleHealth) // No auth for health checks

	// Admin API endpoints (require "admin" permission)
	mux.HandleFunc("/admin/register_shard", wrapAuth(c.handleRegisterShard, "admin"))
	mux.HandleFunc("/admin/unregister_shard", wrapAuth(c.handleUnregisterShard, "admin"))
	mux.HandleFunc("/admin/heartbeat", c.handleHeartbeat) // No auth for heartbeats from shards
	mux.HandleFunc("/admin/cluster_status", wrapAuth(c.handleClusterStatus, "admin"))
	mux.HandleFunc("/admin/failover_stats", wrapAuth(c.handleFailoverStats, "admin"))
	mux.HandleFunc("/admin/failover_trigger", wrapAuth(c.handleFailoverTrigger, "admin"))

	// Internal coordinator-to-coordinator endpoints (no auth for inter-coordinator communication)
	mux.HandleFunc("/internal/vote", c.handleVoteRequest)

	// Wrap entire handler with metrics middleware if enabled
	var handler http.Handler = mux
	if c.metrics != nil {
		handler = c.metrics.HTTPMiddleware(handler)
	}

	c.server = &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
	}

	return c
}

// Start starts the coordinator server
func (c *CoordinatorServer) Start(ctx context.Context) error {
	// Start failover manager if enabled
	if c.failoverMgr != nil {
		if err := c.failoverMgr.Start(ctx); err != nil {
			return fmt.Errorf("failed to start failover manager: %w", err)
		}
	}

	go func() {
		fmt.Printf("Coordinator HTTP API listening on %s\n", c.addr)
		if err := c.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("Coordinator server error: %v\n", err)
		}
	}()

	// Handle context cancellation
	go func() {
		<-ctx.Done()
		_ = c.Shutdown(context.Background())
	}()

	return nil
}

// Shutdown gracefully shuts down the coordinator
func (c *CoordinatorServer) Shutdown(ctx context.Context) error {
	fmt.Println("Shutting down coordinator...")

	// Stop failover manager
	if c.failoverMgr != nil {
		c.failoverMgr.Stop()
	}

	// Shutdown distributed vectordb
	c.distributed.Shutdown()

	// Shutdown HTTP server
	if c.server != nil {
		return c.server.Shutdown(ctx)
	}

	return nil
}

// ===========================================================================================
// CLIENT API HANDLERS
// ===========================================================================================

// handleInsert handles POST /insert
func (c *CoordinatorServer) handleInsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Doc        string            `json:"doc"`
		ID         string            `json:"id"`
		Meta       map[string]string `json:"meta"`
		Collection string            `json:"collection"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Doc == "" {
		http.Error(w, "doc required", http.StatusBadRequest)
		return
	}

	if req.Collection == "" {
		req.Collection = "default"
	}

	// Insert via distributed vectordb
	id, err := c.distributed.Add(req.Doc, req.ID, req.Meta, req.Collection)
	if err != nil {
		http.Error(w, fmt.Sprintf("insert failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id":         id,
		"collection": req.Collection,
	})
}

// handleBatchInsert handles POST /batch_insert
func (c *CoordinatorServer) handleBatchInsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Docs []struct {
			Doc        string            `json:"doc"`
			ID         string            `json:"id"`
			Meta       map[string]string `json:"meta"`
			Collection string            `json:"collection"`
		} `json:"docs"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	ids := make([]string, 0, len(req.Docs))
	var errors []string

	for i, doc := range req.Docs {
		if doc.Doc == "" {
			errors = append(errors, fmt.Sprintf("doc %d: empty document", i))
			continue
		}

		collection := doc.Collection
		if collection == "" {
			collection = "default"
		}

		id, err := c.distributed.Add(doc.Doc, doc.ID, doc.Meta, collection)
		if err != nil {
			errors = append(errors, fmt.Sprintf("doc %d: %v", i, err))
			continue
		}

		ids = append(ids, id)
	}

	response := map[string]any{"ids": ids}
	if len(errors) > 0 {
		response["errors"] = errors
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleQuery handles POST /query
func (c *CoordinatorServer) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query       string            `json:"query"`
		TopK        int               `json:"top_k"`
		Collections []string          `json:"collections"`
		MetaFilter  map[string]string `json:"meta_filter"`
		Mode        string            `json:"mode"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	if req.TopK == 0 {
		req.TopK = 10
	}

	if req.Mode == "" {
		req.Mode = "hybrid"
	}

	// Query via distributed vectordb
	results, err := c.distributed.Query(req.Query, req.TopK, req.Collections, req.MetaFilter, req.Mode)
	if err != nil {
		http.Error(w, fmt.Sprintf("query failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to response format
	ids := make([]string, len(results))
	docs := make([]string, len(results))
	scores := make([]float64, len(results))
	metas := make([]map[string]string, len(results))

	for i, result := range results {
		ids[i] = result["id"].(string)
		docs[i] = result["doc"].(string)
		scores[i] = result["score"].(float64)
		if meta, ok := result["meta"].(map[string]string); ok {
			metas[i] = meta
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"ids":    ids,
		"docs":   docs,
		"scores": scores,
		"meta":   metas,
		"count":  len(results),
	})
}

// handleDelete handles POST /delete
func (c *CoordinatorServer) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ID         string `json:"id"`
		Collection string `json:"collection"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	if req.ID == "" {
		http.Error(w, "id required", http.StatusBadRequest)
		return
	}

	if req.Collection == "" {
		req.Collection = "default"
	}

	// Delete via distributed vectordb
	if err := c.distributed.Delete(req.ID, req.Collection); err != nil {
		http.Error(w, fmt.Sprintf("delete failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"deleted":    req.ID,
		"collection": req.Collection,
	})
}

// handleHealth handles GET /health
func (c *CoordinatorServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"ok":        true,
		"timestamp": time.Now().Unix(),
	})
}

// ===========================================================================================
// ADMIN API HANDLERS
// ===========================================================================================

// handleRegisterShard handles POST /admin/register_shard
func (c *CoordinatorServer) handleRegisterShard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var node ShardNode
	if err := json.NewDecoder(r.Body).Decode(&node); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	if err := c.distributed.RegisterShard(&node); err != nil {
		http.Error(w, fmt.Sprintf("registration failed: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":  "registered",
		"node_id": node.NodeID,
	})
}

// handleUnregisterShard handles POST /admin/unregister_shard
func (c *CoordinatorServer) handleUnregisterShard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		NodeID string `json:"node_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	if err := c.distributed.UnregisterShard(req.NodeID); err != nil {
		http.Error(w, fmt.Sprintf("unregistration failed: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":  "unregistered",
		"node_id": req.NodeID,
	})
}

// handleHeartbeat handles POST /admin/heartbeat
func (c *CoordinatorServer) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		NodeID       string `json:"node_id"`
		ShardID      int    `json:"shard_id"`
		Healthy      bool   `json:"healthy"`
		VectorCount  int    `json:"vector_count"`
		DeletedCount int    `json:"deleted_count"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	// Update node health (simplified - in production you'd update metrics)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status": "received",
	})
}

// handleClusterStatus handles GET /admin/cluster_status
func (c *CoordinatorServer) handleClusterStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	status := c.distributed.GetClusterStatus()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// handleFailoverStats handles GET /admin/failover_stats
func (c *CoordinatorServer) handleFailoverStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if c.failoverMgr == nil {
		http.Error(w, "failover manager not enabled", http.StatusServiceUnavailable)
		return
	}

	stats := c.failoverMgr.GetFailoverStats()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// handleFailoverTrigger handles POST /admin/failover_trigger
func (c *CoordinatorServer) handleFailoverTrigger(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if c.failoverMgr == nil {
		http.Error(w, "failover manager not enabled", http.StatusServiceUnavailable)
		return
	}

	var req struct {
		ShardID int `json:"shard_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	if err := c.failoverMgr.ManualFailover(req.ShardID); err != nil {
		http.Error(w, fmt.Sprintf("failover failed: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":   "failover_triggered",
		"shard_id": req.ShardID,
	})
}

// ===========================================================================================
// INTERNAL COORDINATOR ENDPOINTS
// ===========================================================================================

// handleVoteRequest handles POST /internal/vote - receives vote requests from peer coordinators
func (c *CoordinatorServer) handleVoteRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check if quorum is enabled
	if c.distributed.quorum == nil {
		http.Error(w, "quorum not enabled on this coordinator", http.StatusServiceUnavailable)
		return
	}

	var req VoteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	// Process vote request
	response := c.distributed.quorum.HandleVoteRequest(&req)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RunCoordinator runs the coordinator server as a standalone process
func RunCoordinator(cfg CoordinatorServerConfig) {
	// Create coordinator
	coordinator := NewCoordinatorServer(cfg)

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	// Start server
	if err := coordinator.Start(ctx); err != nil {
		fmt.Printf("Failed to start coordinator: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Coordinator started. Press Ctrl+C to shutdown.")

	// Wait for shutdown signal
	sig := <-sigCh
	fmt.Printf("\nReceived signal %v, shutting down...\n", sig)

	cancel()

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := coordinator.Shutdown(shutdownCtx); err != nil {
		fmt.Printf("Shutdown error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Coordinator shutdown complete")
}
