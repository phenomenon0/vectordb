package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"agentscope/transport"
)

// =============================================================================
// VectorDB Transport Adapter
// =============================================================================

// VectorDBTransport wraps VectorDB with multi-protocol support.
type VectorDBTransport struct {
	store       *VectorStore
	embedder    Embedder
	reranker    Reranker
	indexPath   string
	server      *transport.Server
	httpHandler http.Handler
}

// TransportConfig configures the transport layer.
type TransportConfig struct {
	HTTPAddr     string
	GRPCAddr     string
	WSAddr       string
	UnixSocket   string
	EnableAuth   bool
	JWTSecret    string
	RateLimit    float64
	RateBurst    int
}

// DefaultTransportConfig returns default configuration.
func DefaultTransportConfig() TransportConfig {
	return TransportConfig{
		HTTPAddr:   ":8080",
		GRPCAddr:   ":8081",
		WSAddr:     "",      // Shares HTTP port if empty
		UnixSocket: "",      // Optional
		RateLimit:  100,
		RateBurst:  200,
	}
}

// NewVectorDBTransport creates a new transport-enabled VectorDB.
func NewVectorDBTransport(store *VectorStore, embedder Embedder, reranker Reranker, indexPath string, cfg TransportConfig) *VectorDBTransport {
	// Create underlying HTTP handler
	httpHandler := newHTTPHandler(store, embedder, reranker, indexPath)

	// Create transport server config
	transportCfg := transport.Config{
		HTTPAddr:   cfg.HTTPAddr,
		GRPCAddr:   cfg.GRPCAddr,
		UnixSocket: cfg.UnixSocket,
	}

	// Create transport server
	srv := transport.NewServer(transportCfg)

	vt := &VectorDBTransport{
		store:       store,
		embedder:    embedder,
		reranker:    reranker,
		indexPath:   indexPath,
		server:      srv,
		httpHandler: httpHandler,
	}

	// Register methods
	vt.registerMethods()

	// Add middleware
	vt.registerMiddleware(cfg)

	return vt
}

// registerMethods registers VectorDB operations as transport methods.
func (vt *VectorDBTransport) registerMethods() {
	// Insert method
	vt.server.Handle("insert", func(req *transport.Request) *transport.Response {
		var input struct {
			ID         string            `json:"id"`
			Doc        string            `json:"doc"`
			Meta       map[string]string `json:"meta"`
			Upsert     bool              `json:"upsert"`
			Collection string            `json:"collection"`
		}
		if err := transport.ParseJSON(req, &input); err != nil {
			return transport.ErrorResponse(err, 400)
		}

		if input.Doc == "" {
			return transport.ErrorResponse(fmt.Errorf("doc required"), 400)
		}

		// Get tenant from request metadata
		tenantID := req.Meta["X-Tenant-ID"]
		if tenantID == "" {
			tenantID = "default"
		}

		vec, err := vt.embedder.Embed(input.Doc)
		if err != nil {
			return transport.ErrorResponse(fmt.Errorf("embed error: %w", err), 500)
		}

		var id string
		if input.Upsert {
			id, err = vt.store.Upsert(vec, input.Doc, input.ID, input.Meta, input.Collection, tenantID)
		} else {
			id, err = vt.store.Add(vec, input.Doc, input.ID, input.Meta, input.Collection, tenantID)
		}
		if err != nil {
			return transport.ErrorResponse(err, 500)
		}

		return transport.JSONResponse(map[string]any{"id": id})
	})

	// Query method
	vt.server.Handle("query", func(req *transport.Request) *transport.Response {
		var input struct {
			Query       string              `json:"query"`
			TopK        int                 `json:"top_k"`
			Mode        string              `json:"mode"`
			Meta        map[string]string   `json:"meta"`
			MetaAny     []map[string]string `json:"meta_any"`
			MetaNot     map[string]string   `json:"meta_not"`
			IncludeMeta bool                `json:"include_meta"`
			Collection  string              `json:"collection"`
			HybridAlpha float64             `json:"hybrid_alpha"`
			ScoreMode   string              `json:"score_mode"`
		}
		if err := transport.ParseJSON(req, &input); err != nil {
			return transport.ErrorResponse(err, 400)
		}

		if input.TopK <= 0 {
			input.TopK = 3
		}
		if input.Mode == "" {
			input.Mode = "ann"
		}
		if input.HybridAlpha == 0 {
			input.HybridAlpha = 0.5
		}
		if input.ScoreMode == "" {
			input.ScoreMode = "vector"
		}

		qVec, err := vt.embedder.Embed(input.Query)
		if err != nil {
			return transport.ErrorResponse(fmt.Errorf("embed error: %w", err), 500)
		}

		var ids []int
		switch input.Mode {
		case "scan":
			ids = vt.store.Search(qVec, input.TopK)
		case "lex":
			ids = vt.store.SearchLex(tokenize(input.Query), input.TopK)
		default:
			ids = vt.store.SearchANN(qVec, input.TopK)
		}

		// Build response
		docs := make([]string, 0, len(ids))
		respIDs := make([]string, 0, len(ids))
		respScores := make([]float32, 0, len(ids))
		var respMeta []map[string]string

		for _, idx := range ids {
			hid := hashID(vt.store.GetID(idx))
			if vt.store.Deleted[hid] {
				continue
			}

			// Filter by metadata
			meta := vt.store.Meta[hid]
			if !matchesMeta(meta, input.Meta) {
				continue
			}
			if len(input.MetaAny) > 0 && !matchesAny(meta, input.MetaAny) {
				continue
			}
			if len(input.MetaNot) > 0 && matchesMeta(meta, input.MetaNot) {
				continue
			}
			if input.Collection != "" && vt.store.Coll[hid] != input.Collection {
				continue
			}

			docs = append(docs, vt.store.GetDoc(idx))
			respIDs = append(respIDs, vt.store.GetID(idx))
			respScores = append(respScores, DotProduct(qVec, vt.store.Data[idx*vt.store.Dim:(idx+1)*vt.store.Dim]))

			if input.IncludeMeta {
				cp := make(map[string]string, len(meta))
				for k, v := range meta {
					cp[k] = v
				}
				respMeta = append(respMeta, cp)
			}
		}

		// Rerank results
		rDocs, rerankScores, stats, err := vt.reranker.Rerank(input.Query, docs, len(docs))
		if err != nil {
			return transport.ErrorResponse(fmt.Errorf("rerank error: %w", err), 500)
		}
		if len(respScores) == 0 {
			respScores = rerankScores
		}

		return transport.JSONResponse(map[string]any{
			"ids":    respIDs,
			"docs":   rDocs,
			"scores": respScores,
			"stats":  stats,
			"meta":   respMeta,
		})
	})

	// Delete method
	vt.server.Handle("delete", func(req *transport.Request) *transport.Response {
		var input struct {
			ID string `json:"id"`
		}
		if err := transport.ParseJSON(req, &input); err != nil {
			return transport.ErrorResponse(err, 400)
		}

		if input.ID == "" {
			return transport.ErrorResponse(fmt.Errorf("id required"), 400)
		}

		if err := vt.store.Delete(input.ID); err != nil {
			return transport.ErrorResponse(err, 500)
		}

		return transport.JSONResponse(map[string]any{"deleted": input.ID})
	})

	// Health method
	vt.server.Handle("health", func(req *transport.Request) *transport.Response {
		vt.store.RLock()
		total := vt.store.Count
		deleted := len(vt.store.Deleted)
		active := total - deleted
		vt.store.RUnlock()

		return transport.JSONResponse(map[string]any{
			"ok":      true,
			"total":   total,
			"active":  active,
			"deleted": deleted,
		})
	})

	// Batch insert method
	vt.server.Handle("batch_insert", func(req *transport.Request) *transport.Response {
		var input struct {
			Docs []struct {
				ID         string            `json:"id"`
				Doc        string            `json:"doc"`
				Meta       map[string]string `json:"meta"`
				Collection string            `json:"collection"`
			} `json:"docs"`
			Upsert bool `json:"upsert"`
		}
		if err := transport.ParseJSON(req, &input); err != nil {
			return transport.ErrorResponse(err, 400)
		}

		if len(input.Docs) == 0 {
			return transport.ErrorResponse(fmt.Errorf("no docs provided"), 400)
		}

		tenantID := req.Meta["X-Tenant-ID"]
		if tenantID == "" {
			tenantID = "default"
		}

		ids := make([]string, 0, len(input.Docs))
		var errors []string

		for i, d := range input.Docs {
			if d.Doc == "" {
				errors = append(errors, fmt.Sprintf("doc %d: empty document", i))
				continue
			}

			vec, err := vt.embedder.Embed(d.Doc)
			if err != nil {
				errors = append(errors, fmt.Sprintf("doc %d: embed error: %v", i, err))
				continue
			}

			collection := d.Collection
			if collection == "" {
				collection = "default"
			}

			var id string
			if input.Upsert {
				id, err = vt.store.Upsert(vec, d.Doc, d.ID, d.Meta, collection, tenantID)
			} else {
				id, err = vt.store.Add(vec, d.Doc, d.ID, d.Meta, collection, tenantID)
			}
			if err != nil {
				errors = append(errors, fmt.Sprintf("doc %d: insert error: %v", i, err))
				continue
			}
			ids = append(ids, id)
		}

		response := map[string]any{"ids": ids}
		if len(errors) > 0 {
			response["errors"] = errors
		}

		return transport.JSONResponse(response)
	})

	// Compact method
	vt.server.Handle("compact", func(req *transport.Request) *transport.Response {
		if err := vt.store.Compact(vt.indexPath); err != nil {
			return transport.ErrorResponse(err, 500)
		}
		return transport.JSONResponse(map[string]any{"ok": true})
	})

	// Stream search - for real-time query updates
	vt.server.HandleStream("search_stream", func(req *transport.Request, send func(*transport.Response) error) error {
		var input struct {
			Query  string `json:"query"`
			TopK   int    `json:"top_k"`
			Stream bool   `json:"stream"` // If true, stream results as they're found
		}
		if err := transport.ParseJSON(req, &input); err != nil {
			return err
		}

		if input.TopK <= 0 {
			input.TopK = 10
		}

		qVec, err := vt.embedder.Embed(input.Query)
		if err != nil {
			return err
		}

		ids := vt.store.SearchANN(qVec, input.TopK)

		// Stream results one by one
		for i, idx := range ids {
			hid := hashID(vt.store.GetID(idx))
			if vt.store.Deleted[hid] {
				continue
			}

			doc := vt.store.GetDoc(idx)
			id := vt.store.GetID(idx)
			score := DotProduct(qVec, vt.store.Data[idx*vt.store.Dim:(idx+1)*vt.store.Dim])

			result := map[string]any{
				"index": i,
				"id":    id,
				"doc":   doc,
				"score": score,
				"final": i == len(ids)-1,
			}

			data, _ := json.Marshal(result)
			if err := send(&transport.Response{Body: data}); err != nil {
				return err
			}

			// Small delay between streamed results
			if input.Stream {
				time.Sleep(10 * time.Millisecond)
			}
		}

		return nil
	})
}

// registerMiddleware adds standard middleware.
func (vt *VectorDBTransport) registerMiddleware(cfg TransportConfig) {
	// Recovery middleware
	vt.server.Use(transport.RecoveryMiddleware())

	// Logging middleware
	vt.server.Use(transport.LoggingMiddleware(nil))

	// Rate limiting
	if cfg.RateLimit > 0 {
		limiter := transport.NewRateLimiter(cfg.RateLimit, cfg.RateBurst)
		vt.server.Use(transport.RateLimitMiddleware(limiter))
	}

	// Optional JWT authentication
	if cfg.EnableAuth && cfg.JWTSecret != "" {
		authFunc := func(token string) (string, error) {
			if vt.store.jwtMgr == nil {
				return "", fmt.Errorf("JWT not configured")
			}
			claims, err := vt.store.jwtMgr.ValidateTenantToken(token)
			if err != nil {
				return "", err
			}
			return claims.TenantID, nil
		}
		vt.server.Use(transport.AuthMiddleware(authFunc, "Authorization"))
	}

	// Metrics middleware
	metrics := transport.NewMetrics()
	vt.server.Use(transport.MetricsMiddleware(metrics))
}

// Start starts all transport protocols.
func (vt *VectorDBTransport) Start(ctx context.Context) error {
	return vt.server.Start(ctx)
}

// Stop gracefully stops all transports.
func (vt *VectorDBTransport) Stop(ctx context.Context) error {
	return vt.server.Stop(ctx)
}

// Server returns the underlying transport server.
func (vt *VectorDBTransport) Server() *transport.Server {
	return vt.server
}

// =============================================================================
// Integration with main.go
// =============================================================================

// runWithTransport starts VectorDB with multi-protocol transport support.
func runWithTransport() {
	fmt.Println(">>> Initializing VectorDB with Multi-Protocol Transport...")

	const indexPath = "vectordb/index.gob"
	defaultDim := 384
	if envDim := os.Getenv("EMBED_DIM"); envDim != "" {
		if v, err := fmt.Sscanf(envDim, "%d", &defaultDim); err != nil || v != 1 {
			defaultDim = 384
		}
	}

	embedder := initEmbedder(defaultDim)
	// Make capacity configurable for low-memory deployments
	initialCapacity := envInt("VECTOR_CAPACITY", 1000) // Reduced from 100000
	store, loaded := loadOrInitStore(indexPath, initialCapacity, embedder.Dim())
	store.walMaxBytes = envInt64("WAL_MAX_BYTES", 5*1024*1024)
	store.walMaxOps = envInt("WAL_MAX_OPS", 1000)

	if !loaded {
		fmt.Println(">>> Index not found, starting fresh")
	}
	fmt.Printf(">>> Index Ready. %d Vectors.\n", store.Count)

	reranker := initReranker(embedder)
	warmupModels(embedder, reranker)

	// Transport configuration
	cfg := TransportConfig{
		HTTPAddr:   os.Getenv("HTTP_ADDR"),
		GRPCAddr:   os.Getenv("GRPC_ADDR"),
		UnixSocket: os.Getenv("UNIX_SOCKET"),
		EnableAuth: os.Getenv("ENABLE_AUTH") == "1",
		JWTSecret:  os.Getenv("JWT_SECRET"),
		RateLimit:  float64(envInt("RATE_LIMIT", 100)),
		RateBurst:  envInt("RATE_BURST", 200),
	}
	if cfg.HTTPAddr == "" {
		cfg.HTTPAddr = ":8080"
	}

	// Create transport-enabled VectorDB
	vt := NewVectorDBTransport(store, embedder, reranker, indexPath, cfg)

	// Start transport
	ctx := context.Background()
	if err := vt.Start(ctx); err != nil {
		fmt.Printf("Failed to start transport: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf(">>> VectorDB Transport listening:\n")
	fmt.Printf("    HTTP:  %s\n", cfg.HTTPAddr)
	if cfg.GRPCAddr != "" {
		fmt.Printf("    gRPC:  %s\n", cfg.GRPCAddr)
	}
	if cfg.UnixSocket != "" {
		fmt.Printf("    Unix:  %s\n", cfg.UnixSocket)
	}
	fmt.Println()
	fmt.Println(">>> Methods available: insert, query, delete, health, batch_insert, compact, search_stream")
	fmt.Println(">>> Press Ctrl+C to stop")

	// Wait for shutdown
	select {}
}
