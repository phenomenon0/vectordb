package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"

	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/telemetry"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"net"
)

// ======================================================================================
// Main: bootstrap, HTTP API
// ======================================================================================

func main() {
	// CLI flag parsing — strip "serve" subcommand if present
	args := os.Args[1:]
	// `routes` prints the contract's HTTP surface and exits. It reads only
	// the embedded operations list, so it needs no data directory, no
	// environment and no server; the docs linter runs it to generate the
	// route table in internal/collection/API.md (DOC-03).
	if len(args) > 0 && args[0] == "routes" {
		if err := printRoutes(os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if len(args) > 0 && args[0] == "replicate" {
		os.Exit(runReplicate(args[1:], logging.Default()))
	}
	if len(args) > 0 && args[0] == "serve" {
		args = args[1:]
	}

	cfg, configErrs := loadServerConfig(args, os.Getenv)

	// Initialize structured logging (JSON by default, LOG_FORMAT=text for dev)
	logConfig := logging.DefaultConfig()
	if cfg.LogFormat == "text" {
		logConfig.Format = "text"
	}
	switch cfg.LogLevel {
	case "debug":
		logConfig.Level = logging.LevelDebug
	case "warn":
		logConfig.Level = logging.LevelWarn
	case "error":
		logConfig.Level = logging.LevelError
	default:
		// "info" or unset → LevelInfo (already the default)
	}
	logger := logging.Init(logConfig)
	logger.Info("initializing vector engine")

	// ==========================================================================
	// Startup Config Validation — fail fast on invalid env var values
	// ==========================================================================
	if len(configErrs) > 0 {
		for _, e := range configErrs {
			logger.Error("invalid configuration", "detail", e)
		}
		fmt.Fprintf(os.Stderr, "FATAL: %d configuration error(s) — fix the environment variables above and restart\n", len(configErrs))
		os.Exit(1)
	}

	// The production server exposes only the caller-supplied-vector V3/gRPC
	// collection engine. Historical handlers remain in source for offline
	// migration tests, but no runtime environment switch may re-enable them in
	// the RC binary.
	if err := cfg.validateServe(); err != nil {
		logger.Error("canonical authentication configuration rejected", "error", err)
		os.Exit(1)
	}

	// The server sets HNSW's default ef_search once, before any collection
	// exists, instead of each Collection reading the environment itself.
	vcollection.DefaultEfSearch = cfg.HNSWEfSearch

	// Ensure data directory exists
	if err := os.MkdirAll(cfg.DataDir, 0755); err != nil {
		logger.Error("failed to create data directory", "error", err)
		os.Exit(1)
	}
	logger.Info("data directory ready", "path", cfg.DataDir)
	indexPath := cfg.IndexPath

	initMetrics()

	// Initialize OpenTelemetry tracing
	// Configured via environment variables:
	//   OTEL_SERVICE_NAME - service name (default: "vectordb")
	//   OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint (optional)
	//   OTEL_TRACE_SAMPLE_RATE - sampling rate (default: 1.0)
	//   OTEL_ENABLE_CONSOLE - enable console exporter (default: false)
	if err := telemetry.SetupSimple(); err != nil {
		logger.Warn("telemetry setup failed", "error", err)
	} else {
		logger.Info("opentelemetry tracing initialized")
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := telemetry.Shutdown(ctx); err != nil {
				logger.Warn("telemetry shutdown failed", "error", err)
			}
		}()
	}

	// One text embedder per process, named by DEEPDATA_EMBEDDER (default none:
	// callers send vectors). A configured-but-unreachable embedder refuses to
	// start, like unreadable persistence below.
	var embedder *serverEmbedder
	serverEmb, embErr := newServerEmbedder(cfg.Embedder)
	if embErr != nil {
		logger.Error("refusing to start with an unusable text embedder", "error", embErr)
		os.Exit(1)
	}
	if serverEmb != nil {
		embedder = serverEmb
		logger.Info("text embedder ready", "embedder", serverEmb.Label(), "dim", serverEmb.Dim())
	} else {
		logger.Info("no text embedder configured (DEEPDATA_EMBEDDER=none); clients must provide vectors")
	}

	legacyArtifacts, inspectErr := existingLegacyRootArtifacts(indexPath)
	if inspectErr != nil {
		logger.Error("failed to inspect unsupported legacy persistence", "error", inspectErr)
		os.Exit(1)
	}
	if len(legacyArtifacts) > 0 {
		logger.Error("legacy root persistence requires an explicit offline migration before canonical RC startup", "artifacts", legacyArtifacts)
		os.Exit(1)
	}

	// The V3 surface keeps its authentication and limit state here; the legacy
	// engine is never constructed.
	rt := newServerRuntime()

	// HTTP API with graceful shutdown
	handler, collectionHTTP := newCanonicalHTTPHandler(rt, embedder, indexPath)
	if err := collectionHTTP.PersistenceError(); err != nil {
		logger.Error("refusing to start with unreadable collection persistence state", "path", indexPath+".collections", "error", err)
		os.Exit(1)
	}
	legacyCollectionCount, inspectErr := collectionHTTP.LegacyCollectionCount()
	if inspectErr != nil {
		logger.Error("failed to inspect unified collection state", "error", inspectErr)
		if abortErr := collectionHTTP.Abort(); abortErr != nil {
			logger.Error("failed to release collection store after inspection failure", "error", abortErr)
		}
		os.Exit(1)
	}
	if legacyCollectionCount != 0 {
		logger.Error("refusing canonical startup with legacy V2 collections; migrate them into tenant-aware V3 collections first",
			"path", indexPath+".collections", "legacy_collections", legacyCollectionCount)
		if abortErr := collectionHTTP.Abort(); abortErr != nil {
			logger.Error("failed to release collection store after migration refusal", "error", abortErr)
		}
		os.Exit(1)
	}
	addr, grpcAddr, err := canonicalListenerAddresses(
		cfg.HTTPPort,
		cfg.GRPCPort,
		cfg.InsecureDevMode,
		cfg.BindHost,
	)
	if err != nil {
		logger.Error("refusing invalid API bind host", "error", err)
		if closeErr := collectionHTTP.Abort(); closeErr != nil {
			logger.Error("failed to release collection store after bind-host refusal", "error", closeErr)
		}
		os.Exit(1)
	}
	httpListener, grpcListener, err := bindAPIListeners(addr, grpcAddr)
	if err != nil {
		logger.Error("refusing to start without the complete API listener set", "error", err)
		if closeErr := collectionHTTP.Abort(); closeErr != nil {
			logger.Error("failed to release collection store after listener failure", "error", closeErr)
		}
		os.Exit(1)
	}

	handler, err = canonicalReplicationSurface(handler, collectionHTTP, indexPath, logger)
	if err != nil {
		logger.Error("refusing to start with an unusable replication configuration", "error", err)
		if closeErr := collectionHTTP.Abort(); closeErr != nil {
			logger.Error("failed to release collection store after replication refusal", "error", closeErr)
		}
		os.Exit(1)
	}

	// Wrap handler with h2c (HTTP/2 cleartext) for connection multiplexing
	// without TLS. HTTP/1.1 clients continue to work transparently.
	// Set HTTP_H2C=0 to disable.
	var finalHandler http.Handler = handler
	if cfg.H2C {
		finalHandler = h2c.NewHandler(handler, &http2.Server{})
	}
	var httpRequests sync.WaitGroup
	trackedHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpRequests.Add(1)
		defer httpRequests.Done()
		finalHandler.ServeHTTP(w, r)
	})

	srv := &http.Server{
		Addr:              addr,
		Handler:           trackedHandler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       cfg.ReadTimeout,
		WriteTimeout:      cfg.WriteTimeout,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	// gRPC server (GRPC_PORT=0 to disable, default 50051)
	var grpcSrv *grpc.Server
	if grpcListener != nil {
		grpcSrv = grpc.NewServer(
			grpc.MaxRecvMsgSize(canonicalGRPCMaxReceiveBytes),
			grpc.MaxSendMsgSize(64*1024*1024),
			grpc.UnaryInterceptor(rt.grpcInterceptor(logger)),
		)
		deepdatav3.RegisterDeepDataServer(grpcSrv, &CollectionGRPCServer{
			tenants:  collectionHTTP.TenantManager(),
			embedder: collectionHTTP.embedder,
			persistenceHealth: func() error {
				if !collectionHTTP.IsDurable() {
					return errors.New("durable collection persistence is not initialized")
				}
				return collectionHTTP.PersistenceError()
			},
		})
	}

	type apiServeFailure struct {
		surface string
		err     error
	}
	serverErrCh := make(chan apiServeFailure, 2)
	logger.Info("http api listening", "addr", httpListener.Addr())
	go func() {
		if err := srv.Serve(httpListener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErrCh <- apiServeFailure{surface: "http", err: err}
		}
	}()
	if grpcSrv != nil {
		logger.Info("grpc api listening", "addr", grpcListener.Addr())
		go func() {
			if err := grpcSrv.Serve(grpcListener); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
				serverErrCh <- apiServeFailure{surface: "grpc", err: err}
			}
		}()
	}

	// Setup graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	// Wait for a shutdown signal or any unexpected listener/server failure.
	logging.Default().Info("server running, press Ctrl+C to stop")
	serveFailed := false
	select {
	case sig := <-sigCh:
		logging.Default().Info("received signal, initiating graceful shutdown", "signal", sig)
	case failure := <-serverErrCh:
		serveFailed = true
		logging.Default().Error("API server failed; initiating coordinated shutdown", "surface", failure.surface, "error", failure.err)
	}
	signal.Stop(sigCh)

	// Graceful shutdown sequence. The final collection checkpoint runs only
	// when every handler has drained.
	allHandlersDrained := true
	if grpcSrv != nil {
		logging.Default().Info("shutting down gRPC server")
		grpcDone := make(chan struct{})
		go func() {
			grpcSrv.GracefulStop()
			close(grpcDone)
		}()
		select {
		case <-grpcDone:
		case <-time.After(30 * time.Second):
			logging.Default().Warn("gRPC graceful shutdown timed out, forcing stop")
			grpcSrv.Stop()
			select {
			case <-grpcDone:
			case <-time.After(5 * time.Second):
				allHandlersDrained = false
				logging.Default().Error("gRPC handlers did not drain after forced stop")
			}
		}
	}
	logging.Default().Info("shutting down HTTP server")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logging.Default().Error("HTTP server shutdown error; forcing connection close", "error", err)
		if closeErr := srv.Close(); closeErr != nil && closeErr != http.ErrServerClosed {
			logging.Default().Error("HTTP server forced close error", "error", closeErr)
		}
	}
	httpDone := make(chan struct{})
	go func() {
		httpRequests.Wait()
		close(httpDone)
	}()
	select {
	case <-httpDone:
	case <-time.After(5 * time.Second):
		allHandlersDrained = false
		logging.Default().Error("HTTP handlers did not drain after shutdown")
	}

	if allHandlersDrained {
		if err := collectionHTTP.Close(); err != nil {
			logger.Error("failed to checkpoint and close canonical collection state", "error", err)
		} else {
			logger.Info("canonical collection state checkpointed and closed successfully")
		}
	} else {
		logger.Error("skipping final persistence checkpoint because handlers are still active; WAL artifacts retained")
	}

	logger.Info("shutdown complete")
	if serveFailed {
		logger.Error("exiting non-zero after API server failure")
		os.Exit(1)
	}
}

// bindAPIListeners proves the complete configured API surface is available
// before either protocol begins serving. If the second bind fails, the first
// listener is closed so a replacement process can start immediately.
func bindAPIListeners(httpAddr, grpcAddr string) (net.Listener, net.Listener, error) {
	httpListener, err := net.Listen("tcp", httpAddr)
	if err != nil {
		return nil, nil, fmt.Errorf("bind HTTP listener %q: %w", httpAddr, err)
	}
	if grpcAddr == "" {
		return httpListener, nil, nil
	}
	grpcListener, err := net.Listen("tcp", grpcAddr)
	if err != nil {
		return nil, nil, errors.Join(
			fmt.Errorf("bind gRPC listener %q: %w", grpcAddr, err),
			httpListener.Close(),
		)
	}
	return httpListener, grpcListener, nil
}

// canonicalListenerAddresses keeps the explicit credentialless development
// escape hatch loopback-only. Authenticated deployments retain wildcard binds
// by default so containers and orchestrators can publish the configured ports,
// while DEEPDATA_BIND_HOST lets an operator reduce exposure to one IP literal.
func canonicalListenerAddresses(httpPort, grpcPort int, insecureDevelopment bool, configuredHost string) (string, string, error) {
	host := ""
	if configuredHost != strings.TrimSpace(configuredHost) {
		return "", "", errors.New("DEEPDATA_BIND_HOST must not contain surrounding whitespace")
	}
	if configuredHost != "" {
		ip := net.ParseIP(configuredHost)
		if ip == nil {
			return "", "", fmt.Errorf("DEEPDATA_BIND_HOST=%q must be an IP literal", configuredHost)
		}
		if insecureDevelopment && !ip.IsLoopback() {
			return "", "", errors.New("DEEPDATA_INSECURE_DEV_MODE may bind only to a loopback IP")
		}
		host = configuredHost
	} else if insecureDevelopment {
		host = "127.0.0.1"
	}
	httpAddr := net.JoinHostPort(host, strconv.Itoa(httpPort))
	grpcAddr := ""
	if grpcPort > 0 {
		grpcAddr = net.JoinHostPort(host, strconv.Itoa(grpcPort))
	}
	return httpAddr, grpcAddr, nil
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			logging.Default().Warn("invalid integer env var, using default", "key", key, "value", v, "default", def)
			return def
		}
		return n
	}
	return def
}

// grpcAuthInterceptor is retained for grpc_auth_test.go coverage.
func grpcAuthInterceptor(jwtMgr *security.JWTManager, apiToken string, requireAuth bool, logger *logging.Logger) grpc.UnaryServerInterceptor {
	return grpcAuthInterceptorWithRateLimiters(jwtMgr, apiToken, requireAuth, logger, nil, nil)
}

// grpcAuthInterceptorWithTenantLimiter is retained for canonical_resource_controls_test.go coverage.
func grpcAuthInterceptorWithTenantLimiter(jwtMgr *security.JWTManager, apiToken string, requireAuth bool, logger *logging.Logger, tenantLimiter *rateLimiter) grpc.UnaryServerInterceptor {
	return grpcAuthInterceptorWithRateLimiters(jwtMgr, apiToken, requireAuth, logger, tenantLimiter, nil)
}

func grpcAuthInterceptorWithRateLimiters(
	jwtMgr *security.JWTManager,
	apiToken string,
	requireAuth bool,
	logger *logging.Logger,
	tenantLimiter *rateLimiter,
	authFailureLimiter *authFailureLimiter,
) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
		// Panic recovery — same as before, prevents crashes from taking down the process
		defer func() {
			if r := recover(); r != nil {
				logger.Error("panic recovered in gRPC handler", "error", r, "method", info.FullMethod)
				err = apierror.New(apierror.CodeInternal, "internal error").GRPC(ctx)
			}
		}()

		// Auth token and request id from gRPC metadata (mirrors the HTTP
		// Authorization header and X-Request-ID middleware): honour the
		// caller's x-request-id or mint one, echo it as a response header and
		// carry it in ctx so every apierror quotes it.
		token, requestID := "", ""
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if vals := md.Get("authorization"); len(vals) > 0 {
				token = strings.TrimPrefix(vals[0], "Bearer ")
			}
			if vals := md.Get("x-request-id"); len(vals) > 0 {
				requestID = truncateRequestID(strings.TrimSpace(vals[0]), 128)
			}
		}
		if requestID == "" {
			requestID = generateRequestID()
		}
		ctx = context.WithValue(ctx, logging.RequestIDKey, requestID)
		_ = grpc.SetHeader(ctx, metadata.Pairs("x-request-id", requestID))
		authPeerKey := grpcAuthPeerKey(ctx)
		authAttempt, allowed := authFailureLimiter.begin(authPeerKey)
		if !allowed {
			return nil, apierror.New(apierror.CodeRateLimited, "authentication rate limited").GRPC(ctx)
		}
		finishAuthAttempt := func(failed bool) {
			if authAttempt != nil {
				authAttempt.finish(failed)
				authAttempt = nil
			}
		}
		defer func() { finishAuthAttempt(false) }()

		var tenantCtx *security.TenantContext

		if jwtMgr != nil {
			if token == "" {
				if requireAuth {
					finishAuthAttempt(true)
					return nil, apierror.New(apierror.CodeUnauthenticated, "missing authentication token").GRPC(ctx)
				}
				tenantCtx = &security.TenantContext{
					TenantID:    "default",
					Permissions: map[string]bool{"read": true, "write": true},
					Collections: make(map[string]bool),
				}
			} else {
				var valErr error
				tenantCtx, valErr = jwtMgr.ValidateTenantToken(token)
				if valErr != nil {
					logging.Default().Error("gRPC JWT validation failed", "error", valErr)
					finishAuthAttempt(true)
					return nil, apierror.New(apierror.CodeUnauthenticated, "invalid token").GRPC(ctx)
				}
			}
		} else {
			authenticated := false
			if apiToken != "" {
				if security.SecureCompare(token, apiToken) {
					authenticated = true
				} else if token != "" {
					finishAuthAttempt(true)
					return nil, apierror.New(apierror.CodeUnauthenticated, "unauthorized").GRPC(ctx)
				}
			}
			if requireAuth && !authenticated {
				finishAuthAttempt(true)
				return nil, apierror.New(apierror.CodeUnauthenticated, "unauthorized").GRPC(ctx)
			}
			serverAdmin := authenticated || (jwtMgr == nil && apiToken == "")
			tenantCtx = &security.TenantContext{
				TenantID:    "default",
				Permissions: map[string]bool{"read": true, "write": true},
				Collections: make(map[string]bool),
				// A configured static server token is intentionally full control;
				// JWT claims provide scoped tenant/collection roles.
				IsAdmin:       serverAdmin,
				IsServerAdmin: serverAdmin,
			}
		}
		finishAuthAttempt(false)

		if tenantLimiter != nil {
			targetTenant := ""
			if request, ok := req.(interface{ GetTenantId() string }); ok {
				targetTenant = request.GetTenantId()
			}
			tenantKey := canonicalRateLimitTenant(tenantCtx, targetTenant)
			if !tenantLimiter.allow(tenantKey) {
				return nil, apierror.New(apierror.CodeRateLimited, "tenant rate limited").GRPC(ctx)
			}
		}

		ctx = context.WithValue(ctx, security.TenantContextKey, tenantCtx)
		return handler(ctx, req)
	}
}
