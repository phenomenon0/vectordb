package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/releaseinfo"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/telemetry"
)

// ===========================================================================================
// REQUEST ID
// ===========================================================================================

// generateRequestID returns a 16-byte random hex string (32 chars).
func generateRequestID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// requestIDFromContext extracts the request ID from the context, or returns "".
func requestIDFromContext(ctx context.Context) string {
	if id, ok := ctx.Value(logging.RequestIDKey).(string); ok {
		return id
	}
	return ""
}

// truncateRequestID caps id to maxBytes without splitting a multi-byte UTF-8
// rune, so the echoed header and the logged request_id remain valid UTF-8 even
// after an arbitrary-length client-supplied ID is trimmed.
func truncateRequestID(id string, maxBytes int) string {
	if len(id) <= maxBytes {
		return id
	}
	cut := id[:maxBytes]
	for len(cut) > 0 && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut
}

// ===========================================================================================
// CONFIGURABLE REQUEST LIMITS
// Override via environment variables. Defaults are safe for most deployments.
// ===========================================================================================

// envInt reads an integer environment variable, falling back to def when
// unset or invalid. Used by the package-level LIMIT_* configuration below and
// by config.go's kind-gated DEEPDATA_EMBED_DIM read; every other setting is
// validated once into serverConfig via loadServerConfig's fatal posInt.
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

var (
	// HTTP body size limits (bytes)
	limitInsertBody   = int64(envInt("LIMIT_INSERT_BODY_MB", 10)) * 1024 * 1024
	limitBatchBody    = int64(envInt("LIMIT_BATCH_BODY_MB", 50)) * 1024 * 1024
	limitQueryBody    = int64(envInt("LIMIT_QUERY_BODY_MB", 1)) * 1024 * 1024
	limitDeleteBody   = int64(envInt("LIMIT_DELETE_BODY_MB", 1)) * 1024 * 1024
	limitSnapshotBody = int64(envInt("LIMIT_SNAPSHOT_BODY_MB", 1024)) * 1024 * 1024
	limitBinaryImport = int64(envInt("LIMIT_BINARY_IMPORT_MB", 500)) * 1024 * 1024

	// Batch processing limits
	limitMaxBatchSize       = envInt("LIMIT_MAX_BATCH_SIZE", 10_000)
	limitMaxDocLength       = envInt("LIMIT_MAX_DOC_LENGTH", 1_000_000)
	limitMaxMetaKeys        = envInt("LIMIT_MAX_META_KEYS", 100)
	limitMaxMetaValueLength = envInt("LIMIT_MAX_META_VALUE_LEN", 10_000)
	limitMaxTotalBatchBytes = envInt("LIMIT_MAX_BATCH_BYTES", 100_000_000)

	// List/query result limits
	limitMaxListResults = envInt("LIMIT_MAX_LIST_RESULTS", 100_000)

	// Dimension limit
	limitMaxDimension = envInt("LIMIT_MAX_DIMENSION", 65_536)
)

// newCanonicalHTTPHandler serves the RC surface from the server runtime alone;
// the legacy engine is never constructed for it.
func newCanonicalHTTPHandler(rt *serverRuntime, embedder Embedder, indexPath string) (http.Handler, *CollectionHTTPServer) {
	mux := http.NewServeMux()
	var collectionHTTP *CollectionHTTPServer
	rt.ensureLimiters()

	// Authentication, global rate limiting and per-tenant rate limiting all
	// live on the server runtime.
	guard := rt.httpGuard()

	// Prometheus metrics are an operational surface like the health probes,
	// but unlike probes they can leak request-volume/operation detail, so they
	// are gated behind the same guard used for API routes when REQUIRE_AUTH is
	// on. In credentialless dev mode the guard authorizes anonymous access.
	mux.Handle("/metrics", guard(globalMetrics.Handler().ServeHTTP))

	// GET /v3/status — the server describing itself: version, the operation
	// list, the embedder it will use, its limits and its capabilities. It
	// names no tenant, so it is gated on the caller's own read permission
	// (CTL-04).
	mux.Handle("/v3/status", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
			return
		}
		tenantCtx, _ := security.GetTenantContextFromContext(r.Context())
		var ownTenant string
		if tenantCtx != nil {
			ownTenant = tenantCtx.TenantID
		}
		if !writeCanonicalHTTPAuthorizationResult(w, security.AuthorizeTenantPermission(tenantCtx, ownTenant, "read")) {
			return
		}
		var embedder *serverEmbedder
		usageLoaded := true
		readOnly := false
		if collectionHTTP != nil {
			embedder = collectionHTTP.embedder
			usageLoaded = collectionHTTP.UsageLoaded()
			if store := collectionHTTP.DurableStore(); store != nil {
				readOnly = store.IsReplica()
			}
		}
		payload, err := statusPayload(embedder, rt.limits, usageLoaded, readOnly, requestIDFromContext(r.Context()))
		if err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInternal, "status unavailable"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(payload)
	}))

	// /healthz, /livez - liveness probe. No legacy engine left to deadlock on.
	livenessHandler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	}
	mux.HandleFunc("/healthz", livenessHandler)
	mux.HandleFunc("/livez", livenessHandler)

	// /readyz - Readiness probe: Is the service ready to accept traffic?
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		issues := []string{}
		type canonicalHealth struct {
			durable  bool
			readOnly bool
			err      error
		}
		health := make(chan canonicalHealth, 1)
		go func() {
			if collectionHTTP == nil {
				health <- canonicalHealth{}
				return
			}
			durable := collectionHTTP.IsDurable()
			var err error
			var readOnly bool
			if durable {
				err = collectionHTTP.PersistenceError()
				readOnly = collectionHTTP.DurableStore().IsReplica()
			}
			health <- canonicalHealth{durable: durable, readOnly: readOnly, err: err}
		}()
		var state canonicalHealth
		select {
		case state = <-health:
			if !state.durable {
				issues = append(issues, "durable collection store not initialized")
			} else if state.err != nil {
				logging.Default().Error("readyz: durable collection fault", "error", state.err)
				issues = append(issues, "durable collection store faulted")
			}
		case <-time.After(5 * time.Second):
			issues = append(issues, "durable collection health check timed out")
		}
		w.Header().Set("Content-Type", "application/json")
		if len(issues) == 0 {
			w.WriteHeader(http.StatusOK)
			// embedder names the process text embedder ("none" when
			// callers must send vectors); no live embedding call here.
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ready":  true,
				"checks": []string{"collection_snapshot", "mutation_journal", "lifetime_lock"},
				// A read replica is ready -- it answers reads correctly --
				// so it stays 200 and stays in the pool. read_only is how
				// it declines writes to a load balancer that would
				// otherwise treat every ready backend as interchangeable.
				"read_only": state.readOnly,
				"embedder":  collectionHTTP.embedder.Label(),
				"version":   releaseinfo.Version(),
			})
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]any{"ready": false, "issues": issues})
		}
	})

	// Multi-vector collection API (v2) - hybrid search with dense + sparse vectors.
	collectionBasePath := ""
	if indexPath != "" {
		collectionBasePath = indexPath + ".collections"
	}
	collectionHTTP = NewCollectionHTTPServer(collectionBasePath)
	collectionHTTP.embedder, _ = embedder.(*serverEmbedder)
	if collectionBasePath != "" {
		err := collectionHTTP.LoadDurableWithLimits(collectionBasePath, vcollection.StoreLimits{
			MaxTenants:     rt.limits.MaxTenants,
			MaxCollections: rt.limits.MaxCollections,
		})
		if err != nil {
			collectionHTTP.setPersistenceError(fmt.Errorf("load collection state: %w", err))
		} else {
			// Before a single route is registered: a replica directory that
			// came up unbound would take one local write and fork its history
			// from the leader's at the same LSN.
			if err := bindReplicaReadOnly(collectionHTTP, collectionBasePath); err != nil {
				collectionHTTP.setPersistenceError(err)
			}
		}
	}
	// Only the canonical tenant-aware V3 surface is served. Legacy V2/root
	// collection routes were removed; unsupported paths cannot be registered.
	collectionHTTP.RegisterCanonicalHandlers(mux, guard)

	// canonicalRCSurface wraps corsMiddleware so OPTIONS can't reach an unsupported route.
	otelMiddleware := telemetry.HTTPMiddleware()
	return requestIDMiddleware(recoveryMiddleware(requestTimeoutMiddleware(canonicalRCSurface(corsMiddleware(otelMiddleware(mux), rt.corsAllowedOrigins)), rt.requestTimeout))), collectionHTTP
}

// corsMiddleware allows browser requests from different origins. Default:
// wildcard without credentials. To allow credentialed requests, set
// CORS_ALLOWED_ORIGINS to a comma-separated allowlist.
func corsMiddleware(next http.Handler, allowedOriginsRaw string) http.Handler {
	corsAllowedOriginsRaw := strings.TrimSpace(allowedOriginsRaw)
	corsAllowAllOrigins := corsAllowedOriginsRaw == ""
	corsAllowedOrigins := make(map[string]struct{})
	if !corsAllowAllOrigins {
		for _, item := range strings.Split(corsAllowedOriginsRaw, ",") {
			origin := strings.TrimSpace(item)
			if origin == "" {
				continue
			}
			if origin == "*" {
				corsAllowAllOrigins = true
				corsAllowedOrigins = map[string]struct{}{}
				break
			}
			corsAllowedOrigins[origin] = struct{}{}
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := strings.TrimSpace(r.Header.Get("Origin"))
		if corsAllowAllOrigins {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		} else if origin != "" {
			if _, ok := corsAllowedOrigins[origin]; ok {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Credentials", "true")
				w.Header().Add("Vary", "Origin")
			} else if r.Method == http.MethodOptions {
				http.Error(w, "forbidden origin", http.StatusForbidden)
				return
			}
		}

		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, X-Tenant-ID")
		w.Header().Set("Access-Control-Max-Age", "86400") // 24 hours

		// Handle preflight OPTIONS requests
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// recoveryMiddleware prevents a panic in a handler from crashing the server.
func recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				stack := make([]byte, 4096)
				n := runtime.Stack(stack, false)
				logging.Default().Error("panic recovered in HTTP handler",
					"error", err,
					"path", r.URL.Path,
					"method", r.Method,
					"request_id", requestIDFromContext(r.Context()),
					"stack", string(stack[:n]),
				)
				// Return 500 error to client instead of closing connection — never expose panic details
				http.Error(w, "internal server error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// requestTimeoutMiddleware cancels the handler context after the deadline.
// This is separate from HTTP server WriteTimeout (which is a hard TCP-level
// cutoff); it lets handlers cooperatively abort long operations. Streaming
// endpoints (snapshot, export, import) and /metrics are exempt.
func requestTimeoutMiddleware(next http.Handler, timeout time.Duration) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Path
		if strings.HasPrefix(p, "/snapshot") || strings.HasPrefix(p, "/export") ||
			strings.HasPrefix(p, "/import") || p == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), timeout)
		defer cancel()
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// requestIDMiddleware generates a unique ID for every request, stores it in
// the context for structured logging, and echoes it in the X-Request-ID
// response header so clients can correlate responses with server-side logs.
// If the client or reverse proxy already provides X-Request-ID, it is reused.
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = generateRequestID()
		}
		// Cap client-supplied IDs to prevent log inflation and strip
		// non-printable characters to avoid log injection.
		if len(id) > 128 {
			id = truncateRequestID(id, 128)
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), logging.RequestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func canonicalTenantIDFromPath(path string) string {
	const prefix = "/v3/tenants/"
	if !strings.HasPrefix(path, prefix) {
		return ""
	}
	tenantID := strings.TrimPrefix(path, prefix)
	if slash := strings.IndexByte(tenantID, '/'); slash >= 0 {
		tenantID = tenantID[:slash]
	}
	return tenantID
}

func canonicalRateLimitTenant(tenantCtx *security.TenantContext, targetTenant string) string {
	if tenantCtx == nil {
		return "default"
	}
	tenantKey := tenantCtx.TenantID
	if tenantCtx.IsServerAdmin && isValidTenantID(targetTenant) {
		tenantKey = targetTenant
	}
	if tenantKey == "" {
		return "default"
	}
	return tenantKey
}

func canonicalRCSurface(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if strings.HasPrefix(path, "/v3/tenants/") || path == "/v3/status" || path == "/healthz" || path == "/readyz" || path == "/livez" || path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		e := apierror.New(apierror.CodeNotFound, "no such route on the RC surface: "+path)
		e.Hint = "the RC serves /v3/tenants/{tenant}/collections..., /v3/status, /healthz, /readyz, /livez and /metrics; the route table is in the contract"
		apierror.WriteHTTP(w, e)
	})
}
