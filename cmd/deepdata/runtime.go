package main

import (
	"context"
	"net/http"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"

	"github.com/phenomenon0/vectordb/internal/apierror"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/security"
)

// serverRuntime is the process-wide authentication and limit state the V3
// surface actually needs. The RC used to borrow these fields from the legacy
// v1 engine; they live here now so the live server never constructs that
// engine (SYS-01).
type serverRuntime struct {
	apiToken          string
	jwtMgr            *security.JWTManager  // JWT token manager
	requireAuth       bool                  // Require JWT authentication
	acl               *security.ACL         // access control lists
	quotas            *security.TenantQuota // storage quotas per tenant
	rl                *rateLimiter          // global limiter keyed by token or peer IP
	canonicalTenantRL *rateLimiter          // shared V3 HTTP/gRPC limiter keyed by authenticated tenant
	authFailureRL     *authFailureLimiter   // shared HTTP/gRPC failed-auth budget keyed by peer IP
}

// newServerRuntime reads exactly the environment the legacy engine constructor read for these
// fields: JWT_SECRET and JWT_ISSUER for the token manager, API_TOKEN for the
// static credential, and REQUIRE_AUTH to force authentication when neither is
// configured.
func newServerRuntime() *serverRuntime {
	var jwtMgr *security.JWTManager
	if secret := os.Getenv("JWT_SECRET"); secret != "" {
		issuer := os.Getenv("JWT_ISSUER")
		if issuer == "" {
			issuer = "vectordb"
		}
		jwtMgr = security.NewJWTManager(secret, issuer)
	}
	apiToken := os.Getenv("API_TOKEN")
	return &serverRuntime{
		apiToken:    apiToken,
		jwtMgr:      jwtMgr,
		requireAuth: os.Getenv("REQUIRE_AUTH") == "1" || jwtMgr != nil || apiToken != "",
		acl:         security.NewACL(),
		quotas:      security.NewTenantQuota(),
	}
}

// ensureLimiters fills in any limiter the caller did not preset. Tests preset
// them; the server leaves them nil and gets the env-configured defaults.
func (rt *serverRuntime) ensureLimiters() {
	if rt.rl == nil {
		rps := envInt("API_RPS", 100)
		rt.rl = newRateLimiter(rps, rps, envInt("MAX_RATE_LIMIT_KEYS", 100_000), time.Minute)
	}
	if rt.authFailureRL == nil {
		rt.authFailureRL = newAuthFailureLimiter(
			envInt("AUTH_FAILURE_RPS", 1),
			envInt("AUTH_FAILURE_BURST", 5),
			envInt("MAX_RATE_LIMIT_KEYS", 100_000),
			time.Second,
		)
	}
	if rt.canonicalTenantRL == nil {
		rt.canonicalTenantRL = newRateLimiter(
			envInt("TENANT_RPS", 100),
			envInt("TENANT_BURST", 100),
			envInt("MAX_RATE_LIMIT_KEYS", 100_000),
			time.Second,
		)
	}
}

// httpGuard is the HTTP authentication and rate-limit middleware. TRUST_PROXY
// is read once, when the handler is built.
func (rt *serverRuntime) httpGuard() func(http.HandlerFunc) http.HandlerFunc {
	trustProxy := os.Getenv("TRUST_PROXY") == "1"
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			token := r.Header.Get("Authorization")
			authPeerKey := httpAuthPeerKey(r, trustProxy)
			authAttempt, allowed := rt.authFailureRL.begin(authPeerKey)
			if !allowed {
				apierror.WriteHTTP(w, apierror.New(apierror.CodeRateLimited, "authentication rate limited"))
				return
			}
			finishAuthAttempt := func(failed bool) {
				if authAttempt != nil {
					authAttempt.finish(failed)
					authAttempt = nil
				}
			}
			defer func() { finishAuthAttempt(false) }()

			authenticated := false
			var tenantCtx *security.TenantContext

			// JWT authentication (when enabled) - SECURE VERSION
			if rt.jwtMgr != nil {
				if token == "" {
					// No token provided, but JWT is configured
					if rt.requireAuth {
						finishAuthAttempt(true)
						apierror.WriteHTTP(w, apierror.New(apierror.CodeUnauthenticated, "unauthorized: missing authentication token"))
						return
					}
					// If not required, use default context (backward compatibility)
					tenantCtx = &security.TenantContext{
						TenantID:    "default",
						Permissions: map[string]bool{"read": true, "write": true},
						Collections: make(map[string]bool),
						IsAdmin:     false,
					}
				} else {
					// Token provided - MUST be valid
					jwtToken := strings.TrimPrefix(token, "Bearer ")
					var err error
					tenantCtx, err = rt.jwtMgr.ValidateTenantToken(jwtToken)
					if err != nil {
						logging.Default().Warn("JWT validation failed", "error", err, "path", r.URL.Path)
						finishAuthAttempt(true)
						apierror.WriteHTTP(w, apierror.New(apierror.CodeUnauthenticated, "unauthorized: invalid token"))
						return
					}
					authenticated = true
				}
			} else {
				// No JWT manager configured - fallback to legacy API token auth
				// Simple API token authentication (legacy)
				if rt.apiToken != "" {
					candidate := strings.TrimPrefix(token, "Bearer ")
					if security.SecureCompare(candidate, rt.apiToken) {
						authenticated = true
					} else if token != "" {
						finishAuthAttempt(true)
						apierror.WriteHTTP(w, apierror.New(apierror.CodeUnauthenticated, "unauthorized"))
						return
					}
				}

				if rt.requireAuth && !authenticated {
					finishAuthAttempt(true)
					apierror.WriteHTTP(w, apierror.New(apierror.CodeUnauthenticated, "unauthorized"))
					return
				}

				requestedTenantID := strings.TrimSpace(r.Header.Get("X-Tenant-ID"))
				if requestedTenantID != "" && !isValidTenantID(requestedTenantID) {
					finishAuthAttempt(false)
					apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "invalid X-Tenant-ID header"))
					return
				}
				tenantID := "default"
				if requestedTenantID != "" {
					tenantID = requestedTenantID
				}

				// Use default tenant context for non-JWT mode
				if tenantCtx == nil {
					serverAdmin := authenticated || (rt.jwtMgr == nil && rt.apiToken == "" && requestedTenantID == "")
					tenantCtx = &security.TenantContext{
						TenantID:    tenantID,
						Permissions: map[string]bool{"read": true, "write": true},
						Collections: make(map[string]bool),
						// A configured static server token is an explicit full-control
						// credential. JWTs remain the path for scoped tenant roles.
						IsAdmin:       serverAdmin,
						IsServerAdmin: serverAdmin,
					}
				}
			}
			finishAuthAttempt(false)

			// Global rate limiting (per-IP or per-token)
			if rt.rl != nil {
				// Use IP address for anonymous users instead of shared "anon" key
				key := token
				if key == "" {
					// Extract client IP (handle X-Forwarded-For and X-Real-IP headers)
					clientIP := ""
					if trustProxy {
						clientIP = r.Header.Get("X-Forwarded-For")
						if clientIP == "" {
							clientIP = r.Header.Get("X-Real-IP")
						}
					}
					if clientIP == "" {
						clientIP = r.RemoteAddr
					}
					// Use first IP in X-Forwarded-For chain
					if idx := strings.Index(clientIP, ","); idx > 0 {
						clientIP = clientIP[:idx]
					}
					// Strip port from RemoteAddr
					if idx := strings.LastIndex(clientIP, ":"); idx > 0 {
						clientIP = clientIP[:idx]
					}
					key = "ip:" + strings.TrimSpace(clientIP)
				}
				if !rt.rl.allow(key) {
					apierror.WriteHTTP(w, apierror.New(apierror.CodeRateLimited, "rate limited"))
					return
				}
			}

			if rt.canonicalTenantRL != nil {
				tenantKey := canonicalRateLimitTenant(tenantCtx, canonicalTenantIDFromPath(r.URL.Path))
				if !rt.canonicalTenantRL.allow(tenantKey) {
					apierror.WriteHTTP(w, apierror.New(apierror.CodeRateLimited, "tenant rate limited"))
					return
				}
			}

			// Add tenant context to request context for handlers to use
			if tenantCtx != nil {
				ctx := r.Context()
				ctx = context.WithValue(ctx, security.TenantContextKey, tenantCtx)
				r = r.WithContext(ctx)
			}

			next(w, r)
		}
	}
}

// grpcInterceptor is the gRPC half of the same auth and limit state, so both
// transports share one credential set, one tenant limiter and one failed-auth
// budget.
func (rt *serverRuntime) grpcInterceptor(logger *logging.Logger) grpc.UnaryServerInterceptor {
	return grpcAuthInterceptorWithRateLimiters(
		rt.jwtMgr,
		rt.apiToken,
		rt.requireAuth,
		logger,
		rt.canonicalTenantRL,
		rt.authFailureRL,
	)
}
