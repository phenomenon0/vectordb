package main

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// ===========================================================================================
// CONTEXT KEYS FOR AUTH DATA PROPAGATION
// ===========================================================================================

// contextKey is a private type for context keys to avoid collisions
type contextKey int

const (
	// APIKeyContextKey is the context key for API key data
	APIKeyContextKey contextKey = iota
	// JWTClaimsContextKey is the context key for JWT claims
	JWTClaimsContextKey
	// TenantContextKey is the context key for tenant context
	TenantContextKey
)

// GetAPIKeyFromContext retrieves the API key from request context
func GetAPIKeyFromContext(ctx context.Context) (*APIKey, bool) {
	apiKey, ok := ctx.Value(APIKeyContextKey).(*APIKey)
	return apiKey, ok
}

// GetJWTClaimsFromContext retrieves JWT claims from request context
func GetJWTClaimsFromContext(ctx context.Context) (jwt.MapClaims, bool) {
	claims, ok := ctx.Value(JWTClaimsContextKey).(jwt.MapClaims)
	return claims, ok
}

// GetTenantContextFromContext retrieves tenant context from request context
func GetTenantContextFromContext(ctx context.Context) (*TenantContext, bool) {
	tc, ok := ctx.Value(TenantContextKey).(*TenantContext)
	return tc, ok
}

// ===========================================================================================
// TLS & AUTHENTICATION
// Secure communication and API key management
// ===========================================================================================

// TLSConfig holds TLS configuration
type TLSConfig struct {
	Enabled  bool
	CertFile string
	KeyFile  string
	ClientCA string // Optional: for mTLS
}

// LoadTLSConfig loads TLS configuration from certificates
func LoadTLSConfig(cfg TLSConfig) (*tls.Config, error) {
	if !cfg.Enabled {
		return nil, nil
	}

	cert, err := tls.LoadX509KeyPair(cfg.CertFile, cfg.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load TLS certificate: %w", err)
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		},
	}

	// Load client CA for mTLS if provided
	if cfg.ClientCA != "" {
		caCert, err := os.ReadFile(cfg.ClientCA)
		if err != nil {
			return nil, fmt.Errorf("failed to read client CA: %w", err)
		}

		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse client CA certificate")
		}

		tlsConfig.ClientCAs = caCertPool
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	}

	return tlsConfig, nil
}

// ===========================================================================================
// API KEY MANAGEMENT
// ===========================================================================================

// APIKey represents an API key with metadata
type APIKey struct {
	Key         string    `json:"key"`
	Name        string    `json:"name"`
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
	Permissions []string  `json:"permissions"` // e.g., ["read", "write", "admin"]
	RateLimit   int       `json:"rate_limit"`  // requests per minute
}

// APIKeyManager manages API keys
type APIKeyManager struct {
	mu   sync.RWMutex
	keys map[string]*APIKey // key -> APIKey
}

// NewAPIKeyManager creates a new API key manager
func NewAPIKeyManager() *APIKeyManager {
	return &APIKeyManager{
		keys: make(map[string]*APIKey),
	}
}

// GenerateAPIKey generates a new API key
func (am *APIKeyManager) GenerateAPIKey(name string, permissions []string, expiresIn *time.Duration) (*APIKey, error) {
	// Generate random key
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		return nil, fmt.Errorf("failed to generate random key: %w", err)
	}
	key := "vdb_" + base64.URLEncoding.EncodeToString(keyBytes)

	var expiresAt *time.Time
	if expiresIn != nil {
		t := time.Now().Add(*expiresIn)
		expiresAt = &t
	}

	apiKey := &APIKey{
		Key:         key,
		Name:        name,
		CreatedAt:   time.Now(),
		ExpiresAt:   expiresAt,
		Permissions: permissions,
		RateLimit:   100, // Default rate limit
	}

	am.mu.Lock()
	am.keys[key] = apiKey
	am.mu.Unlock()

	return apiKey, nil
}

// ValidateAPIKey validates an API key and returns the associated metadata
func (am *APIKeyManager) ValidateAPIKey(key string) (*APIKey, error) {
	am.mu.RLock()
	defer am.mu.RUnlock()

	apiKey, exists := am.keys[key]
	if !exists {
		return nil, fmt.Errorf("invalid API key")
	}

	// Check expiration
	if apiKey.ExpiresAt != nil && time.Now().After(*apiKey.ExpiresAt) {
		return nil, fmt.Errorf("API key expired")
	}

	return apiKey, nil
}

// RevokeAPIKey revokes an API key
func (am *APIKeyManager) RevokeAPIKey(key string) error {
	am.mu.Lock()
	defer am.mu.Unlock()

	if _, exists := am.keys[key]; !exists {
		return fmt.Errorf("API key not found")
	}

	delete(am.keys, key)
	return nil
}

// ListAPIKeys returns all API keys (without the actual key value)
func (am *APIKeyManager) ListAPIKeys() []*APIKey {
	am.mu.RLock()
	defer am.mu.RUnlock()

	keys := make([]*APIKey, 0, len(am.keys))
	for _, apiKey := range am.keys {
		// Return copy without full key
		keyCopy := *apiKey
		keyCopy.Key = keyCopy.Key[:10] + "..." // Show only prefix
		keys = append(keys, &keyCopy)
	}

	return keys
}

// SaveToFile saves API keys to a file
func (am *APIKeyManager) SaveToFile(path string) error {
	am.mu.RLock()
	defer am.mu.RUnlock()

	data, err := json.MarshalIndent(am.keys, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0600)
}

// LoadFromFile loads API keys from a file
func (am *APIKeyManager) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No file yet, that's ok
		}
		return err
	}

	am.mu.Lock()
	defer am.mu.Unlock()

	return json.Unmarshal(data, &am.keys)
}

// ===========================================================================================
// JWT TOKEN SUPPORT WITH MULTI-TENANCY
// ===========================================================================================

// TenantClaims represents JWT claims with tenant information
type TenantClaims struct {
	TenantID    string   `json:"tenant_id"`
	Permissions []string `json:"permissions"`  // e.g., ["read", "write", "admin"]
	Collections []string `json:"collections"`  // allowed collections (empty = all)
	jwt.RegisteredClaims
}

// TenantContext holds tenant information for a request
type TenantContext struct {
	TenantID    string
	Permissions map[string]bool // permission -> true
	Collections map[string]bool // collection -> true (empty = all allowed)
	IsAdmin     bool
}

// JWTManager manages JWT tokens
type JWTManager struct {
	secretKey []byte
	issuer    string
}

// NewJWTManager creates a new JWT manager
func NewJWTManager(secretKey, issuer string) *JWTManager {
	return &JWTManager{
		secretKey: []byte(secretKey),
		issuer:    issuer,
	}
}

// GenerateToken generates a JWT token (legacy - use GenerateTenantToken for multi-tenancy)
func (jm *JWTManager) GenerateToken(userID string, permissions []string, expiresIn time.Duration) (string, error) {
	now := time.Now()
	claims := jwt.MapClaims{
		"sub":         userID,
		"iss":         jm.issuer,
		"iat":         now.Unix(),
		"exp":         now.Add(expiresIn).Unix(),
		"permissions": permissions,
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(jm.secretKey)
}

// GenerateTenantToken generates a JWT token with tenant claims
func (jm *JWTManager) GenerateTenantToken(tenantID string, permissions []string, collections []string, expiresIn time.Duration) (string, error) {
	claims := &TenantClaims{
		TenantID:    tenantID,
		Permissions: permissions,
		Collections: collections,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(expiresIn)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			Issuer:    jm.issuer,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(jm.secretKey)
}

// ValidateToken validates a JWT token
func (jm *JWTManager) ValidateToken(tokenString string) (*jwt.Token, error) {
	return jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jm.secretKey, nil
	})
}

// ValidateTenantToken validates a JWT token and returns tenant context
func (jm *JWTManager) ValidateTenantToken(tokenString string) (*TenantContext, error) {
	token, err := jwt.ParseWithClaims(tokenString, &TenantClaims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jm.secretKey, nil
	})

	if err != nil {
		return nil, fmt.Errorf("invalid token: %w", err)
	}

	claims, ok := token.Claims.(*TenantClaims)
	if !ok || !token.Valid {
		return nil, fmt.Errorf("invalid token claims")
	}

	// Validate issuer
	if jm.issuer != "" && claims.Issuer != jm.issuer {
		return nil, fmt.Errorf("invalid issuer: %s", claims.Issuer)
	}

	// Build tenant context
	ctx := &TenantContext{
		TenantID:    claims.TenantID,
		Permissions: make(map[string]bool),
		Collections: make(map[string]bool),
	}

	for _, perm := range claims.Permissions {
		ctx.Permissions[perm] = true
		if perm == "admin" {
			ctx.IsAdmin = true
		}
	}

	for _, coll := range claims.Collections {
		ctx.Collections[coll] = true
	}

	return ctx, nil
}

// ===========================================================================================
// ACCESS CONTROL LISTS (ACLs)
// ===========================================================================================

// ACL represents access control for collections per tenant
type ACL struct {
	mu          sync.RWMutex
	permissions map[string]map[string]bool // tenantID -> collection -> can access
	tenantPerms map[string]map[string]bool // tenantID -> permission (read/write/admin)
}

// NewACL creates a new ACL manager
func NewACL() *ACL {
	return &ACL{
		permissions: make(map[string]map[string]bool),
		tenantPerms: make(map[string]map[string]bool),
	}
}

// GrantCollectionAccess grants a tenant access to a collection
func (acl *ACL) GrantCollectionAccess(tenantID, collection string) {
	acl.mu.Lock()
	defer acl.mu.Unlock()
	if acl.permissions[tenantID] == nil {
		acl.permissions[tenantID] = make(map[string]bool)
	}
	acl.permissions[tenantID][collection] = true
}

// RevokeCollectionAccess revokes a tenant's access to a collection
func (acl *ACL) RevokeCollectionAccess(tenantID, collection string) {
	acl.mu.Lock()
	defer acl.mu.Unlock()
	if acl.permissions[tenantID] != nil {
		delete(acl.permissions[tenantID], collection)
	}
}

// HasCollectionAccess checks if a tenant can access a collection
func (acl *ACL) HasCollectionAccess(tenantID, collection string) bool {
	acl.mu.RLock()
	defer acl.mu.RUnlock()

	// Admin has access to all collections
	if acl.hasPermission(tenantID, "admin") {
		return true
	}

	// Check if tenant has explicit access
	if colls, ok := acl.permissions[tenantID]; ok {
		if len(colls) == 0 {
			// Empty map means access to all collections
			return true
		}
		return colls[collection]
	}
	return false
}

// GrantPermission grants a permission to a tenant (read/write/admin)
func (acl *ACL) GrantPermission(tenantID, permission string) {
	acl.mu.Lock()
	defer acl.mu.Unlock()
	if acl.tenantPerms[tenantID] == nil {
		acl.tenantPerms[tenantID] = make(map[string]bool)
	}
	acl.tenantPerms[tenantID][permission] = true
}

// RevokePermission revokes a permission from a tenant
func (acl *ACL) RevokePermission(tenantID, permission string) {
	acl.mu.Lock()
	defer acl.mu.Unlock()
	if acl.tenantPerms[tenantID] != nil {
		delete(acl.tenantPerms[tenantID], permission)
	}
}

// HasPermission checks if a tenant has a specific permission
func (acl *ACL) HasPermission(tenantID, permission string) bool {
	acl.mu.RLock()
	defer acl.mu.RUnlock()
	return acl.hasPermission(tenantID, permission)
}

func (acl *ACL) hasPermission(tenantID, permission string) bool {
	if perms, ok := acl.tenantPerms[tenantID]; ok {
		// Admin implies all permissions
		if perms["admin"] {
			return true
		}
		return perms[permission]
	}
	return false
}

// ===========================================================================================
// TENANT STORAGE QUOTAS
// ===========================================================================================

// TenantQuota tracks storage quotas per tenant
type TenantQuota struct {
	mu          sync.RWMutex
	quotas      map[string]int64 // tenantID -> max bytes
	usage       map[string]int64 // tenantID -> current bytes
	vectorCount map[string]int   // tenantID -> vector count
}

// NewTenantQuota creates a new quota manager
func NewTenantQuota() *TenantQuota {
	return &TenantQuota{
		quotas:      make(map[string]int64),
		usage:       make(map[string]int64),
		vectorCount: make(map[string]int),
	}
}

// SetQuota sets the storage quota for a tenant (in bytes)
func (tq *TenantQuota) SetQuota(tenantID string, maxBytes int64) {
	tq.mu.Lock()
	defer tq.mu.Unlock()
	tq.quotas[tenantID] = maxBytes
}

// AddUsage adds to a tenant's storage usage
func (tq *TenantQuota) AddUsage(tenantID string, bytes int64, vectors int) error {
	tq.mu.Lock()
	defer tq.mu.Unlock()

	newUsage := tq.usage[tenantID] + bytes
	if quota, ok := tq.quotas[tenantID]; ok && newUsage > quota {
		return fmt.Errorf("quota exceeded: %d/%d bytes used", newUsage, quota)
	}

	tq.usage[tenantID] = newUsage
	tq.vectorCount[tenantID] += vectors
	return nil
}

// RemoveUsage removes from a tenant's storage usage
func (tq *TenantQuota) RemoveUsage(tenantID string, bytes int64, vectors int) {
	tq.mu.Lock()
	defer tq.mu.Unlock()

	tq.usage[tenantID] -= bytes
	if tq.usage[tenantID] < 0 {
		tq.usage[tenantID] = 0
	}

	tq.vectorCount[tenantID] -= vectors
	if tq.vectorCount[tenantID] < 0 {
		tq.vectorCount[tenantID] = 0
	}
}

// GetUsage returns current usage for a tenant
func (tq *TenantQuota) GetUsage(tenantID string) (bytes int64, vectors int) {
	tq.mu.RLock()
	defer tq.mu.RUnlock()
	return tq.usage[tenantID], tq.vectorCount[tenantID]
}

// GetQuota returns the quota for a tenant
func (tq *TenantQuota) GetQuota(tenantID string) int64 {
	tq.mu.RLock()
	defer tq.mu.RUnlock()
	return tq.quotas[tenantID]
}

// ===========================================================================================
// PER-TENANT RATE LIMITING
// ===========================================================================================

type tenantRateLimiter struct {
	mu       sync.RWMutex
	limiters map[string]*rateLimiter // tenantID -> rate limiter
	rps      int
	burst    int
	window   time.Duration
}

func newTenantRateLimiter(rps, burst int, window time.Duration) *tenantRateLimiter {
	return &tenantRateLimiter{
		limiters: make(map[string]*rateLimiter),
		rps:      rps,
		burst:    burst,
		window:   window,
	}
}

func (trl *tenantRateLimiter) allow(tenantID string) bool {
	trl.mu.RLock()
	limiter, ok := trl.limiters[tenantID]
	trl.mu.RUnlock()

	if !ok {
		trl.mu.Lock()
		// Double-check after acquiring write lock
		limiter, ok = trl.limiters[tenantID]
		if !ok {
			limiter = newRateLimiter(trl.rps, trl.burst, trl.window)
			trl.limiters[tenantID] = limiter
		}
		trl.mu.Unlock()
	}

	return limiter.allow(tenantID)
}

func (trl *tenantRateLimiter) setLimit(tenantID string, rps, burst int) {
	trl.mu.Lock()
	defer trl.mu.Unlock()
	trl.limiters[tenantID] = newRateLimiter(rps, burst, trl.window)
}

// ===========================================================================================
// AUTHENTICATION MIDDLEWARE
// ===========================================================================================

// AuthMiddleware provides authentication and authorization
type AuthMiddleware struct {
	apiKeyMgr *APIKeyManager
	jwtMgr    *JWTManager
	disabled  bool
}

// NewAuthMiddleware creates a new auth middleware
func NewAuthMiddleware(apiKeyMgr *APIKeyManager, jwtMgr *JWTManager) *AuthMiddleware {
	return &AuthMiddleware{
		apiKeyMgr: apiKeyMgr,
		jwtMgr:    jwtMgr,
		disabled:  false,
	}
}

// Middleware returns HTTP middleware that enforces authentication
func (am *AuthMiddleware) Middleware(requiredPermissions ...string) func(http.HandlerFunc) http.HandlerFunc {
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if am.disabled {
				next(w, r)
				return
			}

			// Extract token from header or query parameter
			token := extractToken(r)
			if token == "" {
				http.Error(w, "missing authentication token", http.StatusUnauthorized)
				return
			}

			// Try API key first
			if strings.HasPrefix(token, "vdb_") {
				apiKey, err := am.apiKeyMgr.ValidateAPIKey(token)
				if err != nil {
					http.Error(w, "invalid API key", http.StatusUnauthorized)
					return
				}

				// Check permissions
				if !hasPermissions(apiKey.Permissions, requiredPermissions) {
					http.Error(w, "insufficient permissions", http.StatusForbidden)
					return
				}

				// Add API key to context for downstream handlers
				ctx := context.WithValue(r.Context(), APIKeyContextKey, apiKey)
				next(w, r.WithContext(ctx))
				return
			}

			// Try JWT token
			if am.jwtMgr != nil {
				jwtToken, err := am.jwtMgr.ValidateToken(token)
				if err != nil || !jwtToken.Valid {
					http.Error(w, "invalid token", http.StatusUnauthorized)
					return
				}

				// Extract permissions from claims
				claims := jwtToken.Claims.(jwt.MapClaims)
				perms, _ := claims["permissions"].([]interface{})
				permissions := make([]string, len(perms))
				for i, p := range perms {
					permissions[i] = p.(string)
				}

				// Check permissions
				if !hasPermissions(permissions, requiredPermissions) {
					http.Error(w, "insufficient permissions", http.StatusForbidden)
					return
				}

				// Add JWT claims to context for downstream handlers
				ctx := context.WithValue(r.Context(), JWTClaimsContextKey, claims)
				next(w, r.WithContext(ctx))
				return
			}

			http.Error(w, "authentication failed", http.StatusUnauthorized)
		}
	}
}

// extractToken extracts authentication token from request
func extractToken(r *http.Request) string {
	// Check Authorization header
	auth := r.Header.Get("Authorization")
	if auth != "" {
		// Handle "Bearer <token>" format
		if strings.HasPrefix(auth, "Bearer ") {
			return strings.TrimPrefix(auth, "Bearer ")
		}
		return auth
	}

	// Check query parameter
	return r.URL.Query().Get("token")
}

// hasPermissions checks if user has required permissions
func hasPermissions(userPerms, requiredPerms []string) bool {
	if len(requiredPerms) == 0 {
		return true // No permissions required
	}

	// Check if user has "admin" permission (grants all)
	for _, perm := range userPerms {
		if perm == "admin" {
			return true
		}
	}

	// Check each required permission
	for _, required := range requiredPerms {
		found := false
		for _, perm := range userPerms {
			if perm == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// SecureCompare performs constant-time comparison of two strings
func SecureCompare(a, b string) bool {
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}

// ===========================================================================================
// CONTEXT EXTRACTION HELPERS
// ===========================================================================================

// GetTenantFromRequest extracts tenant ID from JWT token in HTTP request
// Returns "default" if no JWT is present or JWT is not enabled
func GetTenantFromRequest(r *http.Request, jwtMgr *JWTManager) string {
	if jwtMgr == nil {
		return "default"
	}

	token := extractToken(r)
	if token == "" {
		return "default"
	}

	// Try to validate as tenant token
	tenantCtx, err := jwtMgr.ValidateTenantToken(token)
	if err != nil {
		return "default"
	}

	return tenantCtx.TenantID
}

// GetTenantContextFromRequest extracts full tenant context from JWT token
// Returns default context if no JWT is present or invalid
func GetTenantContextFromRequest(r *http.Request, jwtMgr *JWTManager) *TenantContext {
	if jwtMgr == nil {
		return &TenantContext{
			TenantID:    "default",
			Permissions: map[string]bool{"read": true, "write": true, "admin": true},
			Collections: make(map[string]bool),
			IsAdmin:     true,
		}
	}

	token := extractToken(r)
	if token == "" {
		return &TenantContext{
			TenantID:    "default",
			Permissions: map[string]bool{"read": true},
			Collections: make(map[string]bool),
			IsAdmin:     false,
		}
	}

	tenantCtx, err := jwtMgr.ValidateTenantToken(token)
	if err != nil {
		return &TenantContext{
			TenantID:    "default",
			Permissions: map[string]bool{"read": true},
			Collections: make(map[string]bool),
			IsAdmin:     false,
		}
	}

	return tenantCtx
}
