package main

import (
	"crypto/rand"
	"crypto/subtle"
	"crypto/tls"
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

	// TODO: Load client CA for mTLS if provided
	// if cfg.ClientCA != "" {
	//     ... load and configure client CA
	// }

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
// JWT TOKEN SUPPORT
// ===========================================================================================

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

// GenerateToken generates a JWT token
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

// ValidateToken validates a JWT token
func (jm *JWTManager) ValidateToken(tokenString string) (*jwt.Token, error) {
	return jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jm.secretKey, nil
	})
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

				// TODO: Add API key to context for downstream handlers
				next(w, r)
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

				// TODO: Add JWT claims to context for downstream handlers
				next(w, r)
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
