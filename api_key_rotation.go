package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

// ===========================================================================================
// API KEY ROTATION SYSTEM
// Provides secure key lifecycle management with rotation policies
// ===========================================================================================

// KeyRotationPolicy defines when keys should be rotated
type KeyRotationPolicy struct {
	// MaxAge is the maximum age of a key before it should be rotated
	MaxAge time.Duration `json:"max_age"`
	// WarningPeriod is how long before expiry to warn
	WarningPeriod time.Duration `json:"warning_period"`
	// GracePeriod allows old keys to work after rotation
	GracePeriod time.Duration `json:"grace_period"`
	// AutoRotate enables automatic rotation
	AutoRotate bool `json:"auto_rotate"`
}

// DefaultRotationPolicy returns sensible defaults
func DefaultRotationPolicy() KeyRotationPolicy {
	return KeyRotationPolicy{
		MaxAge:        90 * 24 * time.Hour, // 90 days
		WarningPeriod: 14 * 24 * time.Hour, // 14 days warning
		GracePeriod:   7 * 24 * time.Hour,  // 7 days grace
		AutoRotate:    false,
	}
}

// RotatableAPIKey extends APIKey with rotation metadata
type RotatableAPIKey struct {
	*APIKey
	RotatedAt   *time.Time `json:"rotated_at,omitempty"`
	PreviousKey string     `json:"-"` // Previous key (for grace period validation)
	RotationID  string     `json:"rotation_id"`
	IsActive    bool       `json:"is_active"`
	LastUsedAt  *time.Time `json:"last_used_at,omitempty"`
	UsageCount  int64      `json:"usage_count"`
}

// KeyRotationEvent records a rotation event for auditing
type KeyRotationEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	KeyName     string    `json:"key_name"`
	RotationID  string    `json:"rotation_id"`
	Reason      string    `json:"reason"`
	OldKeyHash  string    `json:"old_key_hash"` // SHA256 hash for audit trail
	NewKeyHash  string    `json:"new_key_hash"`
	InitiatedBy string    `json:"initiated_by"` // "system", "admin", or user ID
}

// KeyRotationManager handles API key lifecycle and rotation
type KeyRotationManager struct {
	mu sync.RWMutex

	// Key storage
	keys      map[string]*RotatableAPIKey // keyID -> key
	keysByKey map[string]string           // actual key -> keyID (for lookup)

	// Grace period keys (old keys that still work temporarily)
	graceKeys map[string]*graceKey // old key -> grace info

	// Configuration
	policy KeyRotationPolicy

	// Rotation history
	rotationEvents []KeyRotationEvent

	// Background processing
	stopChan    chan struct{}
	stoppedChan chan struct{}
	running     bool

	// Callbacks
	onRotation func(event KeyRotationEvent)
	onWarning  func(keyName string, expiresIn time.Duration)
}

type graceKey struct {
	keyID     string
	expiresAt time.Time
}

// NewKeyRotationManager creates a new key rotation manager
func NewKeyRotationManager(policy KeyRotationPolicy) *KeyRotationManager {
	return &KeyRotationManager{
		keys:           make(map[string]*RotatableAPIKey),
		keysByKey:      make(map[string]string),
		graceKeys:      make(map[string]*graceKey),
		policy:         policy,
		rotationEvents: make([]KeyRotationEvent, 0),
		stopChan:       make(chan struct{}),
		stoppedChan:    make(chan struct{}),
	}
}

// Start begins background monitoring for key rotation
func (krm *KeyRotationManager) Start() {
	krm.mu.Lock()
	if krm.running {
		krm.mu.Unlock()
		return
	}
	krm.running = true
	krm.mu.Unlock()

	go krm.backgroundLoop()
}

// Stop halts background processing
func (krm *KeyRotationManager) Stop() {
	krm.mu.Lock()
	if !krm.running {
		krm.mu.Unlock()
		return
	}
	krm.running = false
	krm.mu.Unlock()

	close(krm.stopChan)
	<-krm.stoppedChan
}

// CreateKey creates a new API key
func (krm *KeyRotationManager) CreateKey(name string, permissions []string, expiresIn *time.Duration) (*RotatableAPIKey, error) {
	key, err := krm.generateKey()
	if err != nil {
		return nil, err
	}

	keyID := krm.generateKeyID()

	var expiresAt *time.Time
	if expiresIn != nil {
		t := time.Now().Add(*expiresIn)
		expiresAt = &t
	}

	rotatable := &RotatableAPIKey{
		APIKey: &APIKey{
			Key:         key,
			Name:        name,
			CreatedAt:   time.Now(),
			ExpiresAt:   expiresAt,
			Permissions: permissions,
			RateLimit:   100,
		},
		RotationID: keyID,
		IsActive:   true,
	}

	krm.mu.Lock()
	krm.keys[keyID] = rotatable
	krm.keysByKey[key] = keyID
	krm.mu.Unlock()

	return rotatable, nil
}

// RotateKey rotates an existing key, returning the new key
func (krm *KeyRotationManager) RotateKey(keyID string, initiatedBy string, reason string) (*RotatableAPIKey, error) {
	krm.mu.Lock()
	defer krm.mu.Unlock()

	oldKey, exists := krm.keys[keyID]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	// Generate new key
	newKeyValue, err := krm.generateKey()
	if err != nil {
		return nil, err
	}

	// Record rotation event
	event := KeyRotationEvent{
		Timestamp:   time.Now(),
		KeyName:     oldKey.Name,
		RotationID:  keyID,
		Reason:      reason,
		OldKeyHash:  hashKey(oldKey.Key),
		NewKeyHash:  hashKey(newKeyValue),
		InitiatedBy: initiatedBy,
	}
	krm.rotationEvents = append(krm.rotationEvents, event)

	// Invoke callback
	if krm.onRotation != nil {
		go krm.onRotation(event)
	}

	// Set up grace period for old key
	if krm.policy.GracePeriod > 0 {
		krm.graceKeys[oldKey.Key] = &graceKey{
			keyID:     keyID,
			expiresAt: time.Now().Add(krm.policy.GracePeriod),
		}
	}

	// Remove old key from lookup
	delete(krm.keysByKey, oldKey.Key)

	// Update key
	now := time.Now()
	oldKey.PreviousKey = oldKey.Key
	oldKey.Key = newKeyValue
	oldKey.RotatedAt = &now

	// Reset expiry if using policy-based expiration
	if krm.policy.MaxAge > 0 {
		t := time.Now().Add(krm.policy.MaxAge)
		oldKey.ExpiresAt = &t
	}

	// Add new key to lookup
	krm.keysByKey[newKeyValue] = keyID

	return oldKey, nil
}

// ValidateKey validates a key and returns its metadata
func (krm *KeyRotationManager) ValidateKey(key string) (*RotatableAPIKey, error) {
	krm.mu.Lock()
	defer krm.mu.Unlock()

	// Try direct lookup
	if keyID, exists := krm.keysByKey[key]; exists {
		rotatable := krm.keys[keyID]
		if rotatable == nil {
			return nil, fmt.Errorf("key metadata not found")
		}

		// Check if active
		if !rotatable.IsActive {
			return nil, fmt.Errorf("key is deactivated")
		}

		// Check expiration
		if rotatable.ExpiresAt != nil && time.Now().After(*rotatable.ExpiresAt) {
			return nil, fmt.Errorf("key expired")
		}

		// Update usage stats
		now := time.Now()
		rotatable.LastUsedAt = &now
		rotatable.UsageCount++

		return rotatable, nil
	}

	// Check grace period keys
	if grace, exists := krm.graceKeys[key]; exists {
		if time.Now().Before(grace.expiresAt) {
			// Grace period still valid
			rotatable := krm.keys[grace.keyID]
			if rotatable != nil && rotatable.IsActive {
				now := time.Now()
				rotatable.LastUsedAt = &now
				rotatable.UsageCount++
				return rotatable, nil
			}
		} else {
			// Grace period expired, clean up
			delete(krm.graceKeys, key)
		}
	}

	return nil, fmt.Errorf("invalid key")
}

// RevokeKey immediately revokes a key
func (krm *KeyRotationManager) RevokeKey(keyID string, initiatedBy string, reason string) error {
	krm.mu.Lock()
	defer krm.mu.Unlock()

	key, exists := krm.keys[keyID]
	if !exists {
		return fmt.Errorf("key not found: %s", keyID)
	}

	// Record event
	event := KeyRotationEvent{
		Timestamp:   time.Now(),
		KeyName:     key.Name,
		RotationID:  keyID,
		Reason:      "revoked: " + reason,
		OldKeyHash:  hashKey(key.Key),
		InitiatedBy: initiatedBy,
	}
	krm.rotationEvents = append(krm.rotationEvents, event)

	// Remove from lookups
	delete(krm.keysByKey, key.Key)
	delete(krm.keys, keyID)

	// Also remove any grace keys
	for k, g := range krm.graceKeys {
		if g.keyID == keyID {
			delete(krm.graceKeys, k)
		}
	}

	return nil
}

// DeactivateKey temporarily deactivates a key (can be reactivated)
func (krm *KeyRotationManager) DeactivateKey(keyID string) error {
	krm.mu.Lock()
	defer krm.mu.Unlock()

	key, exists := krm.keys[keyID]
	if !exists {
		return fmt.Errorf("key not found: %s", keyID)
	}

	key.IsActive = false
	return nil
}

// ReactivateKey reactivates a deactivated key
func (krm *KeyRotationManager) ReactivateKey(keyID string) error {
	krm.mu.Lock()
	defer krm.mu.Unlock()

	key, exists := krm.keys[keyID]
	if !exists {
		return fmt.Errorf("key not found: %s", keyID)
	}

	key.IsActive = true
	return nil
}

// ListKeys returns all keys (without actual key values)
func (krm *KeyRotationManager) ListKeys() []*RotatableAPIKey {
	krm.mu.RLock()
	defer krm.mu.RUnlock()

	keys := make([]*RotatableAPIKey, 0, len(krm.keys))
	for _, key := range krm.keys {
		// Return sanitized copy
		keyCopy := &RotatableAPIKey{
			APIKey: &APIKey{
				Key:         key.Key[:10] + "...", // Show only prefix
				Name:        key.Name,
				CreatedAt:   key.CreatedAt,
				ExpiresAt:   key.ExpiresAt,
				Permissions: key.Permissions,
				RateLimit:   key.RateLimit,
			},
			RotatedAt:  key.RotatedAt,
			RotationID: key.RotationID,
			IsActive:   key.IsActive,
			LastUsedAt: key.LastUsedAt,
			UsageCount: key.UsageCount,
		}
		keys = append(keys, keyCopy)
	}

	// Sort by creation time
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].CreatedAt.After(keys[j].CreatedAt)
	})

	return keys
}

// GetKeyByID returns a key by its rotation ID (without actual key value)
func (krm *KeyRotationManager) GetKeyByID(keyID string) (*RotatableAPIKey, error) {
	krm.mu.RLock()
	defer krm.mu.RUnlock()

	key, exists := krm.keys[keyID]
	if !exists {
		return nil, fmt.Errorf("key not found")
	}

	// Return sanitized copy
	return &RotatableAPIKey{
		APIKey: &APIKey{
			Key:         key.Key[:10] + "...",
			Name:        key.Name,
			CreatedAt:   key.CreatedAt,
			ExpiresAt:   key.ExpiresAt,
			Permissions: key.Permissions,
			RateLimit:   key.RateLimit,
		},
		RotatedAt:  key.RotatedAt,
		RotationID: key.RotationID,
		IsActive:   key.IsActive,
		LastUsedAt: key.LastUsedAt,
		UsageCount: key.UsageCount,
	}, nil
}

// GetRotationHistory returns rotation events
func (krm *KeyRotationManager) GetRotationHistory(keyID string, limit int) []KeyRotationEvent {
	krm.mu.RLock()
	defer krm.mu.RUnlock()

	events := make([]KeyRotationEvent, 0)
	for _, event := range krm.rotationEvents {
		if keyID == "" || event.RotationID == keyID {
			events = append(events, event)
		}
	}

	// Most recent first
	sort.Slice(events, func(i, j int) bool {
		return events[i].Timestamp.After(events[j].Timestamp)
	})

	if limit > 0 && len(events) > limit {
		events = events[:limit]
	}

	return events
}

// GetKeysNeedingRotation returns keys that should be rotated based on policy
func (krm *KeyRotationManager) GetKeysNeedingRotation() []*RotatableAPIKey {
	krm.mu.RLock()
	defer krm.mu.RUnlock()

	var needsRotation []*RotatableAPIKey

	for _, key := range krm.keys {
		if !key.IsActive {
			continue
		}

		// Check age-based rotation
		if krm.policy.MaxAge > 0 {
			age := time.Since(key.CreatedAt)
			if key.RotatedAt != nil {
				age = time.Since(*key.RotatedAt)
			}

			if age > krm.policy.MaxAge-krm.policy.WarningPeriod {
				needsRotation = append(needsRotation, key)
			}
		}

		// Check expiry-based rotation
		if key.ExpiresAt != nil {
			remaining := time.Until(*key.ExpiresAt)
			if remaining < krm.policy.WarningPeriod {
				needsRotation = append(needsRotation, key)
			}
		}
	}

	return needsRotation
}

// SetOnRotation sets a callback for rotation events
func (krm *KeyRotationManager) SetOnRotation(callback func(event KeyRotationEvent)) {
	krm.mu.Lock()
	defer krm.mu.Unlock()
	krm.onRotation = callback
}

// SetOnWarning sets a callback for expiry warnings
func (krm *KeyRotationManager) SetOnWarning(callback func(keyName string, expiresIn time.Duration)) {
	krm.mu.Lock()
	defer krm.mu.Unlock()
	krm.onWarning = callback
}

// Stats returns rotation manager statistics
func (krm *KeyRotationManager) Stats() map[string]interface{} {
	krm.mu.RLock()
	defer krm.mu.RUnlock()

	activeCount := 0
	expiredCount := 0
	for _, key := range krm.keys {
		if key.IsActive {
			activeCount++
		}
		if key.ExpiresAt != nil && time.Now().After(*key.ExpiresAt) {
			expiredCount++
		}
	}

	return map[string]interface{}{
		"total_keys":          len(krm.keys),
		"active_keys":         activeCount,
		"expired_keys":        expiredCount,
		"grace_period_keys":   len(krm.graceKeys),
		"total_rotations":     len(krm.rotationEvents),
		"auto_rotate_enabled": krm.policy.AutoRotate,
	}
}

// SaveToFile persists keys to file
func (krm *KeyRotationManager) SaveToFile(path string) error {
	krm.mu.RLock()
	defer krm.mu.RUnlock()

	data, err := json.MarshalIndent(krm.keys, "", "  ")
	if err != nil {
		return err
	}

	return writeFileSecure(path, data, 0600)
}

// LoadFromFile loads keys from file
func (krm *KeyRotationManager) LoadFromFile(path string) error {
	data, err := readFileSecure(path)
	if err != nil {
		return err
	}

	krm.mu.Lock()
	defer krm.mu.Unlock()

	if err := json.Unmarshal(data, &krm.keys); err != nil {
		return err
	}

	// Rebuild key lookup
	krm.keysByKey = make(map[string]string)
	for keyID, key := range krm.keys {
		krm.keysByKey[key.Key] = keyID
	}

	return nil
}

// Background loop for auto-rotation and cleanup
func (krm *KeyRotationManager) backgroundLoop() {
	defer close(krm.stoppedChan)

	ticker := time.NewTicker(1 * time.Hour) // Check every hour
	defer ticker.Stop()

	cleanupTicker := time.NewTicker(5 * time.Minute) // Cleanup every 5 minutes
	defer cleanupTicker.Stop()

	for {
		select {
		case <-krm.stopChan:
			return

		case <-ticker.C:
			krm.checkAndWarn()
			if krm.policy.AutoRotate {
				krm.autoRotate()
			}

		case <-cleanupTicker.C:
			krm.cleanupGraceKeys()
		}
	}
}

func (krm *KeyRotationManager) checkAndWarn() {
	keysNeedingRotation := krm.GetKeysNeedingRotation()

	if krm.onWarning != nil {
		for _, key := range keysNeedingRotation {
			var remaining time.Duration
			if key.ExpiresAt != nil {
				remaining = time.Until(*key.ExpiresAt)
			} else if krm.policy.MaxAge > 0 {
				age := time.Since(key.CreatedAt)
				if key.RotatedAt != nil {
					age = time.Since(*key.RotatedAt)
				}
				remaining = krm.policy.MaxAge - age
			}
			krm.onWarning(key.Name, remaining)
		}
	}
}

func (krm *KeyRotationManager) autoRotate() {
	krm.mu.Lock()
	defer krm.mu.Unlock()

	now := time.Now()
	for keyID, key := range krm.keys {
		if !key.IsActive {
			continue
		}

		var age time.Duration
		if key.RotatedAt != nil {
			age = now.Sub(*key.RotatedAt)
		} else {
			age = now.Sub(key.CreatedAt)
		}

		if age >= krm.policy.MaxAge {
			// Need to unlock for RotateKey
			krm.mu.Unlock()
			_, _ = krm.RotateKey(keyID, "system", "auto-rotation: max age exceeded")
			krm.mu.Lock()
		}
	}
}

func (krm *KeyRotationManager) cleanupGraceKeys() {
	krm.mu.Lock()
	defer krm.mu.Unlock()

	now := time.Now()
	for key, grace := range krm.graceKeys {
		if now.After(grace.expiresAt) {
			delete(krm.graceKeys, key)
		}
	}
}

// Helper functions

func (krm *KeyRotationManager) generateKey() (string, error) {
	keyBytes := make([]byte, 32)
	if _, err := rand.Read(keyBytes); err != nil {
		return "", fmt.Errorf("failed to generate random key: %w", err)
	}
	return "vdb_" + base64.URLEncoding.EncodeToString(keyBytes), nil
}

func (krm *KeyRotationManager) generateKeyID() string {
	idBytes := make([]byte, 8)
	if _, err := rand.Read(idBytes); err != nil {
		// Fallback to time-based ID if crypto/rand fails (extremely rare)
		return fmt.Sprintf("key-%d", time.Now().UnixNano())
	}
	return base64.URLEncoding.EncodeToString(idBytes)
}

func hashKey(key string) string {
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:8]) // Just first 8 bytes for logging
}

func writeFileSecure(path string, data []byte, perm os.FileMode) error {
	tempPath := path + ".tmp"
	if err := os.WriteFile(tempPath, data, perm); err != nil {
		return err
	}
	return os.Rename(tempPath, path)
}

func readFileSecure(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}

// ===========================================================================================
// HTTP HANDLERS FOR KEY ROTATION API
// ===========================================================================================

// KeyRotationHandler provides HTTP handlers for key rotation
type KeyRotationHandler struct {
	manager *KeyRotationManager
}

// NewKeyRotationHandler creates a new handler
func NewKeyRotationHandler(manager *KeyRotationManager) *KeyRotationHandler {
	return &KeyRotationHandler{manager: manager}
}

// RegisterRoutes registers HTTP routes
func (h *KeyRotationHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/admin/keys", h.handleKeys)
	mux.HandleFunc("/admin/keys/create", h.handleCreateKey)
	mux.HandleFunc("/admin/keys/rotate", h.handleRotateKey)
	mux.HandleFunc("/admin/keys/revoke", h.handleRevokeKey)
	mux.HandleFunc("/admin/keys/deactivate", h.handleDeactivateKey)
	mux.HandleFunc("/admin/keys/reactivate", h.handleReactivateKey)
	mux.HandleFunc("/admin/keys/history", h.handleRotationHistory)
	mux.HandleFunc("/admin/keys/stats", h.handleStats)
	mux.HandleFunc("/admin/keys/needs-rotation", h.handleNeedsRotation)
}

func (h *KeyRotationHandler) handleKeys(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	keys := h.manager.ListKeys()
	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status": "success",
		"keys":   keys,
	})
}

func (h *KeyRotationHandler) handleCreateKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Name        string   `json:"name"`
		Permissions []string `json:"permissions"`
		ExpiresIn   string   `json:"expires_in"` // e.g., "720h" for 30 days
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		http.Error(w, "name is required", http.StatusBadRequest)
		return
	}

	var expiresIn *time.Duration
	if req.ExpiresIn != "" {
		d, err := time.ParseDuration(req.ExpiresIn)
		if err != nil {
			http.Error(w, "invalid expires_in format", http.StatusBadRequest)
			return
		}
		expiresIn = &d
	}

	key, err := h.manager.CreateKey(req.Name, req.Permissions, expiresIn)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Return full key only on creation
	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status":      "success",
		"key":         key.Key, // Full key shown only once
		"rotation_id": key.RotationID,
		"name":        key.Name,
		"created_at":  key.CreatedAt,
		"expires_at":  key.ExpiresAt,
		"message":     "Store this key securely. It will not be shown again.",
	})
}

func (h *KeyRotationHandler) handleRotateKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		KeyID  string `json:"key_id"`
		Reason string `json:"reason"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.KeyID == "" {
		http.Error(w, "key_id is required", http.StatusBadRequest)
		return
	}

	// Get initiator from context (set by auth middleware)
	initiator := "admin" // Default
	if apiKey, ok := GetAPIKeyFromContext(r.Context()); ok {
		initiator = apiKey.Name
	}

	key, err := h.manager.RotateKey(req.KeyID, initiator, req.Reason)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status":      "success",
		"key":         key.Key, // New key shown only once
		"rotation_id": key.RotationID,
		"rotated_at":  key.RotatedAt,
		"message":     "Key rotated. Store the new key securely.",
	})
}

func (h *KeyRotationHandler) handleRevokeKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		KeyID  string `json:"key_id"`
		Reason string `json:"reason"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	initiator := "admin"
	if apiKey, ok := GetAPIKeyFromContext(r.Context()); ok {
		initiator = apiKey.Name
	}

	if err := h.manager.RevokeKey(req.KeyID, initiator, req.Reason); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Key revoked successfully",
	})
}

func (h *KeyRotationHandler) handleDeactivateKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		KeyID string `json:"key_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if err := h.manager.DeactivateKey(req.KeyID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Key deactivated",
	})
}

func (h *KeyRotationHandler) handleReactivateKey(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		KeyID string `json:"key_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if err := h.manager.ReactivateKey(req.KeyID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Key reactivated",
	})
}

func (h *KeyRotationHandler) handleRotationHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	keyID := r.URL.Query().Get("key_id")
	events := h.manager.GetRotationHistory(keyID, 100)

	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status": "success",
		"events": events,
	})
}

func (h *KeyRotationHandler) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	stats := h.manager.Stats()
	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status": "success",
		"stats":  stats,
	})
}

func (h *KeyRotationHandler) handleNeedsRotation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	keys := h.manager.GetKeysNeedingRotation()

	// Sanitize keys
	sanitized := make([]map[string]interface{}, len(keys))
	for i, key := range keys {
		var remaining time.Duration
		if key.ExpiresAt != nil {
			remaining = time.Until(*key.ExpiresAt)
		}

		sanitized[i] = map[string]interface{}{
			"rotation_id": key.RotationID,
			"name":        key.Name,
			"expires_at":  key.ExpiresAt,
			"remaining":   remaining.String(),
		}
	}

	respondJSONKey(w, http.StatusOK, map[string]interface{}{
		"status": "success",
		"keys":   sanitized,
	})
}

// respondJSONKey writes a JSON response
func respondJSONKey(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}
