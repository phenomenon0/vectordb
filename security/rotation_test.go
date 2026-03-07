package security

import (
	"testing"
	"time"
)

// TestDefaultRotationPolicy tests default policy values
func TestDefaultRotationPolicy(t *testing.T) {
	policy := DefaultRotationPolicy()

	if policy.MaxAge != 90*24*time.Hour {
		t.Errorf("expected 90 days, got %v", policy.MaxAge)
	}
	if policy.WarningPeriod != 14*24*time.Hour {
		t.Errorf("expected 14 days warning, got %v", policy.WarningPeriod)
	}
	if policy.GracePeriod != 7*24*time.Hour {
		t.Errorf("expected 7 days grace, got %v", policy.GracePeriod)
	}
	if policy.AutoRotate {
		t.Error("auto rotate should be disabled by default")
	}
}

// TestKeyRotationManagerCreateKey tests key creation
func TestKeyRotationManagerCreateKey(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, err := krm.CreateKey("test-key", []string{"read", "write"}, nil)
	if err != nil {
		t.Fatalf("failed to create key: %v", err)
	}

	if key.Name != "test-key" {
		t.Errorf("expected name 'test-key', got %s", key.Name)
	}
	if len(key.Key) < 20 {
		t.Error("key should be sufficiently long")
	}
	if !key.IsActive {
		t.Error("new key should be active")
	}
	if key.RotationID == "" {
		t.Error("key should have rotation ID")
	}
	if len(key.Permissions) != 2 {
		t.Errorf("expected 2 permissions, got %d", len(key.Permissions))
	}
}

// TestKeyRotationManagerCreateKeyWithExpiry tests key creation with expiry
func TestKeyRotationManagerCreateKeyWithExpiry(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	expiry := 24 * time.Hour
	key, err := krm.CreateKey("expiring-key", []string{"read"}, &expiry)
	if err != nil {
		t.Fatalf("failed to create key: %v", err)
	}

	if key.ExpiresAt == nil {
		t.Fatal("key should have expiry")
	}

	// Should expire within 25 hours
	if time.Until(*key.ExpiresAt) > 25*time.Hour {
		t.Error("expiry too far in future")
	}
}

// TestKeyRotationManagerValidateKey tests key validation
func TestKeyRotationManagerValidateKey(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)

	// Valid key should work
	validated, err := krm.ValidateKey(key.Key)
	if err != nil {
		t.Fatalf("validation failed: %v", err)
	}
	if validated.Name != "test-key" {
		t.Error("validated key has wrong name")
	}

	// Invalid key should fail
	_, err = krm.ValidateKey("invalid-key")
	if err == nil {
		t.Error("invalid key should fail validation")
	}
}

// TestKeyRotationManagerValidateExpiredKey tests expired key rejection
func TestKeyRotationManagerValidateExpiredKey(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	// Create key that expires in 1ms
	expiry := 1 * time.Millisecond
	key, _ := krm.CreateKey("expiring-key", []string{"read"}, &expiry)

	// Wait for expiry
	time.Sleep(10 * time.Millisecond)

	// Should fail validation
	_, err := krm.ValidateKey(key.Key)
	if err == nil {
		t.Error("expired key should fail validation")
	}
}

// TestKeyRotationManagerRotateKey tests key rotation
func TestKeyRotationManagerRotateKey(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)
	originalKey := key.Key
	keyID := key.RotationID

	// Rotate key
	rotated, err := krm.RotateKey(keyID, "admin", "scheduled rotation")
	if err != nil {
		t.Fatalf("rotation failed: %v", err)
	}

	// New key should be different
	if rotated.Key == originalKey {
		t.Error("rotated key should be different from original")
	}

	// Old key should still work during grace period
	validated, err := krm.ValidateKey(originalKey)
	if err != nil {
		t.Fatalf("old key should work during grace period: %v", err)
	}
	if validated.Name != "test-key" {
		t.Error("grace period key returned wrong metadata")
	}

	// New key should also work
	validated, err = krm.ValidateKey(rotated.Key)
	if err != nil {
		t.Fatalf("new key validation failed: %v", err)
	}
}

// TestKeyRotationManagerRevokeKey tests key revocation
func TestKeyRotationManagerRevokeKey(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)

	// Revoke key
	err := krm.RevokeKey(key.RotationID, "admin", "security concern")
	if err != nil {
		t.Fatalf("revocation failed: %v", err)
	}

	// Key should no longer validate
	_, err = krm.ValidateKey(key.Key)
	if err == nil {
		t.Error("revoked key should fail validation")
	}
}

// TestKeyRotationManagerDeactivateReactivate tests key deactivation/reactivation
func TestKeyRotationManagerDeactivateReactivate(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)

	// Deactivate key
	err := krm.DeactivateKey(key.RotationID)
	if err != nil {
		t.Fatalf("deactivation failed: %v", err)
	}

	// Key should fail validation
	_, err = krm.ValidateKey(key.Key)
	if err == nil {
		t.Error("deactivated key should fail validation")
	}

	// Reactivate key
	err = krm.ReactivateKey(key.RotationID)
	if err != nil {
		t.Fatalf("reactivation failed: %v", err)
	}

	// Key should work again
	_, err = krm.ValidateKey(key.Key)
	if err != nil {
		t.Fatalf("reactivated key should pass validation: %v", err)
	}
}

// TestKeyRotationManagerListKeys tests key listing
func TestKeyRotationManagerListKeys(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	// Create multiple keys
	krm.CreateKey("key1", []string{"read"}, nil)
	krm.CreateKey("key2", []string{"write"}, nil)
	krm.CreateKey("key3", []string{"admin"}, nil)

	keys := krm.ListKeys()

	if len(keys) != 3 {
		t.Errorf("expected 3 keys, got %d", len(keys))
	}

	// Keys should be sanitized (partial key shown)
	for _, key := range keys {
		if len(key.Key) > 15 {
			t.Error("listed key should be truncated")
		}
	}
}

// TestKeyRotationManagerGetKeyByID tests getting key by ID
func TestKeyRotationManagerGetKeyByID(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)

	// Get by ID
	retrieved, err := krm.GetKeyByID(key.RotationID)
	if err != nil {
		t.Fatalf("failed to get key by ID: %v", err)
	}

	if retrieved.Name != "test-key" {
		t.Error("retrieved key has wrong name")
	}

	// Non-existent ID should fail
	_, err = krm.GetKeyByID("non-existent")
	if err == nil {
		t.Error("non-existent ID should return error")
	}
}

// TestKeyRotationManagerRotationHistory tests rotation history
func TestKeyRotationManagerRotationHistory(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)

	// Rotate a few times
	krm.RotateKey(key.RotationID, "admin", "reason1")
	krm.RotateKey(key.RotationID, "admin", "reason2")
	krm.RotateKey(key.RotationID, "admin", "reason3")

	// Get history
	history := krm.GetRotationHistory(key.RotationID, 10)

	if len(history) != 3 {
		t.Errorf("expected 3 rotation events, got %d", len(history))
	}

	// Most recent should be first
	if history[0].Reason != "reason3" {
		t.Error("history not in correct order")
	}
}

// TestKeyRotationManagerKeysNeedingRotation tests finding keys needing rotation
func TestKeyRotationManagerKeysNeedingRotation(t *testing.T) {
	policy := KeyRotationPolicy{
		MaxAge:        1 * time.Hour,
		WarningPeriod: 30 * time.Minute,
		GracePeriod:   5 * time.Minute,
		AutoRotate:    false,
	}
	krm := NewKeyRotationManager(policy)

	// Create key that's already old (simulated by setting CreatedAt in past)
	key, _ := krm.CreateKey("old-key", []string{"read"}, nil)

	// Manually set creation time to past
	krm.mu.Lock()
	if k, ok := krm.keys[key.RotationID]; ok {
		k.CreatedAt = time.Now().Add(-45 * time.Minute) // 45 min old, within warning period
	}
	krm.mu.Unlock()

	needsRotation := krm.GetKeysNeedingRotation()

	if len(needsRotation) != 1 {
		t.Errorf("expected 1 key needing rotation, got %d", len(needsRotation))
	}
}

// TestKeyRotationManagerStats tests statistics
func TestKeyRotationManagerStats(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	krm.CreateKey("key1", []string{"read"}, nil)
	krm.CreateKey("key2", []string{"write"}, nil)

	stats := krm.Stats()

	if stats["total_keys"].(int) != 2 {
		t.Errorf("expected 2 total keys, got %v", stats["total_keys"])
	}
	if stats["active_keys"].(int) != 2 {
		t.Errorf("expected 2 active keys, got %v", stats["active_keys"])
	}
}

// TestKeyRotationManagerGracePeriodExpiry tests grace period expiration
func TestKeyRotationManagerGracePeriodExpiry(t *testing.T) {
	policy := KeyRotationPolicy{
		MaxAge:        90 * 24 * time.Hour,
		WarningPeriod: 14 * 24 * time.Hour,
		GracePeriod:   10 * time.Millisecond, // Very short for testing
		AutoRotate:    false,
	}
	krm := NewKeyRotationManager(policy)

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)
	originalKey := key.Key

	// Rotate key
	_, err := krm.RotateKey(key.RotationID, "admin", "test")
	if err != nil {
		t.Fatalf("rotation failed: %v", err)
	}

	// Wait for grace period to expire
	time.Sleep(20 * time.Millisecond)

	// Old key should no longer work
	_, err = krm.ValidateKey(originalKey)
	if err == nil {
		t.Error("old key should fail after grace period")
	}
}

// TestKeyRotationManagerUsageTracking tests usage tracking
func TestKeyRotationManagerUsageTracking(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)

	// Use the key multiple times
	for i := 0; i < 5; i++ {
		krm.ValidateKey(key.Key)
	}

	// Check usage count
	krm.mu.RLock()
	k := krm.keys[key.RotationID]
	krm.mu.RUnlock()

	if k.UsageCount != 5 {
		t.Errorf("expected 5 usages, got %d", k.UsageCount)
	}
	if k.LastUsedAt == nil {
		t.Error("last used time should be set")
	}
}

// TestKeyRotationManagerCallbacks tests rotation callbacks
func TestKeyRotationManagerCallbacks(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	rotationCalled := make(chan bool, 1)
	krm.SetOnRotation(func(event KeyRotationEvent) {
		rotationCalled <- true
	})

	key, _ := krm.CreateKey("test-key", []string{"read"}, nil)
	krm.RotateKey(key.RotationID, "admin", "test")

	// Wait for callback
	select {
	case <-rotationCalled:
		// Success
	case <-time.After(100 * time.Millisecond):
		t.Error("rotation callback not called")
	}
}

// TestKeyRotationManagerSaveLoad tests persistence
func TestKeyRotationManagerSaveLoad(t *testing.T) {
	tmpFile := t.TempDir() + "/keys.json"

	krm1 := NewKeyRotationManager(DefaultRotationPolicy())
	key, _ := krm1.CreateKey("test-key", []string{"read"}, nil)

	// Save
	err := krm1.SaveToFile(tmpFile)
	if err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// Load in new manager
	krm2 := NewKeyRotationManager(DefaultRotationPolicy())
	err = krm2.LoadFromFile(tmpFile)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	// Validate key works
	validated, err := krm2.ValidateKey(key.Key)
	if err != nil {
		t.Fatalf("validation failed after load: %v", err)
	}
	if validated.Name != "test-key" {
		t.Error("loaded key has wrong name")
	}
}

// TestKeyRotationManagerConcurrency tests thread safety
func TestKeyRotationManagerConcurrency(t *testing.T) {
	krm := NewKeyRotationManager(DefaultRotationPolicy())

	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(id int) {
			key, _ := krm.CreateKey("key"+string(rune('0'+id)), []string{"read"}, nil)
			for j := 0; j < 100; j++ {
				krm.ValidateKey(key.Key)
			}
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	keys := krm.ListKeys()
	if len(keys) != 10 {
		t.Errorf("expected 10 keys, got %d", len(keys))
	}
}
