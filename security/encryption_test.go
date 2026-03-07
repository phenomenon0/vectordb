package security

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestEncryptionConfig tests default config
func TestEncryptionConfig(t *testing.T) {
	config := DefaultEncryptionConfig()

	if config.Algorithm != AlgorithmAES256GCM {
		t.Errorf("expected AES-256-GCM, got %s", config.Algorithm)
	}
	if config.KDF != KDFArgon2id {
		t.Errorf("expected Argon2id, got %s", config.KDF)
	}
	if config.KeyRotationDays != 90 {
		t.Errorf("expected 90 days, got %d", config.KeyRotationDays)
	}
	if config.Enabled {
		t.Error("expected encryption disabled by default")
	}
}

// TestEncryptionManagerInitialize tests initialization with passphrase
func TestEncryptionManagerInitialize(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm:       AlgorithmAES256GCM,
		KDF:             KDFArgon2id,
		KeyRotationDays: 90,
		Enabled:         true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase-12345")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	if !em.IsEnabled() {
		t.Error("encryption should be enabled")
	}

	// Check key files were created
	keyMetaPath := filepath.Join(tmpDir, "encryption_key.json")
	if _, err := os.Stat(keyMetaPath); os.IsNotExist(err) {
		t.Error("encryption_key.json not created")
	}

	wrappedKeyPath := filepath.Join(tmpDir, "wrapped_key.bin")
	if _, err := os.Stat(wrappedKeyPath); os.IsNotExist(err) {
		t.Error("wrapped_key.bin not created")
	}
}

// TestEncryptionManagerInitializeWithKey tests initialization with raw key
func TestEncryptionManagerInitializeWithKey(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)

	// Test with wrong key size
	err := em.InitializeWithKey(make([]byte, 16))
	if err == nil {
		t.Error("should reject 16-byte key")
	}

	// Test with correct key size
	masterKey := make([]byte, 32)
	for i := range masterKey {
		masterKey[i] = byte(i)
	}

	err = em.InitializeWithKey(masterKey)
	if err != nil {
		t.Fatalf("failed to initialize with key: %v", err)
	}
	defer em.Close()

	if !em.IsEnabled() {
		t.Error("encryption should be enabled")
	}
}

// TestEncryptDecrypt tests basic encryption and decryption
func TestEncryptDecrypt(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	testCases := []struct {
		name string
		data []byte
	}{
		{"empty", []byte{}},
		{"small", []byte("hello world")},
		{"medium", bytes.Repeat([]byte("x"), 1024)},
		{"large", bytes.Repeat([]byte("y"), 1024*1024)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ciphertext, err := em.Encrypt(tc.data)
			if err != nil {
				t.Fatalf("encrypt failed: %v", err)
			}

			// Ciphertext should be longer than plaintext (header + auth tag)
			if len(tc.data) > 0 && len(ciphertext) <= len(tc.data) {
				t.Error("ciphertext should be longer than plaintext")
			}

			plaintext, err := em.Decrypt(ciphertext)
			if err != nil {
				t.Fatalf("decrypt failed: %v", err)
			}

			if !bytes.Equal(plaintext, tc.data) {
				t.Error("decrypted data doesn't match original")
			}
		})
	}
}

// TestEncryptDecryptChaCha20 tests ChaCha20-Poly1305 algorithm
func TestEncryptDecryptChaCha20(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmChaCha20Poly1305,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	data := []byte("test data for ChaCha20")

	ciphertext, err := em.Encrypt(data)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}

	plaintext, err := em.Decrypt(ciphertext)
	if err != nil {
		t.Fatalf("decrypt failed: %v", err)
	}

	if !bytes.Equal(plaintext, data) {
		t.Error("decrypted data doesn't match original")
	}
}

// TestEncryptionDisabled tests passthrough when encryption is disabled
func TestEncryptionDisabled(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Enabled: false,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("unused-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	data := []byte("plaintext data")

	// Should return data unchanged
	result, err := em.Encrypt(data)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}

	if !bytes.Equal(result, data) {
		t.Error("disabled encryption should return data unchanged")
	}

	result, err = em.Decrypt(data)
	if err != nil {
		t.Fatalf("decrypt failed: %v", err)
	}

	if !bytes.Equal(result, data) {
		t.Error("disabled decryption should return data unchanged")
	}
}

// TestKeyRotation tests key rotation
func TestKeyRotation(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	// Encrypt some data
	data := []byte("test data before rotation")
	ciphertext1, err := em.Encrypt(data)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}

	// Get initial key version
	meta1 := em.GetKeyMetadata()
	if meta1 == nil {
		t.Fatal("no key metadata")
	}
	version1 := meta1.Version

	// Rotate key
	err = em.RotateKey()
	if err != nil {
		t.Fatalf("key rotation failed: %v", err)
	}

	// Verify version incremented
	meta2 := em.GetKeyMetadata()
	if meta2.Version != version1+1 {
		t.Errorf("expected version %d, got %d", version1+1, meta2.Version)
	}

	// Note: Old ciphertext won't decrypt with new key
	// This is expected behavior - data needs to be re-encrypted

	// New encryption should work
	ciphertext2, err := em.Encrypt(data)
	if err != nil {
		t.Fatalf("encrypt after rotation failed: %v", err)
	}

	// Ciphertexts should be different (different nonces)
	if bytes.Equal(ciphertext1, ciphertext2) {
		t.Error("ciphertexts should be different after rotation")
	}

	// New ciphertext should decrypt
	plaintext, err := em.Decrypt(ciphertext2)
	if err != nil {
		t.Fatalf("decrypt after rotation failed: %v", err)
	}

	if !bytes.Equal(plaintext, data) {
		t.Error("decrypted data doesn't match")
	}
}

// TestEncryptionPersistence tests key persistence across restarts
func TestEncryptionPersistence(t *testing.T) {
	tmpDir := t.TempDir()
	passphrase := "test-passphrase"
	data := []byte("persistent data")

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	// First instance
	em1 := NewEncryptionManager(config, tmpDir)
	err := em1.Initialize(passphrase)
	if err != nil {
		t.Fatalf("failed to initialize first instance: %v", err)
	}

	ciphertext, err := em1.Encrypt(data)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}

	keyID1 := em1.GetKeyMetadata().KeyID
	em1.Close()

	// Second instance (simulating restart)
	em2 := NewEncryptionManager(config, tmpDir)
	err = em2.Initialize(passphrase)
	if err != nil {
		t.Fatalf("failed to initialize second instance: %v", err)
	}
	defer em2.Close()

	// Key ID should be the same
	keyID2 := em2.GetKeyMetadata().KeyID
	if keyID1 != keyID2 {
		t.Error("key ID changed after restart")
	}

	// Should decrypt data from first instance
	plaintext, err := em2.Decrypt(ciphertext)
	if err != nil {
		t.Fatalf("decrypt with new instance failed: %v", err)
	}

	if !bytes.Equal(plaintext, data) {
		t.Error("decrypted data doesn't match")
	}
}

// TestEncryptionWrongPassphrase tests decryption failure with wrong passphrase
func TestEncryptionWrongPassphrase(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	// Initialize with correct passphrase
	em1 := NewEncryptionManager(config, tmpDir)
	err := em1.Initialize("correct-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	em1.Close()

	// Try to initialize with wrong passphrase
	em2 := NewEncryptionManager(config, tmpDir)
	err = em2.Initialize("wrong-passphrase")
	if err == nil {
		em2.Close()
		t.Error("should fail with wrong passphrase")
	}
}

// TestEncryptDecryptFile tests file encryption/decryption
func TestEncryptDecryptFile(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	// Create test file
	testFile := filepath.Join(tmpDir, "test.txt")
	originalData := []byte("file content to encrypt")
	err = os.WriteFile(testFile, originalData, 0644)
	if err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	// Encrypt file
	err = em.EncryptFile(testFile)
	if err != nil {
		t.Fatalf("failed to encrypt file: %v", err)
	}

	// Read encrypted file - should have VDBE header
	encryptedData, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("failed to read encrypted file: %v", err)
	}

	if string(encryptedData[:4]) != "VDBE" {
		t.Error("encrypted file should have VDBE header")
	}

	// Decrypt file
	err = em.DecryptFile(testFile)
	if err != nil {
		t.Fatalf("failed to decrypt file: %v", err)
	}

	// Read decrypted file
	decryptedData, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("failed to read decrypted file: %v", err)
	}

	if !bytes.Equal(decryptedData, originalData) {
		t.Error("decrypted file content doesn't match original")
	}
}

// TestEncryptedStorage tests the encrypted storage wrapper
func TestEncryptedStorage(t *testing.T) {
	tmpDir := t.TempDir()
	storagePath := filepath.Join(tmpDir, "storage")

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	es := NewEncryptedStorage(storagePath, em)

	// Write data
	err = es.Write("test.dat", []byte("encrypted storage data"))
	if err != nil {
		t.Fatalf("write failed: %v", err)
	}

	// Check file exists
	if !es.Exists("test.dat") {
		t.Error("file should exist")
	}

	// Read data
	data, err := es.Read("test.dat")
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if string(data) != "encrypted storage data" {
		t.Error("read data doesn't match")
	}

	// List files
	files, err := es.List()
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}

	if len(files) != 1 || files[0] != "test.dat" {
		t.Error("list returned unexpected files")
	}

	// Delete file
	err = es.Delete("test.dat")
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if es.Exists("test.dat") {
		t.Error("file should not exist after delete")
	}
}

// TestEncryptionStats tests stats collection
func TestEncryptionStats(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFArgon2id,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer em.Close()

	// Perform some operations
	data := []byte("test data")
	ciphertext, _ := em.Encrypt(data)
	em.Decrypt(ciphertext)

	stats := em.Stats()

	if stats["encrypt_ops"].(int64) != 1 {
		t.Error("expected 1 encrypt op")
	}
	if stats["decrypt_ops"].(int64) != 1 {
		t.Error("expected 1 decrypt op")
	}
	if stats["bytes_encrypted"].(int64) != int64(len(data)) {
		t.Error("bytes_encrypted mismatch")
	}
}

// TestKDFSHA256 tests SHA256 KDF (for testing only)
func TestKDFSHA256(t *testing.T) {
	tmpDir := t.TempDir()

	config := EncryptionConfig{
		Algorithm: AlgorithmAES256GCM,
		KDF:       KDFSHA256,
		Enabled:   true,
	}

	em := NewEncryptionManager(config, tmpDir)
	err := em.Initialize("test-passphrase")
	if err != nil {
		t.Fatalf("failed to initialize with SHA256 KDF: %v", err)
	}
	defer em.Close()

	data := []byte("test data")
	ciphertext, err := em.Encrypt(data)
	if err != nil {
		t.Fatalf("encrypt failed: %v", err)
	}

	plaintext, err := em.Decrypt(ciphertext)
	if err != nil {
		t.Fatalf("decrypt failed: %v", err)
	}

	if !bytes.Equal(plaintext, data) {
		t.Error("decrypted data doesn't match")
	}
}
