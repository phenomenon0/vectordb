package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/chacha20poly1305"
)

// ===========================================================================================
// ENCRYPTION AT REST
// AES-256-GCM and ChaCha20-Poly1305 encryption for data files
// ===========================================================================================

// EncryptionAlgorithm defines supported encryption algorithms
type EncryptionAlgorithm string

const (
	// AlgorithmAES256GCM uses AES-256 in GCM mode (NIST standard, hardware accelerated)
	AlgorithmAES256GCM EncryptionAlgorithm = "aes-256-gcm"
	// AlgorithmChaCha20Poly1305 uses ChaCha20-Poly1305 (fast on systems without AES-NI)
	AlgorithmChaCha20Poly1305 EncryptionAlgorithm = "chacha20-poly1305"
)

// KeyDerivationFunction defines how encryption keys are derived
type KeyDerivationFunction string

const (
	// KDFArgon2id uses Argon2id for key derivation (recommended)
	KDFArgon2id KeyDerivationFunction = "argon2id"
	// KDFSHA256 uses simple SHA256 (for testing only, not recommended)
	KDFSHA256 KeyDerivationFunction = "sha256"
)

// EncryptionConfig holds encryption configuration
type EncryptionConfig struct {
	// Algorithm is the encryption algorithm to use
	Algorithm EncryptionAlgorithm `json:"algorithm"`
	// KDF is the key derivation function
	KDF KeyDerivationFunction `json:"kdf"`
	// KeyRotationDays is how often to rotate the data encryption key
	KeyRotationDays int `json:"key_rotation_days"`
	// Enabled enables/disables encryption
	Enabled bool `json:"enabled"`
}

// DefaultEncryptionConfig returns sensible defaults
func DefaultEncryptionConfig() EncryptionConfig {
	return EncryptionConfig{
		Algorithm:       AlgorithmAES256GCM,
		KDF:             KDFArgon2id,
		KeyRotationDays: 90,
		Enabled:         false, // Must be explicitly enabled
	}
}

// EncryptionKeyMetadata stores metadata about encryption keys
type EncryptionKeyMetadata struct {
	KeyID     string    `json:"key_id"`
	Algorithm string    `json:"algorithm"`
	CreatedAt time.Time `json:"created_at"`
	RotatedAt time.Time `json:"rotated_at,omitempty"`
	Version   int       `json:"version"`
	Salt      []byte    `json:"salt"` // For key derivation
	IsActive  bool      `json:"is_active"`
}

// EncryptedFileHeader is written at the start of encrypted files
type EncryptedFileHeader struct {
	Magic     [4]byte  // "VDBE" (VectorDB Encrypted)
	Version   uint16   // Format version
	Algorithm uint16   // Encryption algorithm ID
	KeyID     [32]byte // Key identifier
	Nonce     [12]byte // GCM nonce (or 24 bytes for XChaCha20)
	Reserved  [8]byte  // Reserved for future use
}

const (
	headerMagic     = "VDBE"
	headerVersion   = 1
	algorithmAES    = 1
	algorithmChaCha = 2
)

// EncryptionManager handles data encryption/decryption
type EncryptionManager struct {
	mu sync.RWMutex

	config    EncryptionConfig
	masterKey []byte // Master key (derived from passphrase or loaded from HSM)
	dataKey   []byte // Data encryption key (wrapped by master key)
	keyMeta   *EncryptionKeyMetadata
	cipher    cipher.AEAD

	// Key storage path
	keyStorePath string

	// Metrics
	bytesEncrypted int64
	bytesDecrypted int64
	encryptOps     int64
	decryptOps     int64
}

// NewEncryptionManager creates a new encryption manager
func NewEncryptionManager(config EncryptionConfig, keyStorePath string) *EncryptionManager {
	return &EncryptionManager{
		config:       config,
		keyStorePath: keyStorePath,
	}
}

// Initialize sets up encryption with a passphrase
func (em *EncryptionManager) Initialize(passphrase string) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	if !em.config.Enabled {
		return nil // Encryption disabled
	}

	// Check for existing key metadata
	keyMetaPath := filepath.Join(em.keyStorePath, "encryption_key.json")
	if data, err := os.ReadFile(keyMetaPath); err == nil {
		// Load existing key
		var meta EncryptionKeyMetadata
		if err := json.Unmarshal(data, &meta); err != nil {
			return fmt.Errorf("failed to parse key metadata: %w", err)
		}
		em.keyMeta = &meta

		// Derive master key from passphrase using stored salt
		em.masterKey = em.deriveKey(passphrase, meta.Salt)

		// Load and unwrap data key
		wrappedKeyPath := filepath.Join(em.keyStorePath, "wrapped_key.bin")
		wrappedKey, err := os.ReadFile(wrappedKeyPath)
		if err != nil {
			return fmt.Errorf("failed to read wrapped key: %w", err)
		}

		em.dataKey, err = em.unwrapKey(wrappedKey)
		if err != nil {
			return fmt.Errorf("failed to unwrap data key (wrong passphrase?): %w", err)
		}
	} else {
		// Generate new keys
		salt := make([]byte, 32)
		if _, err := rand.Read(salt); err != nil {
			return fmt.Errorf("failed to generate salt: %w", err)
		}

		em.masterKey = em.deriveKey(passphrase, salt)

		// Generate data encryption key
		em.dataKey = make([]byte, 32)
		if _, err := rand.Read(em.dataKey); err != nil {
			return fmt.Errorf("failed to generate data key: %w", err)
		}

		// Create key metadata
		keyID := make([]byte, 16)
		rand.Read(keyID)

		em.keyMeta = &EncryptionKeyMetadata{
			KeyID:     base64.URLEncoding.EncodeToString(keyID),
			Algorithm: string(em.config.Algorithm),
			CreatedAt: time.Now(),
			Version:   1,
			Salt:      salt,
			IsActive:  true,
		}

		// Save key metadata
		if err := em.saveKeyMetadata(); err != nil {
			return fmt.Errorf("failed to save key metadata: %w", err)
		}

		// Wrap and save data key
		if err := em.saveWrappedKey(); err != nil {
			return fmt.Errorf("failed to save wrapped key: %w", err)
		}
	}

	// Initialize cipher
	return em.initCipher()
}

// InitializeWithKey sets up encryption with a raw key (for HSM/KMS integration)
func (em *EncryptionManager) InitializeWithKey(masterKey []byte) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	if len(masterKey) != 32 {
		return fmt.Errorf("master key must be 32 bytes")
	}

	em.masterKey = make([]byte, 32)
	copy(em.masterKey, masterKey)

	// Similar flow as Initialize, but without key derivation
	keyMetaPath := filepath.Join(em.keyStorePath, "encryption_key.json")
	if data, err := os.ReadFile(keyMetaPath); err == nil {
		var meta EncryptionKeyMetadata
		if err := json.Unmarshal(data, &meta); err != nil {
			return fmt.Errorf("failed to parse key metadata: %w", err)
		}
		em.keyMeta = &meta

		wrappedKeyPath := filepath.Join(em.keyStorePath, "wrapped_key.bin")
		wrappedKey, err := os.ReadFile(wrappedKeyPath)
		if err != nil {
			return fmt.Errorf("failed to read wrapped key: %w", err)
		}

		em.dataKey, err = em.unwrapKey(wrappedKey)
		if err != nil {
			return fmt.Errorf("failed to unwrap data key: %w", err)
		}
	} else {
		// Generate new data key
		em.dataKey = make([]byte, 32)
		if _, err := rand.Read(em.dataKey); err != nil {
			return fmt.Errorf("failed to generate data key: %w", err)
		}

		keyID := make([]byte, 16)
		rand.Read(keyID)

		em.keyMeta = &EncryptionKeyMetadata{
			KeyID:     base64.URLEncoding.EncodeToString(keyID),
			Algorithm: string(em.config.Algorithm),
			CreatedAt: time.Now(),
			Version:   1,
			Salt:      nil, // No salt needed for raw key
			IsActive:  true,
		}

		if err := em.saveKeyMetadata(); err != nil {
			return err
		}
		if err := em.saveWrappedKey(); err != nil {
			return err
		}
	}

	return em.initCipher()
}

// RotateKey generates a new data encryption key
func (em *EncryptionManager) RotateKey() error {
	em.mu.Lock()
	defer em.mu.Unlock()

	if em.masterKey == nil {
		return fmt.Errorf("encryption not initialized")
	}

	// Generate new data key
	newDataKey := make([]byte, 32)
	if _, err := rand.Read(newDataKey); err != nil {
		return fmt.Errorf("failed to generate new data key: %w", err)
	}

	// Update metadata
	em.keyMeta.Version++
	em.keyMeta.RotatedAt = time.Now()

	// Save old key for re-encryption
	oldDataKey := em.dataKey
	oldKeyMeta := *em.keyMeta // Copy old metadata for rollback
	em.dataKey = newDataKey

	// Backup old wrapped key before overwriting
	wrappedKeyPath := filepath.Join(em.keyStorePath, "wrapped_key.bin")
	oldWrappedKey, backupErr := os.ReadFile(wrappedKeyPath)
	hasBackup := backupErr == nil

	// Save new wrapped key
	if err := em.saveWrappedKey(); err != nil {
		em.dataKey = oldDataKey // Rollback
		em.keyMeta = &oldKeyMeta
		return fmt.Errorf("failed to save new wrapped key: %w", err)
	}

	// Update metadata
	if err := em.saveKeyMetadata(); err != nil {
		// Rollback: restore old wrapped key to maintain consistency
		if hasBackup {
			if restoreErr := os.WriteFile(wrappedKeyPath, oldWrappedKey, 0600); restoreErr != nil {
				// Critical: both operations failed, log and return combined error
				return fmt.Errorf("failed to save key metadata (%v) AND failed to restore old key (%v) - manual recovery required", err, restoreErr)
			}
		}
		em.dataKey = oldDataKey
		em.keyMeta = &oldKeyMeta
		return fmt.Errorf("failed to save key metadata (rolled back): %w", err)
	}

	// Reinitialize cipher with new key
	return em.initCipher()
}

// Encrypt encrypts data and returns ciphertext with header
func (em *EncryptionManager) Encrypt(plaintext []byte) ([]byte, error) {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if !em.config.Enabled || em.cipher == nil {
		return plaintext, nil // Passthrough if disabled
	}

	// Generate nonce
	nonce := make([]byte, em.cipher.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Encrypt
	ciphertext := em.cipher.Seal(nil, nonce, plaintext, nil)

	// Build output with header
	header := em.buildHeader(nonce)
	result := make([]byte, len(header)+len(ciphertext))
	copy(result, header)
	copy(result[len(header):], ciphertext)

	em.bytesEncrypted += int64(len(plaintext))
	em.encryptOps++

	return result, nil
}

// Decrypt decrypts data, handling the header
func (em *EncryptionManager) Decrypt(ciphertext []byte) ([]byte, error) {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if !em.config.Enabled || em.cipher == nil {
		return ciphertext, nil // Passthrough if disabled
	}

	// Check for header
	if len(ciphertext) < 64 { // Minimum header size
		return nil, fmt.Errorf("ciphertext too short")
	}

	// Parse header
	header, err := em.parseHeader(ciphertext[:64])
	if err != nil {
		// Might be unencrypted data
		return ciphertext, nil
	}

	// Verify key ID matches
	keyIDBytes, _ := base64.URLEncoding.DecodeString(em.keyMeta.KeyID)
	if len(keyIDBytes) >= 32 {
		for i := 0; i < 32 && i < len(keyIDBytes); i++ {
			if header.KeyID[i] != keyIDBytes[i] {
				return nil, fmt.Errorf("key ID mismatch - data encrypted with different key")
			}
		}
	}

	// Decrypt
	plaintext, err := em.cipher.Open(nil, header.Nonce[:em.cipher.NonceSize()], ciphertext[64:], nil)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: %w", err)
	}

	em.bytesDecrypted += int64(len(plaintext))
	em.decryptOps++

	return plaintext, nil
}

// EncryptFile encrypts a file in place
func (em *EncryptionManager) EncryptFile(path string) error {
	if !em.config.Enabled {
		return nil
	}

	// Read file
	plaintext, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Check if already encrypted
	if len(plaintext) >= 4 && string(plaintext[:4]) == headerMagic {
		return nil // Already encrypted
	}

	// Encrypt
	ciphertext, err := em.Encrypt(plaintext)
	if err != nil {
		return fmt.Errorf("encryption failed: %w", err)
	}

	// Write atomically
	tempPath := path + ".enc.tmp"
	if err := os.WriteFile(tempPath, ciphertext, 0600); err != nil {
		return fmt.Errorf("failed to write temp file: %w", err)
	}

	if err := os.Rename(tempPath, path); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to rename file: %w", err)
	}

	return nil
}

// DecryptFile decrypts a file in place
func (em *EncryptionManager) DecryptFile(path string) error {
	if !em.config.Enabled {
		return nil
	}

	// Read file
	ciphertext, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Check if encrypted
	if len(ciphertext) < 4 || string(ciphertext[:4]) != headerMagic {
		return nil // Not encrypted
	}

	// Decrypt
	plaintext, err := em.Decrypt(ciphertext)
	if err != nil {
		return fmt.Errorf("decryption failed: %w", err)
	}

	// Write atomically
	tempPath := path + ".dec.tmp"
	if err := os.WriteFile(tempPath, plaintext, 0600); err != nil {
		return fmt.Errorf("failed to write temp file: %w", err)
	}

	if err := os.Rename(tempPath, path); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to rename file: %w", err)
	}

	return nil
}

// EncryptReader returns an io.Reader that encrypts data on the fly
func (em *EncryptionManager) EncryptReader(r io.Reader) (io.Reader, error) {
	if !em.config.Enabled {
		return r, nil
	}

	// For streaming, we use a pipe
	pr, pw := io.Pipe()

	go func() {
		defer pw.Close()

		// Read all data (for GCM we need full plaintext)
		plaintext, err := io.ReadAll(r)
		if err != nil {
			pw.CloseWithError(err)
			return
		}

		ciphertext, err := em.Encrypt(plaintext)
		if err != nil {
			pw.CloseWithError(err)
			return
		}

		pw.Write(ciphertext)
	}()

	return pr, nil
}

// DecryptReader returns an io.Reader that decrypts data on the fly
func (em *EncryptionManager) DecryptReader(r io.Reader) (io.Reader, error) {
	if !em.config.Enabled {
		return r, nil
	}

	pr, pw := io.Pipe()

	go func() {
		defer pw.Close()

		ciphertext, err := io.ReadAll(r)
		if err != nil {
			pw.CloseWithError(err)
			return
		}

		plaintext, err := em.Decrypt(ciphertext)
		if err != nil {
			pw.CloseWithError(err)
			return
		}

		pw.Write(plaintext)
	}()

	return pr, nil
}

// IsEnabled returns whether encryption is enabled
func (em *EncryptionManager) IsEnabled() bool {
	em.mu.RLock()
	defer em.mu.RUnlock()
	return em.config.Enabled && em.cipher != nil
}

// GetKeyMetadata returns current key metadata (without sensitive data)
func (em *EncryptionManager) GetKeyMetadata() *EncryptionKeyMetadata {
	em.mu.RLock()
	defer em.mu.RUnlock()

	if em.keyMeta == nil {
		return nil
	}

	// Return copy without salt
	return &EncryptionKeyMetadata{
		KeyID:     em.keyMeta.KeyID,
		Algorithm: em.keyMeta.Algorithm,
		CreatedAt: em.keyMeta.CreatedAt,
		RotatedAt: em.keyMeta.RotatedAt,
		Version:   em.keyMeta.Version,
		IsActive:  em.keyMeta.IsActive,
	}
}

// Stats returns encryption statistics
func (em *EncryptionManager) Stats() map[string]interface{} {
	em.mu.RLock()
	defer em.mu.RUnlock()

	return map[string]interface{}{
		"enabled":         em.config.Enabled,
		"algorithm":       em.config.Algorithm,
		"bytes_encrypted": em.bytesEncrypted,
		"bytes_decrypted": em.bytesDecrypted,
		"encrypt_ops":     em.encryptOps,
		"decrypt_ops":     em.decryptOps,
		"key_version":     em.keyMeta.Version,
		"key_created":     em.keyMeta.CreatedAt,
		"key_rotated":     em.keyMeta.RotatedAt,
	}
}

// Close securely clears keys from memory
func (em *EncryptionManager) Close() error {
	em.mu.Lock()
	defer em.mu.Unlock()

	// Securely clear keys
	if em.masterKey != nil {
		for i := range em.masterKey {
			em.masterKey[i] = 0
		}
		em.masterKey = nil
	}

	if em.dataKey != nil {
		for i := range em.dataKey {
			em.dataKey[i] = 0
		}
		em.dataKey = nil
	}

	em.cipher = nil
	return nil
}

// Internal helpers

func (em *EncryptionManager) deriveKey(passphrase string, salt []byte) []byte {
	switch em.config.KDF {
	case KDFArgon2id:
		// Argon2id with recommended parameters
		// Time: 1, Memory: 64MB, Threads: 4
		return argon2.IDKey([]byte(passphrase), salt, 1, 64*1024, 4, 32)
	case KDFSHA256:
		// Simple SHA256 (not recommended for production)
		h := sha256.Sum256(append([]byte(passphrase), salt...))
		return h[:]
	default:
		// Default to Argon2id
		return argon2.IDKey([]byte(passphrase), salt, 1, 64*1024, 4, 32)
	}
}

func (em *EncryptionManager) initCipher() error {
	switch em.config.Algorithm {
	case AlgorithmAES256GCM:
		block, err := aes.NewCipher(em.dataKey)
		if err != nil {
			return fmt.Errorf("failed to create AES cipher: %w", err)
		}
		em.cipher, err = cipher.NewGCM(block)
		if err != nil {
			return fmt.Errorf("failed to create GCM: %w", err)
		}

	case AlgorithmChaCha20Poly1305:
		var err error
		em.cipher, err = chacha20poly1305.New(em.dataKey)
		if err != nil {
			return fmt.Errorf("failed to create ChaCha20-Poly1305 cipher: %w", err)
		}

	default:
		return fmt.Errorf("unsupported algorithm: %s", em.config.Algorithm)
	}

	return nil
}

func (em *EncryptionManager) wrapKey(plainKey []byte) ([]byte, error) {
	// Use master key to encrypt data key
	block, err := aes.NewCipher(em.masterKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}

	// Prepend nonce to ciphertext
	ciphertext := gcm.Seal(nonce, nonce, plainKey, nil)
	return ciphertext, nil
}

func (em *EncryptionManager) unwrapKey(wrappedKey []byte) ([]byte, error) {
	block, err := aes.NewCipher(em.masterKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	nonceSize := gcm.NonceSize()
	if len(wrappedKey) < nonceSize {
		return nil, fmt.Errorf("wrapped key too short")
	}

	nonce, ciphertext := wrappedKey[:nonceSize], wrappedKey[nonceSize:]
	return gcm.Open(nil, nonce, ciphertext, nil)
}

func (em *EncryptionManager) saveKeyMetadata() error {
	if err := os.MkdirAll(em.keyStorePath, 0700); err != nil {
		return err
	}

	data, err := json.MarshalIndent(em.keyMeta, "", "  ")
	if err != nil {
		return err
	}

	path := filepath.Join(em.keyStorePath, "encryption_key.json")
	return os.WriteFile(path, data, 0600)
}

func (em *EncryptionManager) saveWrappedKey() error {
	wrappedKey, err := em.wrapKey(em.dataKey)
	if err != nil {
		return err
	}

	path := filepath.Join(em.keyStorePath, "wrapped_key.bin")
	return os.WriteFile(path, wrappedKey, 0600)
}

func (em *EncryptionManager) buildHeader(nonce []byte) []byte {
	header := make([]byte, 64)

	// Magic bytes
	copy(header[0:4], headerMagic)

	// Version
	binary.LittleEndian.PutUint16(header[4:6], headerVersion)

	// Algorithm
	var alg uint16
	switch em.config.Algorithm {
	case AlgorithmAES256GCM:
		alg = algorithmAES
	case AlgorithmChaCha20Poly1305:
		alg = algorithmChaCha
	}
	binary.LittleEndian.PutUint16(header[6:8], alg)

	// Key ID
	keyIDBytes, _ := base64.URLEncoding.DecodeString(em.keyMeta.KeyID)
	copy(header[8:40], keyIDBytes)

	// Nonce
	copy(header[40:52], nonce)

	// Reserved (52-64)

	return header
}

func (em *EncryptionManager) parseHeader(data []byte) (*EncryptedFileHeader, error) {
	if len(data) < 64 {
		return nil, fmt.Errorf("header too short")
	}

	if string(data[0:4]) != headerMagic {
		return nil, fmt.Errorf("invalid magic bytes")
	}

	header := &EncryptedFileHeader{}
	copy(header.Magic[:], data[0:4])
	header.Version = binary.LittleEndian.Uint16(data[4:6])
	header.Algorithm = binary.LittleEndian.Uint16(data[6:8])
	copy(header.KeyID[:], data[8:40])
	copy(header.Nonce[:], data[40:52])
	copy(header.Reserved[:], data[52:60])

	return header, nil
}

// ===========================================================================================
// ENCRYPTED STORAGE WRAPPER
// Wraps storage operations with transparent encryption
// ===========================================================================================

// EncryptedStorage wraps a storage path with encryption
type EncryptedStorage struct {
	basePath   string
	encryption *EncryptionManager
}

// NewEncryptedStorage creates an encrypted storage wrapper
func NewEncryptedStorage(basePath string, em *EncryptionManager) *EncryptedStorage {
	return &EncryptedStorage{
		basePath:   basePath,
		encryption: em,
	}
}

// Write writes encrypted data to a file
func (es *EncryptedStorage) Write(filename string, data []byte) error {
	path := filepath.Join(es.basePath, filename)

	encrypted, err := es.encryption.Encrypt(data)
	if err != nil {
		return err
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	// Write atomically
	tempPath := path + ".tmp"
	if err := os.WriteFile(tempPath, encrypted, 0600); err != nil {
		return err
	}

	return os.Rename(tempPath, path)
}

// Read reads and decrypts data from a file
func (es *EncryptedStorage) Read(filename string) ([]byte, error) {
	path := filepath.Join(es.basePath, filename)

	encrypted, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	return es.encryption.Decrypt(encrypted)
}

// Exists checks if a file exists
func (es *EncryptedStorage) Exists(filename string) bool {
	path := filepath.Join(es.basePath, filename)
	_, err := os.Stat(path)
	return err == nil
}

// Delete removes a file
func (es *EncryptedStorage) Delete(filename string) error {
	path := filepath.Join(es.basePath, filename)
	return os.Remove(path)
}

// List returns all files in the storage
func (es *EncryptedStorage) List() ([]string, error) {
	entries, err := os.ReadDir(es.basePath)
	if err != nil {
		return nil, err
	}

	var files []string
	for _, entry := range entries {
		if !entry.IsDir() {
			files = append(files, entry.Name())
		}
	}
	return files, nil
}

// ReEncryptAll re-encrypts all files with the current key (after key rotation)
func (es *EncryptedStorage) ReEncryptAll() error {
	files, err := es.List()
	if err != nil {
		return err
	}

	for _, filename := range files {
		// Read with old key (already loaded if valid)
		data, err := es.Read(filename)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", filename, err)
		}

		// Write with new key
		if err := es.Write(filename, data); err != nil {
			return fmt.Errorf("failed to write %s: %w", filename, err)
		}
	}

	return nil
}
