package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestDefaultTLSManagerConfig tests default config values
func TestDefaultTLSManagerConfig(t *testing.T) {
	config := DefaultTLSManagerConfig()

	if config.CertFile != "certs/server.crt" {
		t.Errorf("expected default cert file 'certs/server.crt', got %s", config.CertFile)
	}
	if config.KeyFile != "certs/server.key" {
		t.Errorf("expected default key file 'certs/server.key', got %s", config.KeyFile)
	}
	if !config.AutoGenerate {
		t.Error("auto generate should be enabled by default")
	}
	if config.MinVersion != "1.2" {
		t.Errorf("expected min version '1.2', got %s", config.MinVersion)
	}
	if config.RequireClientCert {
		t.Error("client cert should not be required by default")
	}
}

// TestTLSManagerAutoGenerate tests auto-generation of certificates
func TestTLSManagerAutoGenerate(t *testing.T) {
	tmpDir := t.TempDir()

	config := TLSManagerConfig{
		CertFile:     filepath.Join(tmpDir, "server.crt"),
		KeyFile:      filepath.Join(tmpDir, "server.key"),
		AutoGenerate: true,
		CommonName:   "test-server",
		DNSNames:     []string{"localhost"},
		IPAddresses:  []string{"127.0.0.1"},
		ValidDays:    30,
		MinVersion:   "1.2",
		AutoReload:   false,
	}

	tm, err := NewTLSManager(config)
	if err != nil {
		t.Fatalf("failed to create TLS manager: %v", err)
	}
	defer tm.Close()

	// Verify cert file was created
	if _, err := os.Stat(config.CertFile); os.IsNotExist(err) {
		t.Error("certificate file was not created")
	}

	// Verify key file was created
	if _, err := os.Stat(config.KeyFile); os.IsNotExist(err) {
		t.Error("key file was not created")
	}
}

// TestTLSManagerLoadExisting tests loading existing certificates
func TestTLSManagerLoadExisting(t *testing.T) {
	tmpDir := t.TempDir()

	// First, generate certs
	config1 := TLSManagerConfig{
		CertFile:     filepath.Join(tmpDir, "server.crt"),
		KeyFile:      filepath.Join(tmpDir, "server.key"),
		AutoGenerate: true,
		CommonName:   "test-server",
		DNSNames:     []string{"localhost"},
		ValidDays:    30,
		MinVersion:   "1.2",
		AutoReload:   false,
	}

	tm1, err := NewTLSManager(config1)
	if err != nil {
		t.Fatalf("failed to create first TLS manager: %v", err)
	}
	tm1.Close()

	// Now load existing certs
	config2 := TLSManagerConfig{
		CertFile:     filepath.Join(tmpDir, "server.crt"),
		KeyFile:      filepath.Join(tmpDir, "server.key"),
		AutoGenerate: false, // Don't regenerate
		MinVersion:   "1.2",
		AutoReload:   false,
	}

	tm2, err := NewTLSManager(config2)
	if err != nil {
		t.Fatalf("failed to load existing certificates: %v", err)
	}
	defer tm2.Close()
}

// TestTLSManagerMissingCertNoAutoGen tests error when certs missing and no auto-gen
func TestTLSManagerMissingCertNoAutoGen(t *testing.T) {
	tmpDir := t.TempDir()

	config := TLSManagerConfig{
		CertFile:     filepath.Join(tmpDir, "nonexistent.crt"),
		KeyFile:      filepath.Join(tmpDir, "nonexistent.key"),
		AutoGenerate: false,
		MinVersion:   "1.2",
	}

	_, err := NewTLSManager(config)
	if err == nil {
		t.Error("should fail when certs missing and auto-generate disabled")
	}
}

// TestTLSManagerGetCertificate tests certificate retrieval
func TestTLSManagerGetCertificate(t *testing.T) {
	tmpDir := t.TempDir()

	config := TLSManagerConfig{
		CertFile:     filepath.Join(tmpDir, "server.crt"),
		KeyFile:      filepath.Join(tmpDir, "server.key"),
		AutoGenerate: true,
		CommonName:   "test-server",
		DNSNames:     []string{"localhost"},
		ValidDays:    30,
		MinVersion:   "1.2",
		AutoReload:   false,
	}

	tm, err := NewTLSManager(config)
	if err != nil {
		t.Fatalf("failed to create TLS manager: %v", err)
	}
	defer tm.Close()

	cert, err := tm.GetCertificate(nil)
	if err != nil {
		t.Fatalf("failed to get certificate: %v", err)
	}
	if cert == nil {
		t.Error("certificate should not be nil")
	}
}

// TestTLSManagerTLSConfigForServer tests server TLS config generation
func TestTLSManagerTLSConfigForServer(t *testing.T) {
	tmpDir := t.TempDir()

	config := TLSManagerConfig{
		CertFile:          filepath.Join(tmpDir, "server.crt"),
		KeyFile:           filepath.Join(tmpDir, "server.key"),
		AutoGenerate:      true,
		CommonName:        "test-server",
		DNSNames:          []string{"localhost"},
		ValidDays:         30,
		MinVersion:        "1.3",
		RequireClientCert: false,
		AutoReload:        false,
	}

	tm, err := NewTLSManager(config)
	if err != nil {
		t.Fatalf("failed to create TLS manager: %v", err)
	}
	defer tm.Close()

	tlsConfig := tm.TLSConfigForServer()

	if tlsConfig == nil {
		t.Fatal("TLS config should not be nil")
	}
	if tlsConfig.GetCertificate == nil {
		t.Error("GetCertificate callback should be set")
	}
}

// TestTLSManagerCertInfo tests certificate info retrieval
func TestTLSManagerCertInfo(t *testing.T) {
	tmpDir := t.TempDir()

	config := TLSManagerConfig{
		CertFile:     filepath.Join(tmpDir, "server.crt"),
		KeyFile:      filepath.Join(tmpDir, "server.key"),
		AutoGenerate: true,
		CommonName:   "test-server",
		DNSNames:     []string{"localhost", "example.com"},
		IPAddresses:  []string{"127.0.0.1", "::1"},
		ValidDays:    30,
		MinVersion:   "1.2",
		AutoReload:   false,
	}

	tm, err := NewTLSManager(config)
	if err != nil {
		t.Fatalf("failed to create TLS manager: %v", err)
	}
	defer tm.Close()

	info := tm.CertInfo()

	// Check subject
	subject, ok := info["subject"].(string)
	if !ok {
		t.Error("subject should be string")
	}
	if subject == "" {
		t.Error("subject should not be empty")
	}

	// Check expiry
	daysUntilExpiry, ok := info["days_until_expiry"].(int)
	if !ok {
		t.Error("days_until_expiry should be int")
	}
	if daysUntilExpiry < 25 || daysUntilExpiry > 35 {
		t.Errorf("days_until_expiry should be around 30, got %d", daysUntilExpiry)
	}

	// Check DNS names
	dnsNames, ok := info["dns_names"].([]string)
	if !ok {
		t.Error("dns_names should be []string")
	}
	if len(dnsNames) != 2 {
		t.Errorf("expected 2 DNS names, got %d", len(dnsNames))
	}

	// Check self-signed flag
	isSelfSigned, ok := info["is_self_signed"].(bool)
	if !ok {
		t.Error("is_self_signed should be bool")
	}
	if !isSelfSigned {
		t.Error("auto-generated cert should be self-signed")
	}
}

// TestTLSManagerMinVersionParsing tests min TLS version parsing
func TestTLSManagerMinVersionParsing(t *testing.T) {
	testCases := []struct {
		version  string
		expected uint16
	}{
		{"1.2", 0x0303},     // TLS 1.2
		{"1.3", 0x0304},     // TLS 1.3
		{"1.1", 0x0302},     // TLS 1.1
		{"invalid", 0x0303}, // Default to 1.2
	}

	for _, tc := range testCases {
		t.Run(tc.version, func(t *testing.T) {
			tmpDir := t.TempDir()

			config := TLSManagerConfig{
				CertFile:     filepath.Join(tmpDir, "server.crt"),
				KeyFile:      filepath.Join(tmpDir, "server.key"),
				AutoGenerate: true,
				CommonName:   "test-server",
				ValidDays:    30,
				MinVersion:   tc.version,
				AutoReload:   false,
			}

			tm, err := NewTLSManager(config)
			if err != nil {
				t.Fatalf("failed to create TLS manager: %v", err)
			}
			defer tm.Close()

			tlsConfig := tm.TLSConfigForServer()
			if tlsConfig.MinVersion != tc.expected {
				t.Errorf("expected min version %x, got %x", tc.expected, tlsConfig.MinVersion)
			}
		})
	}
}

// TestTLSManagerConfigFromEnv tests environment variable configuration
func TestTLSManagerConfigFromEnv(t *testing.T) {
	// Set environment variables
	os.Setenv("TLS_CERT_FILE", "/custom/path/cert.pem")
	os.Setenv("TLS_KEY_FILE", "/custom/path/key.pem")
	os.Setenv("TLS_MIN_VERSION", "1.3")
	os.Setenv("TLS_REQUIRE_CLIENT_CERT", "true")
	defer func() {
		os.Unsetenv("TLS_CERT_FILE")
		os.Unsetenv("TLS_KEY_FILE")
		os.Unsetenv("TLS_MIN_VERSION")
		os.Unsetenv("TLS_REQUIRE_CLIENT_CERT")
	}()

	config := TLSManagerConfigFromEnv()

	if config.CertFile != "/custom/path/cert.pem" {
		t.Errorf("expected cert file '/custom/path/cert.pem', got %s", config.CertFile)
	}
	if config.KeyFile != "/custom/path/key.pem" {
		t.Errorf("expected key file '/custom/path/key.pem', got %s", config.KeyFile)
	}
	if config.MinVersion != "1.3" {
		t.Errorf("expected min version '1.3', got %s", config.MinVersion)
	}
	if !config.RequireClientCert {
		t.Error("expected RequireClientCert to be true")
	}
}

// TestTLSManagerAutoReload tests certificate auto-reload
func TestTLSManagerAutoReload(t *testing.T) {
	tmpDir := t.TempDir()

	config := TLSManagerConfig{
		CertFile:       filepath.Join(tmpDir, "server.crt"),
		KeyFile:        filepath.Join(tmpDir, "server.key"),
		AutoGenerate:   true,
		CommonName:     "test-server",
		ValidDays:      30,
		MinVersion:     "1.2",
		AutoReload:     true,
		ReloadInterval: 50 * time.Millisecond, // Short interval for testing
	}

	tm, err := NewTLSManager(config)
	if err != nil {
		t.Fatalf("failed to create TLS manager: %v", err)
	}
	defer tm.Close()

	// Get original cert
	originalCert, _ := tm.GetCertificate(nil)

	// Regenerate cert by touching the file
	// In real scenario, this would be a new cert
	time.Sleep(100 * time.Millisecond)

	// Cert should still be loadable (watcher should handle changes)
	newCert, err := tm.GetCertificate(nil)
	if err != nil {
		t.Fatalf("failed to get certificate after reload check: %v", err)
	}
	if newCert == nil {
		t.Error("certificate should not be nil after reload check")
	}

	// Original and new should be same (no actual change)
	if originalCert.Certificate[0][0] != newCert.Certificate[0][0] {
		t.Error("certificate should be same (no actual file change)")
	}
}
