package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ===========================================================================================
// ENHANCED TLS MANAGER
// Provides certificate auto-generation, mTLS support, and automatic cert reload
// Uses the basic TLSConfig from auth.go as foundation
// ===========================================================================================

// TLSManagerConfig holds configuration for the enhanced TLS manager
type TLSManagerConfig struct {
	// Certificate paths
	CertFile string `json:"cert_file"`
	KeyFile  string `json:"key_file"`
	CAFile   string `json:"ca_file"` // For client certificate verification (mTLS)

	// Auto-generation options
	AutoGenerate bool     `json:"auto_generate"`
	CommonName   string   `json:"common_name"`
	DNSNames     []string `json:"dns_names"`
	IPAddresses  []string `json:"ip_addresses"`
	ValidDays    int      `json:"valid_days"`

	// Security settings
	MinVersion        string   `json:"min_version"`         // "1.2" or "1.3"
	RequireClientCert bool     `json:"require_client_cert"` // Enable mTLS
	CipherSuites      []string `json:"cipher_suites"`

	// Certificate reload
	AutoReload     bool          `json:"auto_reload"`
	ReloadInterval time.Duration `json:"reload_interval"`
}

// DefaultTLSManagerConfig returns secure defaults
func DefaultTLSManagerConfig() TLSManagerConfig {
	return TLSManagerConfig{
		CertFile:          "certs/server.crt",
		KeyFile:           "certs/server.key",
		AutoGenerate:      true,
		CommonName:        "vectordb",
		DNSNames:          []string{"localhost", "vectordb"},
		IPAddresses:       []string{"127.0.0.1", "::1"},
		ValidDays:         365,
		MinVersion:        "1.2",
		RequireClientCert: false,
		AutoReload:        true,
		ReloadInterval:    time.Hour,
	}
}

// TLSManager manages TLS certificates and configuration
type TLSManager struct {
	mu       sync.RWMutex
	config   TLSManagerConfig
	cert     *tls.Certificate
	certPool *x509.CertPool
	stopCh   chan struct{}
	lastMod  time.Time
}

// NewTLSManager creates a new TLS manager
func NewTLSManager(config TLSManagerConfig) (*TLSManager, error) {
	tm := &TLSManager{
		config: config,
		stopCh: make(chan struct{}),
	}

	// Load or generate certificates
	if err := tm.loadOrGenerateCerts(); err != nil {
		return nil, err
	}

	// Load CA pool for client verification
	if config.RequireClientCert && config.CAFile != "" {
		if err := tm.loadCAPool(); err != nil {
			return nil, err
		}
	}

	// Start certificate watcher if enabled
	if config.AutoReload {
		go tm.watchCerts()
	}

	return tm, nil
}

// loadOrGenerateCerts loads existing certs or generates new ones
func (tm *TLSManager) loadOrGenerateCerts() error {
	certExists := fileExistsTLS(tm.config.CertFile)
	keyExists := fileExistsTLS(tm.config.KeyFile)

	if certExists && keyExists {
		return tm.loadCerts()
	}

	if !tm.config.AutoGenerate {
		return fmt.Errorf("certificate files not found and auto_generate is disabled")
	}

	return tm.generateSelfSigned()
}

// loadCerts loads certificates from files
func (tm *TLSManager) loadCerts() error {
	cert, err := tls.LoadX509KeyPair(tm.config.CertFile, tm.config.KeyFile)
	if err != nil {
		return fmt.Errorf("failed to load certificates: %w", err)
	}

	tm.mu.Lock()
	tm.cert = &cert
	tm.mu.Unlock()

	if info, err := os.Stat(tm.config.CertFile); err == nil {
		tm.lastMod = info.ModTime()
	}

	return nil
}

// generateSelfSigned generates a self-signed certificate
func (tm *TLSManager) generateSelfSigned() error {
	certDir := filepath.Dir(tm.config.CertFile)
	if err := os.MkdirAll(certDir, 0750); err != nil {
		return fmt.Errorf("failed to create cert directory: %w", err)
	}

	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return fmt.Errorf("failed to generate private key: %w", err)
	}

	serialNumber, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return fmt.Errorf("failed to generate serial number: %w", err)
	}

	notBefore := time.Now()
	notAfter := notBefore.AddDate(0, 0, tm.config.ValidDays)

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"VectorDB"},
			CommonName:   tm.config.CommonName,
		},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	template.DNSNames = tm.config.DNSNames

	for _, ipStr := range tm.config.IPAddresses {
		if ip := net.ParseIP(ipStr); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		}
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return fmt.Errorf("failed to create certificate: %w", err)
	}

	certOut, err := os.Create(tm.config.CertFile)
	if err != nil {
		return fmt.Errorf("failed to create cert file: %w", err)
	}
	defer certOut.Close()

	if err := pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		return fmt.Errorf("failed to write certificate: %w", err)
	}

	keyOut, err := os.OpenFile(tm.config.KeyFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("failed to create key file: %w", err)
	}
	defer keyOut.Close()

	privBytes, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		return fmt.Errorf("failed to marshal private key: %w", err)
	}

	if err := pem.Encode(keyOut, &pem.Block{Type: "PRIVATE KEY", Bytes: privBytes}); err != nil {
		return fmt.Errorf("failed to write private key: %w", err)
	}

	fmt.Printf("TLS: Generated self-signed certificate: %s\n", tm.config.CertFile)
	return tm.loadCerts()
}

// loadCAPool loads CA certificates for client verification
func (tm *TLSManager) loadCAPool() error {
	caCert, err := os.ReadFile(tm.config.CAFile)
	if err != nil {
		return fmt.Errorf("failed to read CA file: %w", err)
	}

	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caCert) {
		return fmt.Errorf("failed to parse CA certificate")
	}

	tm.mu.Lock()
	tm.certPool = pool
	tm.mu.Unlock()

	return nil
}

// watchCerts watches for certificate changes and reloads
func (tm *TLSManager) watchCerts() {
	ticker := time.NewTicker(tm.config.ReloadInterval)
	defer ticker.Stop()

	for {
		select {
		case <-tm.stopCh:
			return
		case <-ticker.C:
			tm.checkAndReload()
		}
	}
}

// checkAndReload checks if certificates have changed and reloads them
func (tm *TLSManager) checkAndReload() {
	info, err := os.Stat(tm.config.CertFile)
	if err != nil {
		return
	}

	if info.ModTime().After(tm.lastMod) {
		if err := tm.loadCerts(); err != nil {
			fmt.Printf("TLS: failed to reload certificates: %v\n", err)
		} else {
			fmt.Println("TLS: certificates reloaded")
		}
	}
}

// GetCertificate returns the current certificate (for tls.Config.GetCertificate)
func (tm *TLSManager) GetCertificate(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if tm.cert == nil {
		return nil, fmt.Errorf("no certificate available")
	}

	return tm.cert, nil
}

// TLSConfigForServer returns a tls.Config for servers
func (tm *TLSManager) TLSConfigForServer() *tls.Config {
	tlsConfig := &tls.Config{
		GetCertificate: tm.GetCertificate,
		MinVersion:     tm.parseMinVersion(),
	}

	if tm.config.RequireClientCert {
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
		tm.mu.RLock()
		tlsConfig.ClientCAs = tm.certPool
		tm.mu.RUnlock()
	}

	if len(tm.config.CipherSuites) > 0 {
		tlsConfig.CipherSuites = tm.parseCipherSuites()
	}

	return tlsConfig
}

func (tm *TLSManager) parseMinVersion() uint16 {
	switch tm.config.MinVersion {
	case "1.3":
		return tls.VersionTLS13
	case "1.2":
		return tls.VersionTLS12
	case "1.1":
		return tls.VersionTLS11
	default:
		return tls.VersionTLS12
	}
}

func (tm *TLSManager) parseCipherSuites() []uint16 {
	var suites []uint16
	for _, name := range tm.config.CipherSuites {
		for _, suite := range tls.CipherSuites() {
			if strings.EqualFold(suite.Name, name) {
				suites = append(suites, suite.ID)
				break
			}
		}
	}
	return suites
}

// Close stops the certificate watcher
func (tm *TLSManager) Close() {
	close(tm.stopCh)
}

// CertInfo returns information about the current certificate
func (tm *TLSManager) CertInfo() map[string]interface{} {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if tm.cert == nil {
		return map[string]interface{}{"status": "no certificate loaded"}
	}

	leaf, err := x509.ParseCertificate(tm.cert.Certificate[0])
	if err != nil {
		return map[string]interface{}{"status": "error parsing certificate", "error": err.Error()}
	}

	daysUntilExpiry := time.Until(leaf.NotAfter).Hours() / 24

	return map[string]interface{}{
		"subject":           leaf.Subject.String(),
		"issuer":            leaf.Issuer.String(),
		"serial":            leaf.SerialNumber.String(),
		"not_before":        leaf.NotBefore,
		"not_after":         leaf.NotAfter,
		"days_until_expiry": int(daysUntilExpiry),
		"dns_names":         leaf.DNSNames,
		"ip_addresses":      formatIPsTLS(leaf.IPAddresses),
		"is_self_signed":    leaf.Issuer.String() == leaf.Subject.String(),
	}
}

// ===========================================================================================
// TLS SERVER WRAPPER
// ===========================================================================================

// SecureTLSServer wraps an HTTP server with TLS
type SecureTLSServer struct {
	server     *http.Server
	tlsManager *TLSManager
	listener   net.Listener
}

// NewSecureTLSServer creates a new TLS-enabled HTTP server
func NewSecureTLSServer(addr string, handler http.Handler, config TLSManagerConfig) (*SecureTLSServer, error) {
	tm, err := NewTLSManager(config)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize TLS: %w", err)
	}

	server := &http.Server{
		Addr:      addr,
		Handler:   handler,
		TLSConfig: tm.TLSConfigForServer(),
	}

	return &SecureTLSServer{
		server:     server,
		tlsManager: tm,
	}, nil
}

// ListenAndServeTLS starts the TLS server
func (s *SecureTLSServer) ListenAndServeTLS() error {
	ln, err := net.Listen("tcp", s.server.Addr)
	if err != nil {
		return err
	}

	s.listener = tls.NewListener(ln, s.server.TLSConfig)
	fmt.Printf("TLS: Server listening on %s (HTTPS)\n", s.server.Addr)

	return s.server.Serve(s.listener)
}

// Shutdown gracefully shuts down the server
func (s *SecureTLSServer) Shutdown(ctx context.Context) error {
	s.tlsManager.Close()
	return s.server.Shutdown(ctx)
}

// CertInfo returns certificate information
func (s *SecureTLSServer) CertInfo() map[string]interface{} {
	return s.tlsManager.CertInfo()
}

// ===========================================================================================
// HELPER FUNCTIONS
// ===========================================================================================

func fileExistsTLS(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func formatIPsTLS(ips []net.IP) []string {
	result := make([]string, len(ips))
	for i, ip := range ips {
		result[i] = ip.String()
	}
	return result
}

// ===========================================================================================
// TLS HTTP HANDLERS
// ===========================================================================================

// TLSHTTPHandlers provides HTTP handlers for TLS management
type TLSHTTPHandlers struct {
	manager *TLSManager
}

// NewTLSHTTPHandlers creates TLS HTTP handlers
func NewTLSHTTPHandlers(manager *TLSManager) *TLSHTTPHandlers {
	return &TLSHTTPHandlers{manager: manager}
}

// HandleCertInfo returns certificate information (GET /admin/tls/info)
func (h *TLSHTTPHandlers) HandleCertInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	info := h.manager.CertInfo()
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"cert_info": %+v}`, info)
}

// HandleReloadCert forces certificate reload (POST /admin/tls/reload)
func (h *TLSHTTPHandlers) HandleReloadCert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := h.manager.loadCerts(); err != nil {
		http.Error(w, fmt.Sprintf("failed to reload: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status": "reloaded"}`)
}

// ===========================================================================================
// ENVIRONMENT CONFIGURATION
// ===========================================================================================

// TLSManagerConfigFromEnv creates TLS config from environment variables
func TLSManagerConfigFromEnv() TLSManagerConfig {
	config := DefaultTLSManagerConfig()

	if v := os.Getenv("TLS_CERT_FILE"); v != "" {
		config.CertFile = v
	}
	if v := os.Getenv("TLS_KEY_FILE"); v != "" {
		config.KeyFile = v
	}
	if v := os.Getenv("TLS_CA_FILE"); v != "" {
		config.CAFile = v
	}
	if v := os.Getenv("TLS_AUTO_GENERATE"); v == "false" {
		config.AutoGenerate = false
	}
	if v := os.Getenv("TLS_MIN_VERSION"); v != "" {
		config.MinVersion = v
	}
	if v := os.Getenv("TLS_REQUIRE_CLIENT_CERT"); v == "true" {
		config.RequireClientCert = true
	}
	if v := os.Getenv("TLS_COMMON_NAME"); v != "" {
		config.CommonName = v
	}
	if v := os.Getenv("TLS_DNS_NAMES"); v != "" {
		config.DNSNames = strings.Split(v, ",")
	}

	return config
}

// ===========================================================================================
// MTLS CLIENT CONFIGURATION
// For internal service-to-service communication
// ===========================================================================================

// MTLSClientTLSConfig creates a TLS config for mTLS clients
func MTLSClientTLSConfig(certFile, keyFile, caFile string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load client certificate: %w", err)
	}

	caCert, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read CA certificate: %w", err)
	}

	caCertPool := x509.NewCertPool()
	if !caCertPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA certificate")
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      caCertPool,
		MinVersion:   tls.VersionTLS12,
	}, nil
}

// CreateMTLSClient creates an HTTP client with mTLS
func CreateMTLSClient(certFile, keyFile, caFile string) (*http.Client, error) {
	tlsConfig, err := MTLSClientTLSConfig(certFile, keyFile, caFile)
	if err != nil {
		return nil, err
	}

	return &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: tlsConfig,
		},
	}, nil
}
