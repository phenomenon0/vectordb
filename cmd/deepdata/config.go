package main

import (
	"errors"
	"flag"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// authConfig is the credential and access-control configuration the V3
// surface authenticates requests against.
type authConfig struct {
	APIToken    string // API_TOKEN
	JWTSecret   string // JWT_SECRET
	JWTIssuer   string // JWT_ISSUER (default "vectordb")
	RequireAuth bool   // REQUIRE_AUTH == "1"
	TrustProxy  bool   // TRUST_PROXY == "1"
}

// limitConfig is the rate-limit and tenancy ceiling configuration.
type limitConfig struct {
	APIRPS           int // API_RPS
	MaxRateLimitKeys int // MAX_RATE_LIMIT_KEYS
	AuthFailureRPS   int // AUTH_FAILURE_RPS
	AuthFailureBurst int // AUTH_FAILURE_BURST
	TenantRPS        int // TENANT_RPS
	TenantBurst      int // TENANT_BURST
	MaxTenants       int // MAX_TENANTS
	MaxCollections   int // MAX_COLLECTIONS

	// MaxTenantDocuments, MaxTenantBytes and MaxTenantCollections are the
	// server-wide per-tenant quota defaults (0 = unlimited); see
	// vcollection.StoreLimits.
	MaxTenantDocuments   int64 // MAX_TENANT_DOCUMENTS
	MaxTenantBytes       int64 // MAX_TENANT_BYTES
	MaxTenantCollections int64 // MAX_TENANT_COLLECTIONS
}

// embedderConfig selects and configures the one text embedder a process runs.
type embedderConfig struct {
	Kind          string // DEEPDATA_EMBEDDER, lowercased and trimmed
	Dim           int    // DEEPDATA_EMBED_DIM
	OllamaURL     string // OLLAMA_URL (or --embedder-url)
	OllamaModel   string // OLLAMA_EMBED_MODEL (or --embedder-model)
	OpenAIAPIKey  string // OPENAI_API_KEY
	OnnxModel     string // ONNX_EMBED_MODEL
	OnnxTokenizer string // ONNX_EMBED_TOKENIZER
	OnnxMaxLen    int    // ONNX_EMBED_MAX_LEN
}

// serverConfig is everything the process reads from its flags and
// environment, read exactly once in loadServerConfig.
type serverConfig struct {
	HTTPPort, GRPCPort                        int
	BindHost                                  string
	InsecureDevMode                           bool
	H2C                                       bool
	ReadTimeout, WriteTimeout, RequestTimeout time.Duration
	LogFormat, LogLevel                       string
	DataDir, IndexPath                        string
	HNSWEfSearch                              int
	Auth                                      authConfig
	Limits                                    limitConfig
	CORSAllowedOrigins                        string
	Embedder                                  embedderConfig
	ReplicationToken                          string

	// mode is VECTORDB_MODE / --mode, lowercased and trimmed. "" and "local"
	// are the only values validateServe accepts.
	mode string
}

// loadServerConfig parses flags (flag beats env beats default) and reads the
// environment through getenv. It returns every configuration error it finds,
// in the same wording the historical environment validator used.
func loadServerConfig(args []string, getenv func(string) string) (*serverConfig, []string) {
	fs := flag.NewFlagSet("vectordb", flag.ExitOnError)
	flagPort := fs.String("port", "", "HTTP port (env: PORT)")
	flagMode := fs.String("mode", "", "Engine mode: local (env: VECTORDB_MODE)")
	flagDataDir := fs.String("data-dir", "", "Data directory (env: VECTORDB_DATA_DIR)")
	flagEmbModel := fs.String("embedder-model", "", "Embedder model name (env: OLLAMA_EMBED_MODEL)")
	flagEmbURL := fs.String("embedder-url", "", "Embedder URL (env: OLLAMA_URL)")
	fs.Parse(args)

	// env resolves a key from its flag override (when one exists) first, then
	// the real environment. Every other key falls straight through to getenv.
	env := func(key string) string {
		switch key {
		case "PORT":
			if *flagPort != "" {
				return *flagPort
			}
		case "VECTORDB_MODE":
			if *flagMode != "" {
				return *flagMode
			}
		case "VECTORDB_DATA_DIR":
			if *flagDataDir != "" {
				return *flagDataDir
			}
		case "OLLAMA_EMBED_MODEL":
			if *flagEmbModel != "" {
				return *flagEmbModel
			}
		case "OLLAMA_URL":
			if *flagEmbURL != "" {
				return *flagEmbURL
			}
		}
		return getenv(key)
	}

	var errs []string
	posInt := func(key string, def int) int {
		v := env(key)
		if v == "" {
			return def
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s=%q is not a valid integer", key, v))
			return def
		}
		if n <= 0 {
			errs = append(errs, fmt.Sprintf("%s=%d must be positive", key, n))
			return def
		}
		return n
	}
	nonNegInt := func(key string, def int) int {
		v := env(key)
		if v == "" {
			return def
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s=%q is not a valid integer", key, v))
			return def
		}
		if n < 0 {
			errs = append(errs, fmt.Sprintf("%s=%d must be non-negative", key, n))
			return def
		}
		return n
	}
	strDefault := func(key, def string) string {
		if v := env(key); v != "" {
			return v
		}
		return def
	}

	if v := env("LOG_LEVEL"); v != "" {
		switch strings.ToLower(v) {
		case "debug", "info", "warn", "error":
		default:
			errs = append(errs, fmt.Sprintf("LOG_LEVEL=%q is not valid (use: debug, info, warn, error)", v))
		}
	}

	// DEEPDATA_EMBED_DIM only matters to the hash and onnx embedders (see
	// embed_text.go); for every other DEEPDATA_EMBEDDER kind it is dead, so
	// the read is kind-gated and tolerant (envIntFrom warns and falls back
	// to the default) rather than a fatal config error like the keys above.
	embKind := strings.ToLower(strings.TrimSpace(env("DEEPDATA_EMBEDDER")))
	embDim := 384
	if embKind == "hash" || embKind == "onnx" {
		embDim = envIntFrom(env, "DEEPDATA_EMBED_DIM", 384)
	}

	cfg := &serverConfig{
		HTTPPort:        posInt("PORT", 8080),
		GRPCPort:        nonNegInt("GRPC_PORT", 50051),
		BindHost:        env("DEEPDATA_BIND_HOST"),
		InsecureDevMode: env("DEEPDATA_INSECURE_DEV_MODE") == "1",
		H2C:             env("HTTP_H2C") != "0",
		ReadTimeout:     time.Duration(posInt("HTTP_READ_TIMEOUT_SEC", 60)) * time.Second,
		WriteTimeout:    time.Duration(posInt("HTTP_WRITE_TIMEOUT_SEC", 300)) * time.Second,
		RequestTimeout:  time.Duration(posInt("HTTP_REQUEST_TIMEOUT_SEC", 120)) * time.Second,
		LogFormat:       env("LOG_FORMAT"),
		LogLevel:        env("LOG_LEVEL"),
		HNSWEfSearch:    posInt("HNSW_EFSEARCH", 200),
		Auth: authConfig{
			APIToken:    env("API_TOKEN"),
			JWTSecret:   env("JWT_SECRET"),
			JWTIssuer:   strDefault("JWT_ISSUER", "vectordb"),
			RequireAuth: env("REQUIRE_AUTH") == "1",
			TrustProxy:  env("TRUST_PROXY") == "1",
		},
		Limits: limitConfig{
			APIRPS:           posInt("API_RPS", 100),
			MaxRateLimitKeys: posInt("MAX_RATE_LIMIT_KEYS", 100_000),
			AuthFailureRPS:   posInt("AUTH_FAILURE_RPS", 1),
			AuthFailureBurst: posInt("AUTH_FAILURE_BURST", 5),
			TenantRPS:        posInt("TENANT_RPS", 100),
			TenantBurst:      posInt("TENANT_BURST", 100),
			MaxTenants:       posInt("MAX_TENANTS", 100_000),
			MaxCollections:   posInt("MAX_COLLECTIONS", 10_000),

			MaxTenantDocuments:   int64(nonNegInt("MAX_TENANT_DOCUMENTS", 0)),
			MaxTenantBytes:       int64(nonNegInt("MAX_TENANT_BYTES", 0)),
			MaxTenantCollections: int64(nonNegInt("MAX_TENANT_COLLECTIONS", 0)),
		},
		CORSAllowedOrigins: env("CORS_ALLOWED_ORIGINS"),
		Embedder: embedderConfig{
			Kind:          embKind,
			Dim:           embDim,
			OllamaURL:     strings.TrimSpace(env("OLLAMA_URL")),
			OllamaModel:   strings.TrimSpace(env("OLLAMA_EMBED_MODEL")),
			OpenAIAPIKey:  env("OPENAI_API_KEY"),
			OnnxModel:     strings.TrimSpace(env("ONNX_EMBED_MODEL")),
			OnnxTokenizer: strings.TrimSpace(env("ONNX_EMBED_TOKENIZER")),
			OnnxMaxLen:    posInt("ONNX_EMBED_MAX_LEN", 512),
		},
		ReplicationToken: env(replicationTokenEnv),
		mode:             strings.ToLower(strings.TrimSpace(env("VECTORDB_MODE"))),
	}
	cfg.DataDir = resolveDataDir(env("VECTORDB_BASE_DIR"), env("VECTORDB_DATA_DIR"))
	cfg.IndexPath = filepath.Join(cfg.DataDir, "index.gob")

	return cfg, errs
}

const canonicalCredentialMinBytes = 32

// validateCanonicalCredential is retained for grpc_auth_test.go and
// canonical_process_test.go coverage of the exact rejection wording.
func validateCanonicalCredential(name, value string) error {
	if strings.TrimSpace(value) != value {
		return fmt.Errorf("%s must not contain leading or trailing whitespace", name)
	}
	if len([]byte(value)) < canonicalCredentialMinBytes {
		return fmt.Errorf("%s must be at least %d bytes", name, canonicalCredentialMinBytes)
	}
	return nil
}

// validateServe is the serve-only check: exactly one of API_TOKEN / JWT_SECRET,
// >=32 bytes, no surrounding whitespace, or DEEPDATA_INSECURE_DEV_MODE=1; and
// VECTORDB_MODE / --mode must be "" or "local". Same error strings as
// validateCanonicalAuthEnvironment and the historical main.go mode rejection.
func (c *serverConfig) validateServe() error {
	hasStaticToken := c.Auth.APIToken != ""
	hasJWTSecret := c.Auth.JWTSecret != ""
	if hasStaticToken && hasJWTSecret {
		return errors.New("configure exactly one of API_TOKEN or JWT_SECRET; combined credential modes are unsupported")
	}
	if hasStaticToken {
		if err := validateCanonicalCredential("API_TOKEN", c.Auth.APIToken); err != nil {
			return err
		}
	}
	if hasJWTSecret {
		if err := validateCanonicalCredential("JWT_SECRET", c.Auth.JWTSecret); err != nil {
			return err
		}
	}
	if !hasStaticToken && !hasJWTSecret && !c.InsecureDevMode {
		return errors.New("API_TOKEN or JWT_SECRET is required; set DEEPDATA_INSECURE_DEV_MODE=1 only for isolated development")
	}
	if c.mode != "" && c.mode != "local" {
		return fmt.Errorf("canonical RC accepts caller-supplied vectors and supports only the local persistence path (VECTORDB_MODE=%s)", c.mode)
	}
	return nil
}
