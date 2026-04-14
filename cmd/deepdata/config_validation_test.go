package main

import (
	"os"
	"testing"

	"github.com/phenomenon0/vectordb/internal/logging"
)

func TestValidateEnvConfig_Clean(t *testing.T) {
	// With no env vars set, validation should pass with zero errors
	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors with clean env, got %d: %v", len(errs), errs)
	}
}

func TestValidateEnvConfig_InvalidPort(t *testing.T) {
	os.Setenv("PORT", "not-a-number")
	defer os.Unsetenv("PORT")

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0] != `PORT="not-a-number" is not a valid integer` {
		t.Errorf("unexpected error message: %s", errs[0])
	}
}

func TestValidateEnvConfig_NegativePort(t *testing.T) {
	os.Setenv("PORT", "-1")
	defer os.Unsetenv("PORT")

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0] != `PORT=-1 must be positive` {
		t.Errorf("unexpected error message: %s", errs[0])
	}
}

func TestValidateEnvConfig_InvalidStorageFormat(t *testing.T) {
	os.Setenv("STORAGE_FORMAT", "badformat")
	defer os.Unsetenv("STORAGE_FORMAT")

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0] == "" || errs[0][:len("STORAGE_FORMAT")] != "STORAGE_FORMAT" {
		t.Errorf("expected STORAGE_FORMAT error, got: %s", errs[0])
	}
}

func TestValidateEnvConfig_InvalidLogLevel(t *testing.T) {
	os.Setenv("LOG_LEVEL", "verbose")
	defer os.Unsetenv("LOG_LEVEL")

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0] != `LOG_LEVEL="verbose" is not valid (use: debug, info, warn, error)` {
		t.Errorf("unexpected error message: %s", errs[0])
	}
}

func TestValidateEnvConfig_ValidLogLevels(t *testing.T) {
	logger := logging.Init(logging.DefaultConfig())
	for _, level := range []string{"debug", "info", "warn", "error"} {
		os.Setenv("LOG_LEVEL", level)
		errs := validateEnvConfig(logger)
		if len(errs) != 0 {
			t.Errorf("LOG_LEVEL=%s should be valid, got errors: %v", level, errs)
		}
	}
	os.Unsetenv("LOG_LEVEL")
}

func TestValidateEnvConfig_InvalidHNSWFloat(t *testing.T) {
	os.Setenv("HNSW_ML", "abc")
	defer os.Unsetenv("HNSW_ML")

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if errs[0] != `HNSW_ML="abc" is not a valid float` {
		t.Errorf("unexpected error message: %s", errs[0])
	}
}

func TestValidateEnvConfig_ZeroVectorCapacity(t *testing.T) {
	// VECTOR_CAPACITY=0 should be allowed (non-negative)
	os.Setenv("VECTOR_CAPACITY", "0")
	defer os.Unsetenv("VECTOR_CAPACITY")

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 0 {
		t.Errorf("VECTOR_CAPACITY=0 should be valid, got errors: %v", errs)
	}
}

func TestValidateEnvConfig_MultipleErrors(t *testing.T) {
	os.Setenv("PORT", "abc")
	os.Setenv("HNSW_M", "-5")
	os.Setenv("LOG_LEVEL", "trace")
	defer func() {
		os.Unsetenv("PORT")
		os.Unsetenv("HNSW_M")
		os.Unsetenv("LOG_LEVEL")
	}()

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 3 {
		t.Fatalf("expected 3 errors, got %d: %v", len(errs), errs)
	}
}

func TestValidateEnvConfig_ValidIntValues(t *testing.T) {
	os.Setenv("PORT", "8080")
	os.Setenv("HNSW_M", "32")
	os.Setenv("HNSW_ML", "0.5")
	os.Setenv("HNSW_EFSEARCH", "128")
	os.Setenv("WAL_MAX_OPS", "5000")
	defer func() {
		os.Unsetenv("PORT")
		os.Unsetenv("HNSW_M")
		os.Unsetenv("HNSW_ML")
		os.Unsetenv("HNSW_EFSEARCH")
		os.Unsetenv("WAL_MAX_OPS")
	}()

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 0 {
		t.Errorf("all values are valid, expected 0 errors, got %d: %v", len(errs), errs)
	}
}

func TestValidateEnvConfig_InvalidWALMaxBytes(t *testing.T) {
	os.Setenv("WAL_MAX_BYTES", "0")
	defer os.Unsetenv("WAL_MAX_BYTES")

	logger := logging.Init(logging.DefaultConfig())
	errs := validateEnvConfig(logger)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error for WAL_MAX_BYTES=0, got %d: %v", len(errs), errs)
	}
}

func TestEnvInt_InvalidValue_ReturnsDefault(t *testing.T) {
	os.Setenv("TEST_ENV_INT", "notanumber")
	defer os.Unsetenv("TEST_ENV_INT")

	result := envInt("TEST_ENV_INT", 42)
	if result != 42 {
		t.Errorf("expected default 42, got %d", result)
	}
}

func TestEnvInt_ValidValue(t *testing.T) {
	os.Setenv("TEST_ENV_INT", "99")
	defer os.Unsetenv("TEST_ENV_INT")

	result := envInt("TEST_ENV_INT", 42)
	if result != 99 {
		t.Errorf("expected 99, got %d", result)
	}
}

func TestEnvInt64_InvalidValue_ReturnsDefault(t *testing.T) {
	os.Setenv("TEST_ENV_INT64", "notanumber")
	defer os.Unsetenv("TEST_ENV_INT64")

	result := envInt64("TEST_ENV_INT64", 1000)
	if result != 1000 {
		t.Errorf("expected default 1000, got %d", result)
	}
}

func TestEnvInt64_ZeroValue_ReturnsDefault(t *testing.T) {
	// envInt64 requires positive values
	os.Setenv("TEST_ENV_INT64", "0")
	defer os.Unsetenv("TEST_ENV_INT64")

	result := envInt64("TEST_ENV_INT64", 1000)
	if result != 1000 {
		t.Errorf("expected default 1000 for zero value, got %d", result)
	}
}
