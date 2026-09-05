package main

import (
	"os"
	"strings"
	"testing"
)

func mapGetenv(env map[string]string) func(string) string {
	return func(k string) string { return env[k] }
}

// A flag is a stronger signal than the ambient environment: an operator
// invoking `deepdata --port 9000` means 9000 even if a stale PORT=8000 is
// still exported in the shell that launched it.
func TestLoadServerConfigFlagBeatsEnv(t *testing.T) {
	t.Parallel()
	cfg, errs := loadServerConfig([]string{"--port", "9000"}, mapGetenv(map[string]string{"PORT": "8000"}))
	if len(errs) != 0 {
		t.Fatalf("unexpected config errors: %v", errs)
	}
	if cfg.HTTPPort != 9000 {
		t.Fatalf("HTTPPort = %d, want 9000 (flag must beat env)", cfg.HTTPPort)
	}
}

// A live key (one loadServerConfig actually consumes) must fail fast with
// the same wording operators have always seen, so existing deploy tooling
// that greps startup logs keeps working.
func TestLoadServerConfigInvalidLiveKeyExactMessage(t *testing.T) {
	t.Parallel()
	_, errs := loadServerConfig(nil, mapGetenv(map[string]string{"PORT": "-1"}))
	if len(errs) != 1 || errs[0] != `PORT=-1 must be positive` {
		t.Fatalf("errs = %v, want exactly [\"PORT=-1 must be positive\"]", errs)
	}
}

// HNSW_M was validated at the old boundary but nothing ever read it. A dead
// key must not be able to block startup: if it silently starts validating
// again, that is this refactor regressing the very knob it was meant to bury.
func TestLoadServerConfigDeadKeyNoError(t *testing.T) {
	t.Parallel()
	_, errs := loadServerConfig(nil, mapGetenv(map[string]string{"HNSW_M": "abc"}))
	if len(errs) != 0 {
		t.Fatalf("dead key HNSW_M must not produce a config error, got: %v", errs)
	}
}

// The canonical RC serves only the local persistence path; validateServe is
// the one place that still enforces it now that the mode check moved out of
// main().
func TestServerConfigValidateServeRejectsNonLocalMode(t *testing.T) {
	t.Parallel()
	env := mapGetenv(map[string]string{"DEEPDATA_INSECURE_DEV_MODE": "1"})

	cfg, errs := loadServerConfig([]string{"--mode", "pro"}, env)
	if len(errs) != 0 {
		t.Fatalf("unexpected config errors: %v", errs)
	}
	err := cfg.validateServe()
	if err == nil || !strings.Contains(err.Error(), "canonical RC accepts caller-supplied vectors and supports only the local persistence path") {
		t.Fatalf("validateServe() = %v, want the canonical mode rejection", err)
	}

	cfg, errs = loadServerConfig([]string{"--mode", "local"}, env)
	if len(errs) != 0 {
		t.Fatalf("unexpected config errors: %v", errs)
	}
	if err := cfg.validateServe(); err != nil {
		t.Fatalf("validateServe() with --mode local = %v, want nil", err)
	}
}

// DEEPDATA_EMBED_DIM only matters to the hash/onnx embedders (embed_text.go).
// A garbage value must stay inert and never abort startup, whether or not
// the configured embedder is one that reads it — same tolerance the
// pre-refactor lazy envInt read inside newServerEmbedderFromEnv had.
func TestLoadServerConfigEmbedDimIsKindGatedAndTolerant(t *testing.T) {
	t.Setenv("DEEPDATA_EMBED_DIM", "not-a-number")

	cfg, errs := loadServerConfig(nil, os.Getenv)
	if len(errs) != 0 {
		t.Fatalf("garbage DEEPDATA_EMBED_DIM must not produce a config error with no embedder configured, got: %v", errs)
	}
	if cfg.Embedder.Dim != 384 {
		t.Fatalf("Embedder.Dim = %d, want default 384", cfg.Embedder.Dim)
	}

	t.Setenv("DEEPDATA_EMBEDDER", "hash")
	cfg, errs = loadServerConfig(nil, os.Getenv)
	if len(errs) != 0 {
		t.Fatalf("garbage DEEPDATA_EMBED_DIM must not produce a config error even for a kind that reads it, got: %v", errs)
	}
	if cfg.Embedder.Dim != 384 {
		t.Fatalf("Embedder.Dim = %d, want default 384 on parse failure", cfg.Embedder.Dim)
	}
}
