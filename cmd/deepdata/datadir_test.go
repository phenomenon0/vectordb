package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveDataDirHonorsExactOverride(t *testing.T) {
	base := filepath.Join(t.TempDir(), "base")
	override := filepath.Join(t.TempDir(), "state")

	if got := resolveDataDir(base, override); got != override {
		t.Fatalf("data directory = %q, want exact override %q", got, override)
	}
}

func TestResolveDataDirResolvesRelativeOverrideBelowBase(t *testing.T) {
	base := filepath.Join(t.TempDir(), "base")

	if got, want := resolveDataDir(base, "custom"), filepath.Join(base, "custom"); got != want {
		t.Fatalf("data directory = %q, want %q", got, want)
	}
}

func TestResolveDataDirFallsBackToLocalSubdirectory(t *testing.T) {
	base := filepath.Join(t.TempDir(), "base")

	if got, want := resolveDataDir(base, ""), filepath.Join(base, "local"); got != want {
		t.Fatalf("data directory = %q, want %q", got, want)
	}
}

func TestLoadServerConfigDataDirMatchesRuntimePath(t *testing.T) {
	override := filepath.Join(t.TempDir(), "state")
	ignoredBase := filepath.Join(t.TempDir(), "ignored")
	env := map[string]string{
		"VECTORDB_MODE":     "local",
		"VECTORDB_DATA_DIR": override,
		"VECTORDB_BASE_DIR": ignoredBase,
	}
	getenv := func(k string) string { return env[k] }

	cfg, errs := loadServerConfig(nil, getenv)
	if len(errs) != 0 {
		t.Fatalf("unexpected config errors: %v", errs)
	}
	if cfg.DataDir != override {
		t.Fatalf("loaded config directory = %q, want %q", cfg.DataDir, override)
	}
	if got := resolveDataDir(ignoredBase, override); got != override {
		t.Fatalf("reported directory = %q, want %q", got, override)
	}
}

func TestResolveDataDirDefaultsUnderHomeVectordbLocal(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(home, ".vectordb", "local")
	if got := resolveDataDir("", ""); got != want {
		t.Fatalf("default data directory = %q, want %q", got, want)
	}
}
