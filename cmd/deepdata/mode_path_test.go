package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGetDataDirectoryHonorsExactOverride(t *testing.T) {
	base := filepath.Join(t.TempDir(), "base")
	override := filepath.Join(t.TempDir(), "state")
	t.Setenv("VECTORDB_BASE_DIR", base)
	t.Setenv("VECTORDB_DATA_DIR", override)

	if got := GetDataDirectory(ModeLocal); got != override {
		t.Fatalf("data directory = %q, want exact override %q", got, override)
	}
	if got := GetIndexPath(ModeLocal); got != filepath.Join(override, "index.gob") {
		t.Fatalf("index path = %q", got)
	}
	dir, err := EnsureDataDirectory(ModeLocal)
	if err != nil {
		t.Fatal(err)
	}
	if dir != override {
		t.Fatalf("created directory = %q, want %q", dir, override)
	}
	if info, err := os.Stat(override); err != nil || !info.IsDir() {
		t.Fatalf("exact override was not created: info=%v err=%v", info, err)
	}
}

func TestGetDataDirectoryResolvesRelativeOverrideBelowBase(t *testing.T) {
	base := filepath.Join(t.TempDir(), "base")
	t.Setenv("VECTORDB_BASE_DIR", base)
	t.Setenv("VECTORDB_DATA_DIR", "custom")

	if got, want := GetDataDirectory(ModeLocal), filepath.Join(base, "custom"); got != want {
		t.Fatalf("data directory = %q, want %q", got, want)
	}
}

func TestGetDataDirectoryFallsBackToModeSubdirectory(t *testing.T) {
	base := filepath.Join(t.TempDir(), "base")
	t.Setenv("VECTORDB_BASE_DIR", base)
	t.Setenv("VECTORDB_DATA_DIR", "")

	if got, want := GetDataDirectory(ModeLocal), filepath.Join(base, "local"); got != want {
		t.Fatalf("data directory = %q, want %q", got, want)
	}
}

func TestLoadModeDataDirectoryMatchesRuntimePath(t *testing.T) {
	override := filepath.Join(t.TempDir(), "state")
	t.Setenv("VECTORDB_MODE", "local")
	t.Setenv("VECTORDB_DATA_DIR", override)
	t.Setenv("VECTORDB_BASE_DIR", filepath.Join(t.TempDir(), "ignored"))

	config, err := LoadModeFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if config.DataDirectory != override {
		t.Fatalf("loaded config directory = %q, want %q", config.DataDirectory, override)
	}
	if got := GetModeInfo(config).DataDirectory; got != override {
		t.Fatalf("reported directory = %q, want %q", got, override)
	}
}
