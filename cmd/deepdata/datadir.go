package main

import (
	"os"
	"path/filepath"
	"strings"
)

// resolveDataDir returns the primary state directory: dataDir (flag or
// VECTORDB_DATA_DIR) absolute as given, relative below base (VECTORDB_BASE_DIR,
// default ~/.vectordb), empty -> <base>/local.
func resolveDataDir(baseDir, dataDir string) string {
	if baseDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			home = "."
		}
		baseDir = filepath.Join(home, ".vectordb")
	}
	baseDir = filepath.Clean(baseDir)

	if dataDir = strings.TrimSpace(dataDir); dataDir != "" {
		if filepath.IsAbs(dataDir) {
			return filepath.Clean(dataDir)
		}
		return filepath.Join(baseDir, dataDir)
	}
	return filepath.Join(baseDir, "local")
}
