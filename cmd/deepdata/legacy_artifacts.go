package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func syncParentDirectory(path string) error {
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	if err := dir.Sync(); err != nil {
		_ = dir.Close()
		return err
	}
	return dir.Close()
}

func removeDurableFile(path string) error {
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if err := syncParentDirectory(path); err != nil {
		return fmt.Errorf("sync directory after removing %q: %w", path, err)
	}
	return nil
}

// removeWALArtifact is a narrow test seam for cleanup-failure recovery tests.
var removeWALArtifact = removeDurableFile

// syncWALFile is a narrow test seam for indeterminate append failures.
var syncWALFile = func(f *os.File) error { return f.Sync() }

// syncWALDirectory is a narrow test seam for WAL create/rotation durability.
var syncWALDirectory = syncParentDirectory

// renameSnapshotFile is a narrow test seam for snapshot commit ordering.
var renameSnapshotFile = os.Rename

func existingLegacyRootArtifacts(indexPath string) ([]string, error) {
	if indexPath == "" {
		return nil, nil
	}
	paths := make([]string, 0, 3)
	for _, path := range []string{indexPath, indexPath + ".wal.frozen", indexPath + ".wal"} {
		if _, err := os.Lstat(path); err == nil {
			paths = append(paths, path)
		} else if !os.IsNotExist(err) {
			return nil, fmt.Errorf("inspect legacy root artifact %q: %w", path, err)
		}
	}
	return paths, nil
}

// ======================================================================================
// Embedder selection (Hash by default; ONNX under build tag)
// ======================================================================================
