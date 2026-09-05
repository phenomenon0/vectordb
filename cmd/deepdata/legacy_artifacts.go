package main

import (
	"fmt"
	"os"
)

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
