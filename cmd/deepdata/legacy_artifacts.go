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

// detectStoreLayout distinguishes a raw pre-0.1 legacy collection root, the
// 0.1 single-store layout (basePath.*), and the canonical StoreSet layout
// (indexPath+".tenants"/) so canonical startup never silently reinterprets an
// old data directory. It never migrates anything itself.
func detectStoreLayout(indexPath string) error {
	basePath := indexPath + ".collections"
	tenantDir := indexPath + ".tenants"

	rawLegacy := make([]string, 0, 2)
	for _, path := range []string{basePath + ".manager", basePath + ".tenants"} {
		if _, err := os.Lstat(path); err == nil {
			rawLegacy = append(rawLegacy, path)
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("inspect legacy V2 collection artifact %q: %w", path, err)
		}
	}
	if len(rawLegacy) > 0 {
		return fmt.Errorf(
			"legacy V2 collection persistence requires an explicit offline migration before canonical startup: %v",
			rawLegacy,
		)
	}

	singleStore := false
	for _, path := range []string{basePath + ".snapshot", basePath + ".initialized"} {
		if _, err := os.Lstat(path); err == nil {
			singleStore = true
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("inspect single-store collection artifact %q: %w", path, err)
		}
	}

	tenantDirInfo, err := os.Stat(tenantDir)
	tenantDirExists := err == nil && tenantDirInfo.IsDir()
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("inspect tenant store directory %q: %w", tenantDir, err)
	}

	switch {
	case singleStore && tenantDirExists:
		return fmt.Errorf("ambiguous layout: both %s.* and %s exist; keep one", basePath, tenantDir)
	case singleStore:
		return fmt.Errorf("single-store layout found at %s; run: deepdata migrate-tenants <data-dir> (see docs/upgrade-to-0.2-rc.md)", basePath)
	default:
		return nil
	}
}
