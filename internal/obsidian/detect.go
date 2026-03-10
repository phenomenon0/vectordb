package obsidian

import (
	"os"
	"path/filepath"
)

// DetectVaults searches common locations for Obsidian vaults (directories
// containing a .obsidian/ subdirectory). Returns all found vault paths.
func DetectVaults() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	// Search directories, in priority order
	searchDirs := []string{
		filepath.Join(home, "Documents"),
		home,
		filepath.Join(home, "Obsidian"),
		filepath.Join(home, "vaults"),
	}

	seen := make(map[string]bool)
	var vaults []string

	for _, dir := range searchDirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			candidate := filepath.Join(dir, entry.Name())
			if seen[candidate] {
				continue
			}
			// Check for .obsidian/ subdirectory
			obsDir := filepath.Join(candidate, ".obsidian")
			if info, err := os.Stat(obsDir); err == nil && info.IsDir() {
				seen[candidate] = true
				vaults = append(vaults, candidate)
			}
		}
	}

	return vaults
}

// DetectVault returns the first detected Obsidian vault path, or empty string.
func DetectVault() string {
	vaults := DetectVaults()
	if len(vaults) > 0 {
		return vaults[0]
	}
	return ""
}
