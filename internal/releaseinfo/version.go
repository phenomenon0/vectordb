package releaseinfo

import (
	_ "embed"
	"strings"
)

//go:embed version.txt
var rawVersion string

// Version returns the canonical semantic version from the embedded source of
// truth. Packaging metadata is checked against the same file in CI.
func Version() string {
	return strings.TrimSpace(rawVersion)
}
