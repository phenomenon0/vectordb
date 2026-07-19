package releaseinfo

import (
	"regexp"
	"testing"
)

func TestVersionIsReleaseCandidateSemver(t *testing.T) {
	if !regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+-rc\.[1-9][0-9]*$`).MatchString(Version()) {
		t.Fatalf("release version %q is not an rc SemVer", Version())
	}
}
