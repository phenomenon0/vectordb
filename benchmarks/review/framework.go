package review

import (
	"fmt"
	"strings"
	"testing"
)

// Severity levels for review findings.
type Severity int

const (
	SeverityInfo     Severity = iota // Informational, no action needed
	SeverityLow                      // Minor issue, low priority
	SeverityMedium                   // Should be fixed before release
	SeverityHigh                     // Must be fixed, potential data loss or security issue
	SeverityCritical                 // Blocking, system unusable or unsafe
)

func (s Severity) String() string {
	switch s {
	case SeverityInfo:
		return "INFO"
	case SeverityLow:
		return "LOW"
	case SeverityMedium:
		return "MEDIUM"
	case SeverityHigh:
		return "HIGH"
	case SeverityCritical:
		return "CRITICAL"
	default:
		return "UNKNOWN"
	}
}

// Check represents a single review checkpoint.
type Check struct {
	Name        string
	Description string
	Severity    Severity
	Passed      bool
	Details     string
}

// PersonaReview aggregates checks from a specific reviewer persona.
type PersonaReview struct {
	Persona     string
	Description string
	Checks      []Check
}

// NewReview creates a new persona review.
func NewReview(persona, description string) *PersonaReview {
	return &PersonaReview{
		Persona:     persona,
		Description: description,
	}
}

// AddCheck adds a check result to the review.
func (r *PersonaReview) AddCheck(name, description string, severity Severity, passed bool, details string) {
	r.Checks = append(r.Checks, Check{
		Name:        name,
		Description: description,
		Severity:    severity,
		Passed:      passed,
		Details:     details,
	})
}

// Pass records a passing check.
func (r *PersonaReview) Pass(name, description string, severity Severity, details string) {
	r.AddCheck(name, description, severity, true, details)
}

// Fail records a failing check.
func (r *PersonaReview) Fail(name, description string, severity Severity, details string) {
	r.AddCheck(name, description, severity, false, details)
}

// Score returns the pass rate as a percentage (0-100).
func (r *PersonaReview) Score() float64 {
	if len(r.Checks) == 0 {
		return 0
	}
	passed := 0
	for _, c := range r.Checks {
		if c.Passed {
			passed++
		}
	}
	return float64(passed) / float64(len(r.Checks)) * 100
}

// PassCount returns the number of passing checks.
func (r *PersonaReview) PassCount() int {
	n := 0
	for _, c := range r.Checks {
		if c.Passed {
			n++
		}
	}
	return n
}

// FailCount returns the number of failing checks.
func (r *PersonaReview) FailCount() int {
	return len(r.Checks) - r.PassCount()
}

// Findings returns all failing checks.
func (r *PersonaReview) Findings() []Check {
	var findings []Check
	for _, c := range r.Checks {
		if !c.Passed {
			findings = append(findings, c)
		}
	}
	return findings
}

// CriticalFindings returns failing checks with HIGH or CRITICAL severity.
func (r *PersonaReview) CriticalFindings() []Check {
	var findings []Check
	for _, c := range r.Checks {
		if !c.Passed && c.Severity >= SeverityHigh {
			findings = append(findings, c)
		}
	}
	return findings
}

// Summary returns a formatted text summary of the review.
func (r *PersonaReview) Summary() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("\n=== %s Review ===\n", r.Persona))
	sb.WriteString(fmt.Sprintf("Description: %s\n", r.Description))
	sb.WriteString(fmt.Sprintf("Score: %.0f%% (%d/%d passed)\n\n", r.Score(), r.PassCount(), len(r.Checks)))

	// Group by status
	if failures := r.Findings(); len(failures) > 0 {
		sb.WriteString("FINDINGS:\n")
		for _, f := range failures {
			sb.WriteString(fmt.Sprintf("  [%s] %s: %s\n", f.Severity, f.Name, f.Description))
			if f.Details != "" {
				sb.WriteString(fmt.Sprintf("         %s\n", f.Details))
			}
		}
		sb.WriteString("\n")
	}

	sb.WriteString("PASSED:\n")
	for _, c := range r.Checks {
		if c.Passed {
			sb.WriteString(fmt.Sprintf("  [OK] %s: %s\n", c.Name, c.Description))
		}
	}

	return sb.String()
}

// Report outputs the review summary to testing.T.
func (r *PersonaReview) Report(t *testing.T) {
	t.Helper()
	t.Log(r.Summary())

	// Fail the test if there are critical findings
	if critical := r.CriticalFindings(); len(critical) > 0 {
		for _, c := range critical {
			t.Errorf("[%s] %s: %s — %s", c.Severity, c.Name, c.Description, c.Details)
		}
	}
}
