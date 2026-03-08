package review

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/index"
)

// Product Manager Persona: feature completeness, documentation accuracy, competitor positioning.

func TestProductManagerReview(t *testing.T) {
	review := NewReview("Product Manager",
		"Feature completeness, API surface, documentation, competitive positioning")

	// Check 1: Index type coverage
	t.Run("index_type_coverage", func(t *testing.T) {
		requiredTypes := []string{"hnsw", "ivf", "flat", "diskann"}
		supported := index.SupportedTypes()
		supportedSet := make(map[string]bool)
		for _, s := range supported {
			supportedSet[s] = true
		}

		missing := 0
		for _, req := range requiredTypes {
			if !supportedSet[req] {
				t.Logf("Missing index type: %s", req)
				missing++
			}
		}

		if missing == 0 {
			review.Pass("index_types", "All 4 core index types available (HNSW, IVF, FLAT, DiskANN)", SeverityHigh,
				strings.Join(supported, ", "))
		} else {
			review.Fail("index_types", "All 4 core index types available", SeverityHigh,
				strings.Join(supported, ", "))
		}
	})

	// Check 2: Quantization coverage
	t.Run("quantization_coverage", func(t *testing.T) {
		quantTypes := []struct {
			Name   string
			Config map[string]interface{}
		}{
			{"none", nil},
			{"float16", map[string]interface{}{"quantization": map[string]interface{}{"type": "float16"}}},
			{"uint8", map[string]interface{}{"quantization": map[string]interface{}{"type": "uint8"}}},
			{"binary", map[string]interface{}{"quantization": map[string]interface{}{"type": "binary"}}},
		}

		working := 0
		for _, q := range quantTypes {
			config := map[string]interface{}{"m": 16}
			if q.Config != nil {
				for k, v := range q.Config {
					config[k] = v
				}
			}
			_, err := index.Create("hnsw", 64, config)
			if err == nil {
				working++
				t.Logf("Quantization %s: OK", q.Name)
			} else {
				t.Logf("Quantization %s: FAIL (%v)", q.Name, err)
			}
		}

		if working >= 3 {
			review.Pass("quantization", "At least 3 quantization modes available", SeverityMedium,
				"checked: none, float16, uint8, binary")
		} else {
			review.Fail("quantization", "At least 3 quantization modes available", SeverityMedium,
				"Insufficient quantization support")
		}
	})

	// Check 3: Export/Import API exists
	t.Run("export_import_api", func(t *testing.T) {
		idx, err := index.Create("hnsw", 64, map[string]interface{}{"m": 16})
		if err != nil {
			review.Fail("export_import", "Index creation", SeverityMedium, err.Error())
			return
		}

		_, err = idx.Export()
		if err == nil {
			review.Pass("export_import", "Export API available", SeverityMedium, "")
		} else {
			review.Fail("export_import", "Export API available", SeverityMedium, err.Error())
		}
	})

	// Check 4: Stats API provides useful info
	t.Run("stats_api", func(t *testing.T) {
		idx, err := index.Create("hnsw", 64, map[string]interface{}{"m": 16})
		if err != nil {
			review.Fail("stats_api", "Index creation for stats check", SeverityLow, err.Error())
			return
		}
		stats := idx.Stats()

		hasName := stats.Name != ""
		hasDim := stats.Dim > 0

		if hasName && hasDim {
			review.Pass("stats_api", "Stats API provides name and dimension", SeverityLow, "")
		} else {
			review.Fail("stats_api", "Stats API provides name and dimension", SeverityLow,
				"Stats missing basic fields")
		}
	})

	// Check 5: Benchmark documentation exists
	t.Run("benchmark_docs", func(t *testing.T) {
		// Check for README in benchmarks directory
		projectRoot := findProjectRoot()
		readmePath := filepath.Join(projectRoot, "benchmarks", "README.md")
		_, err := os.Stat(readmePath)
		if err == nil {
			review.Pass("benchmark_docs", "Benchmark README exists", SeverityLow, readmePath)
		} else {
			review.Fail("benchmark_docs", "Benchmark README exists", SeverityLow,
				"benchmarks/README.md not found")
		}
	})

	// Check 6: GAP_ANALYSIS exists
	t.Run("gap_analysis", func(t *testing.T) {
		projectRoot := findProjectRoot()
		gapPath := filepath.Join(projectRoot, "benchmarks", "GAP_ANALYSIS.md")
		_, err := os.Stat(gapPath)
		if err == nil {
			review.Pass("gap_analysis", "GAP_ANALYSIS.md exists", SeverityMedium, gapPath)
		} else {
			review.Fail("gap_analysis", "GAP_ANALYSIS.md exists", SeverityLow,
				"benchmarks/GAP_ANALYSIS.md not found — competitive positioning unclear")
		}
	})

	// Check 7: Feature matrix — index × search params combinations
	t.Run("feature_matrix", func(t *testing.T) {
		ctx := context.Background()
		combinations := []struct {
			IndexType string
			Config    map[string]interface{}
			Params    index.SearchParams
		}{
			{"hnsw", map[string]interface{}{"m": 16}, &index.HNSWSearchParams{EfSearch: 64}},
			{"ivf", map[string]interface{}{"nlist": 10, "metric": "cosine"}, &index.IVFSearchParams{NProbe: 5}},
			{"flat", map[string]interface{}{"metric": "cosine"}, &index.DefaultSearchParams{}},
			{"diskann", map[string]interface{}{"max_degree": 16, "memory_limit": 1000, "metric": "cosine"}, &index.DefaultSearchParams{}},
		}

		working := 0
		for _, c := range combinations {
			idx, err := index.Create(c.IndexType, 64, c.Config)
			if err != nil {
				t.Logf("%s: creation failed: %v", c.IndexType, err)
				continue
			}
			// Try basic add+search
			vec := make([]float32, 64)
			for d := range vec {
				vec[d] = float32(d) / 64.0
			}
			if err := idx.Add(ctx, 0, vec); err != nil {
				t.Logf("%s: add failed: %v", c.IndexType, err)
				continue
			}
			_, err = idx.Search(ctx, vec, 1, c.Params)
			if err == nil {
				working++
			} else {
				t.Logf("%s: search failed: %v", c.IndexType, err)
			}
		}

		if working == len(combinations) {
			review.Pass("feature_matrix", "All index types support add+search lifecycle", SeverityHigh, "")
		} else {
			review.Fail("feature_matrix", "All index types support add+search lifecycle", SeverityHigh,
				"working: "+strconv.Itoa(working)+"/"+strconv.Itoa(len(combinations)))
		}
	})

	review.Report(t)
}

func findProjectRoot() string {
	// Walk up from current directory to find go.mod
	dir, _ := os.Getwd()
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

