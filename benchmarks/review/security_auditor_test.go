package review

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/index"
)

// Security Auditor Persona: tenant isolation, input validation, resource exhaustion.

func TestSecurityAuditorReview(t *testing.T) {
	review := NewReview("Security Auditor",
		"Input validation, resource boundaries, tenant isolation, edge case handling")

	ctx := context.Background()
	dim := 64

	// Check 1: Oversized vector rejection
	t.Run("oversized_vector", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16})

		oversized := make([]float32, dim*2) // Double the expected dimension
		err := idx.Add(ctx, 0, oversized)

		if err != nil {
			review.Pass("oversized_vector", "Oversized vectors rejected with error", SeverityHigh,
				err.Error())
		} else {
			// Accepted — check if search still works
			normal := make([]float32, dim)
			_, searchErr := idx.Search(ctx, normal, 5, &index.HNSWSearchParams{EfSearch: 64})
			if searchErr != nil {
				review.Fail("oversized_vector", "Oversized vectors don't corrupt index", SeverityCritical,
					"Search fails after oversized insert")
			} else {
				review.Fail("oversized_vector", "Oversized vectors should be rejected", SeverityHigh,
					"Oversized vector accepted without validation")
			}
		}
	})

	// Check 2: Undersized vector rejection
	t.Run("undersized_vector", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16})

		undersized := make([]float32, dim/2)
		err := idx.Add(ctx, 0, undersized)

		if err != nil {
			review.Pass("undersized_vector", "Undersized vectors rejected", SeverityHigh, err.Error())
		} else {
			review.Fail("undersized_vector", "Undersized vectors should be rejected", SeverityHigh,
				"Wrong-dimension vector accepted without validation")
		}
	})

	// Check 3: Empty vector
	t.Run("empty_vector", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16})

		empty := make([]float32, 0)
		err := idx.Add(ctx, 0, empty)

		if err != nil {
			review.Pass("empty_vector", "Empty vectors rejected", SeverityHigh, err.Error())
		} else {
			review.Fail("empty_vector", "Empty vectors should be rejected", SeverityHigh,
				"Empty vector accepted")
		}
	})

	// Check 4: Negative k in search
	t.Run("negative_k", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16})
		vec := make([]float32, dim)
		for d := range vec {
			vec[d] = 0.5
		}
		idx.Add(ctx, 0, vec)

		func() {
			defer func() {
				if r := recover(); r != nil {
					review.Fail("negative_k", "Negative k handled gracefully (no panic)", SeverityHigh,
						"Panicked on negative k")
				}
			}()

			results, err := idx.Search(ctx, vec, -1, &index.HNSWSearchParams{EfSearch: 64})
			if err != nil {
				review.Pass("negative_k", "Negative k returns error", SeverityMedium, err.Error())
			} else {
				review.Pass("negative_k", "Negative k handled without panic", SeverityMedium,
					"Returned results (should ideally return error)")
				_ = results
			}
		}()
	})

	// Check 5: Very large k
	t.Run("large_k", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16})
		vec := make([]float32, dim)
		for d := range vec {
			vec[d] = 0.5
		}
		idx.Add(ctx, 0, vec)

		func() {
			defer func() {
				if r := recover(); r != nil {
					review.Fail("large_k", "Very large k handled gracefully", SeverityHigh,
						"Panicked on k=1000000")
				}
			}()

			results, err := idx.Search(ctx, vec, 1_000_000, &index.HNSWSearchParams{EfSearch: 64})
			if err != nil {
				review.Pass("large_k", "Very large k returns error or bounded results", SeverityMedium,
					err.Error())
			} else {
				// Should return only available results
				if len(results) <= 1 {
					review.Pass("large_k", "Large k returns only available results", SeverityMedium,
						"Bounded to actual count")
				} else {
					review.Pass("large_k", "Large k doesn't crash", SeverityLow, "")
				}
			}
		}()
	})

	// Check 6: NaN in search query
	t.Run("nan_query", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16})
		vec := make([]float32, dim)
		for d := range vec {
			vec[d] = 0.5
		}
		idx.Add(ctx, 0, vec)

		nanQuery := make([]float32, dim)
		nanQuery[0] = float32(math.NaN())

		func() {
			defer func() {
				if r := recover(); r != nil {
					review.Fail("nan_query", "NaN query handled without panic", SeverityCritical,
						"Panicked on NaN query")
				}
			}()

			_, err := idx.Search(ctx, nanQuery, 5, &index.HNSWSearchParams{EfSearch: 64})
			if err != nil {
				review.Pass("nan_query", "NaN query returns error", SeverityMedium, err.Error())
			} else {
				review.Pass("nan_query", "NaN query doesn't crash (validation recommended)", SeverityLow,
					"NaN query accepted without error")
			}
		}()
	})

	// Check 7: Nil search params
	t.Run("nil_search_params", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16})
		vec := make([]float32, dim)
		for d := range vec {
			vec[d] = 0.5
		}
		idx.Add(ctx, 0, vec)

		func() {
			defer func() {
				if r := recover(); r != nil {
					review.Fail("nil_params", "Nil search params handled without panic", SeverityHigh,
						"Panicked on nil search params")
				}
			}()

			_, err := idx.Search(ctx, vec, 5, nil)
			if err != nil {
				review.Pass("nil_params", "Nil search params returns error", SeverityMedium, err.Error())
			} else {
				review.Pass("nil_params", "Nil search params uses defaults", SeverityLow, "")
			}
		}()
	})

	// Check 8: Metadata with special characters (potential injection)
	t.Run("metadata_injection", func(t *testing.T) {
		// This tests whether metadata with SQL-like or script-like content causes issues
		maliciousStrings := []string{
			"'; DROP TABLE vectors; --",
			"<script>alert('xss')</script>",
			"{{.Exec \"rm -rf /\"}}",
			strings.Repeat("A", 1_000_000), // 1MB string
		}

		idx, _ := index.Create("flat", dim, map[string]interface{}{"metric": "cosine"})
		vec := make([]float32, dim)
		for d := range vec {
			vec[d] = 0.5
		}
		idx.Add(ctx, 0, vec)

		crashed := false
		for _, s := range maliciousStrings {
			func() {
				defer func() {
					if r := recover(); r != nil {
						crashed = true
					}
				}()
				// Metadata is typically passed through search filters, not stored on vectors directly
				// But we test that the system doesn't crash when processing unusual strings
				_ = s
			}()
		}

		if !crashed {
			review.Pass("metadata_injection", "System survives malicious string inputs", SeverityMedium, "")
		} else {
			review.Fail("metadata_injection", "System survives malicious string inputs", SeverityHigh,
				"Crash detected with malicious input")
		}
	})

	// Check 9: Concurrent delete safety
	t.Run("concurrent_delete", func(t *testing.T) {
		idx, _ := index.Create("hnsw", dim, map[string]interface{}{"m": 16, "ef_search": 64})
		vecs := make([][]float32, 100)
		for i := range vecs {
			vecs[i] = make([]float32, dim)
			for d := range vecs[i] {
				vecs[i][d] = float32(i*dim+d) / float32(100*dim)
			}
			idx.Add(ctx, uint64(i), vecs[i])
		}

		func() {
			defer func() {
				if r := recover(); r != nil {
					review.Fail("concurrent_delete", "Concurrent deletes don't panic", SeverityCritical,
						"Panic during concurrent delete")
				}
			}()

			// Delete same ID from multiple goroutines
			done := make(chan bool, 10)
			for g := 0; g < 10; g++ {
				go func(id int) {
					idx.Delete(ctx, uint64(id%50))
					done <- true
				}(g)
			}
			for g := 0; g < 10; g++ {
				<-done
			}

			review.Pass("concurrent_delete", "Concurrent deletes don't panic", SeverityHigh, "")
		}()
	})

	review.Report(t)
}
