package review

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/index"
)

// DB Engineer Persona: data integrity, crash recovery, edge cases, concurrent correctness.

func TestDBEngineerReview(t *testing.T) {
	review := NewReview("Database Engineer",
		"Data integrity, crash recovery, edge cases, and concurrent correctness")

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))
	dim := 64
	scale := 1000
	if testing.Short() {
		scale = 200
	}

	vectors := testdata.GenerateClusteredVectors(scale, dim, 10, 0.15, rng)

	// Check 1: Insert then search returns inserted vectors
	t.Run("insert_search_correctness", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 200, "ef_construction": 200,
		})
		if err != nil {
			review.Fail("insert_search", "Index creation succeeds", SeverityCritical, err.Error())
			return
		}
		for i, v := range vectors {
			if err := idx.Add(ctx, uint64(i), v); err != nil {
				review.Fail("insert_search", "All vectors insert without error", SeverityCritical, err.Error())
				return
			}
		}

		// Search for each of the first 10 vectors — it should find itself
		allFound := true
		for i := 0; i < 10 && i < len(vectors); i++ {
			results, err := idx.Search(ctx, vectors[i], 1, &index.HNSWSearchParams{EfSearch: 200})
			if err != nil || len(results) == 0 || results[0].ID != uint64(i) {
				allFound = false
				break
			}
		}
		if allFound {
			review.Pass("insert_search", "Inserted vectors found via self-query", SeverityMedium, "")
		} else {
			review.Fail("insert_search", "Inserted vectors found via self-query", SeverityMedium,
				"Some vectors not found as nearest neighbor to themselves (expected for approximate indexes with small ef)")
		}
	})

	// Check 2: Delete then search should not return deleted vectors
	t.Run("delete_search_correctness", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 200, "ef_construction": 200,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range vectors[:100] {
			if err := idx.Add(ctx, uint64(i), v); err != nil {
				t.Fatalf("inserting vector %d: %v", i, err)
			}
		}

		// Delete first 10
		for i := 0; i < 10; i++ {
			idx.Delete(ctx, uint64(i))
		}

		// Search should not return deleted IDs
		deletedFound := false
		for i := 0; i < 10; i++ {
			results, _ := idx.Search(ctx, vectors[i], 10, &index.HNSWSearchParams{EfSearch: 200})
			for _, r := range results {
				if r.ID < 10 {
					deletedFound = true
					break
				}
			}
		}
		if !deletedFound {
			review.Pass("delete_search", "Deleted vectors excluded from search results", SeverityHigh, "")
		} else {
			review.Fail("delete_search", "Deleted vectors excluded from search results", SeverityHigh,
				"Deleted vector IDs appeared in search results")
		}
	})

	// Check 3: Concurrent read/write safety
	t.Run("concurrent_rw_safety", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 64, "ef_construction": 200,
		})
		if err != nil {
			t.Fatal(err)
		}
		// Pre-populate
		for i := 0; i < 100; i++ {
			if err := idx.Add(ctx, uint64(i), vectors[i%len(vectors)]); err != nil {
				t.Fatalf("inserting vector %d: %v", i, err)
			}
		}

		var wg sync.WaitGroup
		panicked := false
		errCount := 0
		var mu sync.Mutex

		// 4 readers + 2 writers
		for r := 0; r < 4; r++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				defer func() {
					if rec := recover(); rec != nil {
						mu.Lock()
						panicked = true
						mu.Unlock()
					}
				}()
				localRng := rand.New(rand.NewSource(int64(id)))
				for i := 0; i < 100; i++ {
					q := vectors[localRng.Intn(len(vectors))]
					_, err := idx.Search(ctx, q, 5, &index.HNSWSearchParams{EfSearch: 64})
					if err != nil {
						mu.Lock()
						errCount++
						mu.Unlock()
					}
				}
			}(r)
		}
		for w := 0; w < 2; w++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				defer func() {
					if rec := recover(); rec != nil {
						mu.Lock()
						panicked = true
						mu.Unlock()
					}
				}()
				for i := 0; i < 50; i++ {
					vid := uint64(100 + id*50 + i)
					idx.Add(ctx, vid, vectors[i%len(vectors)])
				}
			}(w)
		}
		wg.Wait()

		if panicked {
			review.Fail("concurrent_rw", "No panics under concurrent read/write", SeverityCritical,
				"Panic detected during concurrent access")
		} else if errCount > 0 {
			review.Fail("concurrent_rw", "No errors under concurrent read/write", SeverityMedium,
				fmt.Sprintf("%d search errors during concurrent access", errCount))
		} else {
			review.Pass("concurrent_rw", "No panics or errors under concurrent read/write", SeverityHigh, "")
		}
	})

	// Check 4: Export/Import round-trip
	t.Run("export_import_roundtrip", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 100, "ef_construction": 200,
		})
		if err != nil {
			review.Fail("export_import", "Index creation", SeverityMedium, err.Error())
			return
		}
		for i := 0; i < 50; i++ {
			if err := idx.Add(ctx, uint64(i), vectors[i]); err != nil {
				review.Fail("export_import", "Vector insertion", SeverityMedium, err.Error())
				return
			}
		}

		exported, err := idx.Export()
		if err != nil {
			review.Fail("export_import", "Index exports successfully", SeverityMedium, err.Error())
			return
		}

		idx2, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 100, "ef_construction": 200,
		})
		if err != nil {
			review.Fail("export_import", "Index creation for import", SeverityMedium, err.Error())
			return
		}
		err = idx2.Import(exported)
		if err != nil {
			review.Fail("export_import", "Index imports successfully", SeverityMedium, err.Error())
			return
		}

		// Verify search works on imported index
		results, err := idx2.Search(ctx, vectors[0], 5, &index.HNSWSearchParams{EfSearch: 100})
		if err != nil || len(results) == 0 {
			review.Fail("export_import", "Imported index returns search results", SeverityMedium,
				"No results from imported index")
		} else {
			review.Pass("export_import", "Export/Import preserves searchability", SeverityMedium, "")
		}
	})

	// Check 5: NaN/Inf vector handling
	t.Run("nan_inf_handling", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 64,
		})
		if err != nil {
			t.Fatal(err)
		}

		nanVec := make([]float32, dim)
		nanVec[0] = float32(math.NaN())
		infVec := make([]float32, dim)
		infVec[0] = float32(math.Inf(1))

		nanErr := idx.Add(ctx, 999, nanVec)
		infErr := idx.Add(ctx, 998, infVec)

		if nanErr != nil || infErr != nil {
			// Good: index rejects invalid vectors
			review.Pass("nan_inf", "Index handles NaN/Inf vectors gracefully", SeverityMedium,
				"Invalid vectors rejected with error")
		} else {
			// Added without error — check if search still works
			normalVec := vectors[0]
			_, searchErr := idx.Search(ctx, normalVec, 5, &index.HNSWSearchParams{EfSearch: 64})
			if searchErr != nil {
				review.Fail("nan_inf", "NaN/Inf vectors don't corrupt search", SeverityHigh,
					"Search fails after NaN/Inf insertion: "+searchErr.Error())
			} else {
				review.Pass("nan_inf", "Index tolerates NaN/Inf without corruption", SeverityLow,
					"NaN/Inf accepted but search still works (validation recommended)")
			}
		}
	})

	// Check 6: Zero vector
	t.Run("zero_vector", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 64,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 20; i++ {
			if err := idx.Add(ctx, uint64(i), vectors[i]); err != nil {
				t.Fatalf("inserting vector %d: %v", i, err)
			}
		}

		zeroVec := make([]float32, dim)
		addErr := idx.Add(ctx, 999, zeroVec)
		if addErr != nil {
			review.Pass("zero_vector", "Zero vector handled with error", SeverityLow, addErr.Error())
		} else {
			_, searchErr := idx.Search(ctx, zeroVec, 5, &index.HNSWSearchParams{EfSearch: 64})
			if searchErr != nil {
				review.Fail("zero_vector", "Zero vector doesn't break search", SeverityMedium,
					searchErr.Error())
			} else {
				review.Pass("zero_vector", "Zero vector search works", SeverityLow,
					"Zero vector accepted and searchable")
			}
		}
	})

	// Check 7: Duplicate ID overwrite
	t.Run("duplicate_id", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_search": 200, "ef_construction": 200,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := idx.Add(ctx, 0, vectors[0]); err != nil {
			t.Fatal(err)
		}
		overwriteErr := idx.Add(ctx, 0, vectors[1]) // Attempt to overwrite with different vector

		if overwriteErr != nil {
			// Index rejects duplicate IDs — valid behavior
			review.Pass("duplicate_id", "Duplicate ID rejected with error", SeverityMedium,
				overwriteErr.Error())
		} else {
			// Index accepted the overwrite — verify the new vector is returned
			results, _ := idx.Search(ctx, vectors[1], 1, &index.HNSWSearchParams{EfSearch: 200})
			if len(results) > 0 && results[0].ID == 0 {
				review.Pass("duplicate_id", "Duplicate ID overwrites previous vector", SeverityMedium, "")
			} else {
				review.Fail("duplicate_id", "Duplicate ID handling", SeverityMedium,
					"Duplicate ID behavior unclear — may cause data inconsistency")
			}
		}
	})

	// Check 8: Stats accuracy
	t.Run("stats_accuracy", func(t *testing.T) {
		idx, err := index.Create("hnsw", dim, map[string]interface{}{
			"m": 16, "ef_construction": 200,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 50; i++ {
			if err := idx.Add(ctx, uint64(i), vectors[i]); err != nil {
				t.Fatalf("inserting vector %d: %v", i, err)
			}
		}
		stats := idx.Stats()
		if stats.Count == 50 {
			review.Pass("stats_accuracy", "Stats.Count matches inserted count", SeverityLow, "")
		} else {
			review.Fail("stats_accuracy", "Stats.Count matches inserted count", SeverityLow,
				fmt.Sprintf("Expected 50, got %d", stats.Count))
		}
	})

	review.Report(t)
}

