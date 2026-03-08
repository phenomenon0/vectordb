package scenarios

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/filter"
	"github.com/phenomenon0/vectordb/internal/index"
)

// Filtered search benchmarks: measure impact of metadata filter selectivity.
// Selectivity = fraction of vectors that pass the filter.

func BenchmarkFiltered_HNSW(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	scale := 50_000
	if testing.Short() {
		scale = 10_000
	}

	vectors := testdata.GenerateClusteredVectors(scale, dim, 30, 0.15, rng)
	queries := testdata.GenerateQueries(100, dim, 30, 0.15, rng)

	// Build index with metadata
	ctx := context.Background()
	idx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_search": 100, "ef_construction": 200,
	})
	if err != nil {
		b.Fatal(err)
	}

	// TODO: attach metadata to vectors when index supports metadata storage.
	// metadata := testdata.GenerateUniformMetadata(scale, rng)
	for i, v := range vectors {
		if err := idx.Add(ctx, uint64(i), v); err != nil {
			b.Fatal(err)
		}
	}

	// Different selectivity levels
	selectivities := []struct {
		Name   string
		Filter filter.Filter
		Pct    string
	}{
		{"10pct", filter.Lt("score", 10.0), "10%"},
		{"50pct", filter.Lt("score", 50.0), "50%"},
		{"90pct", filter.Lt("score", 90.0), "90%"},
	}

	for _, sel := range selectivities {
		b.Run("selectivity="+sel.Pct, func(b *testing.B) {
			params := &index.HNSWSearchParams{
				EfSearch: 100,
				Filter:   sel.Filter,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				_, err := idx.Search(ctx, q, 10, params)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkFiltered_IVF(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	scale := 50_000
	if testing.Short() {
		scale = 10_000
	}

	vectors := testdata.GenerateClusteredVectors(scale, dim, 30, 0.15, rng)
	queries := testdata.GenerateQueries(100, dim, 30, 0.15, rng)

	ctx := context.Background()
	idx, err := index.Create("ivf", dim, map[string]interface{}{
		"nlist": 100, "nprobe": 10, "metric": "cosine",
	})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range vectors {
		if err := idx.Add(ctx, uint64(i), v); err != nil {
			b.Fatal(err)
		}
	}

	selectivities := []struct {
		Name string
		Pct  int
	}{
		{"10pct", 10},
		{"50pct", 50},
		{"90pct", 90},
	}

	for _, sel := range selectivities {
		b.Run("selectivity="+sel.Name, func(b *testing.B) {
			params := &index.IVFSearchParams{
				NProbe: 10,
				Filter: filter.Lt("score", float64(sel.Pct)),
			}

			latencies := make([]time.Duration, 0, b.N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				start := time.Now()
				_, err := idx.Search(ctx, q, 10, params)
				latencies = append(latencies, time.Since(start))
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()

			if len(latencies) > 0 {
				var total time.Duration
				for _, l := range latencies {
					total += l
				}
				mean := total / time.Duration(len(latencies))
				b.ReportMetric(float64(mean.Microseconds()), "mean_us")
			}
		})
	}
}
