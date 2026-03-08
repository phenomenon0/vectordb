package scenarios

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/hybrid"
	"github.com/phenomenon0/vectordb/internal/index"
	"github.com/phenomenon0/vectordb/internal/sparse"
)

// Hybrid search benchmarks: dense+sparse fusion via RRF and weighted strategies.

func BenchmarkHybrid_RRF(b *testing.B) {
	benchmarkHybrid(b, hybrid.FusionRRF, "RRF")
}

func BenchmarkHybrid_Weighted(b *testing.B) {
	benchmarkHybrid(b, hybrid.FusionWeighted, "Weighted")
}

func benchmarkHybrid(b *testing.B, strategy hybrid.FusionStrategy, name string) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	vocabSize := 30_000
	scale := 20_000
	if testing.Short() {
		scale = 5_000
	}

	// Dense vectors
	vectors := testdata.GenerateClusteredVectors(scale, dim, 20, 0.15, rng)
	denseQueries := testdata.GenerateQueries(100, dim, 20, 0.15, rng)

	// Sparse documents
	sparseDocs := testdata.GenerateSparseDocuments(scale, vocabSize, 50, 200, rng)
	sparseQueries := testdata.GenerateSparseQueries(100, vocabSize, 5, 15, rng)

	// Build dense index
	ctx := context.Background()
	denseIdx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_search": 64, "ef_construction": 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range vectors {
		if err := denseIdx.Add(ctx, uint64(i), v); err != nil {
			b.Fatal(err)
		}
	}

	// Build sparse index
	sparseIdx := sparse.NewInvertedIndex(vocabSize)
	for i, doc := range sparseDocs {
		if err := sparseIdx.Add(ctx, uint64(i), doc); err != nil {
			b.Fatal(err)
		}
	}

	fusionParams := hybrid.DefaultFusionParams()
	fusionParams.Strategy = strategy

	b.Run(name, func(b *testing.B) {
		latencies := make([]time.Duration, 0, b.N)
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			start := time.Now()

			// Dense search
			denseResults, err := denseIdx.Search(ctx, denseQueries[i%len(denseQueries)], 20,
				&index.HNSWSearchParams{EfSearch: 64})
			if err != nil {
				b.Fatal(err)
			}

			// Sparse search
			sparseResults, err := sparseIdx.Search(ctx, sparseQueries[i%len(sparseQueries)], 20)
			if err != nil {
				b.Fatal(err)
			}

			// Convert to fusion format
			denseHybrid := make([]hybrid.SearchResult, len(denseResults))
			for j, r := range denseResults {
				denseHybrid[j] = hybrid.SearchResult{DocID: r.ID, Score: r.Score}
			}
			sparseHybrid := make([]hybrid.SearchResult, len(sparseResults))
			for j, r := range sparseResults {
				sparseHybrid[j] = hybrid.SearchResult{DocID: r.DocID, Score: r.Score}
			}

			// Fuse
			_, err = hybrid.HybridSearch(denseHybrid, sparseHybrid, fusionParams, 10)
			if err != nil {
				b.Fatal(err)
			}

			latencies = append(latencies, time.Since(start))
		}
		b.StopTimer()

		if len(latencies) > 0 {
			var total time.Duration
			for _, l := range latencies {
				total += l
			}
			qps := float64(len(latencies)) / (float64(total) / float64(time.Second))
			b.ReportMetric(qps, "qps")
		}
	})
}

func BenchmarkHybrid_WeightSweep(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	vocabSize := 30_000
	scale := 10_000
	if testing.Short() {
		scale = 5_000
	}

	vectors := testdata.GenerateClusteredVectors(scale, dim, 20, 0.15, rng)
	denseQueries := testdata.GenerateQueries(50, dim, 20, 0.15, rng)
	sparseDocs := testdata.GenerateSparseDocuments(scale, vocabSize, 50, 200, rng)
	sparseQueries := testdata.GenerateSparseQueries(50, vocabSize, 5, 15, rng)

	ctx := context.Background()
	denseIdx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_search": 64, "ef_construction": 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range vectors {
		if err := denseIdx.Add(ctx, uint64(i), v); err != nil {
			b.Fatalf("inserting dense vector %d: %v", i, err)
		}
	}
	sparseIdx := sparse.NewInvertedIndex(vocabSize)
	for i, doc := range sparseDocs {
		if err := sparseIdx.Add(ctx, uint64(i), doc); err != nil {
			b.Fatalf("inserting sparse doc %d: %v", i, err)
		}
	}

	weights := []struct {
		Dense  float32
		Sparse float32
		Label  string
	}{
		{0.9, 0.1, "dense_heavy"},
		{0.7, 0.3, "balanced"},
		{0.5, 0.5, "equal"},
		{0.3, 0.7, "sparse_heavy"},
	}

	for _, w := range weights {
		b.Run(w.Label, func(b *testing.B) {
			params := hybrid.FusionParams{
				Strategy:     hybrid.FusionWeighted,
				DenseWeight:  w.Dense,
				SparseWeight: w.Sparse,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				denseResults, err := denseIdx.Search(ctx, denseQueries[i%len(denseQueries)], 20,
					&index.HNSWSearchParams{EfSearch: 64})
				if err != nil {
					b.Fatal(err)
				}
				sparseResults, err := sparseIdx.Search(ctx, sparseQueries[i%len(sparseQueries)], 20)
				if err != nil {
					b.Fatal(err)
				}

				denseH := make([]hybrid.SearchResult, len(denseResults))
				for j, r := range denseResults {
					denseH[j] = hybrid.SearchResult{DocID: r.ID, Score: r.Score}
				}
				sparseH := make([]hybrid.SearchResult, len(sparseResults))
				for j, r := range sparseResults {
					sparseH[j] = hybrid.SearchResult{DocID: r.DocID, Score: r.Score}
				}
				if _, err := hybrid.HybridSearch(denseH, sparseH, params, 10); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
