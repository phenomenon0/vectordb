package scenarios

import (
	"context"
	"strconv"
	"testing"

	"github.com/phenomenon0/vectordb/benchmarks/competitive"
	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/index"
)

// Recall benchmarks: measure recall@1/10/100 against brute-force ground truth.

func TestRecall_HNSW(t *testing.T) {
	runRecallSuite(t, "hnsw", map[string]interface{}{
		"m": 16, "ef_construction": 200,
	}, hnswEfValues())
}

func TestRecall_IVF(t *testing.T) {
	runRecallSuite(t, "ivf", map[string]interface{}{
		"nlist": 100, "metric": "cosine",
	}, ivfNprobeValues())
}

func TestRecall_DiskANN(t *testing.T) {
	runRecallSuite(t, "diskann", map[string]interface{}{
		"max_degree": 32, "ef_construction": 100, "memory_limit": 100000, "metric": "cosine",
	}, diskannEfValues())
}

type paramSweep struct {
	Label  string
	Params index.SearchParams
}

func hnswEfValues() []paramSweep {
	return []paramSweep{
		{"ef=16", &index.HNSWSearchParams{EfSearch: 16}},
		{"ef=32", &index.HNSWSearchParams{EfSearch: 32}},
		{"ef=64", &index.HNSWSearchParams{EfSearch: 64}},
		{"ef=128", &index.HNSWSearchParams{EfSearch: 128}},
		{"ef=256", &index.HNSWSearchParams{EfSearch: 256}},
		{"ef=512", &index.HNSWSearchParams{EfSearch: 512}},
	}
}

func ivfNprobeValues() []paramSweep {
	return []paramSweep{
		{"nprobe=1", &index.IVFSearchParams{NProbe: 1}},
		{"nprobe=5", &index.IVFSearchParams{NProbe: 5}},
		{"nprobe=10", &index.IVFSearchParams{NProbe: 10}},
		{"nprobe=20", &index.IVFSearchParams{NProbe: 20}},
		{"nprobe=50", &index.IVFSearchParams{NProbe: 50}},
	}
}

func diskannEfValues() []paramSweep {
	return []paramSweep{
		{"default", &index.DefaultSearchParams{}},
	}
}

func runRecallSuite(t *testing.T, indexType string, baseConfig map[string]interface{}, sweeps []paramSweep) {
	scale := 10_000
	numQueries := 100
	if testing.Short() {
		scale = 5_000
		numQueries = 50
	}

	for _, dim := range []int{128, 768} {
		vectors, queries := testdata.GenerateClusteredDataset(scale, numQueries, dim, 20, 0.15, 42)

		// Compute ground truth
		gt, err := testdata.ComputeGroundTruth(vectors, queries, 100)
		if err != nil {
			t.Fatalf("computing ground truth: %v", err)
		}

		// Build index
		ctx := context.Background()
		idx, err := index.Create(indexType, dim, baseConfig)
		if err != nil {
			t.Fatalf("creating %s index: %v", indexType, err)
		}
		for i, v := range vectors {
			if err := idx.Add(ctx, uint64(i), v); err != nil {
				t.Fatalf("inserting vector %d: %v", i, err)
			}
		}

		for _, sweep := range sweeps {
			name := indexType + "/" + strconv.Itoa(dim) + "d/" + sweep.Label
			t.Run(name, func(t *testing.T) {
				r1, r10, r100, err := competitive.MeasureRecall(idx, queries, gt.Neighbors, 100, sweep.Params)
				if err != nil {
					t.Fatalf("measuring recall: %v", err)
				}
				t.Logf("Recall@1=%.4f  Recall@10=%.4f  Recall@100=%.4f", r1, r10, r100)

				// Minimum recall expectations (lower for short mode with small datasets)
				minRecall := 0.5
				if testing.Short() {
					minRecall = 0.1
				}
				if r10 < minRecall {
					t.Errorf("recall@10 = %.4f, expected >= %.1f", r10, minRecall)
				}
			})
		}
	}
}

func BenchmarkRecall_HNSW_EfSweep(b *testing.B) {
	scale := 50_000
	dim := 128
	if testing.Short() {
		scale = 10_000
	}

	vectors, queries := testdata.GenerateClusteredDataset(scale, 100, dim, 30, 0.15, 42)

	gt, err := testdata.ComputeGroundTruth(vectors, queries, 100)
	if err != nil {
		b.Fatalf("computing ground truth: %v", err)
	}

	ctx := context.Background()
	idx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_construction": 200,
	})
	if err != nil {
		b.Fatalf("creating index: %v", err)
	}
	for i, v := range vectors {
		if err := idx.Add(ctx, uint64(i), v); err != nil {
			b.Fatalf("inserting %d: %v", i, err)
		}
	}

	for _, ef := range []int{16, 32, 64, 128, 256} {
		b.Run("ef="+strconv.Itoa(ef), func(b *testing.B) {
			params := &index.HNSWSearchParams{EfSearch: ef}
			scenario := competitive.BenchmarkScenario{
				Name:         "recall_hnsw_ef" + strconv.Itoa(ef),
				IndexType:    "hnsw",
				Dimension:    dim,
				Scale:        scale,
				SearchParams: params,
				K:            10,
			}
			result := competitive.RunScenario(b, scenario, vectors, queries)

			// Also measure recall
			r1, r10, r100, _ := competitive.MeasureRecall(idx, queries, gt.Neighbors, 100, params)
			result.Recall1 = r1
			result.Recall10 = r10
			result.Recall100 = r100
			b.ReportMetric(r10, "recall@10")
		})
	}
}

