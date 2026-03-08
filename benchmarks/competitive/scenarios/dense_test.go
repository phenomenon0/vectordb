package scenarios

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/phenomenon0/vectordb/benchmarks/competitive"
	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/index"
)

// Dense search benchmarks: dimension × index type × quantization × scale.

var denseDimensions = []int{128, 768, 1536}

type indexSetup struct {
	Name   string
	Type   string
	Config map[string]interface{}
	Params index.SearchParams
}

func denseIndexes() []indexSetup {
	return []indexSetup{
		{
			Name: "HNSW",
			Type: "hnsw",
			Config: map[string]interface{}{
				"m": 16, "ef_search": 100, "ef_construction": 200,
			},
			Params: &index.HNSWSearchParams{EfSearch: 100},
		},
		{
			Name: "IVF",
			Type: "ivf",
			Config: map[string]interface{}{
				"nlist": 100, "nprobe": 10, "metric": "cosine",
			},
			Params: &index.IVFSearchParams{NProbe: 10},
		},
		{
			Name: "DiskANN",
			Type: "diskann",
			Config: map[string]interface{}{
				"max_degree": 32, "ef_construction": 100, "ef_search": 50,
				"memory_limit": 100000, "metric": "cosine",
			},
			Params: &index.DefaultSearchParams{},
		},
	}
}

type quantSetup struct {
	Name   string
	Config map[string]interface{}
}

func quantizers() []quantSetup {
	return []quantSetup{
		{Name: "none", Config: nil},
		{Name: "fp16", Config: map[string]interface{}{"type": "float16"}},
		{Name: "uint8", Config: map[string]interface{}{"type": "uint8"}},
	}
}

func BenchmarkDense(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	scales := testdata.StandardScales(testing.Short())

	for _, dim := range denseDimensions {
		for _, sc := range scales {
			vectors := testdata.GenerateClusteredVectors(sc.Count, dim, sc.Clusters, sc.Spread, rng)
			queries := testdata.GenerateQueries(100, dim, sc.Clusters, sc.Spread, rng)

			for _, idx := range denseIndexes() {
				for _, q := range quantizers() {
					config := mergeConfig(idx.Config, q.Config)
					name := idx.Name + "_" + q.Name + "_" + strconv.Itoa(dim) + "d_" + sc.Name

					b.Run(name, func(b *testing.B) {
						scenario := competitive.BenchmarkScenario{
							Name:         name,
							IndexType:    idx.Type,
							Dimension:    dim,
							Scale:        sc.Count,
							Config:       config,
							SearchParams: idx.Params,
							K:            10,
							NumQueries:   len(queries),
						}
						competitive.RunScenario(b, scenario, vectors, queries)
					})
				}
			}
		}
	}
}

func BenchmarkDense_HNSW_128d_100K(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	vectors := testdata.GenerateClusteredVectors(100_000, 128, 50, 0.15, rng)
	queries := testdata.GenerateQueries(100, 128, 50, 0.15, rng)

	scenario := competitive.BenchmarkScenario{
		Name:      "HNSW_128d_100K",
		IndexType: "hnsw",
		Dimension: 128,
		Scale:     100_000,
		Config: map[string]interface{}{
			"m": 16, "ef_search": 100, "ef_construction": 200,
		},
		SearchParams: &index.HNSWSearchParams{EfSearch: 100},
		K:            10,
	}
	competitive.RunScenario(b, scenario, vectors, queries)
}

func mergeConfig(base map[string]interface{}, quantConfig map[string]interface{}) map[string]interface{} {
	merged := make(map[string]interface{})
	for k, v := range base {
		merged[k] = v
	}
	if quantConfig != nil {
		merged["quantization"] = quantConfig
	}
	return merged
}

