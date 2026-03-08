package scenarios

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/phenomenon0/vectordb/benchmarks/competitive"
	"github.com/phenomenon0/vectordb/benchmarks/testdata"
)

// Memory benchmarks: footprint per vector across index types and quantization levels.

func TestMemoryFootprint(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	scale := 10_000
	if testing.Short() {
		scale = 5_000
	}

	type memCase struct {
		Index  string
		Config map[string]interface{}
		Quant  string
	}

	cases := []memCase{
		{"hnsw", map[string]interface{}{"m": 16, "ef_construction": 200}, "none"},
		{"hnsw", map[string]interface{}{"m": 16, "ef_construction": 200, "quantization": map[string]interface{}{"type": "float16"}}, "fp16"},
		{"hnsw", map[string]interface{}{"m": 16, "ef_construction": 200, "quantization": map[string]interface{}{"type": "uint8"}}, "uint8"},
		{"ivf", map[string]interface{}{"nlist": 100, "metric": "cosine"}, "none"},
		{"ivf", map[string]interface{}{"nlist": 100, "metric": "cosine", "quantization": map[string]interface{}{"type": "float16"}}, "fp16"},
		{"flat", map[string]interface{}{"metric": "cosine"}, "none"},
		{"diskann", map[string]interface{}{"max_degree": 32, "ef_construction": 100, "memory_limit": 100000, "metric": "cosine"}, "none"},
	}

	for _, dim := range []int{128, 768} {
		vectors := testdata.GenerateClusteredVectors(scale, dim, 20, 0.15, rng)

		for _, tc := range cases {
			name := tc.Index + "_" + tc.Quant + "_" + strconv.Itoa(dim) + "d"
			t.Run(name, func(t *testing.T) {
				memMB, bytesPerVec, err := competitive.MeasureMemory(tc.Index, dim, tc.Config, vectors)
				if err != nil {
					t.Fatalf("measuring memory: %v", err)
				}

				rawBytesPerVec := float64(dim * 4) // fp32 baseline
				overhead := bytesPerVec / rawBytesPerVec

				t.Logf("Memory: %.2f MB | Bytes/vec: %.1f | Raw: %.0f | Overhead: %.2fx",
					memMB, bytesPerVec, rawBytesPerVec, overhead)

				// Sanity: index shouldn't use more than 10x raw vector size
				if overhead > 10.0 {
					t.Errorf("overhead %.2fx exceeds 10x threshold", overhead)
				}
			})
		}
	}
}

func BenchmarkMemory(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	scales := testdata.StandardScales(testing.Short())

	for _, sc := range scales {
		for _, dim := range []int{128, 768} {
			vectors := testdata.GenerateClusteredVectors(sc.Count, dim, sc.Clusters, sc.Spread, rng)

			name := "HNSW_" + strconv.Itoa(dim) + "d_" + sc.Name
			b.Run(name, func(b *testing.B) {
				memMB, bytesPerVec, err := competitive.MeasureMemory("hnsw", dim,
					map[string]interface{}{"m": 16, "ef_construction": 200}, vectors)
				if err != nil {
					b.Fatal(err)
				}
				b.ReportMetric(memMB, "mem_mb")
				b.ReportMetric(bytesPerVec, "bytes/vec")
			})
		}
	}
}
