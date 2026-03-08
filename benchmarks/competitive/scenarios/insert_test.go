package scenarios

import (
	"context"
	"math/rand"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/benchmarks/competitive"
	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/index"
)

// Insert throughput benchmarks: single, batch, and parallel insertion.

func BenchmarkInsert_Single(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	for _, dim := range []int{128, 768} {
		vectors := testdata.GenerateClusteredVectors(b.N+1000, dim, 20, 0.15, rng)
		for _, idx := range denseIndexes() {
			name := idx.Name + "_" + strconv.Itoa(dim) + "d"
			b.Run(name, func(b *testing.B) {
				competitive.MeasureInsertThroughput(b, idx.Type, dim, idx.Config, vectors)
			})
		}
	}
}

func BenchmarkInsert_Batch(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	batchSizes := []int{100, 1000, 10000}
	dim := 128
	vectors := testdata.GenerateClusteredVectors(100_000, dim, 30, 0.15, rng)

	for _, batch := range batchSizes {
		b.Run("batch="+strconv.Itoa(batch), func(b *testing.B) {
			ctx := context.Background()
			idx, err := index.Create("hnsw", dim, map[string]interface{}{
				"m": 16, "ef_construction": 200,
			})
			if err != nil {
				b.Fatal(err)
			}

			total := batch * b.N
			if total > len(vectors) {
				total = len(vectors)
			}

			b.ResetTimer()
			start := time.Now()
			for i := 0; i < total; i++ {
				if err := idx.Add(ctx, uint64(i), vectors[i%len(vectors)]); err != nil {
					b.Fatal(err)
				}
			}
			elapsed := time.Since(start)
			b.StopTimer()

			qps := float64(total) / elapsed.Seconds()
			b.ReportMetric(qps, "insert_qps")
		})
	}
}

func BenchmarkInsert_Parallel(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	vectors := testdata.GenerateClusteredVectors(100_000, dim, 30, 0.15, rng)
	workerCounts := []int{1, 4, 8, 16}

	for _, workers := range workerCounts {
		b.Run("workers="+strconv.Itoa(workers), func(b *testing.B) {
			ctx := context.Background()
			idx, err := index.Create("hnsw", dim, map[string]interface{}{
				"m": 16, "ef_construction": 200,
			})
			if err != nil {
				b.Fatal(err)
			}

			var counter atomic.Uint64
			b.ResetTimer()
			start := time.Now()

			var wg sync.WaitGroup
			perWorker := b.N / workers
			if perWorker == 0 {
				perWorker = 1
			}

			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := 0; i < perWorker; i++ {
						id := counter.Add(1)
						v := vectors[int(id)%len(vectors)]
						if err := idx.Add(ctx, id, v); err != nil {
							// Skip errors from concurrent inserts (expected for some indexes)
							continue
						}
					}
				}()
			}
			wg.Wait()
			elapsed := time.Since(start)
			b.StopTimer()

			total := counter.Load()
			qps := float64(total) / elapsed.Seconds()
			b.ReportMetric(qps, "insert_qps")
			b.ReportMetric(float64(workers), "workers")
		})
	}
}
