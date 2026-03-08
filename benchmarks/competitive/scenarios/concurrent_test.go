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

// Concurrent benchmarks: multi-goroutine search and mixed read/write workloads.

func BenchmarkConcurrent_SearchScale(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	scale := 50_000
	if testing.Short() {
		scale = 10_000
	}

	vectors := testdata.GenerateClusteredVectors(scale, dim, 30, 0.15, rng)
	queries := testdata.GenerateQueries(200, dim, 30, 0.15, rng)

	ctx := context.Background()
	idx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_search": 64, "ef_construction": 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range vectors {
		if err := idx.Add(ctx, uint64(i), v); err != nil {
			b.Fatal(err)
		}
	}

	workerCounts := []int{1, 4, 8, 16, 32}

	for _, workers := range workerCounts {
		b.Run("workers="+strconv.Itoa(workers), func(b *testing.B) {
			var totalOps atomic.Int64
			var totalLatency atomic.Int64
			perWorker := b.N / workers
			if perWorker == 0 {
				perWorker = 1
			}

			b.ResetTimer()
			start := time.Now()

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()
					localRng := rand.New(rand.NewSource(int64(workerID)))
					for i := 0; i < perWorker; i++ {
						q := queries[localRng.Intn(len(queries))]
						t0 := time.Now()
						_, err := idx.Search(ctx, q, 10, &index.HNSWSearchParams{EfSearch: 64})
						elapsed := time.Since(t0)
						if err == nil {
							totalOps.Add(1)
							totalLatency.Add(int64(elapsed))
						}
					}
				}(w)
			}
			wg.Wait()
			wallTime := time.Since(start)
			b.StopTimer()

			ops := totalOps.Load()
			qps := float64(ops) / wallTime.Seconds()
			avgLatUs := float64(0)
			if ops > 0 {
				avgLatUs = float64(totalLatency.Load()) / float64(ops) / float64(time.Microsecond)
			}

			b.ReportMetric(qps, "qps")
			b.ReportMetric(avgLatUs, "avg_lat_us")
			b.ReportMetric(float64(workers), "workers")
		})
	}
}

func BenchmarkConcurrent_MixedReadWrite(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	scale := 20_000
	if testing.Short() {
		scale = 5_000
	}

	vectors := testdata.GenerateClusteredVectors(scale*2, dim, 30, 0.15, rng)
	queries := testdata.GenerateQueries(100, dim, 30, 0.15, rng)

	ctx := context.Background()
	idx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_search": 64, "ef_construction": 200,
	})
	if err != nil {
		b.Fatal(err)
	}

	// Pre-populate with half the vectors
	for i := 0; i < scale; i++ {
		if err := idx.Add(ctx, uint64(i), vectors[i]); err != nil {
			b.Fatal(err)
		}
	}

	readWriteRatios := []struct {
		Name      string
		ReadPct   int
		WritePct  int
	}{
		{"read_heavy_95_5", 95, 5},
		{"balanced_80_20", 80, 20},
		{"write_heavy_50_50", 50, 50},
	}

	for _, ratio := range readWriteRatios {
		b.Run(ratio.Name, func(b *testing.B) {
			var readOps, writeOps atomic.Int64
			var insertID atomic.Uint64
			insertID.Store(uint64(scale))

			workers := 8
			perWorker := b.N / workers
			if perWorker == 0 {
				perWorker = 1
			}

			b.ResetTimer()
			start := time.Now()

			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(wid int) {
					defer wg.Done()
					localRng := rand.New(rand.NewSource(int64(wid)))
					for i := 0; i < perWorker; i++ {
						if localRng.Intn(100) < ratio.ReadPct {
							q := queries[localRng.Intn(len(queries))]
							_, _ = idx.Search(ctx, q, 10, &index.HNSWSearchParams{EfSearch: 64})
							readOps.Add(1)
						} else {
							id := insertID.Add(1)
							v := vectors[int(id)%len(vectors)]
							_ = idx.Add(ctx, id, v)
							writeOps.Add(1)
						}
					}
				}(w)
			}
			wg.Wait()
			elapsed := time.Since(start)
			b.StopTimer()

			totalOps := readOps.Load() + writeOps.Load()
			b.ReportMetric(float64(totalOps)/elapsed.Seconds(), "total_qps")
			b.ReportMetric(float64(readOps.Load())/elapsed.Seconds(), "read_qps")
			b.ReportMetric(float64(writeOps.Load())/elapsed.Seconds(), "write_qps")
		})
	}
}

func BenchmarkConcurrent_LatencyUnderLoad(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	dim := 128
	scale := 50_000
	if testing.Short() {
		scale = 10_000
	}

	vectors := testdata.GenerateClusteredVectors(scale, dim, 30, 0.15, rng)
	queries := testdata.GenerateQueries(200, dim, 30, 0.15, rng)

	ctx := context.Background()
	idx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_search": 64, "ef_construction": 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range vectors {
		_ = idx.Add(ctx, uint64(i), v)
	}

	// Measure single-threaded baseline then multi-threaded
	for _, workers := range []int{1, 8} {
		b.Run("workers="+strconv.Itoa(workers), func(b *testing.B) {
			var mu sync.Mutex
			allLatencies := make([]time.Duration, 0, b.N)
			perWorker := b.N / workers
			if perWorker == 0 {
				perWorker = 1
			}

			b.ResetTimer()
			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(wid int) {
					defer wg.Done()
					local := make([]time.Duration, 0, perWorker)
					for i := 0; i < perWorker; i++ {
						q := queries[(wid*perWorker+i)%len(queries)]
						t0 := time.Now()
						_, _ = idx.Search(ctx, q, 10, &index.HNSWSearchParams{EfSearch: 64})
						local = append(local, time.Since(t0))
					}
					mu.Lock()
					allLatencies = append(allLatencies, local...)
					mu.Unlock()
				}(w)
			}
			wg.Wait()
			b.StopTimer()

			_, p50, _, p99, _ := competitive.LatencyStats(allLatencies)
			b.ReportMetric(p50, "p50_us")
			b.ReportMetric(p99, "p99_us")
			if p50 > 0 {
				b.ReportMetric(p99/p50, "p99/p50_ratio")
			}
		})
	}
}
