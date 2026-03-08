# DeepData Benchmark Suite

Competitive benchmarks, recall measurements, and persona-based quality reviews for DeepData vector database.

## Quick Start

```bash
# Run all benchmarks (short mode, ~5 min)
cd DeepData && go test ./benchmarks/... -v -short

# Run persona reviews only
go test ./benchmarks/review/ -v -short

# Run competitive dense search benchmarks
go test ./benchmarks/competitive/scenarios/ -bench=BenchmarkDense -benchtime=3s -short

# Run specific benchmark
go test ./benchmarks/competitive/scenarios/ -bench=BenchmarkDense_HNSW_128d_100K -benchtime=3s
```

## Directory Structure

```
benchmarks/
  competitive/
    runner.go                    # Core benchmark execution engine
    results.go                   # JSON/markdown output, percentile calculations
    competitor_baselines.go      # Published numbers from Qdrant/Milvus/Weaviate/Chroma/Pinecone
    scenarios/
      dense_test.go              # 128d/768d/1536d x HNSW/IVF/DiskANN x quantization
      sparse_test.go             # BM25 inverted index benchmarks
      hybrid_test.go             # Dense+sparse fusion (RRF, weighted)
      filtered_test.go           # Metadata filter selectivity impact
      insert_test.go             # Single/batch/parallel insert throughput
      concurrent_test.go         # Multi-goroutine scaling (1-32 workers)
      memory_test.go             # Memory footprint per configuration
      recall_test.go             # Recall@1/10/100 vs brute-force ground truth
  testdata/
    vectors.go                   # Clustered synthetic vector generators
    ground_truth.go              # FLAT-index brute-force for recall reference
    corpus.go                    # Text generators (academic/code/product/legal/multilingual)
    metadata.go                  # Metadata generators for filtered search
  review/
    framework.go                 # PersonaReview scoring infrastructure
    db_engineer_test.go          # Data integrity, crash recovery, edge cases
    ml_researcher_test.go        # Recall curves, statistical significance
    devops_sre_test.go           # Latency shape, memory growth, GC pressure
    product_manager_test.go      # Feature matrix, documentation completeness
    security_auditor_test.go     # Input validation, resource boundaries
  GAP_ANALYSIS.md                # DeepData vs competitors comparison
  README.md                      # This file
```

## Benchmark Categories

### Competitive Benchmarks (`competitive/scenarios/`)

Standard `testing.B` benchmarks compatible with `go test -bench` and `benchstat`.

| Benchmark | What it measures |
|-----------|-----------------|
| `BenchmarkDense` | Search QPS across dimensions, index types, and quantization |
| `BenchmarkRecall_HNSW_EfSweep` | Recall vs throughput tradeoff at varying ef_search |
| `BenchmarkInsert_*` | Insert throughput: single, batch, parallel |
| `BenchmarkSparse_BM25_*` | BM25 inverted index performance |
| `BenchmarkHybrid_*` | Dense+sparse fusion overhead |
| `BenchmarkFiltered_*` | Impact of metadata filter selectivity |
| `BenchmarkConcurrent_*` | Multi-goroutine scaling and mixed workloads |
| `BenchmarkMemory` | Memory footprint per vector |

### Recall Tests (`competitive/scenarios/`)

Standard `testing.T` tests that verify recall quality.

| Test | What it checks |
|------|---------------|
| `TestRecall_HNSW` | Recall@1/10/100 at varying ef_search |
| `TestRecall_IVF` | Recall at varying nprobe |
| `TestRecall_DiskANN` | DiskANN recall quality |
| `TestMemoryFootprint` | Memory overhead per configuration |

### Persona Reviews (`review/`)

Five expert personas review DeepData from different angles:

| Persona | Focus Areas |
|---------|-------------|
| Database Engineer | Insert/search correctness, delete safety, concurrent R/W, export/import |
| ML Researcher | Recall monotonicity, quantization degradation, distance metric correctness |
| DevOps/SRE | P99/P50 ratio, memory growth, GC pressure, goroutine leaks, sustained load |
| Product Manager | Feature completeness, API surface, documentation, competitive positioning |
| Security Auditor | Input validation (oversized/NaN vectors), resource boundaries, concurrent safety |

## Key Design Decisions

1. **Clustered vectors** — Uniform random vectors make ANN trivially easy due to concentration of measure in high dimensions. Our generators use K-means clustered distributions for realistic recall/QPS tradeoffs.

2. **Ground truth via FLAT** — DeepData's own `FLATIndex` provides exact nearest neighbors. No external dependencies needed.

3. **Published baselines** — Competitor numbers are hard-coded from published benchmarks with URL citations. Self-contained and reproducible.

4. **Direct internal API** — Benchmarks import `internal/index` directly, avoiding HTTP overhead for pure algorithm comparison.

## Running Full Benchmarks

```bash
# Full benchmark suite (30+ minutes at default scale)
go test ./benchmarks/competitive/scenarios/ -bench=. -benchtime=5s -timeout=60m 2>&1 | tee results.txt

# Compare with benchstat
go test ./benchmarks/competitive/scenarios/ -bench=BenchmarkDense -benchtime=5s -count=5 > old.txt
# ... make changes ...
go test ./benchmarks/competitive/scenarios/ -bench=BenchmarkDense -benchtime=5s -count=5 > new.txt
benchstat old.txt new.txt
```

## Output

Benchmarks report custom metrics via `b.ReportMetric()`:
- `qps` — queries per second
- `p50_us` / `p99_us` — latency percentiles in microseconds
- `mem_mb` — memory usage in megabytes
- `recall@10` — recall at k=10 vs ground truth
- `insert_qps` — insert throughput
- `bytes/vec` — memory per vector
