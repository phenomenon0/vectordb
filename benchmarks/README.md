# DeepData benchmarks

Harnesses, generators and historical reports. Numbers live in one place,
[docs/BENCHMARKS.md](../docs/BENCHMARKS.md); competitive positioning lives in
[docs/GAP_ANALYSIS.md](../docs/GAP_ANALYSIS.md). This file restates neither.

## Layout

```
benchmarks/
  README.md                    this file (review/product_manager_test.go:124 stats it)
  ddload/main.go               equal-effort Go client against DeepData V3; the headline evidence
  ddload-qdrant/main.go        same client shape against Qdrant
  recall_test.py               real-dataset recall through the Python client
  comprehensive_bench.py       feature sweep over HTTP/gRPC configs, optional Qdrant comparison
  mega_bench.py                autonomous multi-VDB sweep with checkpoint/resume
  test_mega_bench.py           unit tests for mega_bench.py
  download_datasets.py         SIFT / GloVe / code datasets into ~/.vectordb_bench as .fvecs
  competitive/                 Go testing.B scenarios on internal/index (runner.go, results.go, competitor_baselines.go, scenarios/*_test.go)
    live/                      Python head-to-head harness: adapters/, compose file, results/
  testdata/                    vectors.go, ground_truth.go, corpus.go, metadata.go
  review/                      persona review tests: framework.go + five *_test.go
  vectordbbench/               VectorDBBench plugin (deepdata/), install.sh, run_all.sh, results/
  results/                     historical reports (2026-03-11_*.md, mega/); like live/results/ and vectordbbench/results/, records, not evidence
```

## Go: index-level scenarios (no server)

`competitive/` and `review/` import `internal/index` directly
(`competitive/runner.go:9`). Go commands need `GOTOOLCHAIN=go1.25.12`.

```bash
go test ./benchmarks/... -short                         # everything at reduced scales
go test ./benchmarks/review/ -v -short                  # the five persona reviews
# 27-cell sweep: 3 dims x 3 index types x 3 quantizers, 100K vectors each (dense_test.go:15-64); slow
go test ./benchmarks/competitive/scenarios/ -run '^$' -bench BenchmarkDense -benchtime=3s -short
go test ./benchmarks/competitive/scenarios/ -run '^$' -bench BenchmarkDense_HNSW_128d_100K -benchtime=3s
```

Index types other than HNSW and FLAT are not in the RC (non-RC; see the
non-goals in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)); the rows that
exercised IVF and DiskANN were deleted under SYS-03 together with those index
types, so every row below runs on an RC index type.

| Benchmark (`competitive/scenarios/`) | Measures |
|---|---|
| `BenchmarkDense` | search qps by dimension x index type x quantizer; HNSW across the fp16 and uint8 quantizers, which are non-RC |
| `BenchmarkDense_HNSW_128d_100K` | one HNSW configuration, 128d, 100K vectors |
| `BenchmarkRecall_HNSW_EfSweep` | recall vs throughput across `ef_search` |
| `BenchmarkInsert_Single`, `_Batch`, `_Parallel` | insert throughput on HNSW |
| `BenchmarkSparse_BM25_Insert`, `_Search`, `BenchmarkSparse_Corpus` | BM25 inverted index |
| `BenchmarkHybrid_RRF`, `_Weighted`, `_WeightSweep` | dense+sparse fusion overhead |
| `BenchmarkFiltered_HNSW` | metadata filter selectivity on HNSW |
| `BenchmarkConcurrent_SearchScale`, `_MixedReadWrite`, `_LatencyUnderLoad` | goroutine scaling and mixed load |
| `BenchmarkMemory` | bytes per vector |

| Test (`competitive/scenarios/`) | Checks |
|---|---|
| `TestRecall_HNSW` | recall@1/10/100 across `ef_search` |
| `TestMemoryFootprint` | memory per configuration |

| Persona review (`review/`) | File | Checks |
|---|---|---|
| `TestDBEngineerReview` | `db_engineer_test.go` | insert/search/delete correctness, concurrent R/W, export/import, NaN/Inf, zero vector, duplicate id, stats |
| `TestMLResearcherReview` | `ml_researcher_test.go` | recall monotonicity and stability, cosine correctness, and quantization degradation; quantization is non-RC |
| `TestDevOpsSREReview` | `devops_sre_test.go` | latency shape, memory growth, GC pressure, goroutine leaks, sustained load |
| `TestProductManagerReview` | `product_manager_test.go` | index coverage (HNSW and FLAT) and quantizer coverage (non-RC), export/import and stats APIs, this README, GAP_ANALYSIS |
| `TestSecurityAuditorReview` | `security_auditor_test.go` | negative/large k, NaN query, nil params, metadata injection, concurrent delete |

## Go: equal-effort clients (server-side truth)

`ddload` drives a running DeepData through the canonical V3 HTTP API with
4 workers and 2000-vector batches (`ddload/main.go:21-28`); it expects the
server on 127.0.0.1:8093 with the bearer token hard-coded at
`ddload/main.go:23`, tenant and collection `bench`. `ddload-qdrant` does the
same against Qdrant on 127.0.0.1:6333 (`ddload-qdrant/main.go:24`).
`-segments` sets the HNSW segment count; 0 keeps the server default
(`ddload/main.go:95`).

```bash
go run ./benchmarks/ddload        -base ~/.vectordb_bench/dataset/sift/sift_base_100k_norm.fvecs [-segments N]
go run ./benchmarks/ddload-qdrant -base ~/.vectordb_bench/dataset/sift/sift_base_100k_norm.fvecs
```

## Python harnesses (client cost included)

```bash
python3 benchmarks/download_datasets.py --dataset sift       # choices: sift, glove, code
python3 benchmarks/recall_test.py --vdb deepdata --dataset sift-100k   # --json, --skip-build, --port, --n-search, --concurrency, --duration
python3 benchmarks/comprehensive_bench.py --quick            # or --dataset, --config, --compare-qdrant, --json
python3 benchmarks/mega_bench.py --quick                     # or --vdb, --dataset, --resume, --report-only
python3 -m unittest -q benchmarks/test_mega_bench.py         # what scripts/hardening_check.sh benchmark-unit runs
```

`competitive/live/` is a five-phase pipeline: `prepare_data.py` chunks the
repo's Go sources, `embed_data.py` embeds them through the OpenAI API
(text-embedding-3-small), `compute_ground_truth.py` brute-forces top-100,
`benchmark.py --all` (or `--suite`, `--vdb`) runs the suites against the
adapters, `report.py` renders the report. `docker-compose.benchmark.yml`
pins Weaviate 1.28.4, Milvus 2.5.4, Qdrant 1.12.5 and Chroma 1.0.12.

`vectordbbench/` plugs DeepData into VectorDBBench: `install.sh` clones it to
/tmp/VectorDBBench and registers `vectordbbench/deepdata/`, `run_all.sh`
starts the competitor set with podman compose, `run_comprehensive.py --all`
(or `--dataset`, `--vdb`, `--report-only`) runs the sweep, and
`BENCHMARK_CONFIG.md` lists the environment variables a fair run must set.
