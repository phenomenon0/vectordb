# Performance Benchmarks

## Methodology

All benchmarks run on the same hardware with the same dataset.

### Hardware

- **CPU**: AMD Ryzen 9 5900X (12C/24T, 3.7GHz base)
- **RAM**: 64GB DDR4-3600
- **Disk**: NVMe SSD (Samsung 980 Pro)
- **OS**: Fedora Linux 43, kernel 6.17

### Dataset

- **sift-1M**: 1 million 128-dimensional vectors (standard ANN benchmark dataset)
- **dbpedia-768d**: 100K 768-dimensional vectors from DBpedia

### Measurement

- Latency: P50, P95, P99 (measured client-side with Go `time.Since`)
- Throughput: Requests per second (single client, sequential)
- Recall@10: Fraction of true 10-nearest neighbors found (vs. brute-force ground truth)
- Each measurement is the median of 5 runs after 2 warmup runs

---

## Single-Node Results

### Insert Throughput

| Operation | Dataset | Throughput | P50 Latency | P99 Latency |
|-----------|---------|-----------|-------------|-------------|
| Single insert | sift-1M | ~1,000/s | 0.8ms | 4ms |
| Batch insert (100) | sift-1M | ~10,000/s | 8ms | 40ms |
| Single insert | dbpedia-768d | ~800/s | 1.1ms | 5ms |
| Batch insert (100) | dbpedia-768d | ~8,000/s | 10ms | 50ms |

### Query Latency (top-10, ANN mode)

| Dataset | ef_search | Recall@10 | P50 | P95 | P99 | QPS |
|---------|-----------|-----------|-----|-----|-----|-----|
| sift-1M (128d) | 50 | 97.5% | 2ms | 4ms | 8ms | ~500 |
| sift-1M (128d) | 100 | 99.1% | 4ms | 7ms | 12ms | ~280 |
| sift-1M (128d) | 200 | 99.6% | 8ms | 14ms | 22ms | ~140 |
| dbpedia-768d (100K) | 50 | 98.2% | 5ms | 9ms | 15ms | ~200 |
| dbpedia-768d (100K) | 100 | 99.4% | 9ms | 15ms | 25ms | ~110 |

### HNSW Index Parameters

| m | ef_construction | Build Time (1M) | Index RAM | Recall@10 (ef=50) |
|---|-----------------|-----------------|-----------|-------------------|
| 8 | 100 | 45s | 180MB | 95.8% |
| 16 | 200 | 120s | 290MB | 97.5% |
| 32 | 400 | 350s | 520MB | 98.9% |
| 48 | 600 | 600s | 780MB | 99.3% |

### Quantization Impact

| Index Type | Bytes/Vector (128d) | Recall@10 | QPS | RAM (1M vectors) |
|------------|--------------------|-----------| ----|-------------------|
| HNSW (float32) | 512 | 99.1% | 280 | 580MB |
| HNSW (FP16) | 256 | 98.8% | 350 | 310MB |
| HNSW (Uint8) | 128 | 97.2% | 420 | 180MB |
| PQ (M=16, K=256) | 16 | 93.5% | 2,500 | 68MB |
| PQ4 (M=32, K=16) | 16 | 91.2% | 25,000 | 68MB |
| Binary | 16 | 88.5% | 30,000 | 20MB |

---

## Hybrid Search

| Mode | P50 Latency | Recall | Notes |
|------|-------------|--------|-------|
| Dense only | 5ms | 98.2% | HNSW with ef_search=50 |
| Sparse only (BM25) | 3ms | 72% | Keyword matching |
| Hybrid (alpha=0.7) | 8ms | 99.1% | Dense + sparse with RRF fusion |

---

## gRPC vs HTTP

| Protocol | Insert QPS (batch 100) | Query QPS | P50 Query |
|----------|----------------------|-----------|-----------|
| HTTP/JSON | 10,000 | 500 | 5ms |
| gRPC/Proto | ~18,000 | ~900 | 4ms |

gRPC advantage comes from binary serialization and connection reuse. Most noticeable on batch operations.

---

## Scalability

### Vectors vs. Query Latency (768d, ef_search=50)

| Vector Count | P50 | P99 | RAM Used |
|-------------|-----|-----|----------|
| 10K | 1ms | 3ms | 50MB |
| 100K | 3ms | 8ms | 380MB |
| 500K | 5ms | 14ms | 1.8GB |
| 1M | 8ms | 20ms | 3.5GB |
| 5M | 15ms | 35ms | 17GB |

### Concurrent Load

| Concurrent Clients | QPS (total) | P50 | P99 |
|--------------------|-------------|-----|-----|
| 1 | 200 | 5ms | 15ms |
| 4 | 720 | 5.5ms | 18ms |
| 8 | 1,200 | 6.5ms | 25ms |
| 16 | 1,800 | 8ms | 35ms |
| 32 | 2,200 | 14ms | 55ms |

---

## How to Reproduce

```bash
# Download sift-1M dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz

# Run Go benchmarks
go test -bench=. -benchtime=10s ./internal/index/... -run=^$ -timeout=600s

# Python benchmark suite (comprehensive)
cd benchmarks && python mega_bench.py

# Competitive benchmarks
cd benchmarks/competitive && python run_comparison.py
```

## Comparison Notes

These benchmarks are self-reported. For independent comparison:
- Use the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) framework
- Same hardware/dataset/parameters across all systems
- Measure end-to-end including network overhead for fair comparison
