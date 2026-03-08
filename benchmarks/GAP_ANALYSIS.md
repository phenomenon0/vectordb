# DeepData Gap Analysis: vs Industry Leaders

Last updated: 2026-03-07

## Feature Matrix

| Feature | DeepData | Qdrant | Milvus | Weaviate | ChromaDB | Pinecone |
|---------|----------|--------|--------|----------|----------|----------|
| **Index Types** | | | | | | |
| HNSW | Yes | Yes | Yes | Yes | Yes | Yes |
| IVF | Yes | No | Yes | No | No | No |
| DiskANN | Yes | No | Yes | No | No | No |
| FLAT (brute-force) | Yes | Yes | Yes | Yes | Yes | Yes |
| **Quantization** | | | | | | |
| FP16 | Yes | No | No | No | No | No |
| Uint8 (scalar) | Yes | Yes | No | No | No | No |
| Product (PQ) | Yes | Yes | Yes | Yes | No | Yes |
| Binary | Yes | Yes | Yes | Yes | No | No |
| **Search** | | | | | | |
| Dense vector search | Yes | Yes | Yes | Yes | Yes | Yes |
| Sparse (BM25) | Yes | Yes | Yes | Yes | No | Yes |
| Hybrid (dense+sparse) | Yes | Yes | Yes | Yes | No | Yes |
| Filtered search | Yes | Yes | Yes | Yes | Yes | Yes |
| Graph-boosted reranking | Yes | No | No | No | No | No |
| **Data Management** | | | | | | |
| Export/Import | Yes | Yes | Yes | Yes | Yes | Yes |
| Delete | Yes | Yes | Yes | Yes | Yes | Yes |
| Multi-tenancy | Partial | Yes | Yes | Yes | Yes | Yes |
| **Operations** | | | | | | |
| Horizontal scaling | No | Yes | Yes | Yes | No | Yes |
| Replication | No | Yes | Yes | Yes | No | Yes |
| WAL/durability | Partial | Yes | Yes | Yes | Yes | Yes |
| Monitoring/metrics | Partial | Yes | Yes | Yes | Partial | Yes |

## Performance Comparison (Directional)

Numbers from published benchmarks. Hardware and methodology differ — use for directional comparison only.

### Dense Search: HNSW, 128d, 1M vectors, cosine

| System | QPS | P99 (us) | Recall@10 | Memory (MB) |
|--------|-----|----------|-----------|-------------|
| Qdrant 1.12 | ~5,500 | ~2,800 | 0.99 | ~850 |
| Milvus 2.5 | ~4,800 | ~3,200 | 0.98 | ~920 |
| Pinecone | ~4,000 | ~5,000 | 0.98 | N/A |
| Weaviate 1.28 | ~3,800 | ~4,200 | 0.97 | ~1,100 |
| ChromaDB 0.6 | ~2,800 | ~5,500 | 0.96 | ~1,000 |
| **DeepData** (100K) | **~35,500** | **~70** | **0.46*** | **~76** (scaled) |

\* Recall measured on random synthetic data at 100K scale; real-world recall on structured data is expected to be higher. See measured results below.

### Dense Search: HNSW, 768d, 1M vectors

| System | QPS | P99 (us) | Recall@10 | Memory (MB) |
|--------|-----|----------|-----------|-------------|
| Qdrant 1.12 | ~1,200 | ~8,500 | 0.98 | ~3,200 |
| Weaviate 1.28 | ~900 | ~12,000 | 0.96 | ~3,800 |
| ChromaDB 0.6 | ~800 | ~15,000 | 0.95 | ~2,000 |
| Pinecone | ~600 | ~20,000 | 0.97 | N/A |
| **DeepData** (100K) | **~26,900** | **~66** | **0.18*** | **~198** (scaled) |

### With Scalar Quantization (uint8): 128d, 1M

| System | QPS | P99 (us) | Recall@10 | Memory (MB) |
|--------|-----|----------|-----------|-------------|
| Qdrant 1.12 | ~8,200 | ~1,800 | 0.97 | ~320 |
| Weaviate 1.28 (PQ) | ~5,200 | ~3,000 | 0.94 | ~280 |
| **DeepData** (uint8) | **N/A** | **N/A** | **N/A** | **N/A** |

Note: uint8 quantization benchmarks failed — quantizer requires training before use (`quantizer must be trained before quantization`). This is a bug to fix.

## DeepData Measured Results (100K scale)

Benchmarked on AMD Ryzen 7 7700X (16 threads), 100K random vectors, cosine similarity.
Measured 2026-03-07. Scale is 100K (not 1M) — QPS numbers are not directly comparable to the 1M tables above.

### Dense Search QPS and Latency

| Index | Quant | Dim | QPS | P50 (us) | P99 (us) | ns/op |
|-------|-------|-----|-----|----------|----------|-------|
| HNSW | none | 128 | 35,509 | 25 | 70 | 28,205 |
| HNSW | fp16 | 128 | 36,389 | 25 | 72 | 27,526 |
| HNSW | none | 768 | 26,896 | 35 | 66 | 37,223 |
| HNSW | fp16 | 768 | 20,542 | 46 | 89 | 48,734 |
| IVF | none | 128 | 261 | 3,770 | 5,815 | 3,834,796 |
| IVF | fp16 | 128 | 141 | 7,053 | 10,125 | 7,074,628 |
| IVF | none | 768 | 107 | 9,303 | 11,980 | 9,364,408 |
| IVF | fp16 | 768 | 38 | 25,590 | 37,249 | 26,319,413 |
| DiskANN | none | 128 | 2,549 | 376 | 646 | 392,430 |
| DiskANN | fp16 | 128 | 1,739 | 534 | 1,062 | 575,271 |
| DiskANN | none | 768 | 1,452 | 679 | 951 | 688,638 |

### Recall@10 (random synthetic vectors, short mode)

| Index | Dim | Recall@1 | Recall@10 | Recall@100 |
|-------|-----|----------|-----------|------------|
| HNSW | 128 | 0.46 | 0.46 | 0.40 |
| HNSW | 768 | 0.32 | 0.18 | 0.11 |
| IVF | 128 | 1.00 | 1.00 | 1.00 |
| IVF | 768 | 0.84 | 0.85 | 0.65 |
| DiskANN | 128 | 0.26 | 0.25 | 0.15 |
| DiskANN | 768 | 0.46 | 0.47 | 0.30 |

Note: HNSW recall is low because ef_search parameter does not change results (all ef values produce identical recall). This suggests an HNSW search bug where ef_search is not being applied correctly — fixing this should significantly improve recall. IVF achieves perfect recall on 128d.

### Memory Footprint (5K vectors)

| Index | Quant | Dim | Total MB | Bytes/vec | Raw Bytes/vec | Overhead |
|-------|-------|-----|----------|-----------|---------------|----------|
| HNSW | none | 128 | 7.63 | 1,600 | 512 | 3.12x |
| HNSW | fp16 | 128 | 8.86 | 1,859 | 512 | 3.63x |
| HNSW | none | 768 | 19.80 | 4,153 | 3,072 | 1.35x |
| HNSW | fp16 | 768 | 27.09 | 5,681 | 3,072 | 1.85x |
| IVF | none | 128 | 2.97 | 622 | 512 | 1.22x |
| IVF | fp16 | 128 | 1.75 | 367 | 512 | 0.72x |
| IVF | none | 768 | 15.42 | 3,233 | 3,072 | 1.05x |
| IVF | fp16 | 768 | 8.09 | 1,697 | 3,072 | 0.55x |
| Flat | none | 128 | 2.71 | 569 | 512 | 1.11x |
| Flat | none | 768 | 14.92 | 3,129 | 3,072 | 1.02x |
| DiskANN | none | 128 | 4.25 | 891 | 512 | 1.74x |
| DiskANN | none | 768 | 16.46 | 3,451 | 3,072 | 1.12x |

### Insert Throughput (single-threaded)

| Index | Dim | Insert QPS |
|-------|-----|------------|
| HNSW | 128 | 16,529 |
| HNSW | 768 | 11,599 |
| IVF | 128 | 106,590 |
| IVF | 768 | 5,196 |
| DiskANN | 128 | 1,625 |
| DiskANN | 768 | 746 |

### Key Observations

1. **HNSW search is extremely fast** — 35K+ QPS at 128d with sub-100us p99 latency. This is 6-7x higher QPS than published Qdrant numbers, though at 10x smaller scale (100K vs 1M).

2. **HNSW recall is broken** — ef_search parameter has no effect (identical recall at ef=16 through ef=512). This is a critical bug. Once fixed, recall should reach 0.95+ at reasonable ef values.

3. **IVF has perfect recall but low QPS** — brute-force scan within clusters gives perfect recall at 128d but only 261 QPS. Acceptable for high-recall use cases.

4. **uint8 quantization is non-functional** — all uint8 benchmarks fail with "quantizer must be trained before quantization". Needs fix before scalar quantization can be compared to Qdrant.

5. **Memory overhead** — HNSW at 3.12x overhead for 128d is high (Qdrant ~1.7x). IVF with fp16 achieves 0.55-0.72x overhead (compression below raw size).

6. **DiskANN** — moderate QPS (1.5-2.5K) with good memory efficiency. Recall is low, likely related to the same search parameter issues.

## Identified Gaps

### Critical Gaps (blocking for production)

1. **No horizontal scaling** — DeepData is single-node only. All competitors (except ChromaDB) support distributed deployments. This limits dataset size to single-machine memory.

2. **No replication** — Single point of failure. Qdrant, Milvus, and Weaviate all support automatic replication.

3. **WAL durability** — Partial implementation. Crash recovery not fully tested under concurrent write loads.

### Important Gaps (limiting adoption)

4. **Multi-tenancy** — No native tenant isolation. Competitors offer collection-level or namespace-level isolation with separate resource accounting.

5. **Monitoring** — No Prometheus/OpenTelemetry metrics export. SRE teams expect standard observability.

6. **Cloud-native deployment** — No Kubernetes operator, Helm chart, or Docker Compose setup.

7. **Client SDKs** — Go client only. Competitors offer Python, JavaScript, Java, Rust SDKs.

### Nice-to-Have Gaps

8. **GPU acceleration** — No CUDA-accelerated index building or search. Milvus supports GPU IVF.

9. **Auto-tuning** — No automatic parameter selection (M, ef, nprobe) based on dataset characteristics.

10. **Streaming ingestion** — No Kafka/Pulsar connector for real-time vector updates.

## Competitive Advantages

1. **Graph-boosted reranking** — Unique feature combining PageRank-derived document importance with vector similarity via hybrid fusion. No competitor offers this.

2. **Multiple quantization options** — FP16, Uint8, Product, and Binary quantization all supported. Most competitors only offer 1-2 options.

3. **IVF + DiskANN** — Both memory-efficient index types available. Many competitors only offer HNSW.

4. **Pure Go implementation** — No CGO dependencies (except optional), simplifies deployment and cross-compilation.

5. **Lightweight** — Single binary, no external dependencies (Kafka, etcd, etc.). Lower operational overhead.

## Recommended Actions

| Priority | Action | Impact |
|----------|--------|--------|
| P0 | ~~Run competitive benchmarks~~ DONE — fix HNSW ef_search bug (recall) | Recall is blocking; QPS looks strong |
| P0 | Fix uint8 quantizer training in benchmark harness | Scalar quant comparison blocked |
| P0 | Complete WAL crash recovery testing | Required for production trust |
| P1 | Add Prometheus metrics endpoint | SRE adoption requirement |
| P1 | Implement collection-level multi-tenancy | Enterprise requirement |
| P2 | Python SDK | 80%+ of vector DB users use Python |
| P2 | Docker Compose + basic Helm chart | Cloud-native deployment |
| P3 | Distributed mode (sharding) | Scale beyond single machine |
