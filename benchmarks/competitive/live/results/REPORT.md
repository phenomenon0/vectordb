# VDB Competitive Benchmark Report

Real-world head-to-head benchmark: DeepData vs Weaviate vs Milvus vs Qdrant vs ChromaDB

- **Dataset**: DeepData Go source code (~800-1200 chunks)
- **Embeddings**: OpenAI text-embedding-3-small (1536d)
- **HNSW**: M=16, ef_construction=200, ef_search=128
- **Memory limit**: 2GB per service

## Summary

| Metric             | DEEPDATA     | DEEPDATA-GRPC   | WEAVIATE   | MILVUS   | QDRANT   | CHROMADB   |
|--------------------|--------------|-----------------|------------|----------|----------|------------|
| Insert (docs/sec)  | 661.2        | N/A             | 44.8       | 4.7      | 2508.7   | 724.4      |
| Recall@10          | 1.0          | N/A             | 1.0        | 1.0      | 1.0      | 1.0        |
| Recall@100         | 0.9974       | N/A             | 0.9936     | 1.0      | 1.0      | 0.9956     |
| Latency p50 (ms)   | 21.7ms       | N/A             | 1.3ms      | 3.6ms    | 1.6ms    | 1.7ms      |
| Latency p95 (ms)   | 25.8ms       | N/A             | 2.1ms      | 4.2ms    | 2.2ms    | 1.9ms      |
| QPS (top_k=10)     | 273.3        | N/A             | 1828.1     | 272.3    | 799.1    | 674.8      |
| Filtered Recall@10 | 1.0          | N/A             | 1.0        | 1.0      | 1.0      | 1.0        |
| Hybrid Recall@10   | 1.0          | N/A             | 0.8200     | 1.0      | N/A      | N/A        |
| Memory (bytes/vec) | N/A          | N/A             | N/A        | N/A      | N/A      | N/A        |
| Graph Search       | YES (unique) | N/A             | No         | No       | No       | No         |

## Suite 1: Insert Throughput

| VDB           | Total Docs   | Time (s)   | Docs/sec   |
|---------------|--------------|------------|------------|
| DEEPDATA      | 562          | 0.85       | 661.2      |
| DEEPDATA-GRPC | ERROR        |            |            |
| WEAVIATE      | 562          | 12.55      | 44.8       |
| MILVUS        | 562          | 119.334    | 4.7        |
| QDRANT        | 562          | 0.224      | 2508.7     |
| CHROMADB      | 562          | 0.776      | 724.4      |

## Suite 2: Search Recall & Latency

| VDB           | R@1   | R@10   | R@100   | p50ms   | p95ms   | p99ms   | QPS    |
|---------------|-------|--------|---------|---------|---------|---------|--------|
| DEEPDATA      | 1.0   | 1.0    | 0.9974  | 21.7    | 25.8    | 34.5    | 273.3  |
| DEEPDATA-GRPC | N/A   | N/A    | N/A     | N/A     | N/A     | N/A     | N/A    |
| WEAVIATE      | 1.0   | 1.0    | 0.9936  | 1.3     | 2.1     | 3.3     | 1828.1 |
| MILVUS        | 1.0   | 1.0    | 1.0     | 3.6     | 4.2     | 626.5   | 272.3  |
| QDRANT        | 1.0   | 1.0    | 1.0     | 1.6     | 2.2     | 3.0     | 799.1  |
| CHROMADB      | 1.0   | 1.0    | 0.9956  | 1.7     | 1.9     | 13.5    | 674.8  |

## Suite 3: Filtered Search

| VDB           | Filter               | Recall@10   | p50ms   | p95ms   |
|---------------|----------------------|-------------|---------|---------|
| DEEPDATA      | {'package': 'index'} | 1.0         | 22.3    | 24.4    |
| DEEPDATA-GRPC |                      | N/A         | N/A     | N/A     |
| WEAVIATE      | {'package': 'index'} | 1.0         | 1.1     | 1.3     |
| MILVUS        | {'package': 'index'} | 1.0         | 3.7     | 4.3     |
| QDRANT        | {'package': 'index'} | 1.0         | 1.7     | 2.0     |
| CHROMADB      | {'package': 'index'} | 1.0         | 2.0     | 2.3     |

## Suite 4: Hybrid Search

| VDB           | Hybrid R@10   | Hybrid p50ms   | Status        |
|---------------|---------------|----------------|---------------|
| DEEPDATA      | 1.0           | 22.0           | ok            |
| DEEPDATA-GRPC | N/A           | N/A            | ok            |
| WEAVIATE      | 0.8200        | 1.7            | ok            |
| MILVUS        | 1.0           | 3.6            | ok            |
| QDRANT        | N/A           | N/A            | not_supported |
| CHROMADB      | N/A           | N/A            | not_supported |

## Suite 5: Graph-Boosted Search

Graph-boosted search is a **DeepData-only differentiator**.
It combines knowledge graph entity relationships with vector similarity
using weighted fusion scoring. No competitor offers equivalent functionality.

## Suite 6: Memory Footprint

| VDB           | Docker RSS   | Bytes/Vector   | Corpus Size   |
|---------------|--------------|----------------|---------------|
| DEEPDATA      | N/A          | N/A            | 562           |
| DEEPDATA-GRPC | N/A          | N/A            |               |
| WEAVIATE      | N/A          | N/A            | 562           |
| MILVUS        | N/A          | N/A            | 562           |
| QDRANT        | N/A          | N/A            | 562           |
| CHROMADB      | N/A          | N/A            | 562           |

## DeepData Analysis

### Wins
- **Graph search**: Unique feature with no competitor equivalent
- **Built-in hybrid**: Dense + sparse + graph fusion in a single query
- **Single binary**: No external dependencies (etcd, minio) unlike Milvus
- **Deployment simplicity**: One Docker image vs Milvus's 3-service stack

### Known Optimization Opportunities

| Area | Issue | Fix Complexity |
|------|-------|---------------|
| Insert API | v1 `/insert` re-embeds; v2 supports pre-computed vectors | Minor (use v2) |
| SCAN_THRESHOLD | Auto-switches to brute force for small collections | Minor (set env var) |
| Batch concurrency | `/batch_insert` may process sequentially | Medium |
| Memory | Stores full doc text in-memory | Medium (HYDRATION_COUNT) |
| Hybrid sparse | BM25 requires separate sparse insert | Minor |
| Graph extraction | Requires LLM calls, adds latency | Expected (unique feature) |
