# VDB Competitive Benchmark Report

Real-world head-to-head benchmark: DeepData vs Weaviate vs Milvus vs Qdrant vs ChromaDB

- **Dataset**: DeepData Go source code (~800-1200 chunks)
- **Embeddings**: OpenAI text-embedding-3-small (1536d)
- **HNSW**: M=16, ef_construction=200, ef_search=128
- **Memory limit**: 2GB per service

## Summary

| Metric             | DEEPDATA     | WEAVIATE   | MILVUS   | QDRANT   | CHROMADB   |
|--------------------|--------------|------------|----------|----------|------------|
| Insert (docs/sec)  | 689.4        | 44.8       | 4.8      | 2313.9   | 463.8      |
| Recall@10          | 0.9900       | 1.0        | 1.0      | 1.0      | 0.9980     |
| Recall@100         | 0.9602       | 0.9946     | 1.0      | 1.0      | 0.9960     |
| Latency p50 (ms)   | 21.6ms       | 1.5ms      | 4.5ms    | 1.8ms    | 1.7ms      |
| Latency p95 (ms)   | 24.3ms       | 2.1ms      | 5.5ms    | 3.2ms    | 1.9ms      |
| QPS (top_k=10)     | 287.6        | 1809.4     | 262.1    | 813.1    | 683.1      |
| Filtered Recall@10 | 0.9920       | 1.0        | 1.0      | 1.0      | 1.0        |
| Hybrid Recall@10   | 0.9900       | 0.8200     | 1.0      | N/A      | N/A        |
| Memory (bytes/vec) | N/A          | N/A        | N/A      | N/A      | N/A        |
| Graph Search       | YES (unique) | No         | No       | No       | No         |

## Suite 1: Insert Throughput

| VDB      |   Total Docs |   Time (s) |   Docs/sec |
|----------|--------------|------------|------------|
| DEEPDATA |          562 |      0.815 |      689.4 |
| WEAVIATE |          562 |     12.535 |       44.8 |
| MILVUS   |          562 |    116.252 |        4.8 |
| QDRANT   |          562 |      0.243 |     2313.9 |
| CHROMADB |          562 |      1.212 |      463.8 |

## Suite 2: Search Recall & Latency

| VDB      |   R@1 |   R@10 |   R@100 |   p50ms |   p95ms |   p99ms |    QPS |
|----------|-------|--------|---------|---------|---------|---------|--------|
| DEEPDATA |     1 |  0.99  |  0.9602 |    21.6 |    24.3 |    26.2 |  287.6 |
| WEAVIATE |     1 |  1     |  0.9946 |     1.5 |     2.1 |     3.1 | 1809.4 |
| MILVUS   |     1 |  1     |  1      |     4.5 |     5.5 |   729.3 |  262.1 |
| QDRANT   |     1 |  1     |  1      |     1.8 |     3.2 |     3.2 |  813.1 |
| CHROMADB |     1 |  0.998 |  0.996  |     1.7 |     1.9 |    13.2 |  683.1 |

## Suite 3: Filtered Search

| VDB      | Filter               |   Recall@10 |   p50ms |   p95ms |
|----------|----------------------|-------------|---------|---------|
| DEEPDATA | {'package': 'index'} |       0.992 |    22.3 |    25.1 |
| WEAVIATE | {'package': 'index'} |       1     |     1.1 |     1.2 |
| MILVUS   | {'package': 'index'} |       1     |     3.5 |     3.9 |
| QDRANT   | {'package': 'index'} |       1     |     1.7 |     2   |
| CHROMADB | {'package': 'index'} |       1     |     2.2 |     2.6 |

## Suite 4: Hybrid Search

| VDB      | Hybrid R@10   | Hybrid p50ms   | Status        |
|----------|---------------|----------------|---------------|
| DEEPDATA | 0.9900        | 21.5           | ok            |
| WEAVIATE | 0.8200        | 1.6            | ok            |
| MILVUS   | 1.0           | 3.6            | ok            |
| QDRANT   | N/A           | N/A            | not_supported |
| CHROMADB | N/A           | N/A            | not_supported |

## Suite 5: Graph-Boosted Search

Graph-boosted search is a **DeepData-only differentiator**.
It combines knowledge graph entity relationships with vector similarity
using weighted fusion scoring. No competitor offers equivalent functionality.

## Suite 6: Memory Footprint

| VDB      | Docker RSS   | Bytes/Vector   |   Corpus Size |
|----------|--------------|----------------|---------------|
| DEEPDATA | N/A          | N/A            |           562 |
| WEAVIATE | N/A          | N/A            |           562 |
| MILVUS   | N/A          | N/A            |           562 |
| QDRANT   | N/A          | N/A            |           562 |
| CHROMADB | N/A          | N/A            |           562 |

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
