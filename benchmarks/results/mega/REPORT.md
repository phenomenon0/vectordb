# Mega Benchmark: Wide-Sweep All-VDB Results

**Report generated:** 2026-07-17 22:43  |  **CPU:** AMD Ryzen 7 7700X 8-Core Processor
**Systems:** DeepData, Qdrant, Weaviate, Milvus, ChromaDB
**Additional DeepData baseline:** HTTP search at full precision; the canonical gRPC row enables float16 quantization, so this is not a transport-only comparison.
**Coverage:** 108/108 matrix cells  |  **Requested queries:** up to 100 per dataset
**Method:** Recall@10 and latency@100 come from top-100 searches (effective ef is at least 100); QPS@10 is a 5-second sequential loop using the requested ef.

**Provenance:** resumed legacy checkpoint; 90 retained rows predate per-cell timestamps.
**Manifest created:** 2026-07-17T21:39:41-0500  |  **Python:** 3.14.2
---

## Executive Summary

> This report covers the benchmark repair only. For overall product readiness, see [DeepData Pre-Release Status](../../../docs/PRE_RELEASE_STATUS.md).

- **Matrix status:** 108/108 intended cells complete; 0 missing, 0 failed, and 0 pending reruns.
- **Scope:** 6 benchmark targets across 3 datasets and 6 `ef_search` settings.
- **Failure-mode validation:** 7/7 DeepData probes passed.
- **Milvus repair:** source vector IDs are preserved, data is flushed before HNSW creation, indexed rows are verified before loading, and effective `ef` values are recorded.
- **Harness hardening:** checkpoints are atomic and deduplicated; reruns are scoped and preserve last-good rows; manifests validate datasets, dependencies, and intended cells; concurrent runs are locked.

## What Is Left for This Benchmark Repair

- **Required:** nothing remains for the current benchmark repair; the intended matrix and recorded failure-mode suite are complete.
- **Optional provenance upgrade:** run a fresh 108-cell sweep to replace the 90 retained legacy rows with uniformly timestamped measurements.
- **Optional apples-to-apples transport test:** rerun DeepData gRPC and HTTP with identical quantization; the current gRPC float16 and HTTP full-precision rows are not transport-only comparisons.
- **Optional statistical tightening:** reuse one Milvus index per dataset for the full `ef_search` sweep and add repeated trials with variance or confidence intervals.

---

## sift-100k

### Recall@10 vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        | 0.9978 | 0.9967 | 1.0000 | 0.9989 | 0.9967 | 0.9989 |
| 32        | 0.9978 | 0.9967 | 1.0000 | 0.9989 | 0.9978 | 0.9967 |
| 64        | 0.9978 | 0.9978 | 1.0000 | 0.9989 | 0.9956 | 0.9989 |
| 128       | 0.9989 | 0.9989 | 1.0000 | 0.9989 | 0.9978 | 0.9989 |
| 256       | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 512       | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### P50 Latency@100 (ms) vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        |    0.5 |    1.5 |    1.9 |    1.2 |    1.2 |    1.5 |
| 32        |    0.5 |    1.6 |    1.8 |    1.4 |    0.8 |    1.6 |
| 64        |    0.5 |    1.6 |    1.8 |    1.3 |    0.8 |    1.4 |
| 128       |    0.6 |    1.6 |    1.9 |    1.4 |    0.8 |    1.5 |
| 256       |    0.9 |    2.2 |    2.2 |    1.8 |    1.0 |    1.6 |
| 512       |    1.8 |    2.9 |    1.9 |    2.5 |    1.2 |    1.8 |

### QPS@10 vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        |   6683 |   1844 |    547 |   2454 |    839 |   1138 |
| 32        |   5101 |   1718 |    535 |   2218 |   1732 |   1124 |
| 64        |   3782 |   1460 |    534 |   2085 |   1677 |   1086 |
| 128       |   2385 |   1208 |    538 |   1905 |   1561 |   1064 |
| 256       |   1418 |    914 |    545 |   1438 |   1365 |    998 |
| 512       |    675 |    512 |    546 |   1085 |   1134 |    942 |

## glove-100d

### Recall@10 vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        | 0.9889 | 0.9889 | 1.0000 | 0.9878 | 0.9878 | 0.9856 |
| 32        | 0.9900 | 0.9911 | 1.0000 | 0.9867 | 0.9911 | 0.9844 |
| 64        | 0.9911 | 0.9889 | 1.0000 | 0.9867 | 0.9889 | 0.9867 |
| 128       | 0.9944 | 0.9956 | 1.0000 | 0.9911 | 0.9967 | 0.9922 |
| 256       | 0.9989 | 0.9989 | 1.0000 | 0.9978 | 0.9967 | 0.9989 |
| 512       | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### P50 Latency@100 (ms) vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        |    0.3 |    1.3 |    1.0 |    1.0 |    0.7 |    1.2 |
| 32        |    0.3 |    1.4 |    0.9 |    1.2 |    0.8 |    1.2 |
| 64        |    0.4 |    1.4 |    1.1 |    1.0 |    0.9 |    1.2 |
| 128       |    0.4 |    1.4 |    1.0 |    1.2 |    0.8 |    1.3 |
| 256       |    0.6 |    1.6 |    1.1 |    1.2 |    0.8 |    1.3 |
| 512       |    1.4 |    2.6 |    1.1 |    1.7 |    0.9 |    1.5 |

### QPS@10 vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        |   8320 |   2034 |   1106 |   2651 |   1730 |   1123 |
| 32        |   6936 |   1961 |   1080 |   2536 |   1542 |   1148 |
| 64        |   5572 |   1673 |   1120 |   2421 |   1727 |   1097 |
| 128       |   3408 |   1453 |   1060 |   2158 |   1577 |   1078 |
| 256       |   1867 |   1053 |   1036 |   1702 |   1476 |   1036 |
| 512       |    852 |    599 |   1048 |   1305 |   1265 |    933 |

## code-562

### Recall@10 vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 32        | 1.0000 | 1.0000 | 1.0000 | 0.9975 | 1.0000 | 1.0000 |
| 64        | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 128       | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 |
| 256       | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 512       | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### P50 Latency@100 (ms) vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        |    0.3 |   11.5 |    1.1 |    1.1 |    0.8 |    1.4 |
| 32        |    0.3 |   11.2 |    1.1 |    1.1 |    0.7 |    1.4 |
| 64        |    0.3 |   11.2 |    1.2 |    1.1 |    0.8 |    1.5 |
| 128       |    0.3 |   11.1 |    1.1 |    1.0 |    0.8 |    1.4 |
| 256       |    0.4 |   11.7 |    1.2 |    1.2 |    0.8 |    1.5 |
| 512       |    0.6 |   11.6 |    1.1 |    1.2 |    0.8 |    1.6 |

### QPS@10 vs ef_search

| ef_search | deepdata-grpc | deepdata-http | qdrant | weaviate | milvus | chromadb |
|-----------|--------|--------|--------|--------|--------|--------|
| 16        |   6102 |    410 |    939 |   2457 |   1481 |    979 |
| 32        |   5475 |    396 |    887 |   2295 |   1491 |    894 |
| 64        |   4401 |    390 |    926 |   2258 |   1452 |    904 |
| 128       |   3538 |    384 |    932 |   2089 |   1495 |    898 |
| 256       |   2601 |    377 |    917 |   1948 |   1490 |    883 |
| 512       |   1879 |    352 |    920 |   1692 |   1435 |    842 |

---

## Best ef_search per VDB and Dataset (highest QPS@10 with R@10 >= 0.95)

| VDB | Dataset | ef_search | R@10 | QPS | P50 ms |
|-----|---------|-----------|------|-----|--------|
| deepdata-grpc | sift-100k | 16 | 0.9978 | 6683 | 0.5 |
| deepdata-grpc | glove-100d | 16 | 0.9889 | 8320 | 0.3 |
| deepdata-grpc | code-562 | 16 | 1.0000 | 6102 | 0.3 |
| deepdata-http | sift-100k | 16 | 0.9967 | 1844 | 1.5 |
| deepdata-http | glove-100d | 16 | 0.9889 | 2034 | 1.3 |
| deepdata-http | code-562 | 16 | 1.0000 | 410 | 11.5 |
| qdrant | sift-100k | 16 | 1.0000 | 547 | 1.9 |
| qdrant | glove-100d | 64 | 1.0000 | 1120 | 1.1 |
| qdrant | code-562 | 16 | 1.0000 | 939 | 1.1 |
| weaviate | sift-100k | 16 | 0.9989 | 2454 | 1.2 |
| weaviate | glove-100d | 16 | 0.9878 | 2651 | 1.0 |
| weaviate | code-562 | 16 | 1.0000 | 2457 | 1.1 |
| milvus | sift-100k | 32 | 0.9978 | 1732 | 0.8 |
| milvus | glove-100d | 16 | 0.9878 | 1730 | 0.7 |
| milvus | code-562 | 128 | 1.0000 | 1495 | 0.8 |
| chromadb | sift-100k | 16 | 0.9989 | 1138 | 1.5 |
| chromadb | glove-100d | 32 | 0.9844 | 1148 | 1.2 |
| chromadb | code-562 | 16 | 1.0000 | 979 | 1.4 |

---

## Failure Mode Tests (DeepData)

| Test | Passed | Detail |
|------|--------|--------|
| empty_collection_search | PASS | status=200, docs=0 |
| k_greater_than_n | PASS | asked k=1000, got 3 from 3 vectors |
| zero_vector_query | PASS | status=200 |
| duplicate_id_import | PASS | status=200 |
| search_nonexistent_collection | PASS | status=500 |
| grpc_empty_collection_search | PASS | got 0 results |
| very_high_ef_search | PASS | ef_search=10000, status=200 |

**7/7 passed**

---
*Generated by `benchmarks/mega_bench.py`*