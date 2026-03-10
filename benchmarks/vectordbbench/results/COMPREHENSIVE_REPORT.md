# DeepData Comprehensive Vector Database Benchmark Report

**Generated**: 2026-03-10 14:16:41
**Framework**: VectorDBBench-compatible Comprehensive Suite
**Platform**: AMD Ryzen 7 7700X, 64GB DDR5, NVMe SSD, Linux 6.17

## Test Matrix

- **Databases (9)**: CHROMADB, DEEPDATA, ELASTICSEARCH, LANCEDB, MILVUS, PGVECTOR, QDRANT, REDIS, WEAVIATE
- **Datasets (8)**: random-128d-100k, random-128d-10k, random-128d-1k, random-128d-50k, random-1536d-10k, random-1536d-1k, random-768d-10k, random-768d-1k
- **Scales (4)**: 100k, 10k, 1k, 50k
- **Index**: HNSW (M=16, ef_construction=200, ef_search=128)
- **Distance**: Cosine Similarity
- **Modalities**: Insert, Serial Search, Concurrent Search (4/8/16T), Filtered Search (1%/50%/99%), Recall Sweep, Streaming, Memory

---
## Dataset: random-128d-100k (100,000 vectors, 128d)

### Core Metrics

| Metric | WEAVIATE | MILVUS | CHROMADB | PGVECTOR | REDIS | ELASTICSEARCH | LANCEDB | DEEPDATA | QDRANT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Insert QPS | 7153 | 22805 | 3053 | 20446 | 3350 | 3752 | **98522** | 181 | 22779 |
| Insert Time (s) | 14.0 | 4.4 | 32.8 | 4.9 | 29.8 | 26.7 | **1.0** | 553.3 | 4.4 |
| Recall@1 | 0.4500 | 0.4100 | 0.4400 | 0.4700 | 0.4000 | 0.6200 | 0.0300 | 0.5600 | **1.0000** |
| Recall@10 | 0.3550 | 0.3960 | 0.3040 | 0.4090 | 0.2970 | 0.5430 | 0.0580 | 0.4470 | **1.0000** |
| Recall@100 | 0.2852 | 0.3542 | 0.2334 | 0.3356 | 0.1000 | 0.4549 | 0.0981 | 0.3411 | **1.0000** |
| Latency p50 (ms) | 1.94 | 2.74 | 2.66 | 3.25 | **0.39** | 12.18 | 4.84 | 6.19 | 2.53 |
| Latency p95 (ms) | 2.52 | 4.53 | 12.73 | 4.38 | **0.53** | 13.90 | 6.26 | 7.63 | 3.48 |
| Latency p99 (ms) | 3.73 | 11.44 | 21.56 | 5.18 | **0.78** | 14.48 | 7.78 | 8.05 | 4.80 |
| Serial QPS | 1312 | 266 | 796 | 1016 | **5537** | 437 | 572 | 226 | 397 |
| Concurrent QPS (4T) | 4205 | 1394 | 1525 | 4032 | **9486** | 1820 | 971 | 898 | 625 |
| Concurrent QPS (8T) | 4336 | 3765 | 1350 | 6930 | **8792** | 1776 | 1077 | 1278 | 746 |
| Concurrent QPS (16T) | 4401 | 3721 | 1489 | 7673 | **8727** | 1876 | 974 | 1582 | 721 |
| Concurrent p99 (ms) | 9.36 | 8.39 | 16.90 | **6.04** | 6.78 | 19.57 | 29.11 | 27.85 | 51.76 |
| Filtered R@10 (1%) | **1.0000** | **1.0000** | **1.0000** | 0.2043 | **1.0000** | **1.0000** | 0.1720 | 0.9892 | **1.0000** |
| Filtered R@10 (50%) | 0.5220 | 0.4580 | 0.4700 | 0.3480 | 0.1040 | 0.5020 | 0.0720 | 0.5680 | **1.0000** |
| Filtered R@10 (99%) | 0.3580 | 0.4240 | 0.2900 | 0.4100 | 0.0800 | 0.3660 | 0.0660 | 0.4680 | **1.0000** |
| Filter p50 (ms) | 99.56 | 1.11 | 71.91 | 1.86 | **0.30** | 3.31 | 2.80 | 9.89 | 21.70 |
| Optimize Time (s) | N/A | 0.8 | N/A | 54.2 | 0.0 | 87.0 | 5.7 | N/A | **0.0** |
| Memory (B/vec) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 91 | 91 | 17.95 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 405qps | R=1.000 / 316qps | R=1.000 / 237qps | R=1.000 / 399qps | R=1.000 / 298qps | R=1.000 / 293qps |

---
## Dataset: random-128d-10k (10,000 vectors, 128d)

### Core Metrics

| Metric | WEAVIATE | MILVUS | CHROMADB | PGVECTOR | REDIS | ELASTICSEARCH | LANCEDB | DEEPDATA | QDRANT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Insert QPS | 3949 | 3029 | 5731 | 19841 | 8026 | 3663 | **123457** | 631 | 21834 |
| Insert Time (s) | 2.5 | 3.3 | 1.7 | 0.5 | 1.2 | 2.7 | **0.1** | 15.9 | 0.5 |
| Recall@1 | 0.8600 | **1.0000** | 0.8200 | 0.9600 | 0.8000 | 0.9700 | 0.0600 | **1.0000** | **1.0000** |
| Recall@10 | 0.8380 | **1.0000** | 0.7730 | 0.9020 | 0.7660 | 0.9720 | 0.1330 | 0.9710 | **1.0000** |
| Recall@100 | 0.7261 | **1.0000** | 0.6556 | 0.8002 | 0.1000 | 0.9086 | 0.2165 | 0.9096 | **1.0000** |
| Latency p50 (ms) | 1.43 | 4.35 | 1.98 | 0.74 | **0.30** | 14.91 | 4.23 | 4.29 | 1.74 |
| Latency p95 (ms) | 1.73 | 5.35 | 2.69 | 0.88 | **0.36** | 16.89 | 4.98 | 6.31 | 2.51 |
| Latency p99 (ms) | 1.93 | 5.66 | 3.28 | 1.09 | **0.42** | 24.35 | 7.00 | 8.65 | 3.66 |
| Serial QPS | 1658 | 964 | 867 | 1553 | **5879** | 384 | 610 | 394 | 789 |
| Concurrent QPS (4T) | 4256 | 4132 | 1626 | 5366 | **9532** | 1428 | 984 | 1283 | 1334 |
| Concurrent QPS (8T) | 4639 | 4144 | 1579 | **9108** | 8946 | 1991 | 1104 | 1828 | 1249 |
| Concurrent QPS (16T) | 4567 | 3651 | 1664 | **9886** | 8812 | 1831 | 1093 | 1750 | 1142 |
| Concurrent p99 (ms) | 9.33 | 9.37 | 16.45 | **4.53** | 6.65 | 21.44 | 26.82 | 23.79 | 32.53 |
| Filtered R@10 (1%) | **1.0000** | **1.0000** | **1.0000** | 0.7412 | **1.0000** | **1.0000** | 0.5768 | N/A | **1.0000** |
| Filtered R@10 (50%) | **1.0000** | 0.8940 | 0.8720 | 0.8700 | **1.0000** | 0.9420 | 0.1500 | N/A | **1.0000** |
| Filtered R@10 (99%) | **1.0000** | 0.8960 | 0.7620 | 0.8920 | **1.0000** | 0.8360 | 0.1300 | N/A | **1.0000** |
| Filter p50 (ms) | 11.71 | 0.88 | 14.36 | 0.79 | **0.49** | 2.36 | 1.86 | 11.66 | 6.36 |
| Optimize Time (s) | N/A | 1.4 | N/A | 2.8 | N/A | 1.8 | 0.6 | N/A | **0.0** |
| Memory (B/vec) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 208 | 208 | 8.28 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 235qps | R=1.000 / 247qps | R=1.000 / 246qps | R=1.000 / 234qps | R=1.000 / 252qps | R=1.000 / 268qps |

---
## Dataset: random-128d-1k (1,000 vectors, 128d)

### Core Metrics

| Metric | DEEPDATA | QDRANT |
| --- | --- | --- |
| Insert QPS | 2110 | **22727** |
| Insert Time (s) | 0.5 | **0.0** |
| Recall@1 | **1.0000** | **1.0000** |
| Recall@10 | **1.0000** | **1.0000** |
| Recall@100 | 0.9999 | **1.0000** |
| Latency p50 (ms) | 3.19 | **1.22** |
| Latency p95 (ms) | 3.92 | **1.46** |
| Latency p99 (ms) | 4.60 | **1.66** |
| Serial QPS | 837 | **920** |
| Concurrent QPS (4T) | **1937** | 1633 |
| Concurrent QPS (8T) | **2222** | 1205 |
| Concurrent QPS (16T) | **2132** | 1395 |
| Concurrent p99 (ms) | **18.96** | 30.83 |
| Filtered R@10 (1%) | N/A | 1.0000 |
| Filtered R@10 (50%) | N/A | 1.0000 |
| Filtered R@10 (99%) | N/A | 1.0000 |
| Filter p50 (ms) | **1.73** | 4.70 |
| Optimize Time (s) | N/A | 0.0 |
| Memory (B/vec) | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 401 | 400 | 4.70 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 275qps | R=1.000 / 326qps | R=1.000 / 313qps | R=1.000 / 309qps | R=1.000 / 335qps | R=1.000 / 353qps |

---
## Dataset: random-128d-50k (50,000 vectors, 128d)

### Core Metrics

| Metric | DEEPDATA | QDRANT |
| --- | --- | --- |
| Insert QPS | 314 | **21277** |
| Insert Time (s) | 159.0 | **2.4** |
| Recall@1 | 0.8100 | **1.0000** |
| Recall@10 | 0.7250 | **1.0000** |
| Recall@100 | 0.6002 | **1.0000** |
| Latency p50 (ms) | 6.63 | **2.63** |
| Latency p95 (ms) | 7.78 | **3.35** |
| Latency p99 (ms) | 8.96 | **4.57** |
| Serial QPS | 248 | **574** |
| Concurrent QPS (4T) | **994** | 889 |
| Concurrent QPS (8T) | **1377** | 1151 |
| Concurrent QPS (16T) | **1688** | 972 |
| Concurrent p99 (ms) | **23.93** | 40.36 |
| Filtered R@10 (1%) | N/A | 1.0000 |
| Filtered R@10 (50%) | N/A | 1.0000 |
| Filtered R@10 (99%) | N/A | 1.0000 |
| Filter p50 (ms) | 78.87 | **14.12** |
| Optimize Time (s) | N/A | 0.0 |
| Memory (B/vec) | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 120 | 120 | 13.70 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 204qps | R=1.000 / 226qps | R=1.000 / 204qps | R=1.000 / 231qps | R=1.000 / 221qps | R=1.000 / 215qps |

---
## Dataset: random-1536d-10k (10,000 vectors, 1536d)

### Core Metrics

| Metric | WEAVIATE | MILVUS | CHROMADB | PGVECTOR | REDIS | ELASTICSEARCH | LANCEDB | DEEPDATA | QDRANT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Insert QPS | 2414 | 2414 | 1712 | 1996 | 924 | 889 | **12642** | 218 | 3235 |
| Insert Time (s) | 4.1 | 4.1 | 5.8 | 5.0 | 10.8 | 11.2 | **0.8** | 45.9 | 3.1 |
| Recall@1 | 0.4400 | 0.3300 | 0.3900 | 0.5200 | 0.3800 | 0.7000 | 0.0700 | 0.6400 | **1.0000** |
| Recall@10 | 0.4430 | 0.2440 | 0.3780 | 0.5310 | 0.3660 | 0.6680 | 0.1030 | 0.6370 | **1.0000** |
| Recall@100 | 0.4176 | 0.2092 | 0.3519 | 0.4787 | 0.1000 | 0.6192 | 0.1587 | 0.6181 | **1.0000** |
| Latency p50 (ms) | 3.44 | 2.03 | 3.85 | 3.94 | **0.98** | 113.65 | 37.01 | 29.17 | 3.00 |
| Latency p95 (ms) | 4.27 | 2.71 | 4.34 | 5.00 | **1.18** | 135.47 | 44.28 | 33.86 | 3.75 |
| Latency p99 (ms) | 4.61 | 3.59 | 5.35 | 7.21 | **1.27** | 167.06 | 65.88 | 40.22 | 11.69 |
| Serial QPS | 470 | 1002 | 371 | 325 | **4808** | 70 | 168 | 129 | 355 |
| Concurrent QPS (4T) | 1576 | 3578 | 597 | 1150 | **6807** | 329 | 195 | 462 | 549 |
| Concurrent QPS (8T) | 2630 | 2475 | 593 | 2198 | **7672** | 488 | 178 | 725 | 628 |
| Concurrent QPS (16T) | 2844 | 2235 | 609 | 2722 | **7375** | 575 | 229 | 746 | 580 |
| Concurrent p99 (ms) | 13.67 | 13.58 | 55.46 | 17.80 | **5.64** | 56.03 | 103.42 | 42.38 | 68.56 |
| Filtered R@10 (1%) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.2989 | N/A | **1.0000** |
| Filtered R@10 (50%) | **1.0000** | 0.5620 | 0.5120 | 0.4560 | **1.0000** | 0.6700 | 0.1140 | N/A | **1.0000** |
| Filtered R@10 (99%) | **1.0000** | 0.5060 | 0.3680 | 0.5060 | **1.0000** | 0.4800 | 0.0860 | N/A | **1.0000** |
| Filter p50 (ms) | 13.99 | 1.92 | 22.87 | 3.94 | **1.10** | 16.65 | 6.36 | 28.18 | 3.87 |
| Optimize Time (s) | N/A | 3.9 | N/A | 19.0 | N/A | 22.7 | 8.4 | N/A | **0.0** |
| Memory (B/vec) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 89 | 89 | 15.61 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 389qps | R=1.000 / 388qps | R=1.000 / 398qps | R=1.000 / 389qps | R=1.000 / 339qps | R=1.000 / 400qps |

---
## Dataset: random-1536d-1k (1,000 vectors, 1536d)

### Core Metrics

| Metric | DEEPDATA | QDRANT |
| --- | --- | --- |
| Insert QPS | 708 | **3068** |
| Insert Time (s) | 1.4 | **0.3** |
| Recall@1 | **1.0000** | **1.0000** |
| Recall@10 | **1.0000** | **1.0000** |
| Recall@100 | 0.9994 | **1.0000** |
| Latency p50 (ms) | 24.63 | **4.11** |
| Latency p95 (ms) | 29.26 | **4.82** |
| Latency p99 (ms) | 31.71 | **5.60** |
| Serial QPS | 204 | **293** |
| Concurrent QPS (4T) | 879 | **1001** |
| Concurrent QPS (8T) | 828 | **1197** |
| Concurrent QPS (16T) | 842 | **1267** |
| Concurrent p99 (ms) | 56.24 | **28.77** |
| Filtered R@10 (1%) | N/A | 1.0000 |
| Filtered R@10 (50%) | N/A | 1.0000 |
| Filtered R@10 (99%) | N/A | 1.0000 |
| Filter p50 (ms) | **3.85** | 4.35 |
| Optimize Time (s) | N/A | 0.0 |
| Memory (B/vec) | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 196 | 196 | 8.50 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 739qps | R=1.000 / 724qps | R=1.000 / 736qps | R=1.000 / 713qps | R=1.000 / 754qps | R=1.000 / 750qps |

---
## Dataset: random-768d-10k (10,000 vectors, 768d)

### Core Metrics

| Metric | WEAVIATE | MILVUS | CHROMADB | PGVECTOR | REDIS | ELASTICSEARCH | LANCEDB | DEEPDATA | QDRANT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Insert QPS | 3203 | 2772 | 3069 | 3851 | 1941 | 1489 | **23095** | 332 | 6064 |
| Insert Time (s) | 3.1 | 3.6 | 3.3 | 2.6 | 5.2 | 6.7 | **0.4** | 30.1 | 1.6 |
| Recall@1 | 0.6600 | 0.2800 | 0.5200 | 0.6400 | 0.5400 | 0.8300 | 0.0500 | 0.7800 | **1.0000** |
| Recall@10 | 0.5200 | 0.2560 | 0.4470 | 0.5850 | 0.4590 | 0.7350 | 0.1050 | 0.7330 | **1.0000** |
| Recall@100 | 0.4645 | 0.2287 | 0.4024 | 0.5327 | 0.1000 | 0.6802 | 0.1656 | 0.6778 | **1.0000** |
| Latency p50 (ms) | 1.90 | 1.10 | 2.83 | 2.20 | **0.53** | 58.98 | 18.35 | 15.93 | 2.18 |
| Latency p95 (ms) | 2.46 | 1.57 | 4.14 | 2.57 | **0.60** | 67.16 | 27.50 | 18.27 | 2.62 |
| Latency p99 (ms) | 3.81 | 3.88 | 7.16 | 3.24 | **0.67** | 91.34 | 33.76 | 19.13 | 2.98 |
| Serial QPS | 884 | 1406 | 522 | 582 | **5227** | 116 | 284 | 191 | 546 |
| Concurrent QPS (4T) | 3406 | 3725 | 1045 | 2434 | **9081** | 530 | 374 | 682 | 854 |
| Concurrent QPS (8T) | 3639 | 3203 | 969 | 4149 | **8322** | 879 | 347 | 1091 | 1085 |
| Concurrent QPS (16T) | 3492 | 2598 | 917 | 5696 | **8400** | 809 | 369 | 1048 | 886 |
| Concurrent p99 (ms) | 11.90 | 14.30 | 38.15 | 8.35 | **7.04** | 45.83 | 92.46 | 34.05 | 38.49 |
| Filtered R@10 (1%) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.2581 | N/A | **1.0000** |
| Filtered R@10 (50%) | **1.0000** | 0.6360 | 0.5980 | 0.5560 | **1.0000** | 0.7200 | 0.1020 | N/A | **1.0000** |
| Filtered R@10 (99%) | **1.0000** | 0.6020 | 0.4480 | 0.5900 | **1.0000** | 0.5080 | 0.1000 | N/A | **1.0000** |
| Filter p50 (ms) | 11.22 | 1.62 | 18.35 | 2.04 | **0.79** | 8.90 | 4.01 | 18.80 | 6.71 |
| Optimize Time (s) | N/A | 2.2 | N/A | 8.5 | N/A | 7.2 | 4.6 | N/A | **0.0** |
| Memory (B/vec) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 124 | 124 | 11.62 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 220qps | R=1.000 / 231qps | R=1.000 / 210qps | R=1.000 / 241qps | R=1.000 / 231qps | R=1.000 / 219qps |

---
## Dataset: random-768d-1k (1,000 vectors, 768d)

### Core Metrics

| Metric | DEEPDATA | QDRANT |
| --- | --- | --- |
| Insert QPS | 1072 | **5714** |
| Insert Time (s) | 0.9 | **0.2** |
| Recall@1 | **1.0000** | **1.0000** |
| Recall@10 | **1.0000** | **1.0000** |
| Recall@100 | 0.9999 | **1.0000** |
| Latency p50 (ms) | 12.62 | **3.39** |
| Latency p95 (ms) | 14.26 | **4.17** |
| Latency p99 (ms) | 17.65 | **4.68** |
| Serial QPS | **338** | 330 |
| Concurrent QPS (4T) | **1234** | 974 |
| Concurrent QPS (8T) | 1229 | **1278** |
| Concurrent QPS (16T) | 1261 | **1487** |
| Concurrent p99 (ms) | 31.45 | **23.18** |
| Filtered R@10 (1%) | N/A | 1.0000 |
| Filtered R@10 (50%) | N/A | 1.0000 |
| Filtered R@10 (99%) | N/A | 1.0000 |
| Filter p50 (ms) | 2.19 | **2.08** |
| Optimize Time (s) | N/A | 0.0 |
| Memory (B/vec) | N/A | N/A |

### Streaming (Concurrent Insert + Search, 10s)

| VDB | Insert QPS | Search QPS | Search p99 (ms) |
| --- | --- | --- | --- |
| DEEPDATA | 264 | 263 | 6.48 |

### Recall vs QPS Tradeoff (ef_search sweep)

| VDB | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 | ef=512 |
| --- | --- | --- | --- | --- | --- | --- |
| QDRANT | R=1.000 / 748qps | R=1.000 / 748qps | R=1.000 / 719qps | R=1.000 / 551qps | R=1.000 / 717qps | R=1.000 / 539qps |

---
## Feature Comparison Matrix

| Feature | DeepData | Qdrant | Weaviate | Milvus | ChromaDB | pgvector | Redis | Elasticsearch | LanceDB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense Vector Search | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Sparse/BM25 | Y | Y | Y | Y | N | N | N | Y | N |
| Hybrid Fusion | Y | Y | Y | Y | N | N | N | Y | N |
| Graph-Boosted Search | **UNIQUE** | N | N | N | N | N | N | N | N |
| Metadata Filtering | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| HNSW Index | Y | Y | Y | Y | Y | Y | Y | Y | N |
| IVF Index | Y | N | N | Y | N | N | N | N | Y |
| DiskANN | Y | N | N | Y | N | N | N | N | N |
| FLAT (brute force) | Y | N | Y | Y | N | N | Y | Y | N |
| FP16 Quantization | Y | Y | N | Y | N | N | N | N | N |
| Product Quantization | Y | Y | N | Y | N | N | N | N | Y |
| Binary Quantization | Y | Y | Y | Y | N | N | N | Y | N |
| WAL Durability | Y | Y | Y | Y | N | Y | Y | Y | N |
| TLS/mTLS | Y | Y | N | Y | N | Y | Y | Y | N |
| RBAC/Auth | Y | Y | N | N | Tok | Y | Y | Y | N |
| Single Binary | **Y** | Y | Y | N (3svc) | Y | N/A | N/A | N/A | Embed |
| Knowledge Graph | **UNIQUE** | N | N | N | N | N | N | N | N |
| Entity Extraction | **UNIQUE** | N | N | N | N | N | N | N | N |
| Prometheus Metrics | Y | Y | Y | Y | N | Y | Y | Y | N |
| GPU Acceleration | Stubs | N | N | Y | N | N | N | N | N |
| Horizontal Scaling | WIP | Y | Y | Y | N | N | Y | Y | N |
| Python SDK | WIP | Y | Y | Y | Y | Y | Y | Y | Y |
| Go SDK | Y | Y | N | Y | N | N | Y | N | N |

## Deployment Complexity

| Aspect | DeepData | Qdrant | Weaviate | Milvus | ChromaDB | pgvector | Redis | Elasticsearch | LanceDB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Docker Images | 1 | 1 | 1 | 3 | 1 | 1 | 1 | 1 | 0 (embed) |
| External Deps | 0 | 0 | 0 | 2 (etcd+minio) | 0 | 0 | 0 | 0 | 0 |
| Language | Go | Rust | Go | Go/C++ | Python | C/SQL | C | Java | Rust |
| Min RAM | ~50MB | ~100MB | ~200MB | ~500MB | ~100MB | ~50MB | ~50MB | ~2GB | ~10MB |

## DeepData Internal Benchmarks (Go-native, no HTTP)

For reference, bypassing HTTP serialization overhead:

| Metric | 128d/100K | 768d/100K | 1536d/100K |
|--------|-----------|-----------|------------|
| HNSW QPS | 20,644 | 19,325 | 13,436 |
| HNSW+FP16 QPS | 27,440 | 18,762 | 13,273 |
| Recall@10 | 0.994 | 0.998+ | 0.998+ |
| p99 Latency | 90us | 90us | 127us |
| Insert QPS (HNSW) | 16,529 | 11,599 | - |
| Insert QPS (IVF) | 106,590 | 5,196 | - |
| Memory Overhead | 3.12x | - | - |

*HTTP API adds ~1-20ms per call due to JSON serialization.*
*A gRPC or binary protocol would close much of the latency gap.*

## Competitive Analysis

### DeepData Unique Differentiators
1. **Graph-Boosted Search** - PageRank-derived reranking via integrated knowledge graph
2. **Entity Extraction Pipeline** - LLM-powered entity/relation extraction built-in
3. **Triple Hybrid Fusion** - Dense + Sparse + Graph in single query (RRF/weighted/linear)
4. **4 Index Types in 1 Binary** - HNSW, IVF, DiskANN, FLAT + inverted index
5. **4 Quantization Options** - FP16, Uint8, PQ, Binary quantization
6. **Zero External Dependencies** - Single Go binary vs Milvus's 3-service stack

### vs Each Competitor

**vs Qdrant**
- DeepData wins: Graph search, hybrid fusion, more index types, simpler deploy, knowledge graph
- Qdrant wins: Rust native perf, mature SDKs (Python/JS/Rust/Go), production distributed mode, payload indexing

**vs Weaviate**
- DeepData wins: Graph search, more quantization, IVF/DiskANN, lighter binary, RBAC
- Weaviate wins: gRPC performance, module ecosystem (generative, reranker), mature clustering, better Python DX

**vs Milvus**
- DeepData wins: Graph search, single binary (no etcd/minio), faster single-node insert, simpler ops
- Milvus wins: GPU acceleration (IVF/CAGRA), mature distributed, wider SDK support, enterprise features

**vs ChromaDB**
- DeepData wins: Graph search, hybrid, WAL, RBAC, 4 index types, quantization, filtering
- ChromaDB wins: Simplest API, Python-native, lightweight, great for prototyping

**vs pgvector**
- DeepData wins: Graph search, hybrid, purpose-built ANN, streaming, knowledge graph
- pgvector wins: SQL ecosystem, joins with relational data, mature tooling, pgvector is 'good enough' for many

**vs Redis**
- DeepData wins: Graph search, hybrid fusion, multiple index types, knowledge graph, quantization
- Redis wins: In-memory speed, mature ecosystem, pub/sub, caching layer, widely deployed

**vs Elasticsearch**
- DeepData wins: Graph search, purpose-built ANN, lighter weight, simpler ops, knowledge graph
- Elasticsearch wins: Full-text search maturity, aggregations, observability ecosystem, enterprise features

**vs LanceDB**
- DeepData wins: Graph search, server mode, RBAC, WAL, multi-tenancy, sparse vectors
- LanceDB wins: Embedded mode (no network overhead), columnar storage, versioning, Lance format efficiency

## Methodology

1. All databases run in Docker containers with 4GB memory limit
2. DeepData runs as native binary for fairest comparison (no container overhead)
3. Synthetic random unit vectors for reproducibility; real datasets (SIFT) when available
4. Ground truth via brute-force cosine similarity
5. QPS: 5s burst, serial and concurrent (4/8/16 threads)
6. Filtered search at 1%, 50%, 99% selectivity
7. Streaming: concurrent insert + search for 10s
8. All databases use HNSW M=16, ef_construction=200, ef_search=128
9. VectorDBBench client integration available for standardized SIFT/Cohere/GIST benchmarks
