# Benchmark Evidence

All numbers on this page are reproducible from this repository. No number
here is a claim about deployments at scale; DeepData is a single-node RC
(Linux amd64) and these results describe exactly that.

- Host: AMD Ryzen 7 7700X, local NVMe, Linux
- Dataset: sift-100k (100,000 vectors, 128-dim), cosine-normalized,
  ground truth precomputed
- Config parity: HNSW `m=16`, `ef_construction=300` on both engines;
  DeepData searched at `ef_search=200`, Qdrant defaults
- DeepData `0.2.0-rc.1`: canonical V3 contract, durable mode — every
  acknowledged batch is fsynced to the journal AND the HNSW index build
  completes before the response returns
- Qdrant 1.19.0: official container image, single node, default settings
  plus a sane per-collection `indexing_threshold` (see note 3)

## Headline: equal-effort Go clients (`benchmarks/ddload*`)

Both sides driven by equivalent stdlib Go clients, 4 insert workers,
2000-vector batches, identical dataset file.

| Metric (sift-100k) | DeepData | Qdrant 1.19 |
|---|---|---|
| Acknowledged-insert throughput | 3,614 vec/s | 103,965 vec/s |
| What the ack guarantees | durable + fully indexed | written to segment |
| Async index catch-up | n/a (already indexed) | +2.0 s |
| **Time until 100k vectors are fully searchable** | **27.7 s** | **3.0 s** |
| Serial search qps (hot query, top_k=10) | **8,377** | 1,170 |
| Concurrent search qps (8 threads) | **49,020** | 1,469 |

Recall under the rotating-query harness (Python, top_k=100):
DeepData recall@10 = 1.000, recall@100 = 0.997.

## Reading the table honestly

1. **Ingest:** the raw ack rates are not the same work. DeepData's ack
   means the batch is fsynced *and* fully HNSW-indexed; Qdrant appends to
   segments and builds indexes asynchronously (~2 s optimizer catch-up
   here). On the honest common metric — time until everything is
   searchable — the gap is **~9x**, not ~29x. It remains our weakest axis
   and the top engineering priority; the order of work is fixed in
   [ADR 0005](decisions/0005-ingest-levers-segments-then-allocs-then-group-commit.md):
   parallel segment builds first, allocation reduction second, group commit
   deferred.
2. **Search:** DeepData serves 7x more serial and 33x more concurrent
   queries per second against the same collection. Caveats before quoting:
   hot repeated query, top_k=10, no payload filtering, and the Qdrant side
   showed near-flat scaling from 1→8 threads in this environment (cause
   not diagnosed; possibly internal search-thread throttling under
   containers). Treat the ratio as directional until reproduced by third
   parties; recall parity itself was verified separately above.
3. **Qdrant got a fair shot.** With default `indexing_threshold` (20 MB)
   its optimizer never built any HNSW for this dataset (segments stayed
   below threshold; searches were brute force over appendable segments).
   We lowered it per-collection so all search rows are against a fully
   built index. Deployments that skip this step silently get unindexed
   reads.
4. **Client matters more than vendors admit.** An earlier harness reported
   682 serial-search qps for DeepData through a Python client; the same
   server does 8,377 via an efficient client. Any vendor number captured
   through a heavyweight client (including our own historical tables and
   most blog benchmarks) measures the client, not the server.

## Historical context

The pre-rework binary-import era recorded DeepData inserts at
1,339 vec/s and concurrent search at 2,251 qps (`benchmarks/results/results-2026-04-12.json`). The
canonical durable path today is 2.7x faster on inserts while adding
durability the old path never had.

## Reproduce

```bash
# Engine-level durability microbenchmarks
go test -count=1 -run '^$' -bench BenchmarkDurableInsert ./internal/collection/

# End-to-end via canonical V3 (Python client; includes client cost)
python3 benchmarks/recall_test.py --vdb deepdata --dataset sift-100k

# Server-side truth, equal-effort clients
go run ./benchmarks/ddload       -base ~/.vectordb_bench/dataset/sift/sift_base_100k_norm.fvecs
go run ./benchmarks/ddload-qdrant -base ~/.vectordb_bench/dataset/sift/sift_base_100k_norm.fvecs
```

Evidence discipline for release claims lives in
[PRE_RELEASE_STATUS.md](PRE_RELEASE_STATUS.md); soak and crash-recovery
results are bound to exact SHAs there.
