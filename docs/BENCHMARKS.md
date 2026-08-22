# Benchmark Evidence

All numbers on this page are reproducible from this repository. No number
here is a claim about deployments at scale; DeepData is a single-node RC
(Linux amd64) and these results describe exactly that.

- Host: AMD Ryzen 7 7700X, local NVMe, Linux
- Dataset: sift-100k (100,000 vectors, 128-dim, L2), ground truth precomputed
- DeepData: `0.2.0-rc` line, canonical V3 contract, durable mode
  (every acknowledged batch is fsynced to the journal before response)
- Competitor baseline: Qdrant v1.x single node, default settings,
  measured previously via its Python client on the same host
  (`results.json`, historical)

## Headline results

| Metric (sift-100k, 128d) | DeepData | Qdrant (recorded) |
|---|---|---|
| Recall@10 / @100 (top_k=100, ef=200-class settings) | **1.000 / 0.997** | 1.000 / 1.000 |
| Durable insert throughput (batch JSON, 4 workers)   | **5,707 vec/s** | 30,909 vec/s |
| Serial search                                       | **7,874 qps**   | 480 qps |
| Concurrent search (8 threads)                       | **61,061 qps**  | 866 qps |

Search p50/p99 under the Python harness: 1.5 ms / 2.2 ms.

## Measurement honesty notes

Read these before quoting the table:

1. **Client matters more than vendors admit.** Our first harness reported
   682 serial-search qps; an efficient Go client against the same server
   reports 7,874. The Qdrant read-side figures above were captured through
   a Python client and are almost certainly client-bound, not server
   bounds. A fair read-side comparison requires re-running competitors
   with equal-effort clients; until then the search rows demonstrate
   DeepData capability, not superiority.
2. **Ingest comparison is fair-ish but not perfect.** Both sides used
   their official clients with batching (DeepData 2000-doc batches over
   HTTP/JSON; Qdrant 500-point upserts). DeepData's figure includes an
   fsync per acknowledged batch; many vector stores acknowledge before
   durability by default.
3. **Ingest is our weakest axis** (~5x behind on this setup). The cost is
   dominated by single-collection HNSW construction and the strict
   durability barrier. Group-commit and parallel build work is in
   progress (`tasks/todo.md`).

## Reproduce

```bash
# Engine-level durability microbenchmarks
go test -count=1 -run '^$' -bench BenchmarkDurableInsert ./internal/collection/

# End-to-end vs dataset (Python client; upper bound includes client cost)
python3 benchmarks/recall_test.py --vdb deepdata --dataset sift-100k

# Server-side truth with an efficient client
go run ./benchmarks/ddload -base ~/.vectordb_bench/dataset/sift/sift_base_100k_norm.fvecs
```

Evidence discipline for release claims lives in
[PRE_RELEASE_STATUS.md](PRE_RELEASE_STATUS.md); soak and crash-recovery
results are bound to exact SHAs there.
