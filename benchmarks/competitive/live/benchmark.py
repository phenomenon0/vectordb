"""Main benchmark runner: orchestrates all suites across all VDB adapters.

Usage:
    python benchmark.py --all              # Run all suites
    python benchmark.py --suite insert     # Run specific suite
    python benchmark.py --vdb deepdata     # Run specific VDB only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from adapters import ALL_ADAPTERS
from adapters.base import VDBAdapter

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

COLLECTION_NAME = "deepdata_bench"
EMBEDDING_DIM = 1536
BATCH_SIZE = 50


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_recall(retrieved_ids: list[str], gt_ids: list[str], k: int) -> float:
    """Recall@k: fraction of top-k ground truth IDs found in retrieved results."""
    gt_set = set(gt_ids[:k])
    retrieved_set = set(retrieved_ids[:k])
    if not gt_set:
        return 0.0
    return len(gt_set & retrieved_set) / len(gt_set)


def get_docker_memory(service: str) -> int | None:
    """Get RSS memory of a docker service via docker stats."""
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}",
             service],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            mem_str = result.stdout.strip().split("/")[0].strip()
            # Parse "123.4MiB" or "1.2GiB"
            if "GiB" in mem_str:
                return int(float(mem_str.replace("GiB", "")) * 1024 * 1024 * 1024)
            elif "MiB" in mem_str:
                return int(float(mem_str.replace("MiB", "")) * 1024 * 1024)
            elif "KiB" in mem_str:
                return int(float(mem_str.replace("KiB", "")) * 1024)
    except Exception:
        pass
    return None


# ── Suite 1: Insert Throughput ─────────────────────────────────────────


def suite_insert(adapter: VDBAdapter, corpus: list[dict]) -> dict:
    """Measure insert throughput."""
    adapter.create_collection(COLLECTION_NAME, dim=EMBEDDING_DIM)

    total_docs = len(corpus)
    start = time.perf_counter()

    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc=f"  {adapter.name} insert"):
        batch = corpus[i:i + BATCH_SIZE]
        adapter.insert_batch(
            COLLECTION_NAME,
            ids=[r["id"] for r in batch],
            vectors=[r["embedding"] for r in batch],
            texts=[r["text"] for r in batch],
            metadata=[r["meta"] for r in batch],
        )

    elapsed = time.perf_counter() - start
    inserts_per_sec = total_docs / elapsed

    # Wait for indexing to settle
    time.sleep(2)

    return {
        "total_docs": total_docs,
        "elapsed_sec": round(elapsed, 3),
        "inserts_per_sec": round(inserts_per_sec, 1),
    }


# ── Suite 2: Search Recall & Latency ──────────────────────────────────


def suite_search(
    adapter: VDBAdapter, queries: list[dict], ground_truth: list[dict]
) -> dict:
    """Measure recall@k and latency."""
    gt_map = {gt["query_id"]: [n["id"] for n in gt["neighbors"]] for gt in ground_truth}

    recalls_1 = []
    recalls_10 = []
    recalls_100 = []
    latencies = []

    for query in tqdm(queries, desc=f"  {adapter.name} search"):
        gt_ids = gt_map.get(query["id"], [])

        start = time.perf_counter()
        results = adapter.search(
            COLLECTION_NAME,
            vector=query["embedding"],
            top_k=100,
            ef_search=128,
        )
        elapsed = time.perf_counter() - start

        retrieved_ids = [r.id for r in results]
        recalls_1.append(compute_recall(retrieved_ids, gt_ids, 1))
        recalls_10.append(compute_recall(retrieved_ids, gt_ids, 10))
        recalls_100.append(compute_recall(retrieved_ids, gt_ids, 100))
        latencies.append(elapsed * 1000)  # ms

    latencies_arr = np.array(latencies)

    # QPS measurement: tight loop for ~5 seconds
    qps_query = queries[0]["embedding"]
    qps_count = 0
    qps_start = time.perf_counter()
    qps_duration = 5.0
    while time.perf_counter() - qps_start < qps_duration:
        adapter.search(COLLECTION_NAME, vector=qps_query, top_k=10, ef_search=128)
        qps_count += 1
    qps = qps_count / (time.perf_counter() - qps_start)

    return {
        "recall_at_1": round(float(np.mean(recalls_1)), 4),
        "recall_at_10": round(float(np.mean(recalls_10)), 4),
        "recall_at_100": round(float(np.mean(recalls_100)), 4),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
        "qps": round(qps, 1),
    }


# ── Suite 3: Filtered Search ──────────────────────────────────────────


def suite_filtered(
    adapter: VDBAdapter,
    queries: list[dict],
    ground_truth_filtered: list[dict],
    meta_filter: dict[str, str],
) -> dict:
    """Measure filtered search recall and latency overhead."""
    gt_map = {
        gt["query_id"]: [n["id"] for n in gt["neighbors"]]
        for gt in ground_truth_filtered
    }

    recalls_10 = []
    latencies = []

    for query in tqdm(queries, desc=f"  {adapter.name} filtered"):
        gt_ids = gt_map.get(query["id"], [])

        start = time.perf_counter()
        results = adapter.search(
            COLLECTION_NAME,
            vector=query["embedding"],
            top_k=100,
            ef_search=128,
            meta_filter=meta_filter,
        )
        elapsed = time.perf_counter() - start

        retrieved_ids = [r.id for r in results]
        recalls_10.append(compute_recall(retrieved_ids, gt_ids, 10))
        latencies.append(elapsed * 1000)

    latencies_arr = np.array(latencies)

    return {
        "filter": meta_filter,
        "recall_at_10": round(float(np.mean(recalls_10)), 4),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
    }


# ── Suite 4: Hybrid Search ────────────────────────────────────────────


def suite_hybrid(
    adapter: VDBAdapter, queries: list[dict], ground_truth: list[dict]
) -> dict | None:
    """Measure hybrid search recall improvement over dense-only."""
    if not adapter.supports_hybrid():
        return None

    gt_map = {gt["query_id"]: [n["id"] for n in gt["neighbors"]] for gt in ground_truth}

    recalls_10 = []
    latencies = []

    for query in tqdm(queries, desc=f"  {adapter.name} hybrid"):
        gt_ids = gt_map.get(query["id"], [])

        start = time.perf_counter()
        results = adapter.hybrid_search(
            COLLECTION_NAME,
            text=query["text"],
            vector=query["embedding"],
            top_k=100,
            alpha=0.7,
        )
        elapsed = time.perf_counter() - start

        retrieved_ids = [r.id for r in results]
        recalls_10.append(compute_recall(retrieved_ids, gt_ids, 10))
        latencies.append(elapsed * 1000)

    latencies_arr = np.array(latencies)

    return {
        "hybrid_recall_at_10": round(float(np.mean(recalls_10)), 4),
        "hybrid_latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "hybrid_latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
    }


# ── Suite 5: Graph-Boosted Search (DeepData only) ─────────────────────


def suite_graph(adapter: VDBAdapter, queries: list[dict], ground_truth: list[dict]) -> dict | None:
    """Measure graph-boosted search. DeepData only."""
    if not adapter.supports_graph():
        return None

    # This would require enabling entity extraction and building a knowledge graph.
    # For now, document it as a DeepData differentiator.
    return {
        "note": "Graph-boosted search is a DeepData-only feature. "
                "Requires entity extraction (LLM calls) and knowledge graph build. "
                "No competitor equivalent exists for head-to-head comparison.",
        "status": "documented_differentiator",
    }


# ── Suite 6: Memory Footprint ─────────────────────────────────────────


def suite_memory(adapter: VDBAdapter, corpus_size: int) -> dict:
    """Measure memory footprint."""
    # Try adapter-specific stats first
    mem = adapter.get_memory_usage()

    # Docker container names for stats
    docker_names = {
        "deepdata": "live-deepdata-1",
        "weaviate": "live-weaviate-1",
        "milvus": "live-milvus-1",
        "qdrant": "live-qdrant-1",
        "chromadb": "live-chromadb-1",
    }

    docker_mem = get_docker_memory(docker_names.get(adapter.name, ""))

    result: dict = {"corpus_size": corpus_size}
    if mem is not None:
        result["api_memory_bytes"] = mem
    if docker_mem is not None:
        result["docker_rss_bytes"] = docker_mem
        result["bytes_per_vector"] = round(docker_mem / corpus_size, 1) if corpus_size > 0 else 0

    return result


# ── Main Orchestrator ─────────────────────────────────────────────────


def create_adapter(name: str) -> VDBAdapter:
    """Instantiate an adapter by name."""
    cls = ALL_ADAPTERS[name]
    if name == "deepdata":
        return cls(url="http://localhost:8080")
    elif name == "deepdata-grpc":
        return cls(url="localhost:50051")
    elif name == "weaviate":
        return cls(url="http://localhost:8081")
    elif name == "milvus":
        return cls(host="localhost", port=19530)
    elif name == "qdrant":
        return cls(url="http://localhost:6333")
    elif name == "chromadb":
        return cls(host="localhost", port=8010)
    else:
        raise ValueError(f"Unknown adapter: {name}")


def run_benchmarks(
    vdb_names: list[str],
    suites: list[str],
) -> dict:
    """Run specified suites on specified VDBs."""
    # Load data
    print("Loading benchmark data...")
    corpus = load_jsonl(DATA_DIR / "collection_a.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    ground_truth = load_jsonl(DATA_DIR / "ground_truth.jsonl")

    gt_filtered = []
    meta_filter = {"package": "index"}
    gt_filt_path = DATA_DIR / "ground_truth_filtered.jsonl"
    filter_meta_path = DATA_DIR / "filter_meta.json"
    if gt_filt_path.exists():
        gt_filtered = load_jsonl(gt_filt_path)
    if filter_meta_path.exists():
        with open(filter_meta_path) as f:
            meta_filter = json.load(f)

    print(f"  Corpus: {len(corpus)} chunks, Queries: {len(queries)}, "
          f"GT: {len(ground_truth)}, GT filtered: {len(gt_filtered)}")

    all_results: dict = {}

    for vdb_name in vdb_names:
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {vdb_name.upper()}")
        print(f"{'='*60}")

        try:
            adapter = create_adapter(vdb_name)
        except Exception as e:
            print(f"  SKIP {vdb_name}: failed to connect — {e}")
            all_results[vdb_name] = {"error": str(e)}
            continue

        vdb_results: dict = {}

        try:
            if "insert" in suites:
                print(f"\n  Suite 1: Insert Throughput")
                vdb_results["insert"] = suite_insert(adapter, corpus)
                print(f"    → {vdb_results['insert']['inserts_per_sec']} docs/sec")

            if "search" in suites:
                print(f"\n  Suite 2: Search Recall & Latency")
                # Ensure data is inserted
                if "insert" not in suites:
                    suite_insert(adapter, corpus)
                vdb_results["search"] = suite_search(adapter, queries, ground_truth)
                print(f"    → recall@10={vdb_results['search']['recall_at_10']}, "
                      f"p50={vdb_results['search']['latency_p50_ms']}ms, "
                      f"qps={vdb_results['search']['qps']}")

            if "filtered" in suites and gt_filtered:
                print(f"\n  Suite 3: Filtered Search")
                if "insert" not in suites and "search" not in suites:
                    suite_insert(adapter, corpus)
                vdb_results["filtered"] = suite_filtered(
                    adapter, queries, gt_filtered, meta_filter
                )
                print(f"    → recall@10={vdb_results['filtered']['recall_at_10']}, "
                      f"p50={vdb_results['filtered']['latency_p50_ms']}ms")

            if "hybrid" in suites:
                print(f"\n  Suite 4: Hybrid Search")
                result = suite_hybrid(adapter, queries, ground_truth)
                if result:
                    vdb_results["hybrid"] = result
                    print(f"    → hybrid recall@10={result.get('hybrid_recall_at_10', 'N/A')}")
                else:
                    vdb_results["hybrid"] = {"status": "not_supported"}
                    print(f"    → Not supported by {vdb_name}")

            if "graph" in suites:
                print(f"\n  Suite 5: Graph-Boosted Search")
                result = suite_graph(adapter, queries, ground_truth)
                if result:
                    vdb_results["graph"] = result
                    print(f"    → {result.get('status', 'N/A')}")
                else:
                    vdb_results["graph"] = {"status": "not_supported"}

            if "memory" in suites:
                print(f"\n  Suite 6: Memory Footprint")
                vdb_results["memory"] = suite_memory(adapter, len(corpus))
                bpv = vdb_results["memory"].get("bytes_per_vector", "?")
                print(f"    → bytes/vector={bpv}")

        except Exception as e:
            print(f"  ERROR in {vdb_name}: {e}")
            vdb_results["error"] = str(e)

        finally:
            try:
                adapter.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            adapter.teardown()

        all_results[vdb_name] = vdb_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="VDB Competitive Benchmark")
    parser.add_argument("--all", action="store_true", help="Run all suites on all VDBs")
    parser.add_argument(
        "--suite", nargs="+",
        choices=["insert", "search", "filtered", "hybrid", "graph", "memory"],
        help="Run specific suites",
    )
    parser.add_argument(
        "--vdb", nargs="+",
        choices=list(ALL_ADAPTERS.keys()),
        help="Run specific VDBs only",
    )
    args = parser.parse_args()

    if args.all:
        suites = ["insert", "search", "filtered", "hybrid", "graph", "memory"]
        vdb_names = list(ALL_ADAPTERS.keys())
    else:
        suites = args.suite or ["insert", "search"]
        vdb_names = args.vdb or list(ALL_ADAPTERS.keys())

    print(f"VDB Competitive Benchmark")
    print(f"  VDBs: {', '.join(vdb_names)}")
    print(f"  Suites: {', '.join(suites)}")
    print()

    results = run_benchmarks(vdb_names, suites)

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "raw_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for vdb, data in results.items():
        print(f"\n{vdb}:")
        if "error" in data and isinstance(data["error"], str):
            print(f"  Error: {data['error']}")
            continue
        for suite_name, suite_data in data.items():
            if isinstance(suite_data, dict):
                print(f"  {suite_name}: {json.dumps(suite_data, indent=4)}")


if __name__ == "__main__":
    main()
