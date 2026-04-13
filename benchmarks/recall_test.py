#!/usr/bin/env python3
"""Real-dataset recall benchmark for HNSW validation.

Tests DeepData (and optionally Qdrant) recall on real embeddings:
  - SIFT-10K/100K (128d visual descriptors)
  - Code embeddings (1536d, ~562 vectors)
  - GloVe-50d/100d (word vectors, up to 400K)

Each dataset gets a clean server restart for fair measurement.

Usage:
    python benchmarks/recall_test.py                                           # DeepData, all datasets
    python benchmarks/recall_test.py --vdb deepdata qdrant                     # Compare with Qdrant
    python benchmarks/recall_test.py --dataset sift-10k code-562               # Specific datasets
    python benchmarks/recall_test.py --vdb deepdata qdrant \\
        --dataset sift-100k glove-100d-full \\
        --n-search 200 --warmup 10 --concurrency 8 --duration 5 \\
        --json results.json
"""

import argparse
import concurrent.futures
import contextlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add parent for download_datasets module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_datasets import (
    CACHE_DIR,
    compute_ground_truth,
    read_fvecs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_BINARY = PROJECT_ROOT / "deepdata-server"
BENCH_DATA_DIR = Path("/tmp/deepdata-bench")
DEFAULT_PORT = 8080


# ═══════════════════════════════════════════════════════════════════
# Run configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RunConfig:
    port: int = DEFAULT_PORT
    n_search: int = 200
    warmup: int = 10
    concurrency: int = 8
    concurrent_duration_s: float = 5.0
    deepdata_batch_size: int = 5_000
    qdrant_batch_size: int = 500
    startup_timeout_s: float = 20.0
    settle_time_s: float = 1.0


# ═══════════════════════════════════════════════════════════════════
# Dataset definitions
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Dataset:
    name: str
    dim: int
    base_path: Path
    query_path: Path
    gt_path: Path
    n_base: int  # expected count (0 = load all)
    m: int = 16
    ef_construction: int = 300
    ef_search: int = 200


def get_dataset_configs() -> dict[str, Dataset]:
    sift = CACHE_DIR / "sift"
    glove = CACHE_DIR / "glove"
    code = CACHE_DIR / "code"

    return {
        "sift-10k": Dataset(
            name="sift-10k", dim=128, n_base=10_000, m=16, ef_construction=300, ef_search=200,
            base_path=sift / "siftsmall_base_norm.fvecs",
            query_path=sift / "siftsmall_query_norm.fvecs",
            gt_path=sift / "siftsmall_gt100.npy",
        ),
        "sift-100k": Dataset(
            name="sift-100k", dim=128, n_base=100_000, m=16, ef_construction=300, ef_search=200,
            base_path=sift / "sift_base_100k_norm.fvecs",
            query_path=sift / "sift_query_norm.fvecs",
            gt_path=sift / "sift_100k_gt100.npy",
        ),
        "code-562": Dataset(
            name="code-562", dim=1536, n_base=0, m=16, ef_construction=300, ef_search=200,
            base_path=code / "code_1536d_base_norm.fvecs",
            query_path=code / "code_1536d_query_norm.fvecs",
            gt_path=code / "code_1536d_gt100.npy",
        ),
        "glove-50d": Dataset(
            name="glove-50d", dim=50, n_base=10_000, m=16, ef_construction=300, ef_search=200,
            base_path=glove / "glove_50d_base_norm.fvecs",
            query_path=glove / "glove_50d_query_norm.fvecs",
            gt_path=glove / "glove_50d_gt100.npy",
        ),
        "glove-100d": Dataset(
            name="glove-100d", dim=100, n_base=10_000, m=16, ef_construction=300, ef_search=200,
            base_path=glove / "glove_100d_base_norm.fvecs",
            query_path=glove / "glove_100d_query_norm.fvecs",
            gt_path=glove / "glove_100d_gt100.npy",
        ),
        "glove-100d-full": Dataset(
            name="glove-100d-full", dim=100, n_base=0, m=16, ef_construction=300, ef_search=200,
            base_path=glove / "glove_100d_base_norm.fvecs",
            query_path=glove / "glove_100d_query_norm.fvecs",
            gt_path=glove / "glove_100d_gt100.npy",
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BenchResult:
    vdb: str = ""
    dataset: str = ""
    dim: int = 0
    n_vectors: int = 0
    n_queries: int = 0
    # Recall
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    # Throughput
    insert_qps: float = 0.0
    search_serial_qps: float = 0.0
    search_concurrent_qps: float = 0.0
    # Latency
    search_p50_ms: float = 0.0
    search_p95_ms: float = 0.0
    search_p99_ms: float = 0.0
    # Meta
    error: str = ""


def compute_recall(retrieved: list[int], gt: np.ndarray, k: int) -> float:
    """Recall@k: fraction of true top-k found in retrieved top-k."""
    gt_set = set(int(x) for x in gt[:k])
    ret_set = set(int(x) for x in retrieved[:k])
    return len(gt_set & ret_set) / len(gt_set) if gt_set else 0.0


def percentile_ms(latencies: list[float], pct: float) -> float:
    return round(float(np.percentile(latencies, pct)), 2) if latencies else 0.0


# ═══════════════════════════════════════════════════════════════════
# Server management
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    stderr_path: Path


def kill_servers():
    """Kill any running deepdata-server processes."""
    subprocess.run(["pkill", "-f", "deepdata-server"], capture_output=True)
    time.sleep(0.5)


def build_server() -> bool:
    """Clean compile the server binary."""
    print("  Building server...")
    result = subprocess.run(
        ["go", "build", "-o", str(SERVER_BINARY), "./cmd/deepdata/"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  BUILD FAILED: {result.stderr}")
        return False
    return True


def wait_for_port(port: int, timeout: float) -> bool:
    """Wait until a TCP port is accepting connections."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def tail_text(path: Path, n: int = 20) -> str:
    """Return the last n lines of a file, or empty string."""
    try:
        lines = path.read_text().strip().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def start_server(ds: Dataset, cfg: RunConfig) -> ServerHandle | None:
    """Start a fresh server with dataset-appropriate settings."""
    import shutil

    # Clean data directory
    if BENCH_DATA_DIR.exists():
        shutil.rmtree(BENCH_DATA_DIR)
    BENCH_DATA_DIR.mkdir(parents=True)

    env = {
        **os.environ,
        "API_RPS": "100000",
        "TENANT_RPS": "100000",
        "TENANT_BURST": "100000",
        "SCAN_THRESHOLD": "0",
        "VECTORDB_BASE_DIR": str(BENCH_DATA_DIR),
        "HNSW_M": str(ds.m),
        "HNSW_EF_CONSTRUCTION": str(ds.ef_construction),
        "HNSW_EFSEARCH": str(ds.ef_search),
    }

    stderr_path = BENCH_DATA_DIR / "server.stderr"
    stderr_file = open(stderr_path, "w")

    proc = subprocess.Popen(
        [str(SERVER_BINARY)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
        start_new_session=True,
    )

    base_url = f"http://127.0.0.1:{cfg.port}"
    ready = False
    try:
        import httpx
        if wait_for_port(cfg.port, min(3.0, cfg.startup_timeout_s)):
            with httpx.Client(base_url=base_url, timeout=2.0) as client:
                deadline = time.perf_counter() + cfg.startup_timeout_s
                while time.perf_counter() < deadline:
                    if proc.poll() is not None:
                        break
                    try:
                        resp = client.get("/health")
                        if resp.status_code == 200:
                            ready = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.25)
    finally:
        stderr_file.flush()
        stderr_file.close()

    if ready:
        print("  Server ready")
        return ServerHandle(proc=proc, stderr_path=stderr_path)

    stop_process(proc)
    err_tail = tail_text(stderr_path)
    print("  Server failed to start")
    if err_tail:
        print(err_tail)
    return None


def stop_process(proc: subprocess.Popen) -> None:
    """Gracefully stop a process (SIGTERM then SIGKILL)."""
    if proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════


def build_binary_import_payload(ids: np.ndarray, vecs: np.ndarray) -> bytes:
    """Build the wire-format payload for /v2/import."""
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.uint64)
    count, dim = vecs.shape
    rec_dtype = np.dtype([("id", "<u8"), ("vec", "<f4", (dim,))], align=False)
    recs = np.empty(count, dtype=rec_dtype)
    recs["id"] = ids
    recs["vec"] = vecs
    header = struct.pack("<II", count, dim)
    return header + recs.tobytes()


def measure_concurrent_qps(worker_fn, concurrency: int, duration_s: float) -> float:
    """Run worker_fn on `concurrency` threads for `duration_s`, return total QPS."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker_fn, duration_s, wid) for wid in range(concurrency)]
        total = sum(f.result() for f in futures)
    return round(total / duration_s, 1)


# ═══════════════════════════════════════════════════════════════════
# DeepData benchmark
# ═══════════════════════════════════════════════════════════════════


def bench_deepdata(ds: Dataset, base: np.ndarray, queries: np.ndarray,
                   gt: np.ndarray, cfg: RunConfig) -> BenchResult:
    import httpx

    r = BenchResult(vdb="deepdata", dataset=ds.name, dim=ds.dim,
                    n_vectors=len(base), n_queries=len(queries))
    base_url = f"http://127.0.0.1:{cfg.port}"
    coll = "bench"
    n_search = min(len(queries), cfg.n_search)
    warmup = min(cfg.warmup, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        try:
            with contextlib.suppress(Exception):
                client.delete(f"/v2/collections/{coll}")

            client.post("/v2/collections", json={
                "Name": coll,
                "Fields": [{"Name": "embedding", "Type": 0, "Dim": ds.dim}],
            }).raise_for_status()

            # ── INSERT via binary import ──
            print(f"    Inserting {len(base):,} vectors via binary import...")
            t_insert = time.perf_counter()
            bs = cfg.deepdata_batch_size
            for off in range(0, len(base), bs):
                end = min(off + bs, len(base))
                payload = build_binary_import_payload(
                    ids=np.arange(off, end, dtype=np.uint64),
                    vecs=base[off:end],
                )
                for attempt in range(5):
                    resp = client.post(
                        f"/v2/import?collection={coll}&field=embedding",
                        content=payload,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=120.0,
                    )
                    if resp.status_code == 429:
                        time.sleep(0.25 * (2 ** attempt))
                        continue
                    resp.raise_for_status()
                    break
            insert_time = time.perf_counter() - t_insert
            r.insert_qps = round(len(base) / insert_time, 1) if insert_time > 0 else 0.0
            print(f"    Insert: {insert_time:.2f}s ({r.insert_qps:.0f} vec/s)")
            time.sleep(cfg.settle_time_s)

            # ── SEARCH: warmup + timed recall ──
            measured = n_search - warmup
            print(f"    Searching ({measured} timed queries, {warmup} warmup)...")
            recalls_10: list[float] = []
            recalls_100: list[float] = []
            latencies_ms: list[float] = []

            def do_search(q_list: list[float], top_k: int = 100) -> list[int]:
                p = {"collection": coll, "queries": {"embedding": q_list},
                     "top_k": top_k, "ef_search": ds.ef_search}
                for attempt in range(5):
                    resp = client.post("/v2/search", json=p)
                    if resp.status_code == 429:
                        time.sleep(0.1 * (2 ** attempt))
                        continue
                    resp.raise_for_status()
                    return [int(d.get("id", d.get("ID", 0)))
                            for d in resp.json().get("documents", [])]
                raise RuntimeError("search exhausted retries")

            # warmup
            for qi in range(warmup):
                do_search(query_lists[qi])

            # timed
            for qi in range(warmup, n_search):
                t0 = time.perf_counter()
                retrieved = do_search(query_lists[qi])
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                recalls_10.append(compute_recall(retrieved, gt[qi], 10))
                recalls_100.append(compute_recall(retrieved, gt[qi], 100))

            r.recall_at_10 = round(float(np.mean(recalls_10)), 4) if recalls_10 else 0.0
            r.recall_at_100 = round(float(np.mean(recalls_100)), 4) if recalls_100 else 0.0
            total_s = sum(latencies_ms) / 1000.0
            r.search_serial_qps = round(measured / total_s, 1) if total_s > 0 else 0.0
            r.search_p50_ms = percentile_ms(latencies_ms, 50)
            r.search_p95_ms = percentile_ms(latencies_ms, 95)
            r.search_p99_ms = percentile_ms(latencies_ms, 99)

            # ── CONCURRENT SEARCH ──
            print(f"    Concurrent search ({cfg.concurrency} threads, {cfg.concurrent_duration_s:.0f}s)...")

            def _worker(duration_s: float, worker_id: int) -> int:
                cnt = 0
                deadline = time.perf_counter() + duration_s
                with httpx.Client(base_url=base_url, timeout=30.0) as wc:
                    q_idx = worker_id % max(1, n_search)
                    while time.perf_counter() < deadline:
                        p = {"collection": coll,
                             "queries": {"embedding": query_lists[q_idx]},
                             "top_k": 10, "ef_search": ds.ef_search}
                        resp = wc.post("/v2/search", json=p)
                        if resp.status_code == 429:
                            time.sleep(0.01)
                            continue
                        cnt += 1
                        q_idx = (q_idx + 1) % max(1, n_search)
                return cnt

            r.search_concurrent_qps = measure_concurrent_qps(
                _worker, cfg.concurrency, cfg.concurrent_duration_s)

        except Exception as e:
            r.error = str(e)
            print(f"    ERROR: {e}")
        finally:
            with contextlib.suppress(Exception):
                client.delete(f"/v2/collections/{coll}")

    return r


# ═══════════════════════════════════════════════════════════════════
# Qdrant benchmark
# ═══════════════════════════════════════════════════════════════════


def bench_qdrant(ds: Dataset, base: np.ndarray, queries: np.ndarray,
                 gt: np.ndarray, cfg: RunConfig) -> BenchResult:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (Distance, HnswConfigDiff, PointStruct,
                                          SearchParams, VectorParams)
    except ImportError:
        print("    qdrant-client not installed, skipping")
        return BenchResult(vdb="qdrant", dataset=ds.name, error="qdrant-client not installed")

    r = BenchResult(vdb="qdrant", dataset=ds.name, dim=ds.dim,
                    n_vectors=len(base), n_queries=len(queries))
    client = QdrantClient(url="http://127.0.0.1:6333", timeout=120)
    coll = "bench"
    n_search = min(len(queries), cfg.n_search)
    warmup = min(cfg.warmup, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]

    try:
        with contextlib.suppress(Exception):
            client.delete_collection(coll)

        client.create_collection(
            coll,
            vectors_config=VectorParams(size=ds.dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=ds.m, ef_construct=ds.ef_construction),
        )

        # ── INSERT ──
        print(f"    Inserting {len(base):,} vectors...")
        t_insert = time.perf_counter()
        bs = cfg.qdrant_batch_size
        for off in range(0, len(base), bs):
            end = min(off + bs, len(base))
            pts = [PointStruct(id=i, vector=base[i].tolist()) for i in range(off, end)]
            client.upsert(coll, pts)
        insert_time = time.perf_counter() - t_insert
        r.insert_qps = round(len(base) / insert_time, 1) if insert_time > 0 else 0.0
        print(f"    Insert: {insert_time:.2f}s ({r.insert_qps:.0f} vec/s)")

        # Wait for indexing
        while True:
            info = client.get_collection(coll)
            if info.status.value == "green":
                break
            time.sleep(1)

        # ── SEARCH: warmup + timed ──
        measured = n_search - warmup
        print(f"    Searching ({measured} timed queries, {warmup} warmup)...")
        recalls_10: list[float] = []
        recalls_100: list[float] = []
        latencies_ms: list[float] = []
        params = SearchParams(hnsw_ef=ds.ef_search)

        # warmup
        for qi in range(warmup):
            client.search(coll, query_vector=query_lists[qi], limit=100,
                          search_params=params)

        # timed
        for qi in range(warmup, n_search):
            t0 = time.perf_counter()
            hits = client.search(coll, query_vector=query_lists[qi], limit=100,
                                 search_params=params)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            retrieved = [int(h.id) for h in hits]
            recalls_10.append(compute_recall(retrieved, gt[qi], 10))
            recalls_100.append(compute_recall(retrieved, gt[qi], 100))

        r.recall_at_10 = round(float(np.mean(recalls_10)), 4) if recalls_10 else 0.0
        r.recall_at_100 = round(float(np.mean(recalls_100)), 4) if recalls_100 else 0.0
        total_s = sum(latencies_ms) / 1000.0
        r.search_serial_qps = round(measured / total_s, 1) if total_s > 0 else 0.0
        r.search_p50_ms = percentile_ms(latencies_ms, 50)
        r.search_p95_ms = percentile_ms(latencies_ms, 95)
        r.search_p99_ms = percentile_ms(latencies_ms, 99)

        # ── CONCURRENT SEARCH ──
        print(f"    Concurrent search ({cfg.concurrency} threads, {cfg.concurrent_duration_s:.0f}s)...")

        def _worker(duration_s: float, worker_id: int) -> int:
            cnt = 0
            deadline = time.perf_counter() + duration_s
            qc = QdrantClient(url="http://127.0.0.1:6333", timeout=30)
            try:
                q_idx = worker_id % max(1, n_search)
                while time.perf_counter() < deadline:
                    qc.search(coll, query_vector=query_lists[q_idx], limit=10,
                              search_params=params)
                    cnt += 1
                    q_idx = (q_idx + 1) % max(1, n_search)
                return cnt
            finally:
                qc.close()

        r.search_concurrent_qps = measure_concurrent_qps(
            _worker, cfg.concurrency, cfg.concurrent_duration_s)

    except Exception as e:
        r.error = str(e)
        print(f"    ERROR: {e}")
    finally:
        with contextlib.suppress(Exception):
            client.delete_collection(coll)
        client.close()

    return r


# ═══════════════════════════════════════════════════════════════════
# Load dataset
# ═══════════════════════════════════════════════════════════════════


def load_dataset(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load base vectors, queries, and ground truth for a dataset."""
    if not ds.base_path.exists():
        print(f"  Dataset files not found: {ds.base_path}")
        print("  Run: python benchmarks/download_datasets.py")
        return None

    print(f"  Loading {ds.name}...")
    base = read_fvecs(ds.base_path, max_n=ds.n_base if ds.n_base > 0 else None)
    queries = read_fvecs(ds.query_path)
    gt = np.load(ds.gt_path)

    # Truncate base if needed
    if ds.n_base > 0 and len(base) > ds.n_base:
        base = base[:ds.n_base]

    # Recompute GT if it references indices beyond our base size
    if gt.size and int(gt.max()) >= len(base):
        print(f"  Recomputing ground truth for {len(base):,} vector subset...")
        gt = compute_ground_truth(base, queries, k=min(100, len(base)))

    print(f"  Loaded: {len(base):,} base ({ds.dim}d), {len(queries):,} queries, GT shape {gt.shape}")
    return base, queries, gt


# ═══════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════


def print_results_table(results: list[BenchResult], cfg: RunConfig) -> None:
    """Print a clean markdown table of results."""
    ct = cfg.concurrency
    print()
    print("=" * 132)
    print("RECALL BENCHMARK RESULTS")
    print("=" * 132)
    print()

    header = (
        f"| {'VDB':<10} | {'Dataset':<16} | {'Dim':>4} | {'N':>8} "
        f"| {'R@10':>6} | {'R@100':>6} "
        f"| {'Ins QPS':>8} | {'Srch QPS':>9} | {f'{ct}T QPS':>8} "
        f"| {'p50ms':>6} | {'p95ms':>6} | {'p99ms':>6} |"
    )
    sep = "|" + "|".join("-" * (len(c)) for c in header.split("|")[1:-1]) + "|"

    print(header)
    print(sep)

    for r in results:
        if r.error:
            print(f"| {r.vdb:<10} | {r.dataset:<16} | {'ERROR: ' + r.error:<80} |")
            continue
        print(
            f"| {r.vdb:<10} | {r.dataset:<16} | {r.dim:>4} | {r.n_vectors:>8,} "
            f"| {r.recall_at_10:>6.4f} | {r.recall_at_100:>6.4f} "
            f"| {r.insert_qps:>8.0f} | {r.search_serial_qps:>9.0f} | {r.search_concurrent_qps:>8.0f} "
            f"| {r.search_p50_ms:>6.1f} | {r.search_p95_ms:>6.1f} | {r.search_p99_ms:>6.1f} |"
        )

    print()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


ALL_DATASETS = ["sift-10k", "sift-100k", "code-562", "glove-50d", "glove-100d", "glove-100d-full"]
ALL_VDBS = ["deepdata", "qdrant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-dataset recall benchmark")
    parser.add_argument("--vdb", nargs="*", default=["deepdata"], choices=ALL_VDBS,
                        help="VDBs to benchmark (default: deepdata)")
    parser.add_argument("--dataset", nargs="*", default=None, choices=ALL_DATASETS,
                        help="Datasets to benchmark (default: all available)")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip DeepData server rebuild")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"DeepData port (default: {DEFAULT_PORT})")
    parser.add_argument("--n-search", type=int, default=200,
                        help="Max timed search queries per dataset (default: 200)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup queries before timed measurement (default: 10)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent workers for throughput test (default: 8)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Concurrent throughput test duration in seconds (default: 5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        port=args.port,
        n_search=args.n_search,
        warmup=args.warmup,
        concurrency=args.concurrency,
        concurrent_duration_s=args.duration,
    )

    configs = get_dataset_configs()
    ds_names = args.dataset or [name for name, ds in configs.items() if ds.base_path.exists()]
    if not ds_names:
        print("No datasets found. Run download first:")
        print("  python benchmarks/download_datasets.py")
        sys.exit(1)

    print(f"VDBs: {', '.join(args.vdb)}")
    print(f"Datasets: {', '.join(ds_names)}")
    print(f"Settings: n_search={cfg.n_search}, warmup={cfg.warmup}, "
          f"concurrency={cfg.concurrency}, duration={cfg.concurrent_duration_s}s")
    print()

    if "deepdata" in args.vdb and not args.skip_build:
        if not build_server():
            sys.exit(1)

    results: list[BenchResult] = []

    for ds_name in ds_names:
        ds = configs[ds_name]
        loaded = load_dataset(ds)
        if loaded is None:
            continue
        base, queries, gt = loaded

        for vdb in args.vdb:
            print(f"\n{'─' * 70}")
            print(f"  {vdb} / {ds_name} ({len(base):,} x {ds.dim}d)")
            print(f"{'─' * 70}")

            if vdb == "deepdata":
                kill_servers()
                handle = start_server(ds, cfg)
                if handle is None:
                    results.append(BenchResult(vdb="deepdata", dataset=ds_name,
                                               error="server failed to start"))
                    continue
                try:
                    results.append(bench_deepdata(ds, base, queries, gt, cfg))
                finally:
                    stop_process(handle.proc)

            elif vdb == "qdrant":
                results.append(bench_qdrant(ds, base, queries, gt, cfg))

            lr = results[-1]
            if lr.error:
                print(f"    FAILED: {lr.error}")
            else:
                print(f"    recall@10={lr.recall_at_10:.4f}  recall@100={lr.recall_at_100:.4f}  "
                      f"serial_qps={lr.search_serial_qps:.0f}  "
                      f"{cfg.concurrency}t_qps={lr.search_concurrent_qps:.0f}")

    kill_servers()
    print_results_table(results, cfg)

    if args.json:
        payload = {"settings": asdict(cfg), "results": [asdict(r) for r in results]}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
