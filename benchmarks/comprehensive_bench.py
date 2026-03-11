#!/usr/bin/env python3
"""Comprehensive DeepData benchmark — real data, all features, competitive comparison.

Tests every major DeepData optimization (prenorm, FP16, gRPC, payload filters)
across real datasets and optionally compares against Qdrant.

Usage:
    python benchmarks/comprehensive_bench.py                          # Full suite
    python benchmarks/comprehensive_bench.py --quick                  # Smoke test (~2 min)
    python benchmarks/comprehensive_bench.py --dataset sift-100k      # Single dataset
    python benchmarks/comprehensive_bench.py --config default grpc    # Specific configs
    python benchmarks/comprehensive_bench.py --compare-qdrant         # With Qdrant comparison
    python benchmarks/comprehensive_bench.py --json results.json      # Save JSON
    python benchmarks/comprehensive_bench.py --skip-build             # Reuse existing binary
"""

import argparse
import concurrent.futures
import contextlib
import json
import os
import platform
import signal
import shutil
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Reuse existing infrastructure ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_datasets import CACHE_DIR, compute_ground_truth, read_fvecs

# gRPC stubs path
GRPC_STUBS_DIR = Path(__file__).resolve().parent / "competitive" / "live" / "adapters"
sys.path.insert(0, str(GRPC_STUBS_DIR))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_BINARY = PROJECT_ROOT / "deepdata-server"
BENCH_DATA_DIR = Path("/tmp/deepdata-bench-comprehensive")
DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 50051

# ── Old benchmark results (constants from competitive/live runs pre-upgrade) ─
OLD_RESULTS = {
    "sift-100k": {
        "insert_qps": 689,
        "search_serial_qps": 287,
        "search_p50_ms": 21.6,
        "search_p95_ms": 24.3,
        "recall_at_10": 0.99,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchConfig:
    """A specific benchmark configuration (prenorm + quantization + transport)."""
    name: str
    prenormalize: bool = True
    quantization: str = "none"   # "none" or "float16"
    transport: str = "http"      # "http" or "grpc"
    description: str = ""


ALL_CONFIGS = {
    "default":    BenchConfig("default",    prenormalize=True,  quantization="none",    transport="http",  description="Baseline with all upgrades"),
    "grpc":       BenchConfig("grpc",       prenormalize=True,  quantization="none",    transport="grpc",  description="gRPC transport"),
    "fp16":       BenchConfig("fp16",       prenormalize=True,  quantization="float16", transport="http",  description="FP16 quantization"),
    "fp16-grpc":  BenchConfig("fp16-grpc",  prenormalize=True,  quantization="float16", transport="grpc",  description="FP16 + gRPC"),
    "no-prenorm": BenchConfig("no-prenorm", prenormalize=False, quantization="none",    transport="http",  description="Without prenormalization"),
}


@dataclass
class DatasetDef:
    name: str
    dim: int
    base_path: Path
    query_path: Path
    gt_path: Path
    n_base: int  # 0 = load all


def get_datasets() -> dict[str, DatasetDef]:
    sift = CACHE_DIR / "sift"
    glove = CACHE_DIR / "glove"
    code = CACHE_DIR / "code"
    return {
        "sift-100k": DatasetDef(
            name="sift-100k", dim=128, n_base=100_000,
            base_path=sift / "sift_base_100k_norm.fvecs",
            query_path=sift / "sift_query_norm.fvecs",
            gt_path=sift / "sift_100k_gt100.npy",
        ),
        "glove-100d": DatasetDef(
            name="glove-100d", dim=100, n_base=10_000,
            base_path=glove / "glove_100d_base_norm.fvecs",
            query_path=glove / "glove_100d_query_norm.fvecs",
            gt_path=glove / "glove_100d_gt100.npy",
        ),
        "code-562": DatasetDef(
            name="code-562", dim=1536, n_base=0,
            base_path=code / "code_1536d_base_norm.fvecs",
            query_path=code / "code_1536d_query_norm.fvecs",
            gt_path=code / "code_1536d_gt100.npy",
        ),
    }


@dataclass
class RunSettings:
    http_port: int = DEFAULT_HTTP_PORT
    grpc_port: int = DEFAULT_GRPC_PORT
    n_search: int = 200
    warmup: int = 10
    concurrency: int = 8
    concurrent_duration_s: float = 5.0
    batch_size: int = 5_000
    startup_timeout_s: float = 20.0
    settle_time_s: float = 1.0
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 200


# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchResult:
    dataset: str = ""
    config: str = ""
    dim: int = 0
    n_vectors: int = 0
    n_queries: int = 0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    insert_qps: float = 0.0
    search_serial_qps: float = 0.0
    search_concurrent_qps: float = 0.0
    search_p50_ms: float = 0.0
    search_p95_ms: float = 0.0
    search_p99_ms: float = 0.0
    # Filtered search (only for default config)
    filtered_p50_ms: float = 0.0
    filtered_recall_at_10: float = 0.0
    error: str = ""


def compute_recall(retrieved: list[int], gt: np.ndarray, k: int) -> float:
    gt_set = set(int(x) for x in gt[:k])
    ret_set = set(int(x) for x in retrieved[:k])
    return len(gt_set & ret_set) / len(gt_set) if gt_set else 0.0


def percentile_ms(latencies: list[float], pct: float) -> float:
    return round(float(np.percentile(latencies, pct)), 2) if latencies else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Server management
# ═══════════════════════════════════════════════════════════════════════════════


def kill_servers(http_port: int = DEFAULT_HTTP_PORT):
    subprocess.run(["pkill", "-f", "deepdata-server"], capture_output=True)
    # Wait for port to be released
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", http_port), timeout=0.2):
                time.sleep(0.2)
        except OSError:
            break
    time.sleep(0.3)


def build_server() -> bool:
    print("  Building server...")
    result = subprocess.run(
        ["go", "build", "-o", str(SERVER_BINARY), "./cmd/deepdata/"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  BUILD FAILED: {result.stderr}")
        return False
    print("  Build OK")
    return True


def wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def start_server(settings: RunSettings) -> subprocess.Popen | None:
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
        "HNSW_M": str(settings.hnsw_m),
        "HNSW_EF_CONSTRUCTION": str(settings.hnsw_ef_construction),
        "HNSW_EFSEARCH": str(settings.hnsw_ef_search),
        "GRPC_PORT": str(settings.grpc_port),
    }

    stderr_path = BENCH_DATA_DIR / "server.stderr"
    stderr_file = open(stderr_path, "w")

    proc = subprocess.Popen(
        [str(SERVER_BINARY)],
        env=env, stdout=subprocess.DEVNULL, stderr=stderr_file,
        start_new_session=True,
    )

    import httpx
    ready = False
    if wait_for_port(settings.http_port, min(3.0, settings.startup_timeout_s)):
        base_url = f"http://127.0.0.1:{settings.http_port}"
        with httpx.Client(base_url=base_url, timeout=2.0) as client:
            deadline = time.perf_counter() + settings.startup_timeout_s
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
    stderr_file.flush()
    stderr_file.close()

    if ready:
        print("  Server ready")
        return proc

    stop_process(proc)
    err = ""
    try:
        err = stderr_path.read_text().strip().splitlines()[-10:]
        err = "\n".join(err)
    except Exception:
        pass
    print(f"  Server failed to start\n{err}")
    return None


def stop_process(proc: subprocess.Popen) -> None:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Binary import helper
# ═══════════════════════════════════════════════════════════════════════════════


def build_binary_import_payload(ids: np.ndarray, vecs: np.ndarray) -> bytes:
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.uint64)
    count, dim = vecs.shape
    rec_dtype = np.dtype([("id", "<u8"), ("vec", "<f4", (dim,))], align=False)
    recs = np.empty(count, dtype=rec_dtype)
    recs["id"] = ids
    recs["vec"] = vecs
    header = struct.pack("<II", count, dim)
    return header + recs.tobytes()


# ═══════════════════════════════════════════════════════════════════════════════
# Collection creation
# ═══════════════════════════════════════════════════════════════════════════════


def create_collection_http(client, coll: str, dim: int, cfg: BenchConfig, settings: RunSettings):
    with contextlib.suppress(Exception):
        client.delete(f"/v2/collections/{coll}")

    index_params: dict[str, Any] = {
        "m": settings.hnsw_m,
        "ef_construction": settings.hnsw_ef_construction,
        "ef_search": settings.hnsw_ef_search,
        "prenormalize": cfg.prenormalize,
    }
    if cfg.quantization != "none":
        index_params["quantization"] = {"type": cfg.quantization}

    body = {
        "Name": coll,
        "Fields": [{
            "Name": "embedding",
            "Type": 0,
            "Dim": dim,
            "Index": {
                "type": "hnsw",
                "params": index_params,
            },
        }],
    }
    client.post("/v2/collections", json=body).raise_for_status()


def create_collection_grpc(stub, pb2, coll: str, dim: int, cfg: BenchConfig, settings: RunSettings):
    with contextlib.suppress(Exception):
        stub.DeleteCollection(pb2.DeleteCollectionRequest(name=coll))

    params = {
        "m": float(settings.hnsw_m),
        "ef_construction": float(settings.hnsw_ef_construction),
        "ef_search": float(settings.hnsw_ef_search),
        "prenormalize": 1.0 if cfg.prenormalize else 0.0,
    }
    if cfg.quantization != "none":
        # Encode quantization type as a numeric flag (1 = float16)
        params["quantization_float16"] = 1.0

    field = pb2.VectorFieldConfig(
        name="embedding", type=0, dim=dim,
        index_type="hnsw", index_params=params,
    )
    stub.CreateCollection(pb2.CreateCollectionRequest(name=coll, fields=[field]))


# ═══════════════════════════════════════════════════════════════════════════════
# Search helpers
# ═══════════════════════════════════════════════════════════════════════════════


def search_http(client, coll: str, query: list[float], top_k: int, ef_search: int,
                meta_filter: dict | None = None) -> list[int]:
    payload: dict[str, Any] = {
        "collection": coll,
        "queries": {"embedding": query},
        "top_k": top_k,
        "ef_search": ef_search,
    }
    if meta_filter:
        payload["filter"] = meta_filter
    for attempt in range(5):
        resp = client.post("/v2/search", json=payload)
        if resp.status_code == 429:
            time.sleep(0.1 * (2 ** attempt))
            continue
        resp.raise_for_status()
        return [int(d.get("id", d.get("ID", 0))) for d in resp.json().get("documents", [])]
    raise RuntimeError("search exhausted retries")


def search_grpc(stub, pb2, coll: str, query: list[float], top_k: int, ef_search: int) -> list[int]:
    query_vec = pb2.VectorData(dense=pb2.DenseVector(values=query))
    req = pb2.SearchRequest(
        collection=coll,
        queries={"embedding": query_vec},
        top_k=top_k,
        ef_search=ef_search,
    )
    resp = stub.Search(req)
    return [int(hit.id) for hit in resp.results]


# ═══════════════════════════════════════════════════════════════════════════════
# Concurrent QPS measurement
# ═══════════════════════════════════════════════════════════════════════════════


def measure_concurrent_qps(worker_fn, concurrency: int, duration_s: float) -> float:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker_fn, duration_s, wid) for wid in range(concurrency)]
        total = sum(f.result() for f in futures)
    return round(total / duration_s, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark: DeepData (per config)
# ═══════════════════════════════════════════════════════════════════════════════


def bench_deepdata(ds: DatasetDef, base: np.ndarray, queries: np.ndarray,
                   gt: np.ndarray, cfg: BenchConfig, settings: RunSettings,
                   run_filtered: bool = False) -> BenchResult:
    import httpx

    r = BenchResult(dataset=ds.name, config=cfg.name, dim=ds.dim,
                    n_vectors=len(base), n_queries=len(queries))

    base_url = f"http://127.0.0.1:{settings.http_port}"
    coll = "bench"
    n_search = min(len(queries), settings.n_search)
    warmup = min(settings.warmup, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]
    ef_search = settings.hnsw_ef_search

    # ── gRPC setup (if needed) ──
    grpc_stub = None
    grpc_pb2 = None
    grpc_channel = None
    use_grpc = cfg.transport == "grpc"

    if use_grpc:
        try:
            import grpc
            from deepdata.v1 import deepdata_pb2, deepdata_pb2_grpc
            grpc_pb2 = deepdata_pb2
            # Wait for gRPC port to be ready
            if not wait_for_port(settings.grpc_port, 10.0):
                r.error = f"gRPC port {settings.grpc_port} not ready"
                print(f"    ERROR: {r.error}")
                return r
            grpc_channel = grpc.insecure_channel(
                f"127.0.0.1:{settings.grpc_port}",
                options=[
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ],
            )
            grpc_stub = deepdata_pb2_grpc.DeepDataStub(grpc_channel)
        except Exception as e:
            r.error = f"gRPC setup failed: {e}"
            print(f"    ERROR: {r.error}")
            return r

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        try:
            # ── CREATE COLLECTION (always via HTTP — most reliable) ──
            create_collection_http(client, coll, ds.dim, cfg, settings)

            # ── INSERT via binary import (always HTTP — no gRPC bulk import) ──
            print(f"    Inserting {len(base):,} vectors...")
            t_insert = time.perf_counter()
            bs = settings.batch_size
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
            time.sleep(settings.settle_time_s)

            # ── SEARCH: warmup + timed recall ──
            measured = n_search - warmup
            print(f"    Searching ({measured} timed, {warmup} warmup, transport={cfg.transport})...")
            recalls_10: list[float] = []
            recalls_100: list[float] = []
            latencies_ms: list[float] = []

            def do_search(q: list[float], top_k: int = 100) -> list[int]:
                if use_grpc and grpc_stub and grpc_pb2:
                    return search_grpc(grpc_stub, grpc_pb2, coll, q, top_k, ef_search)
                return search_http(client, coll, q, top_k, ef_search)

            for qi in range(warmup):
                do_search(query_lists[qi])

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
            print(f"    Concurrent search ({settings.concurrency} threads, {settings.concurrent_duration_s:.0f}s)...")

            if use_grpc and grpc_stub and grpc_pb2:
                def _worker(duration_s: float, worker_id: int) -> int:
                    import grpc as _grpc
                    from deepdata.v1 import deepdata_pb2 as _pb2, deepdata_pb2_grpc as _grpc_stub
                    ch = _grpc.insecure_channel(
                        f"127.0.0.1:{settings.grpc_port}",
                        options=[
                            ("grpc.max_send_message_length", 64 * 1024 * 1024),
                            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                        ],
                    )
                    st = _grpc_stub.DeepDataStub(ch)
                    cnt = 0
                    deadline = time.perf_counter() + duration_s
                    q_idx = worker_id % max(1, n_search)
                    while time.perf_counter() < deadline:
                        qv = _pb2.VectorData(dense=_pb2.DenseVector(values=query_lists[q_idx]))
                        req = _pb2.SearchRequest(
                            collection=coll, queries={"embedding": qv},
                            top_k=10, ef_search=ef_search,
                        )
                        st.Search(req)
                        cnt += 1
                        q_idx = (q_idx + 1) % max(1, n_search)
                    ch.close()
                    return cnt
            else:
                def _worker(duration_s: float, worker_id: int) -> int:
                    cnt = 0
                    deadline = time.perf_counter() + duration_s
                    with httpx.Client(base_url=base_url, timeout=30.0) as wc:
                        q_idx = worker_id % max(1, n_search)
                        while time.perf_counter() < deadline:
                            p = {"collection": coll,
                                 "queries": {"embedding": query_lists[q_idx]},
                                 "top_k": 10, "ef_search": ef_search}
                            resp = wc.post("/v2/search", json=p)
                            if resp.status_code == 429:
                                time.sleep(0.01)
                                continue
                            cnt += 1
                            q_idx = (q_idx + 1) % max(1, n_search)
                    return cnt

            r.search_concurrent_qps = measure_concurrent_qps(
                _worker, settings.concurrency, settings.concurrent_duration_s)

            # ── FILTERED SEARCH (default config only) ──
            if run_filtered and len(base) >= 100:
                print("    Filtered search (payload index)...")
                n_meta = min(1000, len(base))
                categories = [f"cat_{i % 5}" for i in range(n_meta)]

                # Add metadata to first n_meta vectors via individual inserts
                for i in range(n_meta):
                    body = {
                        "collection": coll,
                        "id": i,
                        "metadata": {"category": categories[i]},
                    }
                    for attempt in range(3):
                        resp = client.put(f"/v2/documents/{coll}/{i}/metadata",
                                          json={"category": categories[i]})
                        if resp.status_code == 429:
                            time.sleep(0.1 * (2 ** attempt))
                            continue
                        # Accept any 2xx or 404 (endpoint may differ)
                        break

                time.sleep(0.5)

                # Filtered search with $eq
                filt = {"category": {"$eq": "cat_0"}}
                filtered_lats: list[float] = []
                filtered_recalls: list[float] = []
                n_filt = min(50, n_search - warmup)

                for qi in range(warmup, warmup + n_filt):
                    t0 = time.perf_counter()
                    retrieved = search_http(client, coll, query_lists[qi], 100, ef_search, filt)
                    filtered_lats.append((time.perf_counter() - t0) * 1000.0)
                    # Recall against unfiltered GT (approximate — filtered recall is always lower)
                    filtered_recalls.append(compute_recall(retrieved, gt[qi], 10))

                r.filtered_p50_ms = percentile_ms(filtered_lats, 50)
                r.filtered_recall_at_10 = round(float(np.mean(filtered_recalls)), 4) if filtered_recalls else 0.0
                print(f"    Filtered: P50={r.filtered_p50_ms}ms, R@10={r.filtered_recall_at_10:.4f}")

        except Exception as e:
            r.error = str(e)
            print(f"    ERROR: {e}")
        finally:
            with contextlib.suppress(Exception):
                client.delete(f"/v2/collections/{coll}")
            if grpc_channel:
                grpc_channel.close()

    return r


# ═══════════════════════════════════════════════════════════════════════════════
# Qdrant benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def bench_qdrant(ds: DatasetDef, base: np.ndarray, queries: np.ndarray,
                 gt: np.ndarray, settings: RunSettings) -> BenchResult:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (Distance, HnswConfigDiff, PointStruct,
                                          SearchParams, VectorParams)
    except ImportError:
        print("    qdrant-client not installed, skipping")
        return BenchResult(dataset=ds.name, config="qdrant", error="qdrant-client not installed")

    r = BenchResult(dataset=ds.name, config="qdrant", dim=ds.dim,
                    n_vectors=len(base), n_queries=len(queries))
    client = QdrantClient(url="http://127.0.0.1:6333", timeout=120)
    coll = "bench"
    n_search = min(len(queries), settings.n_search)
    warmup = min(settings.warmup, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]

    try:
        with contextlib.suppress(Exception):
            client.delete_collection(coll)

        client.create_collection(
            coll,
            vectors_config=VectorParams(size=ds.dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construction),
        )

        # Insert
        print(f"    Inserting {len(base):,} vectors...")
        t_insert = time.perf_counter()
        bs = 500
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

        # Search
        measured = n_search - warmup
        print(f"    Searching ({measured} timed, {warmup} warmup)...")
        recalls_10: list[float] = []
        recalls_100: list[float] = []
        latencies_ms: list[float] = []
        params = SearchParams(hnsw_ef=settings.hnsw_ef_search)

        for qi in range(warmup):
            client.search(coll, query_vector=query_lists[qi], limit=100, search_params=params)

        for qi in range(warmup, n_search):
            t0 = time.perf_counter()
            hits = client.search(coll, query_vector=query_lists[qi], limit=100, search_params=params)
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

        # Concurrent
        print(f"    Concurrent search ({settings.concurrency} threads, {settings.concurrent_duration_s:.0f}s)...")

        def _worker(duration_s: float, worker_id: int) -> int:
            cnt = 0
            deadline = time.perf_counter() + duration_s
            qc = QdrantClient(url="http://127.0.0.1:6333", timeout=30)
            try:
                q_idx = worker_id % max(1, n_search)
                while time.perf_counter() < deadline:
                    qc.search(coll, query_vector=query_lists[q_idx], limit=10, search_params=params)
                    cnt += 1
                    q_idx = (q_idx + 1) % max(1, n_search)
                return cnt
            finally:
                qc.close()

        r.search_concurrent_qps = measure_concurrent_qps(
            _worker, settings.concurrency, settings.concurrent_duration_s)

    except Exception as e:
        r.error = str(e)
        print(f"    ERROR: {e}")
    finally:
        with contextlib.suppress(Exception):
            client.delete_collection(coll)
        client.close()

    return r


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_dataset(ds: DatasetDef) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not ds.base_path.exists():
        print(f"  Dataset files not found: {ds.base_path}")
        print("  Run: python benchmarks/download_datasets.py")
        return None

    print(f"  Loading {ds.name}...")
    base = read_fvecs(ds.base_path, max_n=ds.n_base if ds.n_base > 0 else None)
    queries = read_fvecs(ds.query_path)
    gt = np.load(ds.gt_path)

    if ds.n_base > 0 and len(base) > ds.n_base:
        base = base[:ds.n_base]

    if gt.size and int(gt.max()) >= len(base):
        print(f"  Recomputing ground truth for {len(base):,} vector subset...")
        gt = compute_ground_truth(base, queries, k=min(100, len(base)))

    print(f"  Loaded: {len(base):,} base ({ds.dim}d), {len(queries):,} queries")
    return base, queries, gt


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════


def get_cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def print_full_results(results: list[BenchResult], settings: RunSettings):
    ct = settings.concurrency
    cpu = get_cpu_name()

    print()
    print("=" * 120)
    print(f"DEEPDATA COMPREHENSIVE BENCHMARK — {cpu}")
    print("=" * 120)
    print()

    header = (
        f"{'Dataset':<14} {'Config':<12} {'R@10':>6} {'R@100':>6} "
        f"{'Ins QPS':>9} {'Serial QPS':>11} {f'{ct}T QPS':>8} "
        f"{'P50':>6} {'P95':>6} {'P99':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if r.error:
            print(f"{r.dataset:<14} {r.config:<12} ERROR: {r.error}")
            continue
        print(
            f"{r.dataset:<14} {r.config:<12} "
            f"{r.recall_at_10:>6.4f} {r.recall_at_100:>6.4f} "
            f"{r.insert_qps:>9.0f} {r.search_serial_qps:>11.0f} {r.search_concurrent_qps:>8.0f} "
            f"{r.search_p50_ms:>6.1f} {r.search_p95_ms:>6.1f} {r.search_p99_ms:>6.1f}"
        )
    print()


def print_before_after(results: list[BenchResult]):
    # Find best "default" config result for datasets that have old data
    for ds_name, old in OLD_RESULTS.items():
        default = next((r for r in results if r.dataset == ds_name and r.config == "default" and not r.error), None)
        if not default:
            continue

        print("=" * 80)
        print(f"UPGRADE IMPACT — {ds_name}, same hardware")
        print("=" * 80)
        print()
        print(f"{'Metric':<20} {'Before (v1)':>14} {'After (v2)':>14} {'Improvement':>16}")
        print("-" * 68)

        def row(name, old_val, new_val, unit="", higher_better=True):
            if old_val == 0:
                return
            if higher_better:
                ratio = new_val / old_val if old_val else 0
                imp = f"{ratio:.1f}x"
            else:
                ratio = old_val / new_val if new_val else 0
                imp = f"{ratio:.1f}x faster"
            print(f"{name:<20} {old_val:>12.1f}{unit:>2} {new_val:>12.1f}{unit:>2} {imp:>16}")

        row("Insert QPS",    old["insert_qps"],        default.insert_qps)
        row("Search QPS",    old["search_serial_qps"],  default.search_serial_qps)
        row("P50 (ms)",      old["search_p50_ms"],      default.search_p50_ms, "", False)
        row("P95 (ms)",      old["search_p95_ms"],      default.search_p95_ms, "", False)
        print(f"{'Recall@10':<20} {old['recall_at_10']:>14.4f} {default.recall_at_10:>14.4f} {'maintained':>16}")
        print()


def print_feature_impact(results: list[BenchResult]):
    # Group by dataset, compare configs
    datasets = sorted(set(r.dataset for r in results))
    for ds_name in datasets:
        ds_results = {r.config: r for r in results if r.dataset == ds_name and not r.error}
        default = ds_results.get("default")
        if not default:
            continue

        print("=" * 80)
        print(f"FEATURE IMPACT — {ds_name}")
        print("=" * 80)
        print()
        print(f"{'Feature':<22} {'Metric':<14} {'Without':>10} {'With':>10} {'Delta':>10}")
        print("-" * 70)

        no_prenorm = ds_results.get("no-prenorm")
        if no_prenorm:
            delta = (default.search_serial_qps / no_prenorm.search_serial_qps - 1) * 100 if no_prenorm.search_serial_qps else 0
            print(f"{'Prenormalization':<22} {'Serial QPS':<14} {no_prenorm.search_serial_qps:>10.0f} {default.search_serial_qps:>10.0f} {delta:>+9.0f}%")

        fp16 = ds_results.get("fp16")
        if fp16:
            delta = (fp16.search_serial_qps / default.search_serial_qps - 1) * 100 if default.search_serial_qps else 0
            print(f"{'FP16 Quantization':<22} {'Serial QPS':<14} {default.search_serial_qps:>10.0f} {fp16.search_serial_qps:>10.0f} {delta:>+9.0f}%")

        grpc = ds_results.get("grpc")
        if grpc:
            delta = (grpc.search_serial_qps / default.search_serial_qps - 1) * 100 if default.search_serial_qps else 0
            print(f"{'gRPC transport':<22} {'Serial QPS':<14} {default.search_serial_qps:>10.0f} {grpc.search_serial_qps:>10.0f} {delta:>+9.0f}%")

        fp16_grpc = ds_results.get("fp16-grpc")
        if fp16_grpc:
            delta = (fp16_grpc.search_serial_qps / default.search_serial_qps - 1) * 100 if default.search_serial_qps else 0
            print(f"{'gRPC + FP16':<22} {'Serial QPS':<14} {default.search_serial_qps:>10.0f} {fp16_grpc.search_serial_qps:>10.0f} {delta:>+9.0f}%")

        if default.filtered_p50_ms > 0:
            overhead = (default.filtered_p50_ms / default.search_p50_ms - 1) * 100 if default.search_p50_ms else 0
            print(f"{'Payload filter':<22} {'P50 (ms)':<14} {default.search_p50_ms:>10.1f} {default.filtered_p50_ms:>10.1f} {overhead:>+9.0f}%")

        print()


def print_competitive(dd_results: list[BenchResult], qd_result: BenchResult | None):
    if not qd_result or qd_result.error:
        return

    ds_name = qd_result.dataset
    # Find best DeepData config for each metric
    dd = [r for r in dd_results if r.dataset == ds_name and not r.error]
    if not dd:
        return

    best_insert = max(dd, key=lambda r: r.insert_qps)
    best_recall = max(dd, key=lambda r: r.recall_at_10)
    best_p50 = min(dd, key=lambda r: r.search_p50_ms)
    best_qps = max(dd, key=lambda r: r.search_concurrent_qps)

    print("=" * 80)
    print(f"DEEPDATA vs QDRANT — {ds_name}")
    print("=" * 80)
    print()
    print(f"{'Metric':<22} {'DeepData (best)':>18} {'Qdrant':>12} {'Winner':>20}")
    print("-" * 76)

    def comp(name, dd_val, qd_val, dd_label, higher_better=True):
        if higher_better:
            winner = "DeepData" if dd_val >= qd_val else "Qdrant"
            ratio = max(dd_val, qd_val) / min(dd_val, qd_val) if min(dd_val, qd_val) > 0 else 0
        else:
            winner = "DeepData" if dd_val <= qd_val else "Qdrant"
            ratio = max(dd_val, qd_val) / min(dd_val, qd_val) if min(dd_val, qd_val) > 0 else 0
        margin = f" ({ratio:.1f}x)" if ratio > 1.01 else ""
        print(f"{name:<22} {dd_label:>18} {qd_val:>12.1f} {winner + margin:>20}")

    comp("Insert QPS", best_insert.insert_qps, qd_result.insert_qps,
         f"{best_insert.insert_qps:,.0f} (batch)")
    comp("Recall@10", best_recall.recall_at_10, qd_result.recall_at_10,
         f"{best_recall.recall_at_10:.4f}")
    comp("P50 (ms)", best_p50.search_p50_ms, qd_result.search_p50_ms,
         f"{best_p50.search_p50_ms:.1f} ({best_p50.config})", higher_better=False)

    ct = 8  # concurrent threads
    comp(f"{ct}T QPS", best_qps.search_concurrent_qps, qd_result.search_concurrent_qps,
         f"{best_qps.search_concurrent_qps:,.0f} ({best_qps.config})")
    print()


def generate_recommendations(dd_results: list[BenchResult], qd_result: BenchResult | None = None):
    dd = [r for r in dd_results if not r.error]
    if not dd:
        return

    print("=" * 80)
    print("DATA-DRIVEN RECOMMENDATIONS")
    print("=" * 80)
    print()

    best_insert = max(dd, key=lambda r: r.insert_qps)
    best_latency = min(dd, key=lambda r: r.search_p50_ms)
    best_qps = max(dd, key=lambda r: r.search_concurrent_qps)
    best_recall = max(dd, key=lambda r: r.recall_at_10)

    strengths: list[str] = []

    if best_insert.insert_qps > 5000:
        strengths.append(
            f"High insert throughput: {best_insert.insert_qps:,.0f} vec/s via batch import "
            f"({best_insert.dataset})")
    if best_latency.search_p50_ms < 2.0:
        strengths.append(
            f"Sub-{best_latency.search_p50_ms:.1f}ms P50 latency "
            f"({best_latency.config} on {best_latency.dataset})")
    if best_qps.search_concurrent_qps > 3000:
        strengths.append(
            f"High throughput under load: {best_qps.search_concurrent_qps:,.0f} QPS @ 8 threads "
            f"({best_qps.config} on {best_qps.dataset})")
    if best_recall.recall_at_10 > 0.99:
        strengths.append(
            f"Excellent recall: {best_recall.recall_at_10:.4f} R@10 "
            f"({best_recall.config} on {best_recall.dataset})")

    # gRPC vs HTTP comparison
    http_results = [r for r in dd if r.config in ("default", "fp16")]
    grpc_results = [r for r in dd if r.config in ("grpc", "fp16-grpc")]
    if http_results and grpc_results:
        avg_http = np.mean([r.search_serial_qps for r in http_results])
        avg_grpc = np.mean([r.search_serial_qps for r in grpc_results])
        if avg_grpc > avg_http * 1.1:
            speedup = avg_grpc / avg_http
            strengths.append(f"gRPC gives {speedup:.1f}x serial QPS over HTTP/JSON")

    # Prenorm impact
    prenorm = [r for r in dd if r.config == "default"]
    no_prenorm = [r for r in dd if r.config == "no-prenorm"]
    if prenorm and no_prenorm:
        avg_pre = np.mean([r.search_serial_qps for r in prenorm])
        avg_no = np.mean([r.search_serial_qps for r in no_prenorm])
        if avg_pre > avg_no * 1.05:
            speedup = (avg_pre / avg_no - 1) * 100
            strengths.append(f"Prenormalization adds +{speedup:.0f}% search QPS (free at insert time)")

    print("USE DEEPDATA WHEN:")
    for s in strengths:
        print(f"  - {s}")
    print()

    if qd_result and not qd_result.error:
        print("COMPETITIVE POSITIONING:")
        dd_best_qps = max(dd, key=lambda r: r.search_concurrent_qps)
        dd_best_insert = max(dd, key=lambda r: r.insert_qps)

        if dd_best_insert.insert_qps > qd_result.insert_qps:
            ratio = dd_best_insert.insert_qps / qd_result.insert_qps
            print(f"  - Insert: DeepData {ratio:.1f}x faster "
                  f"({dd_best_insert.insert_qps:,.0f} vs {qd_result.insert_qps:,.0f} vec/s)")
        if dd_best_qps.search_concurrent_qps > qd_result.search_concurrent_qps:
            ratio = dd_best_qps.search_concurrent_qps / qd_result.search_concurrent_qps
            print(f"  - Throughput: DeepData {ratio:.1f}x higher "
                  f"({dd_best_qps.search_concurrent_qps:,.0f} vs {qd_result.search_concurrent_qps:,.0f} QPS)")
        if qd_result.recall_at_10 > best_recall.recall_at_10 + 0.005:
            print(f"  - Recall: Qdrant marginal edge "
                  f"({qd_result.recall_at_10:.4f} vs {best_recall.recall_at_10:.4f})")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive DeepData benchmark — real data, all features")
    parser.add_argument("--dataset", nargs="*", default=None,
                        choices=list(get_datasets().keys()),
                        help="Datasets to benchmark (default: all available)")
    parser.add_argument("--config", nargs="*", default=None,
                        choices=list(ALL_CONFIGS.keys()),
                        help="Configs to benchmark (default: all)")
    parser.add_argument("--compare-qdrant", action="store_true",
                        help="Compare against Qdrant (must be running on localhost:6333)")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (fewer queries, 2 configs)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Go server rebuild")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT,
                        help=f"HTTP port (default: {DEFAULT_HTTP_PORT})")
    parser.add_argument("--grpc-port", type=int, default=DEFAULT_GRPC_PORT,
                        help=f"gRPC port (default: {DEFAULT_GRPC_PORT})")
    parser.add_argument("--n-search", type=int, default=200,
                        help="Max timed search queries per dataset (default: 200)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup queries (default: 10)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent workers (default: 8)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Concurrent test duration in seconds (default: 5)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    settings = RunSettings(
        http_port=args.port,
        grpc_port=args.grpc_port,
        n_search=args.n_search,
        warmup=args.warmup,
        concurrency=args.concurrency,
        concurrent_duration_s=args.duration,
    )

    # Quick mode overrides
    if args.quick:
        settings.n_search = 50
        settings.warmup = 5
        settings.concurrent_duration_s = 2.0
        args.config = args.config or ["default", "grpc"]
        args.dataset = args.dataset or ["sift-100k"]

    all_datasets = get_datasets()
    ds_names = args.dataset or [n for n, d in all_datasets.items() if d.base_path.exists()]
    config_names = args.config or list(ALL_CONFIGS.keys())

    if not ds_names:
        print("No datasets found. Run: python benchmarks/download_datasets.py")
        sys.exit(1)

    configs = [ALL_CONFIGS[c] for c in config_names]

    print(f"Datasets: {', '.join(ds_names)}")
    print(f"Configs: {', '.join(c.name for c in configs)}")
    print(f"Settings: n_search={settings.n_search}, warmup={settings.warmup}, "
          f"concurrency={settings.concurrency}, duration={settings.concurrent_duration_s}s")
    if args.quick:
        print("(quick mode)")
    print()

    # Build server
    if not args.skip_build:
        if not build_server():
            sys.exit(1)

    all_results: list[BenchResult] = []
    qdrant_results: list[BenchResult] = []

    for ds_name in ds_names:
        ds = all_datasets[ds_name]
        loaded = load_dataset(ds)
        if loaded is None:
            continue
        base, queries, gt = loaded

        for cfg in configs:
            print(f"\n{'─' * 70}")
            print(f"  {ds_name} / {cfg.name} ({cfg.description})")
            print(f"{'─' * 70}")

            kill_servers(settings.http_port)
            proc = start_server(settings)
            if proc is None:
                all_results.append(BenchResult(
                    dataset=ds_name, config=cfg.name, error="server failed to start"))
                continue

            try:
                run_filtered = (cfg.name == "default")
                result = bench_deepdata(ds, base, queries, gt, cfg, settings,
                                        run_filtered=run_filtered)
                all_results.append(result)

                if result.error:
                    print(f"    FAILED: {result.error}")
                else:
                    print(f"    R@10={result.recall_at_10:.4f}  "
                          f"serial={result.search_serial_qps:.0f} QPS  "
                          f"{settings.concurrency}t={result.search_concurrent_qps:.0f} QPS  "
                          f"P50={result.search_p50_ms:.1f}ms")
            finally:
                stop_process(proc)

        # Qdrant comparison
        if args.compare_qdrant:
            print(f"\n{'─' * 70}")
            print(f"  Qdrant / {ds_name}")
            print(f"{'─' * 70}")
            qr = bench_qdrant(ds, base, queries, gt, settings)
            qdrant_results.append(qr)
            if qr.error:
                print(f"    FAILED: {qr.error}")
            else:
                print(f"    R@10={qr.recall_at_10:.4f}  "
                      f"serial={qr.search_serial_qps:.0f} QPS  "
                      f"{settings.concurrency}t={qr.search_concurrent_qps:.0f} QPS  "
                      f"P50={qr.search_p50_ms:.1f}ms")

    kill_servers(settings.http_port)

    # ── Report ──
    print()
    print()
    print_full_results(all_results + qdrant_results, settings)
    print_before_after(all_results)
    print_feature_impact(all_results)

    for qr in qdrant_results:
        ds_dd = [r for r in all_results if r.dataset == qr.dataset]
        print_competitive(ds_dd, qr)

    generate_recommendations(all_results, qdrant_results[0] if qdrant_results else None)

    # JSON output
    if args.json:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cpu": get_cpu_name(),
            "settings": asdict(settings),
            "deepdata": [asdict(r) for r in all_results],
            "qdrant": [asdict(r) for r in qdrant_results],
        }
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
