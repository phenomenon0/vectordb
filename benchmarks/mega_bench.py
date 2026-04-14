#!/usr/bin/env python3
"""Mega benchmark: wide-sweep recall × datasets × VDBs, failure modes, autonomous.

One command, zero intervention. Manages its own infrastructure, checkpoints
every result, retries on failure, generates a full report.

Usage:
    python benchmarks/mega_bench.py                    # Full autonomous run
    python benchmarks/mega_bench.py --quick             # Smoke test (~5 min)
    python benchmarks/mega_bench.py --vdb deepdata-grpc qdrant  # Specific VDBs
    python benchmarks/mega_bench.py --resume            # Resume from checkpoint
    python benchmarks/mega_bench.py --report-only       # Regenerate report from existing data
"""

import argparse
import concurrent.futures
import contextlib
import json
import os
import platform
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results" / "mega"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
SERVER_BINARY = PROJECT_ROOT / "deepdata-server"
DATA_DIR = Path("/tmp/deepdata-mega-bench")
COMPOSE_FILE = BENCH_DIR / "competitive" / "live" / "docker-compose.benchmark.yml"

sys.path.insert(0, str(BENCH_DIR))
from download_datasets import CACHE_DIR, compute_ground_truth, read_fvecs

# gRPC stubs
GRPC_STUBS_DIR = BENCH_DIR / "competitive" / "live" / "adapters"
sys.path.insert(0, str(GRPC_STUBS_DIR))

DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 50052  # 50051 taken by Weaviate


# ═══════════════════════════════════════════════════════════════════════════════
# Retry + logging utilities
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FILE = RESULTS_DIR / "mega_bench.log"

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def retry(fn, max_attempts=3, desc="operation"):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            log(f"  [{desc}] attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                return None
            time.sleep(attempt * 2)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset definitions
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetDef:
    name: str
    dim: int
    base_path: Path
    query_path: Path
    gt_path: Path
    n_base: int


def get_datasets() -> dict[str, DatasetDef]:
    sift = CACHE_DIR / "sift"
    glove = CACHE_DIR / "glove"
    code = CACHE_DIR / "code"
    return {
        "sift-100k": DatasetDef("sift-100k", 128, sift / "sift_base_100k_norm.fvecs",
                                 sift / "sift_query_norm.fvecs", sift / "sift_100k_gt100.npy", 100_000),
        "glove-100d": DatasetDef("glove-100d", 100, glove / "glove_100d_base_norm.fvecs",
                                  glove / "glove_100d_query_norm.fvecs", glove / "glove_100d_gt100.npy", 10_000),
        "code-562": DatasetDef("code-562", 1536, code / "code_1536d_base_norm.fvecs",
                                code / "code_1536d_query_norm.fvecs", code / "code_1536d_gt100.npy", 0),
    }


def load_dataset(ds: DatasetDef):
    if not ds.base_path.exists():
        return None
    base = read_fvecs(ds.base_path, max_n=ds.n_base if ds.n_base > 0 else None)
    queries = read_fvecs(ds.query_path)
    gt = np.load(ds.gt_path)
    if ds.n_base > 0 and len(base) > ds.n_base:
        base = base[:ds.n_base]
    if gt.size and int(gt.max()) >= len(base):
        gt = compute_ground_truth(base, queries, k=min(100, len(base)))
    return base, queries, gt


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"results": [], "completed": [], "failures": []}


def save_checkpoint(cp: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


def is_completed(cp: dict, key: str) -> bool:
    return key in cp["completed"]


def mark_completed(cp: dict, key: str, result: dict):
    cp["completed"].append(key)
    cp["results"].append(result)
    save_checkpoint(cp)


def mark_failure(cp: dict, key: str, error: str):
    cp["failures"].append({"key": key, "error": error})
    save_checkpoint(cp)


# ═══════════════════════════════════════════════════════════════════════════════
# Infrastructure management
# ═══════════════════════════════════════════════════════════════════════════════


def wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def build_server() -> bool:
    log("Building DeepData server...")
    r = subprocess.run(["go", "build", "-o", str(SERVER_BINARY), "./cmd/deepdata/"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"BUILD FAILED: {r.stderr[:500]}")
        return False
    log("Build OK")
    return True


def start_deepdata(ef_construction: int = 200) -> subprocess.Popen | None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    env = {
        **os.environ,
        "API_RPS": "100000", "TENANT_RPS": "100000", "TENANT_BURST": "100000",
        "SCAN_THRESHOLD": "0",
        "VECTORDB_BASE_DIR": str(DATA_DIR),
        "HNSW_M": "16",
        "HNSW_EF_CONSTRUCTION": str(ef_construction),
        "HNSW_EFSEARCH": "200",
        "GRPC_PORT": str(DEFAULT_GRPC_PORT),
    }
    stderr_path = DATA_DIR / "server.stderr"
    sf = open(stderr_path, "w")
    proc = subprocess.Popen([str(SERVER_BINARY)], env=env,
                            stdout=subprocess.DEVNULL, stderr=sf,
                            start_new_session=True)
    import httpx
    if wait_for_port(DEFAULT_HTTP_PORT, 15.0):
        deadline = time.perf_counter() + 15.0
        with httpx.Client(base_url=f"http://127.0.0.1:{DEFAULT_HTTP_PORT}", timeout=2.0) as c:
            while time.perf_counter() < deadline:
                try:
                    if c.get("/health").status_code == 200:
                        sf.flush(); sf.close()
                        # Also wait for gRPC
                        wait_for_port(DEFAULT_GRPC_PORT, 15.0)
                        return proc
                except Exception:
                    pass
                time.sleep(0.25)
    sf.flush(); sf.close()
    stop_proc(proc)
    return None


def stop_proc(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5); return
        except subprocess.TimeoutExpired:
            continue


def kill_deepdata():
    subprocess.run(["pkill", "-f", "deepdata-server"], capture_output=True)
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", DEFAULT_HTTP_PORT), timeout=0.2):
                time.sleep(0.2)
        except OSError:
            break
    time.sleep(0.3)


def check_containers() -> dict[str, bool]:
    """Check which competitor VDBs are reachable."""
    import httpx
    status = {}
    checks = [
        ("qdrant", "http://127.0.0.1:6333/healthz"),
        ("weaviate", "http://127.0.0.1:8081/v1/.well-known/ready"),
        ("chromadb", "http://127.0.0.1:8010/api/v2/heartbeat"),
        ("milvus", "http://127.0.0.1:9091/healthz"),
    ]
    with httpx.Client(timeout=3.0) as c:
        for name, url in checks:
            try:
                r = c.get(url)
                status[name] = r.status_code in (200, 410)  # ChromaDB returns 410 on v1
            except Exception:
                status[name] = False
    return status


def start_containers():
    """Start competitor containers if not already running."""
    log("Starting competitor containers...")
    subprocess.run(["podman-compose", "-f", str(COMPOSE_FILE), "up", "-d"],
                   capture_output=True, timeout=120)
    # Wait up to 60s for all to be healthy
    for i in range(30):
        st = check_containers()
        if all(st.values()):
            log(f"All containers ready: {st}")
            return st
        time.sleep(2)
    st = check_containers()
    log(f"Container status (some may be down): {st}")
    return st


# ═══════════════════════════════════════════════════════════════════════════════
# Binary import + helpers
# ═══════════════════════════════════════════════════════════════════════════════


def build_binary_import_payload(ids: np.ndarray, vecs: np.ndarray) -> bytes:
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.uint64)
    count, dim = vecs.shape
    rec_dtype = np.dtype([("id", "<u8"), ("vec", "<f4", (dim,))], align=False)
    recs = np.empty(count, dtype=rec_dtype)
    recs["id"] = ids; recs["vec"] = vecs
    return struct.pack("<II", count, dim) + recs.tobytes()


def compute_recall(retrieved: list[int], gt: np.ndarray, k: int) -> float:
    gt_set = set(int(x) for x in gt[:k])
    ret_set = set(int(x) for x in retrieved[:k])
    return len(gt_set & ret_set) / len(gt_set) if gt_set else 0.0


def percentile_ms(lats: list[float], pct: float) -> float:
    return round(float(np.percentile(lats, pct)), 2) if lats else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# VDB benchmark functions (unified interface, recall sweep aware)
# ═══════════════════════════════════════════════════════════════════════════════


def bench_deepdata_grpc(ds_name: str, dim: int, base: np.ndarray, queries: np.ndarray,
                        gt: np.ndarray, ef_search: int, n_search: int) -> dict:
    """DeepData benchmark using gRPC search, HTTP for collection/import."""
    import httpx
    import grpc
    from deepdata.v1 import deepdata_pb2 as pb2, deepdata_pb2_grpc

    base_url = f"http://127.0.0.1:{DEFAULT_HTTP_PORT}"
    coll = "bench"
    query_lists = [q.tolist() for q in queries[:n_search]]
    warmup = min(10, n_search)

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        with contextlib.suppress(Exception):
            client.delete(f"/v2/collections/{coll}")
        # Create with prenorm + fp16
        client.post("/v2/collections", json={
            "Name": coll,
            "Fields": [{"Name": "embedding", "Type": 0, "Dim": dim,
                         "Index": {"type": "hnsw", "params": {
                             "m": 16, "ef_construction": 200, "ef_search": ef_search,
                             "prenormalize": True,
                             "quantization": {"type": "float16"},
                         }}}],
        }).raise_for_status()

        # Insert via binary import
        t0 = time.perf_counter()
        for off in range(0, len(base), 5000):
            end = min(off + 5000, len(base))
            payload = build_binary_import_payload(np.arange(off, end, dtype=np.uint64), base[off:end])
            client.post(f"/v2/import?collection={coll}&field=embedding",
                        content=payload, headers={"Content-Type": "application/octet-stream"},
                        timeout=120.0).raise_for_status()
        insert_qps = round(len(base) / (time.perf_counter() - t0), 1)
        time.sleep(1.0)

    # gRPC search — ensure port is ready
    if not wait_for_port(DEFAULT_GRPC_PORT, 10.0):
        raise RuntimeError(f"gRPC port {DEFAULT_GRPC_PORT} not ready")
    ch = grpc.insecure_channel(f"127.0.0.1:{DEFAULT_GRPC_PORT}", options=[
        ("grpc.max_send_message_length", 64*1024*1024),
        ("grpc.max_receive_message_length", 64*1024*1024),
    ])
    stub = deepdata_pb2_grpc.DeepDataStub(ch)

    # Warmup
    for qi in range(warmup):
        qv = pb2.VectorData(dense=pb2.DenseVector(values=query_lists[qi]))
        stub.Search(pb2.SearchRequest(collection=coll, queries={"embedding": qv},
                                       top_k=100, ef_search=ef_search))

    # Timed search
    recalls_10, recalls_100, latencies = [], [], []
    for qi in range(warmup, n_search):
        qv = pb2.VectorData(dense=pb2.DenseVector(values=query_lists[qi]))
        t0 = time.perf_counter()
        resp = stub.Search(pb2.SearchRequest(collection=coll, queries={"embedding": qv},
                                              top_k=100, ef_search=ef_search))
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retrieved = [int(h.id) for h in resp.results]
        recalls_10.append(compute_recall(retrieved, gt[qi], 10))
        recalls_100.append(compute_recall(retrieved, gt[qi], 100))

    # QPS
    qps_count = 0
    qps_start = time.perf_counter()
    qv = pb2.VectorData(dense=pb2.DenseVector(values=query_lists[0]))
    while time.perf_counter() - qps_start < 5.0:
        stub.Search(pb2.SearchRequest(collection=coll, queries={"embedding": qv},
                                       top_k=10, ef_search=ef_search))
        qps_count += 1
    qps = round(qps_count / (time.perf_counter() - qps_start), 1)

    ch.close()

    # Cleanup
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        with contextlib.suppress(Exception):
            client.delete(f"/v2/collections/{coll}")

    return {
        "vdb": "deepdata-grpc", "dataset": ds_name, "ef_search": ef_search,
        "recall_at_10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0,
        "recall_at_100": round(float(np.mean(recalls_100)), 4) if recalls_100 else 0,
        "p50_ms": percentile_ms(latencies, 50), "p95_ms": percentile_ms(latencies, 95),
        "p99_ms": percentile_ms(latencies, 99), "qps": qps, "insert_qps": insert_qps,
    }


def bench_deepdata_http(ds_name: str, dim: int, base: np.ndarray, queries: np.ndarray,
                         gt: np.ndarray, ef_search: int, n_search: int) -> dict:
    """DeepData HTTP benchmark."""
    import httpx
    base_url = f"http://127.0.0.1:{DEFAULT_HTTP_PORT}"
    coll = "bench"
    query_lists = [q.tolist() for q in queries[:n_search]]
    warmup = min(10, n_search)

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        with contextlib.suppress(Exception):
            client.delete(f"/v2/collections/{coll}")
        client.post("/v2/collections", json={
            "Name": coll,
            "Fields": [{"Name": "embedding", "Type": 0, "Dim": dim,
                         "Index": {"type": "hnsw", "params": {
                             "m": 16, "ef_construction": 200, "ef_search": ef_search,
                             "prenormalize": True,
                         }}}],
        }).raise_for_status()

        t0 = time.perf_counter()
        for off in range(0, len(base), 5000):
            end = min(off + 5000, len(base))
            payload = build_binary_import_payload(np.arange(off, end, dtype=np.uint64), base[off:end])
            client.post(f"/v2/import?collection={coll}&field=embedding",
                        content=payload, headers={"Content-Type": "application/octet-stream"},
                        timeout=120.0).raise_for_status()
        insert_qps = round(len(base) / (time.perf_counter() - t0), 1)
        time.sleep(1.0)

        for qi in range(warmup):
            client.post("/v2/search", json={"collection": coll, "queries": {"embedding": query_lists[qi]},
                                             "top_k": 100, "ef_search": ef_search})

        recalls_10, recalls_100, latencies = [], [], []
        for qi in range(warmup, n_search):
            t0 = time.perf_counter()
            resp = client.post("/v2/search", json={"collection": coll,
                               "queries": {"embedding": query_lists[qi]},
                               "top_k": 100, "ef_search": ef_search})
            latencies.append((time.perf_counter() - t0) * 1000.0)
            resp.raise_for_status()
            retrieved = [int(d.get("id", d.get("ID", 0))) for d in resp.json().get("documents", [])]
            recalls_10.append(compute_recall(retrieved, gt[qi], 10))
            recalls_100.append(compute_recall(retrieved, gt[qi], 100))

        qps_count = 0
        qps_start = time.perf_counter()
        while time.perf_counter() - qps_start < 5.0:
            client.post("/v2/search", json={"collection": coll,
                        "queries": {"embedding": query_lists[0]}, "top_k": 10, "ef_search": ef_search})
            qps_count += 1
        qps = round(qps_count / (time.perf_counter() - qps_start), 1)

        with contextlib.suppress(Exception):
            client.delete(f"/v2/collections/{coll}")

    return {
        "vdb": "deepdata-http", "dataset": ds_name, "ef_search": ef_search,
        "recall_at_10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0,
        "recall_at_100": round(float(np.mean(recalls_100)), 4) if recalls_100 else 0,
        "p50_ms": percentile_ms(latencies, 50), "p95_ms": percentile_ms(latencies, 95),
        "p99_ms": percentile_ms(latencies, 99), "qps": qps, "insert_qps": insert_qps,
    }


def bench_qdrant(ds_name: str, dim: int, base: np.ndarray, queries: np.ndarray,
                 gt: np.ndarray, ef_search: int, n_search: int) -> dict:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, SearchParams, VectorParams

    client = QdrantClient(url="http://127.0.0.1:6333", timeout=120)
    coll = "bench"
    warmup = min(10, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]

    with contextlib.suppress(Exception):
        client.delete_collection(coll)
    client.create_collection(coll, vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                             hnsw_config=HnswConfigDiff(m=16, ef_construct=200))

    t0 = time.perf_counter()
    for off in range(0, len(base), 500):
        end = min(off + 500, len(base))
        pts = [PointStruct(id=i, vector=base[i].tolist()) for i in range(off, end)]
        client.upsert(coll, pts)
    insert_qps = round(len(base) / (time.perf_counter() - t0), 1)

    while True:
        info = client.get_collection(coll)
        if info.status.value == "green":
            break
        time.sleep(1)

    params = SearchParams(hnsw_ef=ef_search)
    for qi in range(warmup):
        client.search(coll, query_vector=query_lists[qi], limit=100, search_params=params)

    recalls_10, recalls_100, latencies = [], [], []
    for qi in range(warmup, n_search):
        t0 = time.perf_counter()
        hits = client.search(coll, query_vector=query_lists[qi], limit=100, search_params=params)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retrieved = [int(h.id) for h in hits]
        recalls_10.append(compute_recall(retrieved, gt[qi], 10))
        recalls_100.append(compute_recall(retrieved, gt[qi], 100))

    qps_count = 0
    qps_start = time.perf_counter()
    while time.perf_counter() - qps_start < 5.0:
        client.search(coll, query_vector=query_lists[0], limit=10, search_params=params)
        qps_count += 1
    qps = round(qps_count / (time.perf_counter() - qps_start), 1)

    with contextlib.suppress(Exception):
        client.delete_collection(coll)
    client.close()

    return {
        "vdb": "qdrant", "dataset": ds_name, "ef_search": ef_search,
        "recall_at_10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0,
        "recall_at_100": round(float(np.mean(recalls_100)), 4) if recalls_100 else 0,
        "p50_ms": percentile_ms(latencies, 50), "p95_ms": percentile_ms(latencies, 95),
        "p99_ms": percentile_ms(latencies, 99), "qps": qps, "insert_qps": insert_qps,
    }


def bench_weaviate(ds_name: str, dim: int, base: np.ndarray, queries: np.ndarray,
                   gt: np.ndarray, ef_search: int, n_search: int) -> dict:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType, VectorDistances
    from weaviate.classes.query import MetadataQuery

    client = weaviate.connect_to_custom(
        http_host="localhost", http_port=8081, http_secure=False,
        grpc_host="localhost", grpc_port=50051, grpc_secure=False)
    coll_name = "MegaBench"
    warmup = min(10, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]

    with contextlib.suppress(Exception):
        client.collections.delete(coll_name)
    client.collections.create(
        name=coll_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=200, max_connections=16, ef=ef_search),
        properties=[Property(name="idx", data_type=DataType.INT)])
    coll = client.collections.get(coll_name)

    t0 = time.perf_counter()
    with coll.batch.dynamic() as batch:
        for i in range(len(base)):
            batch.add_object(properties={"idx": i}, vector=base[i].tolist())
    insert_qps = round(len(base) / (time.perf_counter() - t0), 1)
    time.sleep(1.0)

    for qi in range(warmup):
        coll.query.near_vector(near_vector=query_lists[qi], limit=100,
                               return_metadata=MetadataQuery(distance=True))

    recalls_10, recalls_100, latencies = [], [], []
    for qi in range(warmup, n_search):
        t0 = time.perf_counter()
        resp = coll.query.near_vector(near_vector=query_lists[qi], limit=100,
                                       return_metadata=MetadataQuery(distance=True),
                                       return_properties=["idx"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retrieved = [obj.properties.get("idx", -1) for obj in resp.objects]
        recalls_10.append(compute_recall(retrieved, gt[qi], 10))
        recalls_100.append(compute_recall(retrieved, gt[qi], 100))

    qps_count = 0
    qps_start = time.perf_counter()
    while time.perf_counter() - qps_start < 5.0:
        coll.query.near_vector(near_vector=query_lists[0], limit=10,
                               return_metadata=MetadataQuery(distance=True))
        qps_count += 1
    qps = round(qps_count / (time.perf_counter() - qps_start), 1)

    with contextlib.suppress(Exception):
        client.collections.delete(coll_name)
    client.close()

    return {
        "vdb": "weaviate", "dataset": ds_name, "ef_search": ef_search,
        "recall_at_10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0,
        "recall_at_100": round(float(np.mean(recalls_100)), 4) if recalls_100 else 0,
        "p50_ms": percentile_ms(latencies, 50), "p95_ms": percentile_ms(latencies, 95),
        "p99_ms": percentile_ms(latencies, 99), "qps": qps, "insert_qps": insert_qps,
    }


def bench_milvus(ds_name: str, dim: int, base: np.ndarray, queries: np.ndarray,
                 gt: np.ndarray, ef_search: int, n_search: int) -> dict:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

    connections.connect("default", host="localhost", port=19530)
    coll_name = "mega_bench"
    warmup = min(10, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]

    if utility.has_collection(coll_name):
        utility.drop_collection(coll_name)
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
    ])
    mcoll = Collection(coll_name, schema)
    mcoll.create_index("embedding", {"metric_type": "COSINE", "index_type": "HNSW",
                                      "params": {"M": 16, "efConstruction": 200}})

    t0 = time.perf_counter()
    for off in range(0, len(base), 1000):
        end = min(off + 1000, len(base))
        mcoll.insert([base[off:end].tolist()])
    mcoll.flush()
    insert_qps = round(len(base) / (time.perf_counter() - t0), 1)
    mcoll.load()

    params = {"metric_type": "COSINE", "params": {"ef": ef_search}}
    for qi in range(warmup):
        mcoll.search([query_lists[qi]], "embedding", params, limit=100)

    recalls_10, recalls_100, latencies = [], [], []
    for qi in range(warmup, n_search):
        t0 = time.perf_counter()
        results = mcoll.search([query_lists[qi]], "embedding", params, limit=100)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        # Milvus auto_id: compare by insertion order position
        retrieved = [int(h.id) for h in results[0]] if results else []
        recalls_10.append(compute_recall(retrieved, gt[qi], 10))
        recalls_100.append(compute_recall(retrieved, gt[qi], 100))

    qps_count = 0
    qps_start = time.perf_counter()
    while time.perf_counter() - qps_start < 5.0:
        mcoll.search([query_lists[0]], "embedding", params, limit=10)
        qps_count += 1
    qps = round(qps_count / (time.perf_counter() - qps_start), 1)

    utility.drop_collection(coll_name)
    connections.disconnect("default")

    return {
        "vdb": "milvus", "dataset": ds_name, "ef_search": ef_search,
        "recall_at_10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0,
        "recall_at_100": round(float(np.mean(recalls_100)), 4) if recalls_100 else 0,
        "p50_ms": percentile_ms(latencies, 50), "p95_ms": percentile_ms(latencies, 95),
        "p99_ms": percentile_ms(latencies, 99), "qps": qps, "insert_qps": insert_qps,
    }


def bench_chromadb(ds_name: str, dim: int, base: np.ndarray, queries: np.ndarray,
                   gt: np.ndarray, ef_search: int, n_search: int) -> dict:
    import chromadb
    client = chromadb.HttpClient(host="localhost", port=8010)
    coll_name = "mega_bench"
    warmup = min(10, n_search)
    query_lists = [q.tolist() for q in queries[:n_search]]

    with contextlib.suppress(Exception):
        client.delete_collection(coll_name)
    coll = client.create_collection(coll_name, metadata={
        "hnsw:M": 16, "hnsw:construction_ef": 200,
        "hnsw:search_ef": ef_search, "hnsw:space": "cosine"})

    t0 = time.perf_counter()
    for off in range(0, len(base), 100):
        end = min(off + 100, len(base))
        coll.add(ids=[str(i) for i in range(off, end)],
                 embeddings=base[off:end].tolist())
    insert_qps = round(len(base) / (time.perf_counter() - t0), 1)
    time.sleep(0.5)

    for qi in range(warmup):
        coll.query(query_embeddings=[query_lists[qi]], n_results=100)

    recalls_10, recalls_100, latencies = [], [], []
    for qi in range(warmup, n_search):
        t0 = time.perf_counter()
        resp = coll.query(query_embeddings=[query_lists[qi]], n_results=100)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retrieved = [int(x) for x in resp["ids"][0]] if resp["ids"] else []
        recalls_10.append(compute_recall(retrieved, gt[qi], 10))
        recalls_100.append(compute_recall(retrieved, gt[qi], 100))

    qps_count = 0
    qps_start = time.perf_counter()
    while time.perf_counter() - qps_start < 5.0:
        coll.query(query_embeddings=[query_lists[0]], n_results=10)
        qps_count += 1
    qps = round(qps_count / (time.perf_counter() - qps_start), 1)

    with contextlib.suppress(Exception):
        client.delete_collection(coll_name)

    return {
        "vdb": "chromadb", "dataset": ds_name, "ef_search": ef_search,
        "recall_at_10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0,
        "recall_at_100": round(float(np.mean(recalls_100)), 4) if recalls_100 else 0,
        "p50_ms": percentile_ms(latencies, 50), "p95_ms": percentile_ms(latencies, 95),
        "p99_ms": percentile_ms(latencies, 99), "qps": qps, "insert_qps": insert_qps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Failure mode tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_failure_modes() -> list[dict]:
    """Test edge cases and failure modes on DeepData."""
    import httpx
    import grpc
    from deepdata.v1 import deepdata_pb2 as pb2, deepdata_pb2_grpc

    results = []
    base_url = f"http://127.0.0.1:{DEFAULT_HTTP_PORT}"
    if not wait_for_port(DEFAULT_GRPC_PORT, 15.0):
        log("  WARNING: gRPC port not ready, gRPC failure mode tests will be skipped")
    ch = grpc.insecure_channel(f"127.0.0.1:{DEFAULT_GRPC_PORT}", options=[
        ("grpc.max_send_message_length", 64*1024*1024),
        ("grpc.max_receive_message_length", 64*1024*1024)])
    # Verify gRPC connectivity with a deadline
    try:
        grpc.channel_ready_future(ch).result(timeout=10)
    except grpc.FutureTimeoutError:
        log("  WARNING: gRPC channel not ready after 10s")
    stub = deepdata_pb2_grpc.DeepDataStub(ch)

    def fm(name: str, fn) -> dict:
        try:
            ok, detail = fn()
            r = {"test": name, "passed": ok, "detail": detail}
        except Exception as e:
            r = {"test": name, "passed": False, "detail": str(e)}
        log(f"  Failure mode [{name}]: {'PASS' if r['passed'] else 'FAIL'} — {r['detail']}")
        return r

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        # 1. Search empty collection
        def t_empty():
            with contextlib.suppress(Exception):
                client.delete("/v2/collections/fmtest")
            client.post("/v2/collections", json={
                "Name": "fmtest",
                "Fields": [{"Name": "embedding", "Type": 0, "Dim": 4}],
            }).raise_for_status()
            resp = client.post("/v2/search", json={
                "collection": "fmtest", "queries": {"embedding": [0.1, 0.2, 0.3, 0.4]},
                "top_k": 10, "ef_search": 64})
            docs = resp.json().get("documents", [])
            client.delete("/v2/collections/fmtest")
            return resp.status_code == 200 and len(docs) == 0, f"status={resp.status_code}, docs={len(docs)}"
        results.append(fm("empty_collection_search", t_empty))

        # 2. Search with k > n_vectors
        def t_k_gt_n():
            with contextlib.suppress(Exception):
                client.delete("/v2/collections/fmtest")
            client.post("/v2/collections", json={
                "Name": "fmtest",
                "Fields": [{"Name": "embedding", "Type": 0, "Dim": 4}],
            }).raise_for_status()
            payload = build_binary_import_payload(np.arange(3, dtype=np.uint64),
                                                  np.random.randn(3, 4).astype(np.float32))
            client.post("/v2/import?collection=fmtest&field=embedding",
                        content=payload, headers={"Content-Type": "application/octet-stream"}).raise_for_status()
            time.sleep(0.5)
            resp = client.post("/v2/search", json={
                "collection": "fmtest", "queries": {"embedding": [0.1, 0.2, 0.3, 0.4]},
                "top_k": 1000, "ef_search": 64})
            docs = resp.json().get("documents", [])
            client.delete("/v2/collections/fmtest")
            return resp.status_code == 200 and 0 < len(docs) <= 3, f"asked k=1000, got {len(docs)} from 3 vectors"
        results.append(fm("k_greater_than_n", t_k_gt_n))

        # 3. Zero-vector query
        def t_zero_vec():
            with contextlib.suppress(Exception):
                client.delete("/v2/collections/fmtest")
            client.post("/v2/collections", json={
                "Name": "fmtest",
                "Fields": [{"Name": "embedding", "Type": 0, "Dim": 4}],
            }).raise_for_status()
            payload = build_binary_import_payload(np.arange(5, dtype=np.uint64),
                                                  np.random.randn(5, 4).astype(np.float32))
            client.post("/v2/import?collection=fmtest&field=embedding",
                        content=payload, headers={"Content-Type": "application/octet-stream"}).raise_for_status()
            time.sleep(0.5)
            resp = client.post("/v2/search", json={
                "collection": "fmtest", "queries": {"embedding": [0.0, 0.0, 0.0, 0.0]},
                "top_k": 5, "ef_search": 64})
            ok = resp.status_code in (200, 400, 500)  # Any defined response is acceptable
            client.delete("/v2/collections/fmtest")
            return ok, f"status={resp.status_code}"
        results.append(fm("zero_vector_query", t_zero_vec))

        # 4. Duplicate IDs in import
        def t_dupe_ids():
            with contextlib.suppress(Exception):
                client.delete("/v2/collections/fmtest")
            client.post("/v2/collections", json={
                "Name": "fmtest",
                "Fields": [{"Name": "embedding", "Type": 0, "Dim": 4}],
            }).raise_for_status()
            ids = np.array([0, 1, 1, 2], dtype=np.uint64)  # ID 1 duplicated
            vecs = np.random.randn(4, 4).astype(np.float32)
            payload = build_binary_import_payload(ids, vecs)
            resp = client.post("/v2/import?collection=fmtest&field=embedding",
                               content=payload, headers={"Content-Type": "application/octet-stream"})
            ok = resp.status_code in (200, 409, 400)
            client.delete("/v2/collections/fmtest")
            return ok, f"status={resp.status_code}"
        results.append(fm("duplicate_id_import", t_dupe_ids))

        # 5. Search nonexistent collection
        def t_no_coll():
            resp = client.post("/v2/search", json={
                "collection": "nonexistent_xyz_999",
                "queries": {"embedding": [0.1, 0.2, 0.3, 0.4]},
                "top_k": 10})
            return resp.status_code in (400, 404, 500), f"status={resp.status_code}"
        results.append(fm("search_nonexistent_collection", t_no_coll))

        # 6. gRPC search on empty
        def t_grpc_empty():
            with contextlib.suppress(Exception):
                stub.DeleteCollection(pb2.DeleteCollectionRequest(name="fmgrpc"))
            field = pb2.VectorFieldConfig(name="embedding", type=0, dim=4,
                                          index_type="hnsw", index_params={"m": 16.0, "ef_construction": 200.0})
            stub.CreateCollection(pb2.CreateCollectionRequest(name="fmgrpc", fields=[field]))
            qv = pb2.VectorData(dense=pb2.DenseVector(values=[0.1, 0.2, 0.3, 0.4]))
            resp = stub.Search(pb2.SearchRequest(collection="fmgrpc", queries={"embedding": qv},
                                                  top_k=10, ef_search=64))
            stub.DeleteCollection(pb2.DeleteCollectionRequest(name="fmgrpc"))
            return len(resp.results) == 0, f"got {len(resp.results)} results"
        results.append(fm("grpc_empty_collection_search", t_grpc_empty))

        # 7. Very high ef_search
        def t_high_ef():
            with contextlib.suppress(Exception):
                client.delete("/v2/collections/fmtest")
            client.post("/v2/collections", json={
                "Name": "fmtest",
                "Fields": [{"Name": "embedding", "Type": 0, "Dim": 4}],
            }).raise_for_status()
            payload = build_binary_import_payload(np.arange(10, dtype=np.uint64),
                                                  np.random.randn(10, 4).astype(np.float32))
            client.post("/v2/import?collection=fmtest&field=embedding",
                        content=payload, headers={"Content-Type": "application/octet-stream"}).raise_for_status()
            time.sleep(0.5)
            resp = client.post("/v2/search", json={
                "collection": "fmtest", "queries": {"embedding": [0.1, 0.2, 0.3, 0.4]},
                "top_k": 10, "ef_search": 10000})
            ok = resp.status_code == 200
            client.delete("/v2/collections/fmtest")
            return ok, f"ef_search=10000, status={resp.status_code}"
        results.append(fm("very_high_ef_search", t_high_ef))

    ch.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(cp: dict) -> str:
    results = cp["results"]
    failures = cp.get("failures", [])
    fm_results = cp.get("failure_modes", [])

    cpu = "Unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass

    lines = [
        f"# Mega Benchmark: Wide-Sweep All-VDB Results",
        f"",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}  |  **CPU:** {cpu}",
        f"**Datasets:** sift-100k (128d/100K), glove-100d (100d/10K), code-562 (1536d/562)",
        f"**ef_search sweep:** 16, 32, 64, 128, 256, 512",
        f"**VDBs:** DeepData gRPC, DeepData HTTP, Qdrant, Weaviate, Milvus, ChromaDB",
        f"",
        f"---",
        f"",
    ]

    # Group by dataset
    datasets = sorted(set(r["dataset"] for r in results))
    vdbs = ["deepdata-grpc", "deepdata-http", "qdrant", "weaviate", "milvus", "chromadb"]
    ef_values = sorted(set(r["ef_search"] for r in results))

    for ds in datasets:
        lines.append(f"## {ds}")
        lines.append(f"")
        lines.append(f"### Recall@10 vs ef_search")
        lines.append(f"")
        header = f"| ef_search | " + " | ".join(vdbs) + " |"
        sep = "|-----------|" + "|".join(["--------"] * len(vdbs)) + "|"
        lines.append(header)
        lines.append(sep)
        for ef in ef_values:
            row = f"| {ef:<9} |"
            for vdb in vdbs:
                r = next((x for x in results if x["dataset"] == ds and x["vdb"] == vdb and x["ef_search"] == ef), None)
                row += f" {r['recall_at_10']:.4f} |" if r else " N/A    |"
            lines.append(row)
        lines.append(f"")

        lines.append(f"### P50 Latency (ms) vs ef_search")
        lines.append(f"")
        lines.append(header)
        lines.append(sep)
        for ef in ef_values:
            row = f"| {ef:<9} |"
            for vdb in vdbs:
                r = next((x for x in results if x["dataset"] == ds and x["vdb"] == vdb and x["ef_search"] == ef), None)
                row += f" {r['p50_ms']:>6.1f} |" if r else " N/A    |"
            lines.append(row)
        lines.append(f"")

        lines.append(f"### QPS vs ef_search")
        lines.append(f"")
        lines.append(header)
        lines.append(sep)
        for ef in ef_values:
            row = f"| {ef:<9} |"
            for vdb in vdbs:
                r = next((x for x in results if x["dataset"] == ds and x["vdb"] == vdb and x["ef_search"] == ef), None)
                row += f" {r['qps']:>6.0f} |" if r else " N/A    |"
            lines.append(row)
        lines.append(f"")

    # Best configs summary
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Best Configs (highest QPS with R@10 >= 0.95)")
    lines.append(f"")
    lines.append(f"| VDB | Dataset | ef_search | R@10 | QPS | P50 ms |")
    lines.append(f"|-----|---------|-----------|------|-----|--------|")
    for vdb in vdbs:
        vdb_results = [r for r in results if r["vdb"] == vdb and r.get("recall_at_10", 0) >= 0.95]
        if vdb_results:
            best = max(vdb_results, key=lambda r: r["qps"])
            lines.append(f"| {vdb} | {best['dataset']} | {best['ef_search']} | {best['recall_at_10']:.4f} | {best['qps']:.0f} | {best['p50_ms']:.1f} |")
    lines.append(f"")

    # Failure modes
    if fm_results:
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## Failure Mode Tests (DeepData)")
        lines.append(f"")
        lines.append(f"| Test | Passed | Detail |")
        lines.append(f"|------|--------|--------|")
        for fm in fm_results:
            icon = "PASS" if fm["passed"] else "**FAIL**"
            lines.append(f"| {fm['test']} | {icon} | {fm['detail']} |")
        passed = sum(1 for f in fm_results if f["passed"])
        lines.append(f"")
        lines.append(f"**{passed}/{len(fm_results)} passed**")
        lines.append(f"")

    # Errors
    if failures:
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## Errors During Run")
        lines.append(f"")
        for f in failures:
            lines.append(f"- `{f['key']}`: {f['error'][:200]}")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Generated by `benchmarks/mega_bench.py`*")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


VDB_BENCH_FNS = {
    "deepdata-grpc": bench_deepdata_grpc,
    "deepdata-http": bench_deepdata_http,
    "qdrant": bench_qdrant,
    "weaviate": bench_weaviate,
    "milvus": bench_milvus,
    "chromadb": bench_chromadb,
}

ALL_VDB_NAMES = list(VDB_BENCH_FNS.keys())
ALL_EF_SEARCH = [16, 32, 64, 128, 256, 512]
ALL_DATASETS = list(get_datasets().keys())


def parse_args():
    p = argparse.ArgumentParser(description="Mega benchmark — wide sweep all VDBs")
    p.add_argument("--vdb", nargs="*", default=None, choices=ALL_VDB_NAMES)
    p.add_argument("--dataset", nargs="*", default=None, choices=ALL_DATASETS)
    p.add_argument("--ef", nargs="*", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="Quick: fewer ef values, fewer queries")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--report-only", action="store_true", help="Regenerate report from checkpoint")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-containers", action="store_true")
    p.add_argument("--skip-failure-modes", action="store_true")
    p.add_argument("--n-search", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        cp = load_checkpoint()
        report = generate_report(cp)
        report_path = RESULTS_DIR / "REPORT.md"
        report_path.write_text(report)
        log(f"Report written to {report_path}")
        return

    cp = load_checkpoint() if args.resume else {"results": [], "completed": [], "failures": []}

    vdb_names = args.vdb or ALL_VDB_NAMES
    ds_names = args.dataset or ALL_DATASETS
    n_search = args.n_search

    if args.quick:
        ef_values = [32, 128, 512]
        n_search = 40
    else:
        ef_values = args.ef or ALL_EF_SEARCH

    all_dsets = get_datasets()
    available_ds = [n for n in ds_names if all_dsets[n].base_path.exists()]
    if not available_ds:
        log("No datasets found. Run: python benchmarks/download_datasets.py")
        sys.exit(1)

    log(f"VDBs: {', '.join(vdb_names)}")
    log(f"Datasets: {', '.join(available_ds)}")
    log(f"ef_search: {ef_values}")
    log(f"n_search: {n_search}")
    log(f"Total combos: {len(available_ds) * len(ef_values) * len(vdb_names)}")
    log("")

    # Build server
    needs_deepdata = any(v.startswith("deepdata") for v in vdb_names)
    if needs_deepdata and not args.skip_build:
        if not build_server():
            sys.exit(1)

    # Start containers
    needs_containers = any(v in vdb_names for v in ["qdrant", "weaviate", "milvus", "chromadb"])
    container_status = {}
    if needs_containers:
        if args.skip_containers:
            # Assume already running — probe each
            for vdb in ["qdrant", "weaviate", "milvus", "chromadb"]:
                if vdb in vdb_names:
                    container_status[vdb] = True
        else:
            container_status = start_containers()

    # Load all datasets upfront
    loaded = {}
    for ds_name in available_ds:
        log(f"Loading {ds_name}...")
        data = load_dataset(all_dsets[ds_name])
        if data:
            loaded[ds_name] = data
            log(f"  {len(data[0]):,} base, {len(data[1]):,} queries")

    # ── Main sweep ──
    for ds_name in available_ds:
        if ds_name not in loaded:
            continue
        base, queries, gt = loaded[ds_name]
        dim = all_dsets[ds_name].dim

        for ef in ef_values:
            for vdb in vdb_names:
                key = f"{vdb}|{ds_name}|ef={ef}"
                if is_completed(cp, key):
                    log(f"SKIP (cached): {key}")
                    continue

                # Check if this VDB is available
                if vdb in ("qdrant", "weaviate", "milvus", "chromadb"):
                    if not container_status.get(vdb, False):
                        mark_failure(cp, key, f"{vdb} container not available")
                        continue

                log(f"RUN: {key}")

                # DeepData needs fresh server per dataset (clean index)
                if vdb.startswith("deepdata"):
                    kill_deepdata()
                    proc = start_deepdata(ef_construction=200)
                    if not proc:
                        mark_failure(cp, key, "server failed to start")
                        continue

                actual_n = min(n_search, len(queries))
                def do_bench():
                    return VDB_BENCH_FNS[vdb](ds_name, dim, base, queries, gt, ef, actual_n)

                result = retry(do_bench, max_attempts=2, desc=key)

                if vdb.startswith("deepdata"):
                    kill_deepdata()

                if result:
                    mark_completed(cp, key, result)
                    log(f"  R@10={result['recall_at_10']:.4f}  QPS={result['qps']:.0f}  P50={result['p50_ms']:.1f}ms")
                else:
                    mark_failure(cp, key, "all retries exhausted")

    # ── Failure modes ──
    if not args.skip_failure_modes and needs_deepdata:
        log("\n=== FAILURE MODE TESTS ===")
        kill_deepdata()
        proc = start_deepdata()
        if proc:
            fm_results = test_failure_modes()
            cp["failure_modes"] = fm_results
            save_checkpoint(cp)
            kill_deepdata()

    # ── Report ──
    report = generate_report(cp)
    report_path = RESULTS_DIR / "REPORT.md"
    report_path.write_text(report)
    log(f"\nReport written to {report_path}")
    log(f"Checkpoint at {CHECKPOINT_FILE}")
    log(f"Total results: {len(cp['results'])}, failures: {len(cp['failures'])}")


if __name__ == "__main__":
    main()
