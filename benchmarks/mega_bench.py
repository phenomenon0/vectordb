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
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results" / "mega"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
LOCK_FILE = Path("/tmp/deepdata-mega-bench.lock")
SERVER_BINARY = PROJECT_ROOT / "deepdata-server"
DATA_DIR = Path("/tmp/deepdata-mega-bench")
COMPOSE_FILE = BENCH_DIR / "competitive" / "live" / "docker-compose.benchmark.yml"

sys.path.insert(0, str(BENCH_DIR))
from download_datasets import CACHE_DIR, compute_ground_truth, read_fvecs  # noqa: E402

# gRPC stubs
GRPC_STUBS_DIR = BENCH_DIR / "competitive" / "live" / "adapters"
sys.path.insert(0, str(GRPC_STUBS_DIR))

DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 50052  # 50051 taken by Weaviate
CHECKPOINT_SCHEMA_VERSION = 2
METRIC_PROTOCOL = "recall-latency-top100_qps-sequential-top10-v1"


# ═══════════════════════════════════════════════════════════════════════════════
# Retry + logging utilities
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FILE = RESULTS_DIR / "mega_bench.log"

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}" if msg else f"[{ts}]"
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


def result_key(result: dict) -> str:
    """Return the stable identity for one benchmark matrix cell."""
    return f"{result['vdb']}|{result['dataset']}|ef={int(result['ef_search'])}"


def normalize_checkpoint(cp: dict) -> dict:
    """Canonicalize legacy append-only checkpoints in place.

    The last result for a key wins. A stored result is authoritative evidence
    that the cell completed, while completed markers without a result are
    discarded so an interrupted cell will be retried.
    """
    cp.setdefault("results", [])
    cp.setdefault("completed", [])
    cp.setdefault("failures", [])
    cp.setdefault("pending_reruns", [])
    cp["pending_reruns"] = list(dict.fromkeys(cp["pending_reruns"]))
    pending_reruns = set(cp["pending_reruns"])

    results_by_key = {}
    for result in cp["results"]:
        results_by_key[result_key(result)] = result
    cp["results"] = list(results_by_key.values())

    completed = list(dict.fromkeys(cp["completed"]))
    cp["completed"] = [key for key in completed if key in results_by_key]
    completed_set = set(cp["completed"])
    for key in results_by_key:
        if key not in completed_set:
            cp["completed"].append(key)

    failures_by_key = {}
    for failure in cp["failures"]:
        key = failure["key"]
        if key not in results_by_key or key in pending_reruns:
            failures_by_key[key] = failure
    cp["failures"] = list(failures_by_key.values())
    if "run_manifest" in cp:
        cp["run_manifest"]["legacy_rows_retained"] = sum(
            1 for result in cp["results"] if "measured_at" not in result
        )
    return cp


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return normalize_checkpoint(json.load(f))
    return {"results": [], "completed": [], "failures": []}


def save_checkpoint(cp: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    normalize_checkpoint(cp)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=RESULTS_DIR,
                prefix="checkpoint.", suffix=".tmp", delete=False) as f:
            temp_path = Path(f.name)
            json.dump(cp, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, CHECKPOINT_FILE)
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


def is_completed(cp: dict, key: str) -> bool:
    return key in cp["completed"] and key not in cp.get("pending_reruns", [])


def mark_completed(cp: dict, key: str, result: dict):
    if result_key(result) != key:
        raise ValueError(f"result key mismatch: expected {key}, got {result_key(result)}")
    cp["completed"] = [item for item in cp["completed"] if item != key]
    cp["completed"].append(key)
    cp["results"] = [item for item in cp["results"] if result_key(item) != key]
    cp["results"].append(result)
    cp["failures"] = [failure for failure in cp["failures"] if failure["key"] != key]
    cp["pending_reruns"] = [item for item in cp.get("pending_reruns", []) if item != key]
    save_checkpoint(cp)


def mark_failure(cp: dict, key: str, error: str):
    cp["failures"] = [failure for failure in cp["failures"] if failure["key"] != key]
    cp["failures"].append({"key": key, "error": error})
    save_checkpoint(cp)


def schedule_vdb_rerun(cp: dict, vdb_names: list[str],
                       dataset_names: list[str], ef_values: list[int]) -> int:
    """Mark selected cells pending while retaining their last good results."""
    selected_keys = {
        f"{vdb}|{dataset}|ef={ef}"
        for dataset in dataset_names for ef in ef_values for vdb in vdb_names
    }
    pending = set(cp.get("pending_reruns", []))
    before = len(pending)
    pending.update(selected_keys)
    cp["pending_reruns"] = list(pending)
    cp["failures"] = [
        failure for failure in cp["failures"] if failure["key"] not in selected_keys
    ]
    return len(pending) - before


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(ds: DatasetDef) -> dict:
    files = {}
    for name, path in (("base", ds.base_path), ("queries", ds.query_path), ("ground_truth", ds.gt_path)):
        stat = path.stat()
        files[name] = {
            "size": stat.st_size,
            "sha256": file_sha256(path),
        }
    return {"dim": ds.dim, "n_base": ds.n_base, "files": files}


def installed_package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "httpx", "grpcio", "protobuf", "qdrant-client",
                    "weaviate-client", "pymilvus", "chromadb"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return versions


def ensure_run_manifest(cp: dict, datasets: dict[str, DatasetDef],
                        dataset_names: list[str], n_search: int,
                        vdb_names: list[str], ef_values: list[int]):
    """Create or validate the fields that affect resume compatibility."""
    fingerprints = {name: dataset_fingerprint(datasets[name]) for name in dataset_names}
    packages = installed_package_versions()
    selected_cells = [
        f"{vdb}|{dataset}|ef={ef}"
        for dataset in dataset_names for ef in ef_values for vdb in vdb_names
    ]
    existing = cp.get("run_manifest")
    if existing:
        if existing.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("checkpoint schema is incompatible; start a new results directory")
        if existing.get("metric_protocol") != METRIC_PROTOCOL:
            raise RuntimeError("checkpoint metric protocol is incompatible")
        if existing.get("n_search") != n_search:
            raise RuntimeError(
                f"checkpoint uses n_search={existing.get('n_search')}, requested {n_search}")
        recorded_packages = existing.setdefault("packages_at_manifest_creation", {})
        for package in recorded_packages:
            if package in recorded_packages and packages.get(package) != recorded_packages[package]:
                raise RuntimeError(
                    f"dependency changed since checkpoint: {package} "
                    f"{recorded_packages[package]} -> {packages.get(package, 'missing')}")
        changed = False
        for package, version in packages.items():
            if package not in recorded_packages:
                recorded_packages[package] = version
                changed = True
        for name, fingerprint in fingerprints.items():
            previous = existing.get("datasets", {}).get(name)
            if previous is not None and previous != fingerprint:
                raise RuntimeError(f"dataset changed since checkpoint: {name}")
            if previous is None:
                existing.setdefault("datasets", {})[name] = fingerprint
                changed = True
        if "python_version" not in existing:
            existing["python_version"] = platform.python_version()
            changed = True
        elif existing["python_version"] != platform.python_version():
            raise RuntimeError(
                f"Python changed since checkpoint: {existing['python_version']} -> "
                f"{platform.python_version()}")
        if "matrix" not in existing:
            existing["matrix"] = {
                "vdbs": list(dict.fromkeys(r["vdb"] for r in cp["results"])),
                "datasets": list(dict.fromkeys(r["dataset"] for r in cp["results"])),
                "ef_search": sorted({int(r["ef_search"]) for r in cp["results"]}),
            }
            changed = True
        matrix = existing["matrix"]
        for field, values in (("vdbs", vdb_names), ("datasets", dataset_names),
                              ("ef_search", ef_values)):
            merged = list(dict.fromkeys([*matrix.get(field, []), *values]))
            if merged != matrix.get(field, []):
                matrix[field] = merged
                changed = True
        if "intended_cells" not in existing:
            existing["intended_cells"] = list(dict.fromkeys(
                [*(result_key(r) for r in cp["results"]), *selected_cells]))
            changed = True
        else:
            merged_cells = list(dict.fromkeys([*existing["intended_cells"], *selected_cells]))
            if merged_cells != existing["intended_cells"]:
                existing["intended_cells"] = merged_cells
                changed = True
        if changed:
            save_checkpoint(cp)
        return

    result_vdbs = [r["vdb"] for r in cp["results"]]
    result_datasets = [r["dataset"] for r in cp["results"]]
    result_ef = [int(r["ef_search"]) for r in cp["results"]]
    cp["run_manifest"] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metric_protocol": METRIC_PROTOCOL,
        "n_search": n_search,
        "datasets": fingerprints,
        "packages_at_manifest_creation": packages,
        "python_version": platform.python_version(),
        "matrix": {
            "vdbs": list(dict.fromkeys([*result_vdbs, *vdb_names])),
            "datasets": list(dict.fromkeys([*result_datasets, *dataset_names])),
            "ef_search": sorted(set([*result_ef, *ef_values])),
        },
        "intended_cells": list(dict.fromkeys(
            [*(result_key(r) for r in cp["results"]), *selected_cells])),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "legacy_rows_retained": len(cp["results"]),
    }
    save_checkpoint(cp)


@contextlib.contextmanager
def benchmark_lock():
    """Prevent concurrent runs from sharing checkpoint and collection names."""
    import fcntl

    lock_handle = open(LOCK_FILE, "w")
    try:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another mega benchmark holds {LOCK_FILE}") from exc
        yield
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


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


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
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
    occupied = [
        port for port in (DEFAULT_HTTP_PORT, DEFAULT_GRPC_PORT) if port_is_open(port)
    ]
    if occupied:
        log(f"DeepData ports already occupied: {occupied}; refusing to stop unrelated processes")
        return None
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
                        # The process is not ready for a benchmark until both
                        # transports are listening. Previously this result was
                        # ignored, which produced intermittent gRPC failures.
                        if wait_for_port(DEFAULT_GRPC_PORT, 15.0):
                            sf.flush()
                            sf.close()
                            return proc
                        log(f"DeepData gRPC port {DEFAULT_GRPC_PORT} did not become ready")
                        break
                except Exception:
                    pass
                time.sleep(0.25)
    sf.flush()
    sf.close()
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
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def check_containers(vdb_names: list[str] | None = None) -> dict[str, bool]:
    """Check which competitor VDBs are reachable."""
    import httpx
    status = {}
    requested = set(vdb_names) if vdb_names is not None else None
    checks = [
        ("qdrant", "http://127.0.0.1:6333/healthz"),
        ("weaviate", "http://127.0.0.1:8081/v1/.well-known/ready"),
        ("chromadb", "http://127.0.0.1:8010/api/v2/heartbeat"),
        ("milvus", "http://127.0.0.1:9091/healthz"),
    ]
    with httpx.Client(timeout=3.0) as c:
        for name, url in checks:
            if requested is not None and name not in requested:
                continue
            try:
                r = c.get(url)
                status[name] = r.status_code in (200, 410)  # ChromaDB returns 410 on v1
            except Exception:
                status[name] = False
    return status


def start_containers(vdb_names: list[str]):
    """Start only requested competitor services and their dependencies."""
    requested = [name for name in vdb_names if name in {"qdrant", "weaviate", "milvus", "chromadb"}]
    targets = []
    for name in requested:
        if name == "milvus":
            targets.extend(["etcd", "minio"])
        targets.append(name)
    targets = list(dict.fromkeys(targets))
    log(f"Starting competitor containers: {', '.join(targets)}")
    result = subprocess.run(
        ["podman-compose", "-f", str(COMPOSE_FILE), "up", "-d", *targets],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log(f"Container start failed: {(result.stderr or result.stdout)[-500:]}")
    # Wait up to 60s for all to be healthy
    for i in range(30):
        st = check_containers(requested)
        if all(st.get(name, False) for name in requested):
            selected = {name: st[name] for name in requested}
            log(f"Requested containers ready: {selected}")
            return selected
        time.sleep(2)
    st = check_containers(requested)
    selected = {name: st.get(name, False) for name in requested}
    log(f"Container status (some may be down): {selected}")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Binary import + helpers
# ═══════════════════════════════════════════════════════════════════════════════


def build_binary_import_payload(ids: np.ndarray, vecs: np.ndarray) -> bytes:
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.uint64)
    count, dim = vecs.shape
    rec_dtype = np.dtype([("id", "<u8"), ("vec", "<f4", (dim,))], align=False)
    recs = np.empty(count, dtype=rec_dtype)
    recs["id"] = ids
    recs["vec"] = vecs
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
        # Ground truth contains zero-based row IDs. Use those IDs as primary
        # keys instead of Milvus-generated IDs, which are unrelated values.
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
    ])
    mcoll = Collection(coll_name, schema)

    t0 = time.perf_counter()
    for off in range(0, len(base), 1000):
        end = min(off + 1000, len(base))
        mcoll.insert([list(range(off, end)), base[off:end].tolist()])
    mcoll.flush()
    insert_qps = round(len(base) / (time.perf_counter() - t0), 1)
    if mcoll.num_entities != len(base):
        raise RuntimeError(
            f"Milvus row count mismatch: inserted {len(base)}, stored {mcoll.num_entities}")

    # Build only after all sealed segments exist. Creating the index on an
    # empty collection allowed later segments to race the search phase,
    # producing a mix of indexed and brute-force measurements.
    mcoll.create_index(
        "embedding",
        {"metric_type": "COSINE", "index_type": "HNSW",
         "params": {"M": 16, "efConstruction": 200}},
        timeout=300)
    utility.wait_for_index_building_complete(coll_name, timeout=300)
    progress = utility.index_building_progress(coll_name, timeout=30)
    indexed_rows = int(progress.get("indexed_rows", 0))
    if len(base) >= 1024 and indexed_rows != len(base):
        raise RuntimeError(f"Milvus index incomplete: {progress}, expected {len(base)} rows")
    index_mode = "hnsw" if indexed_rows == len(base) else "raw-small-segment"
    mcoll.load(timeout=300)

    # Milvus requires ef >= top-k. Recall/latency request 100 results, while
    # the steady-state QPS probe requests 10, so record the effective values
    # rather than relying on an adapter-specific implicit clamp.
    search_ef = max(ef_search, 100)
    qps_ef = max(ef_search, 10)
    params = {"metric_type": "COSINE", "params": {"ef": search_ef}}
    for qi in range(warmup):
        mcoll.search([query_lists[qi]], "embedding", params, limit=100)

    recalls_10, recalls_100, latencies = [], [], []
    for qi in range(warmup, n_search):
        t0 = time.perf_counter()
        results = mcoll.search([query_lists[qi]], "embedding", params, limit=100)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retrieved = [int(h.id) for h in results[0]] if results else []
        recalls_10.append(compute_recall(retrieved, gt[qi], 10))
        recalls_100.append(compute_recall(retrieved, gt[qi], 100))

    qps_count = 0
    qps_start = time.perf_counter()
    qps_params = {"metric_type": "COSINE", "params": {"ef": qps_ef}}
    while time.perf_counter() - qps_start < 5.0:
        mcoll.search([query_lists[0]], "embedding", qps_params, limit=10)
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
        "effective_ef_at_100": search_ef, "effective_ef_at_10": qps_ef,
        "index_mode": index_mode,
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
    normalize_checkpoint(cp)
    results = cp["results"]
    failures = cp.get("failures", [])
    fm_results = cp.get("failure_modes", [])
    manifest = cp.get("run_manifest", {})
    matrix = manifest.get("matrix", {})

    result_vdbs = {r["vdb"] for r in results}
    intended_vdbs = set(matrix.get("vdbs", [])) or result_vdbs
    vdbs = [
        vdb for vdb in ["deepdata-grpc", "deepdata-http", "qdrant", "weaviate", "milvus", "chromadb"]
        if vdb in intended_vdbs
    ]
    datasets = matrix.get("datasets") or sorted({r["dataset"] for r in results})
    ef_values = sorted(matrix.get("ef_search") or {int(r["ef_search"]) for r in results})
    result_keys = {result_key(result) for result in results}
    intended_keys = set(manifest.get("intended_cells", []))
    if not intended_keys:
        intended_keys = {
            f"{vdb}|{dataset}|ef={ef}"
            for dataset in datasets for ef in ef_values for vdb in vdbs
        }
    expected_cells = len(intended_keys)
    pending_reruns = set(cp.get("pending_reruns", []))
    covered_cells = len((result_keys & intended_keys) - pending_reruns)
    legacy_rows = sum(1 for result in results if "measured_at" not in result)
    missing_cells = max(expected_cells - covered_cells, 0)
    passed_failure_modes = sum(1 for result in fm_results if result["passed"])

    cpu = "Unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    lines = [
        "# Mega Benchmark: Wide-Sweep All-VDB Results",
        "",
        f"**Report generated:** {time.strftime('%Y-%m-%d %H:%M')}  |  **CPU:** {cpu}",
        "**Systems:** DeepData, Qdrant, Weaviate, Milvus, ChromaDB",
        "**Additional DeepData baseline:** HTTP search at full precision; the canonical gRPC row enables float16 quantization, so this is not a transport-only comparison.",
        f"**Coverage:** {covered_cells}/{expected_cells} matrix cells  |  **Requested queries:** up to {manifest.get('n_search', 'unknown')} per dataset",
        "**Method:** Recall@10 and latency@100 come from top-100 searches (effective ef is at least 100); QPS@10 is a 5-second sequential loop using the requested ef.",
        "",
        "---",
        "",
    ]
    if manifest.get("created_at"):
        lines.insert(8, f"**Manifest created:** {manifest['created_at']}  |  **Python:** {manifest.get('python_version', 'unrecorded')}")
    if legacy_rows:
        lines.insert(8, f"**Provenance:** resumed legacy checkpoint; {legacy_rows} retained rows predate per-cell timestamps.")
    if pending_reruns:
        lines.insert(8, f"**Status:** partial — {len(pending_reruns)} scheduled reruns are not complete; displayed values may be superseded.")

    lines.extend([
        "## Executive Summary",
        "",
        "> This report covers the benchmark repair only. For overall product readiness, see [DeepData Pre-Release Status](../../../docs/PRE_RELEASE_STATUS.md).",
        "",
        (
            f"- **Matrix status:** {covered_cells}/{expected_cells} intended cells complete; "
            f"{missing_cells} missing, {len(failures)} failed, and "
            f"{len(pending_reruns)} pending reruns."
        ),
        (
            f"- **Scope:** {len(vdbs)} benchmark targets across {len(datasets)} datasets "
            f"and {len(ef_values)} `ef_search` settings."
        ),
        (
            f"- **Failure-mode validation:** {passed_failure_modes}/{len(fm_results)} "
            "DeepData probes passed."
            if fm_results else
            "- **Failure-mode validation:** not run for this checkpoint."
        ),
        "- **Milvus repair:** source vector IDs are preserved, data is flushed before HNSW creation, indexed rows are verified before loading, and effective `ef` values are recorded.",
        "- **Harness hardening:** checkpoints are atomic and deduplicated; reruns are scoped and preserve last-good rows; manifests validate datasets, dependencies, and intended cells; concurrent runs are locked.",
        "",
        "## What Is Left for This Benchmark Repair",
        "",
    ])

    if missing_cells or failures or pending_reruns:
        lines.append(
            f"- **Required:** resolve {missing_cells} missing cells, {len(failures)} failures, "
            f"and {len(pending_reruns)} pending reruns."
        )
    else:
        lines.append(
            "- **Required:** nothing remains for the current benchmark repair; the intended "
            "matrix and recorded failure-mode suite are complete."
        )

    if legacy_rows:
        lines.append(
            f"- **Optional provenance upgrade:** run a fresh {expected_cells}-cell sweep to replace the "
            f"{legacy_rows} retained legacy rows with uniformly timestamped measurements."
        )
    lines.extend([
        "- **Optional apples-to-apples transport test:** rerun DeepData gRPC and HTTP with identical quantization; the current gRPC float16 and HTTP full-precision rows are not transport-only comparisons.",
        "- **Optional statistical tightening:** reuse one Milvus index per dataset for the full `ef_search` sweep and add repeated trials with variance or confidence intervals.",
        "",
        "---",
        "",
    ])

    # Group by dataset

    for ds in datasets:
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("### Recall@10 vs ef_search")
        lines.append("")
        header = "| ef_search | " + " | ".join(vdbs) + " |"
        sep = "|-----------|" + "|".join(["--------"] * len(vdbs)) + "|"
        lines.append(header)
        lines.append(sep)
        for ef in ef_values:
            row = f"| {ef:<9} |"
            for vdb in vdbs:
                r = next((x for x in results if x["dataset"] == ds and x["vdb"] == vdb and x["ef_search"] == ef), None)
                row += f" {r['recall_at_10']:.4f} |" if r else " N/A    |"
            lines.append(row)
        lines.append("")

        lines.append("### P50 Latency@100 (ms) vs ef_search")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for ef in ef_values:
            row = f"| {ef:<9} |"
            for vdb in vdbs:
                r = next((x for x in results if x["dataset"] == ds and x["vdb"] == vdb and x["ef_search"] == ef), None)
                row += f" {r['p50_ms']:>6.1f} |" if r else " N/A    |"
            lines.append(row)
        lines.append("")

        lines.append("### QPS@10 vs ef_search")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for ef in ef_values:
            row = f"| {ef:<9} |"
            for vdb in vdbs:
                r = next((x for x in results if x["dataset"] == ds and x["vdb"] == vdb and x["ef_search"] == ef), None)
                row += f" {r['qps']:>6.0f} |" if r else " N/A    |"
            lines.append(row)
        lines.append("")

    # Best configs summary
    lines.append("---")
    lines.append("")
    lines.append("## Best ef_search per VDB and Dataset (highest QPS@10 with R@10 >= 0.95)")
    lines.append("")
    lines.append("| VDB | Dataset | ef_search | R@10 | QPS | P50 ms |")
    lines.append("|-----|---------|-----------|------|-----|--------|")
    for vdb in vdbs:
        for dataset in datasets:
            vdb_results = [
                r for r in results
                if r["vdb"] == vdb and r["dataset"] == dataset
                and r.get("recall_at_10", 0) >= 0.95
            ]
            if vdb_results:
                best = max(vdb_results, key=lambda r: r["qps"])
                lines.append(f"| {vdb} | {best['dataset']} | {best['ef_search']} | {best['recall_at_10']:.4f} | {best['qps']:.0f} | {best['p50_ms']:.1f} |")
    lines.append("")

    # Failure modes
    if fm_results:
        lines.append("---")
        lines.append("")
        lines.append("## Failure Mode Tests (DeepData)")
        lines.append("")
        lines.append("| Test | Passed | Detail |")
        lines.append("|------|--------|--------|")
        for fm in fm_results:
            icon = "PASS" if fm["passed"] else "**FAIL**"
            lines.append(f"| {fm['test']} | {icon} | {fm['detail']} |")
        passed = sum(1 for f in fm_results if f["passed"])
        lines.append("")
        lines.append(f"**{passed}/{len(fm_results)} passed**")
        lines.append("")

    # Errors
    if failures:
        lines.append("---")
        lines.append("")
        lines.append("## Errors During Run")
        lines.append("")
        for f in failures:
            lines.append(f"- `{f['key']}`: {f['error'][:200]}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by `benchmarks/mega_bench.py`*")
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
    p.add_argument("--fresh", action="store_true",
                   help="Explicitly replace an existing checkpoint with a new run")
    p.add_argument("--rerun-vdb", nargs="+", choices=ALL_VDB_NAMES,
                   help="With --resume, invalidate and recompute selected VDB rows")
    p.add_argument("--report-only", action="store_true", help="Regenerate report from checkpoint")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-containers", action="store_true")
    p.add_argument("--skip-failure-modes", action="store_true")
    p.add_argument("--n-search", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rerun_vdb and not args.resume:
        raise SystemExit("--rerun-vdb requires --resume")
    if args.resume and args.fresh:
        raise SystemExit("--resume and --fresh are mutually exclusive")
    if args.resume and not CHECKPOINT_FILE.exists():
        raise SystemExit(f"cannot resume: checkpoint does not exist at {CHECKPOINT_FILE}")
    if (CHECKPOINT_FILE.exists() and not args.resume and not args.fresh
            and not args.report_only):
        raise SystemExit("checkpoint exists; use --resume or explicitly pass --fresh")

    if args.report_only:
        if not CHECKPOINT_FILE.exists():
            raise SystemExit(f"checkpoint does not exist at {CHECKPOINT_FILE}")
        cp = load_checkpoint()
        report = generate_report(cp)
        report_path = RESULTS_DIR / "REPORT.md"
        report_path.write_text(report)
        log(f"Report written to {report_path}")
        return

    cp = load_checkpoint() if args.resume else {"results": [], "completed": [], "failures": []}
    vdb_names = args.vdb or (args.rerun_vdb if args.rerun_vdb else ALL_VDB_NAMES)
    ds_names = args.dataset or ALL_DATASETS
    n_search = args.n_search

    if args.rerun_vdb and not set(args.rerun_vdb).issubset(vdb_names):
        raise SystemExit("every --rerun-vdb value must also be selected by --vdb")

    if args.quick:
        ef_values = [32, 128, 512]
        n_search = 40
    else:
        ef_values = args.ef or ALL_EF_SEARCH

    if n_search <= 10:
        raise SystemExit("--n-search must be greater than the 10-query warmup")

    all_dsets = get_datasets()
    available_ds = [n for n in ds_names if all_dsets[n].base_path.exists()]
    if not available_ds:
        log("No datasets found. Run: python benchmarks/download_datasets.py")
        sys.exit(1)
    missing_ds = [name for name in ds_names if name not in available_ds]
    if missing_ds:
        raise SystemExit(f"dataset files missing for: {', '.join(missing_ds)}")

    ensure_run_manifest(cp, all_dsets, available_ds, n_search, vdb_names, ef_values)

    if args.rerun_vdb:
        scheduled = schedule_vdb_rerun(
            cp, args.rerun_vdb, dataset_names=available_ds, ef_values=ef_values)
        save_checkpoint(cp)
        log(f"Scheduled {scheduled} rows for replacement: {', '.join(args.rerun_vdb)}")

    selected_keys = {
        f"{vdb}|{dataset}|ef={ef}"
        for dataset in available_ds for ef in ef_values for vdb in vdb_names
    }
    completed_keys = set(cp["completed"]) - set(cp.get("pending_reruns", []))
    pending_keys = selected_keys - completed_keys
    pending_vdbs = [
        vdb for vdb in vdb_names
        if any(key.startswith(f"{vdb}|") for key in pending_keys)
    ]
    pending_datasets = {
        key.split("|", 2)[1] for key in pending_keys
    }

    log(f"VDBs: {', '.join(vdb_names)}")
    log(f"Datasets: {', '.join(available_ds)}")
    log(f"ef_search: {ef_values}")
    log(f"n_search: {n_search}")
    log(f"Total combos: {len(available_ds) * len(ef_values) * len(vdb_names)}")
    log("")

    # Build only when a selected DeepData cell is actually pending.
    needs_deepdata = any(v.startswith("deepdata") for v in pending_vdbs)
    if needs_deepdata and not args.skip_build:
        if not build_server():
            sys.exit(1)

    # Start containers
    competitor_vdbs = [
        vdb for vdb in pending_vdbs
        if vdb in ("qdrant", "weaviate", "milvus", "chromadb")
    ]
    needs_containers = bool(competitor_vdbs)
    container_status = {}
    if needs_containers:
        if args.skip_containers:
            # Reuse existing services only after proving they are reachable.
            probed = check_containers(competitor_vdbs)
            for vdb in competitor_vdbs:
                container_status[vdb] = probed.get(vdb, False)
        else:
            container_status = start_containers(competitor_vdbs)

    # Load all datasets upfront
    loaded = {}
    for ds_name in available_ds:
        if ds_name not in pending_datasets:
            continue
        log(f"Loading {ds_name}...")
        data = load_dataset(all_dsets[ds_name])
        if data:
            loaded[ds_name] = data
            log(f"  {len(data[0]):,} base, {len(data[1]):,} queries")

    # ── Main sweep ──
    for ds_name in available_ds:
        if ds_name not in pending_datasets:
            continue
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

                actual_n = min(n_search, len(queries))
                def do_bench():
                    proc = None
                    if vdb.startswith("deepdata"):
                        proc = start_deepdata(ef_construction=200)
                        if not proc:
                            raise RuntimeError("server failed to start")
                    try:
                        return VDB_BENCH_FNS[vdb](
                            ds_name, dim, base, queries, gt, ef, actual_n)
                    finally:
                        if proc is not None:
                            stop_proc(proc)

                result = retry(do_bench, max_attempts=2, desc=key)

                if result:
                    result["n_search"] = actual_n
                    result["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    mark_completed(cp, key, result)
                    log(f"  R@10={result['recall_at_10']:.4f}  QPS={result['qps']:.0f}  P50={result['p50_ms']:.1f}ms")
                else:
                    mark_failure(cp, key, "all retries exhausted")

    # ── Failure modes ──
    if not args.skip_failure_modes and needs_deepdata:
        log("")
        log("=== FAILURE MODE TESTS ===")
        proc = start_deepdata()
        if proc:
            try:
                fm_results = test_failure_modes()
                cp["failure_modes"] = fm_results
                save_checkpoint(cp)
            finally:
                stop_proc(proc)

    stored_completed_keys = set(cp["completed"])
    completed_keys = stored_completed_keys - set(cp.get("pending_reruns", []))
    missing_keys = sorted(selected_keys - completed_keys)
    result_keys = {result_key(result) for result in cp["results"]}
    if result_keys != stored_completed_keys:
        raise SystemExit("checkpoint invariant failed: result and completed key sets differ")
    if missing_keys:
        log(f"INCOMPLETE: {len(missing_keys)} selected cells missing")
        for key in missing_keys[:20]:
            log(f"  missing: {key}")
        raise SystemExit(1)

    # ── Report ──
    report = generate_report(cp)
    report_path = RESULTS_DIR / "REPORT.md"
    report_path.write_text(report)
    log("")
    log(f"Report written to {report_path}")
    log(f"Checkpoint at {CHECKPOINT_FILE}")
    log(f"Total results: {len(cp['results'])}, failures: {len(cp['failures'])}")


if __name__ == "__main__":
    with benchmark_lock():
        main()
