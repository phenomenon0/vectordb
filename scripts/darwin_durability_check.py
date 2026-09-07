#!/usr/bin/env python3
"""Durability lifecycle check for the Darwin port of DeepData's persistent store.

Exercises the ONLY durable HTTP surface (/v3/tenants/...) through:
  phase 1  fresh boot -> create collection -> insert N docs -> verify
  phase 2  kill -9 (no graceful shutdown) -> restart -> journal replay must
           restore the exact doc count and the exact search result
  phase 3  SIGTERM (graceful) -> checkpoint into snapshot -> cold restart
           from that snapshot must restore the same state

Run on the Mac, from the repo root. Refuses to run anywhere else.
Writes .deepdata-run/checks/darwin-durability/receipt.json for MAC-01.

    python3 scripts/darwin_durability_check.py bin/deepdata-darwin-arm64 /tmp/dd-mac 18080
"""

import datetime
import hashlib
import json
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# This check is the only evidence that the Darwin port works. Every assertion
# below is platform-neutral, so on Linux it passes without touching darwin and
# the receipt would name a host that proves nothing. Refuse rather than lie.
if platform.system() != "Darwin":
    sys.exit(
        f"darwin-durability: refusing to run on {platform.system()}; this check "
        "exists to prove the Darwin port and would pass here without exercising it"
    )

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STARTED_AT = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

BIN, SCRATCH, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])
BASE = f"http://127.0.0.1:{PORT}"
TENANT, COLL, DIM, NDOCS, BATCH = "default", "docs", 128, 3000, 500
DATA_DIR = os.path.join(SCRATCH, "data")

failures = []


def brief(v):
    r = repr(v)
    return r if len(r) <= 90 else f"{r[:70]}... (len {len(v)})"


def check(label, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {brief(got)}, want {brief(want)}")
    if not ok:
        failures.append(label)
    return ok


def check_close(label, got, want, tol=1e-6):
    """Vectors are stored as float32, so a float64 client value only round-trips
    to within float32 precision. Restart-to-restart comparisons still use exact
    equality - those are server value vs server value."""
    ok = (isinstance(got, list) and len(got) == len(want)
          and all(abs(a - b) <= tol for a, b in zip(got, want)))
    worst = max((abs(a - b) for a, b in zip(got, want)), default=None) if isinstance(got, list) else None
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: len={len(got) if isinstance(got, list) else got}, "
          f"max abs diff={worst:.3e}" if worst is not None else f"  [FAIL] {label}: {brief(got)}")
    if not ok:
        failures.append(label)
    return ok


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return resp.status, {"raw": raw.decode(errors="replace")}
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}


def vec(i):
    """Deterministic and UNIQUE per i, reproducible across processes.

    Component 0 is a strictly increasing function of i so no two documents
    share a vector. An earlier version used only modular arithmetic, which
    made vec(i) == vec(i+1000); the resulting exact score ties reordered
    across restarts and looked like a durability failure when it was really
    just unspecified tie-breaking.
    """
    return [i / float(NDOCS)] + [((i * 31 + j * 17) % 997) / 997.0
                                 for j in range(1, DIM)]


def start(tag):
    env = dict(
        os.environ,
        VECTORDB_DATA_DIR=DATA_DIR,
        VECTORDB_MODE="local",
        PORT=str(PORT),
        EMBEDDER_TYPE="hash",
        DEEPDATA_EMBEDDER="hash",
        DEEPDATA_INSECURE_DEV_MODE="1",
    )
    log = open(os.path.join(SCRATCH, f"server-{tag}.log"), "wb")
    p = subprocess.Popen([BIN], env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(150):
        time.sleep(0.2)
        if p.poll() is not None:
            print(f"  server exited early rc={p.returncode}; log tail:")
            print(open(os.path.join(SCRATCH, f"server-{tag}.log")).read()[-1500:])
            sys.exit(1)
        try:
            st, body = req("GET", "/readyz")
            if st == 200 and body.get("ready") is True:
                if tag == "boot":
                    print(f"  /readyz durable checks: {body.get('checks')}")
                return p
        except Exception:
            pass
    print("  server never became ready")
    sys.exit(1)


def doc_count():
    st, body = req("GET", f"/v3/tenants/{TENANT}/collections/{COLL}")
    if st != 200:
        return f"HTTP {st}: {body}"
    # Wire field is the json tag doc_count, not the Go field name (see
    # internal/collection/manager.go CollectionInfo). Reading the Go name made
    # every phase return the same "unparsed" sentinel, so the post-crash and
    # post-cold-start comparisons against it passed without checking anything.
    return body.get("collection", {}).get("doc_count", f"unparsed:{body}")


def search(qi=7, k=5):
    """Returns (ordered ids, id set, top-hit stored vector, top-hit metadata)."""
    st, body = req(
        "POST",
        f"/v3/tenants/{TENANT}/collections/{COLL}/search",
        {"queries": {"embedding": vec(qi)}, "top_k": k, "include_vectors": True},
    )
    if st != 200:
        return (f"HTTP {st}: {body}", None, None, None)
    docs = body.get("documents", [])
    ids = [d.get("id") for d in docs]
    top = docs[0] if docs else {}
    return (ids, set(ids), (top.get("vectors") or {}).get("embedding"),
            top.get("metadata"))


def artifacts():
    # per-tenant StoreSet layout: TENANT's own files under index.gob.tenants/,
    # named after its tenant id (internal/collection/store_set.go).
    base = os.path.join(DATA_DIR, "index.gob.tenants", TENANT)
    return {
        os.path.basename(base + s): os.path.exists(base + s)
        for s in (
            "",
            ".lock",
            ".journal",
            ".journal.frozen",
            ".snapshot",
            ".initialized",
        )
    }


print("=" * 72)
print("PHASE 1 — fresh boot, create collection, insert docs")
print("=" * 72)
shutil.rmtree(DATA_DIR, ignore_errors=True)
os.makedirs(DATA_DIR, exist_ok=True)
p = start("boot")
print(f"  server pid {p.pid} on :{PORT}, data dir {DATA_DIR}")

st, body = req(
    "POST",
    f"/v3/tenants/{TENANT}/collections",
    {
        "name": COLL,
        "fields": [
            {
                "name": "embedding",
                "type": "dense",
                "dim": DIM,
                "index": {"type": "hnsw", "params": {"m": 16, "ef_construction": 200}},
            }
        ],
    },
)
check("create collection HTTP status", st, 201)
if st != 201:
    print("  body:", body)
    p.kill()
    sys.exit(1)

t0 = time.time()
ids = []
for s in range(0, NDOCS, BATCH):
    docs = [
        {"vectors": {"embedding": vec(i)}, "metadata": {"seq": i}}
        for i in range(s, min(s + BATCH, NDOCS))
    ]
    st, body = req(
        "POST",
        f"/v3/tenants/{TENANT}/collections/{COLL}/docs/batch",
        {"documents": docs},
    )
    if st != 200:
        print(f"  batch at {s} failed: {st} {body}")
        p.kill()
        sys.exit(1)
    ids += body.get("ids", [])
elapsed = time.time() - t0
print(
    f"  inserted {len(ids)} docs in {elapsed:.2f}s "
    f"({len(ids) / elapsed:.0f} docs/s, batch={BATCH}, dim={DIM})"
)

pre_count = doc_count()
pre_ids, pre_set, pre_vec, pre_meta = search()
check("doc count before crash", pre_count, NDOCS)
check("exact search hit is the queried doc", pre_ids[0] if pre_ids else None, 8)
check_close("stored vector round-trips within float32", pre_vec, vec(7))
check("stored metadata round-trips exactly", pre_meta, {"seq": 7})
print(f"  search(query=vec(7)) top-5 ids before crash: {pre_ids}")
print(f"  on-disk artifacts: {artifacts()}")

print()
print("=" * 72)
print("PHASE 2 — kill -9 (hard crash, no graceful shutdown) then restart")
print("=" * 72)
os.kill(p.pid, signal.SIGKILL)
p.wait()
print(f"  sent SIGKILL to pid {p.pid}; process rc={p.returncode} (-9 = killed)")
check("process died via SIGKILL", p.returncode, -9)
post_kill = artifacts()
print(f"  artifacts after kill: {post_kill}")
check("journal survived the crash", post_kill[f"{TENANT}.journal"], True)

p2 = start("after-kill")
print(f"  restarted, pid {p2.pid} — journal replay path")
crash_count = doc_count()
crash_ids, crash_set, crash_vec, crash_meta = search()
check("doc count after crash+replay", crash_count, pre_count)
check("top-k id set after crash+replay", crash_set, pre_set)
check("top-k id order after crash+replay", crash_ids, pre_ids)
check("stored vector intact after crash+replay", crash_vec, pre_vec)
check("stored metadata intact after crash+replay", crash_meta, pre_meta)
st, b = req("GET", f"/v3/tenants/{TENANT}/collections/{COLL}")
check("collection readable after replay", st, 200)

print()
print("=" * 72)
print("PHASE 3 — graceful SIGTERM (checkpoint to snapshot) then cold start")
print("=" * 72)
os.kill(p2.pid, signal.SIGTERM)
for _ in range(150):
    time.sleep(0.2)
    if p2.poll() is not None:
        break
print(f"  graceful shutdown rc={p2.returncode}")
check("graceful shutdown exited cleanly", p2.returncode, 0)
post_term = artifacts()
print(f"  artifacts after SIGTERM: {post_term}")
check(
    "snapshot written on graceful close",
    post_term[f"{TENANT}.snapshot"],
    True,
)
check(
    "journal cleaned after checkpoint",
    post_term[f"{TENANT}.journal"],
    False,
)
check("lock released on clean exit", post_term[f"{TENANT}.lock"], True)

p3 = start("cold")
print(f"  cold-started from snapshot, pid {p3.pid}")
cold_count = doc_count()
cold_ids, cold_set, cold_vec, cold_meta = search()
check("doc count after cold start from snapshot", cold_count, pre_count)
check("top-k id set after cold start", cold_set, pre_set)
check("top-k id order after cold start", cold_ids, pre_ids)
check("stored vector intact after cold start", cold_vec, pre_vec)
check("stored metadata intact after cold start", cold_meta, pre_meta)

os.kill(p3.pid, signal.SIGTERM)
for _ in range(100):
    time.sleep(0.2)
    if p3.poll() is not None:
        break
if p3.poll() is None:
    p3.kill()

print()
print("=" * 72)
print(
    f"RESULT: {'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}"
)
print("=" * 72)


def tree_fingerprint():
    """Same digest hardening_check.sh writes, so a Mac receipt and a Linux
    receipt are comparable and a dirty Mac tree is as disqualifying as a dirty
    Linux one. The binary under test belongs in bin/ or outside the repo; both
    are ignored, so copying it in does not dirty the tree."""
    digest = hashlib.sha256()
    digest.update(subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD", "--"],
        check=True, stdout=subprocess.PIPE).stdout)
    untracked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True, stdout=subprocess.PIPE).stdout.split(b"\0")
    for encoded in sorted(path for path in untracked if path):
        digest.update(encoded)
        digest.update(b"\0")
        path = REPO_ROOT / encoded.decode("utf-8", errors="surrogateescape")
        if path.is_file():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


# hardening_check.sh cannot run here -- BSD date has no -Iseconds, timeout is
# not installed, and its heredocs call `python`, which macOS does not ship -- and
# a receipt it half-wrote would carry an empty finished_at that fails gates.py's
# date regex and breaks `gates.py check` for the whole ledger. So this check
# writes its own, in the same schema-2 shape gates.py promote reads.
uname = os.uname()
receipt_dir = REPO_ROOT / ".deepdata-run/checks/darwin-durability"
receipt_dir.mkdir(parents=True, exist_ok=True)
receipt = {
    "schema_version": 2,
    "check": "darwin-durability",
    "host": {
        "hostname": uname.nodename,
        "kernel": uname.release,
        "arch": uname.machine,
        "cpus": os.cpu_count(),
        "system": platform.system(),
        "mac_version": platform.mac_ver()[0],
    },
    "status": "failed" if failures else "pass",
    "exit_code": 1 if failures else 0,
    "started_at": STARTED_AT,
    "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, text=True).stdout.strip(),
    "tree_fingerprint": tree_fingerprint(),
    "working_directory": str(REPO_ROOT),
    "command": " ".join([sys.executable, *sys.argv]),
    # The terminal transcript is the log; what a reader needs from the receipt
    # is which assertions failed, so carry those instead of a path to a file
    # this check does not write.
    "log_path": None,
    "failures": failures,
}
tmp = receipt_dir / f".receipt.{os.getpid()}.tmp"
with tmp.open("w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, receipt_dir / "receipt.json")
print(f"receipt: {receipt_dir / 'receipt.json'}")

sys.exit(1 if failures else 0)
