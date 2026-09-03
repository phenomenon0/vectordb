#!/usr/bin/env python3
"""Stage C rehearsal: VDB correctness, mixed-load soak, restart-under-load,
memory-drift, and chaos evidence against the canonical RC server.

Invariants enforced:
- Acknowledged (2xx) inserts survive every SIGKILL/restart; acknowledged
  deletes stay deleted; in-flight ops at kill time are indeterminate and
  bounded, never duplicated.
- Flat-index exact-vector search returns the owning doc as top hit.
- HNSW recall@10 vs client-side brute-force ground truth >= 0.95 (cosine).
- Server RSS in the final soak third grows < 30% over the first third.
"""

import json, math, os, random, signal, socket, subprocess, sys, threading, time
import urllib.request, urllib.error

REPO = os.environ.get("DEEPDATA_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SHA = subprocess.run(
    ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
).stdout.strip()
WORK = f"{REPO}/.deepdata-run/rehearsals/soak-{SHA}"
os.makedirs(WORK, exist_ok=True)
STATE = f"{WORK}/state"
HTTP, GRPC = 19082, 59083
BASE = f"http://127.0.0.1:{HTTP}"
TOKEN = "soak-rehearsal-strong-credential"
TENANT = "soak"
# A prebuilt binary (SOAK_BIN) lets the soak run on a host without a Go
# toolchain; it must be built from the same commit the receipt records.
SERVER_BIN = os.environ.get("SOAK_BIN") or f"{WORK}/deepdata"

SOAK_MINUTES = float(os.environ.get("SOAK_MINUTES", "25"))
KILLS = int(os.environ.get("SOAK_KILLS", "5"))

log_f = open(f"{WORK}/soak.log", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()


def api(method, path, body=None, timeout=15):
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def api_retry(method, path, body=None, tries=120):
    last = None
    for _ in range(tries):
        try:
            return api(method, path, body)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                last = e
                time.sleep(0.5)
                continue
            raise
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"api_retry exhausted for {method} {path}: last={last!r}")


proc = None


def start_server():
    global proc
    env = dict(
        os.environ,
        VECTORDB_MODE="local",
        VECTORDB_BASE_DIR=STATE,
        VECTORDB_DATA_DIR="local",
        PORT=str(HTTP),
        GRPC_PORT=str(GRPC),
        API_TOKEN=TOKEN,
        REQUIRE_AUTH="1",
        TENANT_RPS="500",
        TENANT_BURST="500",
        GOTOOLCHAIN="go1.25.12",
    )
    proc = subprocess.Popen(
        [SERVER_BIN, "serve"],
        env=env,
        stdout=open(f"{WORK}/server.log", "a"),
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={proc.returncode}")
        try:
            urllib.request.urlopen(BASE + "/readyz", timeout=1)
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("server never became ready")


def rss_kb():
    try:
        with open(f"/proc/{proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return None


def vec(seed, dim):
    r = random.Random(str(seed))
    return [r.uniform(-1, 1) for _ in range(dim)]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


results = {"sha": SHA, "phases": {}}


def phase_result(name, ok, detail):
    results["phases"][name] = {"ok": ok, "detail": detail}
    log(f"{'PASS' if ok else 'FAIL'}: {name}: {detail}")
    if not ok:
        json.dump(results, open(f"{WORK}/receipt.json", "w"), indent=1)
        sys.exit(1)


# ---- build + start
log(f"building server at {SHA}")
try:
    urllib.request.urlopen(BASE + "/readyz", timeout=1)
    log(f"FATAL: something already listens on {HTTP}; kill it first")
    sys.exit(2)
except OSError:
    pass
import atexit

atexit.register(lambda: proc and proc.poll() is None and proc.kill())
if os.environ.get("SOAK_BIN"):
    log(f"using prebuilt binary {SERVER_BIN}")
else:
    subprocess.run(
        ["go", "build", "-trimpath", "-o", SERVER_BIN, "./cmd/deepdata"],
        cwd=REPO,
        check=True,
        env=dict(os.environ, GOTOOLCHAIN=os.environ.get("GOTOOLCHAIN", "go1.25.12")),
    )
if os.path.exists(STATE):
    subprocess.run(["rm", "-rf", STATE])
start_server()

# ---- phase 1: VDB correctness (brute-force ground truth)
log("phase 1: correctness — seeding 20k HNSW docs + 2k flat docs")
api_retry(
    "POST",
    f"/v3/tenants/{TENANT}/collections",
    {
        "name": "c_hnsw",
        "fields": [
            {"name": "embedding", "type": "dense", "dim": 64, "index": {"type": "hnsw"}}
        ],
    },
)
api_retry(
    "POST",
    f"/v3/tenants/{TENANT}/collections",
    {
        "name": "c_flat",
        "fields": [
            {"name": "embedding", "type": "dense", "dim": 32, "index": {"type": "flat"}}
        ],
    },
)
N_HNSW, N_FLAT = 20000, 2000
hnsw_vecs = {}
for lo in range(1, N_HNSW + 1, 16):
    docs = []
    for i in range(lo, min(lo + 16, N_HNSW + 1)):
        v = vec(("h", i), 64)
        hnsw_vecs[i] = v
        docs.append({"id": i, "vectors": {"embedding": v}})
    api_retry(
        "POST",
        f"/v3/tenants/{TENANT}/collections/c_hnsw/docs/batch",
        {"documents": docs},
    )
    time.sleep(0.013)
flat_vecs = {}
for lo in range(1, N_FLAT + 1, 16):
    docs = []
    for i in range(lo, min(lo + 16, N_FLAT + 1)):
        v = vec(("f", i), 32)
        flat_vecs[i] = v
        docs.append({"id": i, "vectors": {"embedding": v}})
    api_retry(
        "POST",
        f"/v3/tenants/{TENANT}/collections/c_flat/docs/batch",
        {"documents": docs},
    )
    time.sleep(0.013)

recalls = []
for q in range(200):
    qv = vec(("q", q), 64)
    truth = sorted(hnsw_vecs, key=lambda i: -cos(qv, hnsw_vecs[i]))[:10]
    got = [
        d["id"]
        for d in api_retry(
            "POST",
            f"/v3/tenants/{TENANT}/collections/c_hnsw/search",
            {"queries": {"embedding": qv}, "top_k": 10},
        )["documents"]
    ]
    recalls.append(len(set(truth) & set(got)) / 10)
    time.sleep(0.005)
recall = sum(recalls) / len(recalls)
phase_result(
    "hnsw-recall-vs-brute-force",
    recall >= 0.95,
    f"recall@10={recall:.4f} over 200 queries",
)

exact_ok = 0
sample = random.Random(7).sample(sorted(flat_vecs), 100)
for i in sample:
    got = api_retry(
        "POST",
        f"/v3/tenants/{TENANT}/collections/c_flat/search",
        {"queries": {"embedding": flat_vecs[i]}, "top_k": 1},
    )["documents"]
    exact_ok += bool(got and got[0]["id"] == i)
phase_result(
    "flat-exact-top1",
    exact_ok == 100,
    f"{exact_ok}/100 exact-vector queries returned owner as top-1",
)

# ---- phase 2+3+4: mixed-load soak with SIGKILL cycles, RSS tracking
log(f"phase 2-4: {SOAK_MINUTES}min mixed soak, {KILLS} SIGKILLs under load")
ledger_lock = threading.Lock()
acked = dict(flat_vecs)  # id -> vector (acknowledged, not deleted)
deleted = set()
indeterminate = set()  # sent, response unknown at a kill
next_id = [N_FLAT + 1]
stop_evt = threading.Event()
pause_evt = threading.Event()  # set = workers pause (during restart)
stats = {"ins": 0, "del": 0, "srch": 0, "err5xx": 0, "conn_err": 0, "s429": 0}


def worker_insert():
    while not stop_evt.is_set():
        if pause_evt.is_set():
            time.sleep(0.3)
            continue
        with ledger_lock:
            i = next_id[0]
            next_id[0] += 1
        v = vec(("f", i), 32)
        try:
            api(
                "POST",
                f"/v3/tenants/{TENANT}/collections/c_flat/docs",
                {"id": i, "vectors": {"embedding": v}},
                timeout=10,
            )
            with ledger_lock:
                acked[i] = v
                stats["ins"] += 1
        except urllib.error.HTTPError as e:
            if e.code == 429:
                stats["s429"] += 1
                time.sleep(0.3)
            else:
                stats["err5xx"] += 1
        except Exception:
            with ledger_lock:
                indeterminate.add(i)
            stats["conn_err"] += 1
            time.sleep(0.3)
        time.sleep(0.05)


def worker_delete():
    rng = random.Random(11)
    while not stop_evt.is_set():
        if pause_evt.is_set():
            time.sleep(0.3)
            continue
        with ledger_lock:
            live = [i for i in acked if i > N_FLAT]  # only churn soak-era docs
            target = rng.choice(live) if len(live) > 50 else None
        if target is None:
            time.sleep(0.5)
            continue
        try:
            api(
                "DELETE",
                f"/v3/tenants/{TENANT}/collections/c_flat/docs",
                {"doc_id": target},
                timeout=10,
            )
            with ledger_lock:
                acked.pop(target, None)
                deleted.add(target)
                stats["del"] += 1
        except urllib.error.HTTPError as e:
            if e.code == 429:
                stats["s429"] += 1
                time.sleep(0.3)
            else:
                stats["err5xx"] += 1
        except Exception:
            with ledger_lock:
                indeterminate.add(target)
            stats["conn_err"] += 1
            time.sleep(0.3)
        time.sleep(0.15)


def worker_search():
    rng = random.Random(13)
    while not stop_evt.is_set():
        if pause_evt.is_set():
            time.sleep(0.3)
            continue
        try:
            api(
                "POST",
                f"/v3/tenants/{TENANT}/collections/c_hnsw/search",
                {
                    "queries": {"embedding": vec(("q", rng.randrange(10000)), 64)},
                    "top_k": 10,
                },
                timeout=10,
            )
            stats["srch"] += 1
        except urllib.error.HTTPError as e:
            if e.code == 429:
                stats["s429"] += 1
                time.sleep(0.3)
            else:
                stats["err5xx"] += 1
        except Exception:
            stats["conn_err"] += 1
            time.sleep(0.3)
        time.sleep(0.03)


def reconcile(tag):
    """After restart: acked docs present, deleted absent, no duplicates."""
    with ledger_lock:
        check = random.Random(tag).sample(sorted(acked), min(60, len(acked)))
        gone = (
            random.Random(tag).sample(sorted(deleted), min(20, len(deleted)))
            if deleted
            else []
        )
    bad = []
    for i in check:
        with ledger_lock:
            v = acked.get(i)
        if v is None:
            continue
        got = api_retry(
            "POST",
            f"/v3/tenants/{TENANT}/collections/c_flat/search",
            {"queries": {"embedding": v}, "top_k": 3},
        )["documents"]
        ids = [d["id"] for d in got]
        if not ids or ids[0] != i or ids.count(i) > 1:
            bad.append((i, ids))
    for i in gone:
        if i in indeterminate:
            continue
        v = vec(("f", i), 32)
        got = api_retry(
            "POST",
            f"/v3/tenants/{TENANT}/collections/c_flat/search",
            {"queries": {"embedding": v}, "top_k": 1},
        )["documents"]
        if got and got[0]["id"] == i:
            bad.append(("deleted-returned", i))
    return bad


workers = [
    threading.Thread(target=f, daemon=True)
    for f in (worker_insert, worker_insert, worker_delete, worker_search, worker_search)
]
for w in workers:
    w.start()

soak_end = time.time() + SOAK_MINUTES * 60
kill_times = [
    soak_end - (KILLS - k) * (SOAK_MINUTES * 60 / (KILLS + 1)) for k in range(KILLS)
]
rss_series = []
kills_done = 0
stall_done = False
while time.time() < soak_end:
    time.sleep(5)
    r = rss_kb()
    if r:
        rss_series.append((time.time(), r))
    if not stall_done and time.time() > soak_end - SOAK_MINUTES * 30:  # halfway-ish
        log("chaos: SIGSTOP stall 10s")
        proc.send_signal(signal.SIGSTOP)
        time.sleep(10)
        proc.send_signal(signal.SIGCONT)
        stall_done = True
    if kills_done < KILLS and time.time() >= kill_times[kills_done]:
        kills_done += 1
        log(f"chaos: SIGKILL #{kills_done} under load")
        pause_evt.set()
        proc.kill()
        proc.wait()
        start_server()
        bad = reconcile(kills_done)
        if bad:
            phase_result(
                f"restart-under-load-{kills_done}", False, f"violations: {bad[:5]}"
            )
        log(
            f"restart #{kills_done}: reconciliation clean "
            f"(acked={len(acked)} deleted={len(deleted)} indeterminate={len(indeterminate)})"
        )
        pause_evt.clear()

stop_evt.set()
for w in workers:
    w.join(timeout=15)
phase_result(
    "restart-under-load",
    kills_done == KILLS,
    f"{kills_done} SIGKILL/restart cycles, every reconciliation clean",
)

# final reconciliation + count parity
bad = reconcile("final")
phase_result(
    "final-ledger-reconciliation",
    not bad,
    f"violations: {bad[:5] if bad else 'none'}; stats={stats}",
)
info = api_retry("GET", f"/v3/tenants/{TENANT}/collections/c_flat")
# Wire field is the json tag doc_count, not the Go field name; asserted by
# cmd/deepdata/canonical_process_test.go against the real HTTP surface.
server_count = info["collection"]["doc_count"]
with ledger_lock:
    lo = len(acked) - len(indeterminate)
    hi = len(acked) + len(indeterminate)
phase_result(
    "count-parity",
    lo <= server_count <= hi,
    f"server={server_count} ledger acked={len(acked)} indeterminate={len(indeterminate)}",
)

# memory drift
third = max(1, len(rss_series) // 3)
first = sorted(r for _, r in rss_series[:third])[third // 2]
last = sorted(r for _, r in rss_series[-third:])[len(rss_series[-third:]) // 2]
growth = (last - first) / first if first else 0
json.dump([{"t": t, "rss_kb": r} for t, r in rss_series], open(f"{WORK}/rss.json", "w"))
phase_result(
    "memory-drift",
    growth < 0.30,
    f"median RSS first-third={first}KB last-third={last}KB growth={growth:.1%}",
)

# graceful shutdown
proc.terminate()
proc.wait(timeout=30)
phase_result("graceful-shutdown", proc.returncode == 0, f"rc={proc.returncode}")

json.dump(results, open(f"{WORK}/receipt.json", "w"), indent=1)
log("SOAK COMPLETE: all phases passed")
