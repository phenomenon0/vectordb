#!/usr/bin/env python3
"""Two-node replication probe: leader plus a `deepdata replicate` follower.

The spike/multinode branch recorded three two-node failures and was parked
unmerged. All three are about the space between two processes, so no package
test can hold them -- each one needs a real leader, a real follower, and a real
restart. This probe is that, run against the shipping binary:

  CHECK 1  A replica directory is single-writer. runReplicate and
           docs/distributed-architecture.md both once described serving a
           replica with `deepdata serve` against the directory `deepdata
           replicate` is syncing. The collection store takes LOCK_EX, so the
           second process is refused. A replica is a directory kept current for
           a later read, not a live member of a read fleet.
  CHECK 2  A replica refuses a local write. The original finding was HTTP 200
           on a write to the replica: two nodes forking history at the same LSN.
           The marker that prevents it has to be durable, or a fresh `serve`
           opens the directory as an ordinary store and the split brain is back.
  CHECK 3  Writes issued after the leader restarts still reach the follower, and
           the two nodes agree document for document.

Two rules this probe is built around, both learned the expensive way:

Assert content, never presence. An id is a claim both nodes can satisfy while
disagreeing -- a leader that loses its place re-mints ids from 1, collides with
a document the replica already holds, and GetDocument(id) -> ok returns true off
the stale document while the write under test was never delivered.

A count is not an agreement. The mutation under test is deliberately
count-neutral (one delete paired with one insert), so a replica that applied
neither still reports the same document count as the leader and passes every
count-based health probe. Only the document-for-document comparison fails it.

And one rule about the probe itself: a transport failure is not a divergence.
The first version of this comparison folded an HTTP 429 body into the compared
value and reported "0/50 post-restart writes reached the replica" -- a
fabricated split brain produced entirely by outrunning API_RPS. Anything that is
not a document or an honest 404 now aborts instead of being compared.
"""

import json, os, signal, subprocess, sys, time, urllib.error, urllib.request

REPO = os.environ.get("DEEPDATA_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
WORK = os.environ.get("WORK", f"{REPO}/.deepdata-run/rehearsals/multinode")
BIN = os.environ.get("BIN", f"{WORK}/deepdata")
LDIR, RDIR = f"{WORK}/leader", f"{WORK}/replica"
LP, RP = int(os.environ.get("LEADER_PORT", "19096")), int(os.environ.get("REPLICA_PORT", "19097"))
TOKEN = "multinode-rehearsal-api-strong-credential"
NODE_TOKEN = "multinode-rehearsal-node-strong-credential"
TENANT, COLLECTION = "acme", "docs"
COLL_PATH = f"/v3/tenants/{TENANT}/collections/{COLLECTION}"
DIM = 8
PRE_LO, PRE_HI = 1, 100
POST_LO, POST_HI = 101, 150
DELETED_ID, CANARY_ID = 50, 99999
PROBED = list(range(PRE_LO, POST_HI + 1)) + [CANARY_ID]

os.makedirs(LDIR, exist_ok=True)
os.makedirs(RDIR, exist_ok=True)

# The comparison issues ~300 point reads back to back. API_RPS defaults to 100
# per minute, so the probe would rate-limit itself and every 429 it read as a
# document would look like replication loss.
CHILD_ENV = {
    **os.environ,
    "API_RPS": "1000000",
    "TENANT_RPS": "1000000",
    "TENANT_BURST": "1000000",
    "VECTORDB_DATA_DIR": "local",
    "GRPC_PORT": "0",
}

failures = []
notes = {}


def record(ok, name, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")
    return ok


def request(port, method, path, body=None, token=TOKEN, timeout=30):
    """Returns (status, parsed_body). Raises only on transport failure."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method, headers=headers,
        data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw.decode("utf-8", "replace")[:200]}


def wait_ready(port, seconds=90):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def spawn(args, base_dir, extra=None, port=None):
    env = {**CHILD_ENV, "VECTORDB_BASE_DIR": base_dir,
           "DEEPDATA_REPLICATION_TOKEN": NODE_TOKEN, **(extra or {})}
    if port is not None:
        env["PORT"] = str(port)
    log = open(f"{WORK}/{args[0]}-{base_dir.rsplit('/', 1)[-1]}.log", "a")
    return subprocess.Popen([BIN, *args], env=env, stdout=log, stderr=subprocess.STDOUT)


def start_leader():
    proc = spawn(["serve"], LDIR, extra={"API_TOKEN": TOKEN}, port=LP)
    if not wait_ready(LP):
        sys.exit("multinode: leader never became ready; see leader logs under " + WORK)
    return proc


def stop(proc, seconds=60):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def seed(lo, hi, tag):
    docs = [{"id": i,
             "vectors": {"embedding": [float((i * 7 + j) % 13) / 13.0 for j in range(DIM)]},
             "metadata": {"tag": tag, "n": str(i)}} for i in range(lo, hi + 1)]
    status, body = request(LP, "POST", f"{COLL_PATH}/docs/batch", {"documents": docs})
    if status not in (200, 201):
        sys.exit(f"multinode: seeding {lo}..{hi} failed with HTTP {status}: {body}")
    print(f"  seeded ids {lo}..{hi} tag={tag}")


def fetch(port, doc_id):
    """The document, or None for an honest 404. Anything else means the probe is
    broken, not the replica, so say which and stop rather than compare it."""
    for attempt in range(6):
        status, body = request(port, "GET", f"{COLL_PATH}/docs/{doc_id}")
        if status == 200:
            inner = body.get("document", body) if isinstance(body, dict) else body
            vectors = (inner.get("vectors") or {}) if isinstance(inner, dict) else {}
            return {"metadata": inner.get("metadata"),
                    "embedding": [round(float(v), 6) for v in (vectors.get("embedding") or [])]}
        if status == 404:
            return None
        if status == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        sys.exit(f"multinode: PROBE BROKEN -- port {port} id {doc_id} returned HTTP {status} "
                 f"({body}); that is a transport failure, not a divergence, and nothing was compared")
    sys.exit(f"multinode: PROBE BROKEN -- port {port} id {doc_id} stayed rate limited; "
             "raise API_RPS for the probe rather than reading 429 as a missing document")


leader = follower = replica_server = None
try:
    print("== leader boot, collection, pre-restart seed ==")
    leader = start_leader()
    status, body = request(LP, "POST", f"/v3/tenants/{TENANT}/collections", {
        "name": COLLECTION,
        "fields": [{"name": "embedding", "type": "dense", "dim": DIM, "index": {"type": "flat"}}]})
    if status not in (200, 201):
        sys.exit(f"multinode: creating the collection failed with HTTP {status}: {body}")
    seed(PRE_LO, PRE_HI, "pre-restart")

    print("== follower attach ==")
    follower = spawn(["replicate", "--leader", f"http://127.0.0.1:{LP}"], RDIR)
    time.sleep(8)
    if follower.poll() is not None:
        sys.exit(f"multinode: follower exited with {follower.returncode} before syncing")

    print("== CHECK 1: replica directory is single-writer ==")
    contender = subprocess.run(
        [BIN, "serve"],
        env={**CHILD_ENV, "VECTORDB_BASE_DIR": RDIR, "API_TOKEN": TOKEN, "PORT": str(RP)},
        capture_output=True, text=True, timeout=60)
    locked = "collection store is already open" in (contender.stdout + contender.stderr)
    notes["lock_refusal"] = locked
    record(locked, "single-writer",
           "serve refused on a directory the follower holds"
           if locked else f"serve was NOT refused; exit {contender.returncode}, "
                          f"tail: {(contender.stdout + contender.stderr).strip()[-200:]}")

    print("== leader restart with a follower attached ==")
    started = time.monotonic()
    stop(leader, seconds=90)
    drain = round(time.monotonic() - started, 1)
    notes["leader_shutdown_seconds"] = drain
    # A follow stream never goes idle, so http.Server.Shutdown waits out its full
    # deadline. Recorded, not asserted: the Helm chart's 90s grace period covers
    # it, and any shorter one would SIGKILL the leader mid-checkpoint.
    print(f"  leader shutdown took {drain}s with one follower attached")
    if follower.poll() is not None:
        sys.exit(f"multinode: follower died when the leader went away (exit {follower.returncode})")
    time.sleep(2)
    leader = start_leader()
    seed(POST_LO, POST_HI, "post-restart")

    print("== count-neutral mutation: one delete paired with one insert ==")
    status, body = request(LP, "DELETE", f"{COLL_PATH}/docs", {"doc_id": DELETED_ID})
    if status not in (200, 202, 204):
        sys.exit(f"multinode: deleting id {DELETED_ID} returned HTTP {status} ({body}); the "
                 "divergence check would be vacuous, so this is a probe failure not a result")
    seed(CANARY_ID, CANARY_ID, "divergence-canary")

    print("== drain the follower, then serve the replica for reads ==")
    time.sleep(12)
    stop(follower, seconds=60)
    follower = None
    replica_server = spawn(["serve"], RDIR, extra={"API_TOKEN": TOKEN}, port=RP)
    if not wait_ready(RP):
        sys.exit("multinode: the replica would not serve for reads; see replica logs under " + WORK)

    print("== CHECK 2: replica refuses a local write ==")
    status, body = request(RP, "POST", f"{COLL_PATH}/docs", {
        "id": 424242, "vectors": {"embedding": [0.0] * DIM}})
    notes["replica_write_status"] = status
    record(status not in (200, 201), "no-split-brain",
           f"replica rejected a local write with HTTP {status} ({(body or {}).get('message')})"
           if status not in (200, 201) else "SPLIT BRAIN -- the replica accepted a local write")

    print("== CHECK 3: leader and replica compared document for document ==")
    pairs = {i: (fetch(LP, i), fetch(RP, i)) for i in PROBED}
    on_leader = sum(1 for left, _ in pairs.values() if left is not None)
    on_replica = sum(1 for _, right in pairs.values() if right is not None)
    notes["ids_probed"] = len(PROBED)
    notes["on_leader"] = on_leader
    notes["on_replica"] = on_replica
    notes["count_only_verdict"] = "AGREE" if on_leader == on_replica else "DISAGREE"
    print(f"  leader holds {on_leader} of {len(PROBED)} probed ids; replica holds {on_replica}")
    print(f"  a count-only health check would report: {notes['count_only_verdict']}")

    missing = [i for i, (l, r) in pairs.items() if l is not None and r is None]
    extra_ids = [i for i, (l, r) in pairs.items() if l is None and r is not None]
    mismatched = [i for i, (l, r) in pairs.items() if l is not None and r is not None and l != r]
    delivered = [i for i in range(POST_LO, POST_HI + 1)
                 if ((pairs[i][1] or {}).get("metadata") or {}).get("tag") == "post-restart"]
    expected_post = POST_HI - POST_LO + 1
    notes["post_restart_delivered"] = len(delivered)
    notes["deleted_id_on_replica"] = pairs[DELETED_ID][1] is not None
    notes["canary_on_replica"] = pairs[CANARY_ID][1] is not None

    record(not missing and not extra_ids and not mismatched, "no-divergence",
           "every probed id is byte-identical on both nodes" if not (missing or extra_ids or mismatched)
           else f"missing on replica={missing[:10]} extra on replica={extra_ids[:10]} "
                f"differing={mismatched[:10]}")
    record(len(delivered) == expected_post, "post-restart-delivery",
           f"{len(delivered)}/{expected_post} writes issued after the leader restart reached the replica")
    record(not notes["deleted_id_on_replica"], "delete-replicated",
           f"id {DELETED_ID} is absent from the replica"
           if not notes["deleted_id_on_replica"] else f"id {DELETED_ID} is still on the replica")
    record(notes["canary_on_replica"], "insert-replicated",
           f"id {CANARY_ID} reached the replica"
           if notes["canary_on_replica"] else f"id {CANARY_ID} never reached the replica")
finally:
    for proc in (replica_server, follower, leader):
        stop(proc, seconds=45)

print("RESULT: " + ("PASS" if not failures else "FAIL") + "  " + json.dumps(notes, sort_keys=True))
for line in failures:
    print("  failure: " + line)
sys.exit(0 if not failures else 1)
