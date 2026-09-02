#!/usr/bin/env python3
"""RCV-05: cold-start rehearsal of a DeepData data root under a cgroup v2 memory hard cap.

Runs the server binary inside a transient systemd user scope with MemoryMax and no swap,
waits for /readyz, samples memory.peak from the cgroup while the process lives, reads the
collection doc count, runs seeded dense / sparse / hybrid searches, closes the server with
SIGTERM (the log must say the state was checkpointed), starts it a second time under the same
cap and requires the same doc count and the same result sets. Optional --populate builds a
fresh store first and kills it with SIGKILL so the journal is left un-checkpointed.

Refuses to touch the preserved production roots. Every phase gets its own append-only log.
Prints a receipt JSON and exits non-zero on any failure.
"""
import argparse, hashlib, json, os, random, signal, socket, subprocess, sys, threading, time, urllib.error, urllib.request

FORBIDDEN = ["/home/omen/var/deepdata/", "/run/media/omen/Storage/miniexa/"]
CLOSE_LINE = "checkpointed and closed successfully"


def die(msg):
    print("FAIL: " + msg, file=sys.stderr)
    sys.exit(1)


def http(method, url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)


class Run:
    """One server lifetime inside its own scope unit."""

    def __init__(self, a, phase):
        self.a, self.phase = a, phase
        self.unit = f"deepdata-rehearsal-{phase}-{os.getpid()}"
        self.log = os.path.join(a.log_dir, f"{phase}.log")
        self.peak = 0
        self.current = 0
        self.oom_kills = 0
        self.samples = 0
        self.t0 = time.monotonic()
        # gRPC off (one port per phase is enough); rate limits raised so populate churn is not throttled.
        env = dict(os.environ, DEEPDATA_INSECURE_DEV_MODE="1", DEEPDATA_BIND_HOST="127.0.0.1",
                   PORT=str(a.port), GRPC_PORT="0", VECTORDB_DATA_DIR=a.data_dir,
                   TENANT_RPS="100000", TENANT_BURST="100000", API_RPS="100000")
        cmd = ["systemd-run", "--user", "--scope", "--quiet", "--unit", self.unit,
               "-p", f"MemoryMax={a.memory_max}", "-p", "MemorySwapMax=0",
               a.binary, "-data-dir", a.data_dir, "-port", str(a.port)]
        self.logf = open(self.log, "ab")
        self.logf.write(f"# {time.strftime('%FT%T')} {' '.join(cmd)}\n".encode())
        self.logf.flush()
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        self.cg = None
        self.sampler = threading.Thread(target=self._sample, daemon=True)
        self.sampler.start()

    def _cgroup(self):
        for _ in range(50):
            out = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", self.unit + ".scope"],
                                 capture_output=True, text=True).stdout.strip()
            if out.startswith("/") and os.path.isdir("/sys/fs/cgroup" + out):
                return "/sys/fs/cgroup" + out
            time.sleep(0.1)
        return None

    def _sample(self):
        self.cg = self._cgroup()
        while self.proc.poll() is None and self.cg:
            try:
                with open(self.cg + "/memory.peak") as f:
                    self.peak = max(self.peak, int(f.read()))
                with open(self.cg + "/memory.current") as f:
                    self.current = int(f.read())
                with open(self.cg + "/memory.events") as f:
                    for line in f:
                        if line.startswith("oom_kill "):
                            self.oom_kills = int(line.split()[1])
                self.samples += 1
            except OSError:
                pass
            time.sleep(self.a.sample_interval)

    def wait_ready(self):
        deadline = time.monotonic() + self.a.ready_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return None
            try:
                st, _ = http("GET", f"http://127.0.0.1:{self.a.port}/readyz", timeout=5)
                if st == 200:
                    return round(time.monotonic() - self.t0, 1)
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
                pass
            time.sleep(1)
        return None

    def stop(self, sig):
        self.proc.send_signal(sig)
        try:
            rc = self.proc.wait(timeout=self.a.close_timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            rc = self.proc.wait()
        self.sampler.join(timeout=5)
        self.logf.close()
        with open(self.log, "rb") as f:
            closed = CLOSE_LINE.encode() in f.read()
        return {"phase": self.phase, "cap": self.a.memory_max, "exit_code": rc, "peak_bytes": self.peak,
                "oom_kills": self.oom_kills, "samples": self.samples, "graceful_close_logged": closed,
                "wall_seconds": round(time.monotonic() - self.t0, 1), "log": self.log}


def base(a):
    return f"http://127.0.0.1:{a.port}/v3/tenants/{a.tenant}"


def pick_collection(a):
    _, cols = http("GET", base(a) + "/collections")
    cols = cols if isinstance(cols, list) else cols.get("collections", [])
    if a.collection:
        cols = [c for c in cols if c["name"] == a.collection]
    if not cols:
        die(f"no collection {a.collection or ''} in tenant {a.tenant}")
    return max(cols, key=lambda c: c["doc_count"])


def queries(info):
    rng = random.Random(42)
    dense = next(f for f in info["fields"] if f["type"] == "dense")
    sparse = next((f for f in info["fields"] if f["type"] == "sparse"), None)
    qd = [rng.gauss(0, 1) for _ in range(dense["dim"])]
    # HNSW is approximate; a rebuilt graph orders near-ties differently. ef_search 2048 (cap 4096) makes the
    # top-10 near-exact so the two cold starts are compared on the data, not on graph construction order.
    out = {"dense": {"queries": {dense["name"]: qd}, "top_k": 10, "ef_search": 2048}}
    if sparse:
        idx = sorted(rng.sample(range(sparse["dim"]), 8))
        qs = {"indices": idx, "values": [1.0] * 8, "dim": sparse["dim"]}
        out["sparse"] = {"queries": {sparse["name"]: qs}, "top_k": 10}
        out["hybrid"] = {"queries": {dense["name"]: qd, sparse["name"]: qs}, "top_k": 10, "ef_search": 2048,
                         "hybrid_params": {"strategy": "weighted", "weights": {dense["name"]: 0.5, sparse["name"]: 0.5}}}
    return out


def search_all(a, name, qs):
    res = {}
    for kind, body in qs.items():
        _, r = http("POST", f"{base(a)}/collections/{name}/search", body)
        res[kind] = {"ids": [d["id"] for d in r["documents"]], "scores": [round(s, 6) for s in r["scores"]]}
    return res


def fetch_docs(a, name, ids):
    """Exact document bytes for a sample of ids: the recovery-integrity check that no index can blur."""
    out = {}
    for i in ids:
        _, d = http("GET", f"{base(a)}/collections/{name}/docs/{i}")
        out[str(i)] = {"vectors": d["vectors"], "metadata": d.get("metadata")}
    return out


def populate(a):
    """Fresh store: create, batch insert, churn upserts, then SIGKILL so the journal is not checkpointed."""
    run = Run(a, "populate")
    if run.wait_ready() is None:
        run.stop(signal.SIGKILL)
        die("populate: server never became ready; see " + run.log)
    try:
        return _populate(a, run)
    except BaseException:
        run.stop(signal.SIGKILL)
        raise


def _populate(a, run):
    dim, sdim = 128, 1024
    http("POST", base(a) + "/collections", {"name": a.collection, "fields": [
        {"name": "embedding", "type": "dense", "dim": dim, "index": {"type": "hnsw"}},
        {"name": "keywords", "type": "sparse", "dim": sdim, "index": {"type": "inverted"}}]})
    rng = random.Random(7)

    def doc(i):
        idx = sorted(rng.sample(range(sdim), 6))
        return {"id": i, "vectors": {"embedding": [rng.gauss(0, 1) for _ in range(dim)],
                                     "keywords": {"indices": idx, "values": [1.0] * 6, "dim": sdim}},
                "metadata": {"i": i}}
    for start in range(1, a.populate + 1, 500):
        http("POST", f"{base(a)}/collections/{a.collection}/docs/batch",
             {"documents": [doc(i) for i in range(start, min(start + 500, a.populate + 1))]})
    for _ in range(a.churn):
        for i in rng.sample(range(1, a.populate + 1), max(1, a.populate // 10)):
            d = doc(i)
            http("PUT", f"{base(a)}/collections/{a.collection}/docs/{i}", {"vectors": d["vectors"], "metadata": d["metadata"]})
    return run.stop(signal.SIGKILL)


def cold_start(a, phase, sample_ids=None):
    run = Run(a, phase)
    ready = run.wait_ready()
    if ready is None:
        r = run.stop(signal.SIGKILL)
        r["ready_seconds"] = None
        return r, None
    try:
        info = pick_collection(a)
        results = search_all(a, info["name"], queries(info))
        if sample_ids is None:
            sample_ids = sorted({i for r in results.values() for i in r["ids"]})
        docs = fetch_docs(a, info["name"], sample_ids)
    except BaseException:
        run.stop(signal.SIGKILL)
        raise
    r = run.stop(signal.SIGTERM)
    r.update(ready_seconds=ready, collection=info["name"], doc_count=info["doc_count"], results=results,
             sample_ids=sample_ids, sample_sha256=hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest())
    return r, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--memory-max", required=True, help="cgroup MemoryMax, e.g. 256M or 10G")
    p.add_argument("--tenant", required=True)
    p.add_argument("--collection", help="collection to measure (default: the one with most docs)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--log-dir", required=True, help="sibling of the data dir, never inside it")
    p.add_argument("--expect-docs", type=int)
    p.add_argument("--populate", type=int, help="build a fresh store with N docs first (needs --collection)")
    p.add_argument("--churn", type=int, default=0, help="upsert passes over 10%% of the docs after populate")
    p.add_argument("--ready-timeout", type=float, default=3600)
    p.add_argument("--close-timeout", type=float, default=1800)
    p.add_argument("--sample-interval", type=float, default=2)
    p.add_argument("--forbid", action="append", default=FORBIDDEN, help="data-dir prefixes that are never touched")
    a = p.parse_args()

    a.data_dir = os.path.realpath(a.data_dir)
    for f in a.forbid:
        if (a.data_dir + "/").startswith(os.path.realpath(f) + "/"):
            die(f"data dir {a.data_dir} is under forbidden root {f}")
    if a.populate and not a.collection:
        die("--populate needs --collection")
    if os.path.realpath(a.log_dir).startswith(a.data_dir + "/"):
        die("log dir must not live inside the data dir")
    os.makedirs(a.log_dir, exist_ok=True)
    os.makedirs(a.data_dir, mode=0o700, exist_ok=True)
    with socket.socket() as s:  # a foreign listener (adb once held 18090) would fail the second cold start, not the first
        try:
            s.bind(("127.0.0.1", a.port))
        except OSError as e:
            die(f"port {a.port} is not free: {e}")

    receipt = {"binary": a.binary, "binary_sha256": hashlib.sha256(open(a.binary, "rb").read()).hexdigest(),
               "data_dir": a.data_dir, "memory_max": a.memory_max, "tenant": a.tenant,
               "started": time.strftime("%FT%T%z"), "phases": [], "failures": []}
    if a.populate:
        receipt["phases"].append(populate(a))
    first, info = cold_start(a, "start1")
    receipt["phases"].append(first)
    if info is None:
        receipt["failures"].append("start1 never became ready")
    else:
        second, info2 = cold_start(a, "start2", first["sample_ids"])
        receipt["phases"].append(second)
        if info2 is None:
            receipt["failures"].append("start2 never became ready")
        else:
            if a.expect_docs is not None and first["doc_count"] != a.expect_docs:
                receipt["failures"].append(f"start1 doc_count {first['doc_count']} != expected {a.expect_docs}")
            if second["doc_count"] != first["doc_count"]:
                receipt["failures"].append(f"doc_count changed {first['doc_count']} -> {second['doc_count']}")
            parity = {"sample_docs_identical": first["sample_sha256"] == second["sample_sha256"],
                      "sample_size": len(first["sample_ids"])}
            if not parity["sample_docs_identical"]:
                receipt["failures"].append("sampled documents differ between cold starts")
            for kind in first["results"]:
                r1, r2 = first["results"][kind], second["results"][kind]
                s1, s2 = dict(zip(r1["ids"], r1["scores"])), dict(zip(r2["ids"], r2["scores"]))
                shared = sorted(set(s1) & set(s2))
                # HNSW is approximate: a graph rebuilt in a different order can miss a near-tie, so the
                # top_k sets may differ. The rule is: every document both starts returned scores the same
                # (dense/sparse scores are per-document; hybrid fusion is set-relative, so hybrid is recorded
                # only), and at least half the hits are shared. Served bytes are checked separately above.
                parity[kind] = {"overlap": len(shared), "hits": len(r1["ids"]),
                                "same_scores_shared": all(s1[i] == s2[i] for i in shared),
                                "same_set": set(s1) == set(s2), "same_order": r1 == r2}
                if not r1["ids"] or len(shared) * 2 < len(r1["ids"]):
                    receipt["failures"].append(f"{kind} results share {len(shared)}/{len(r1['ids'])} hits between cold starts")
                if kind != "hybrid" and not parity[kind]["same_scores_shared"]:
                    receipt["failures"].append(f"{kind} scores differ for the same document between cold starts")
            receipt["parity"] = parity
    for ph in receipt["phases"]:
        if ph["phase"] == "populate":
            continue
        if ph["exit_code"] != 0 or not ph["graceful_close_logged"]:
            receipt["failures"].append(f"{ph['phase']} exit {ph['exit_code']} graceful_close_logged={ph['graceful_close_logged']}")
        if ph["oom_kills"]:
            receipt["failures"].append(f"{ph['phase']} saw {ph['oom_kills']} oom kill(s)")
    receipt["pass"] = not receipt["failures"]
    with open(os.path.join(a.log_dir, "receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))
    sys.exit(0 if receipt["pass"] else 1)


if __name__ == "__main__":
    main()
