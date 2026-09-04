#!/usr/bin/env bash
# Offline backup/restore rehearsal against a real server process.
# Adapts docs/cookbook.md's systemd procedure to a bare-process drill:
#   seed -> graceful stop -> whole-root copy + manifest -> destroy live
#   state -> restore from copy -> restart -> strict assertions.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRILL="${DRILL_DIR:-/tmp/deepdata-backup-drill}"
PORT="${PORT:-8094}"
TOKEN="drill-token-0123456789abcdef0123456789abcdef"
STATE_ROOT="$DRILL/state"
BACKUP_PARENT="$DRILL/backups"
N_DOCS="${N_DOCS:-500}"
DIM=8

fail() { echo "DRILL FAILED: $*" >&2; exit 1; }

start_server() {
  API_TOKEN="$TOKEN" VECTORDB_BASE_DIR="$STATE_ROOT" VECTORDB_DATA_DIR=local \
    PORT="$PORT" GRPC_PORT=0 \
    "$ROOT/deepdata-server" >>"$DRILL/server.log" 2>&1 &
  SERVER_PID=$!
  local deadline=$((SECONDS + 30))
  while ((SECONDS < deadline)); do
    curl -fsS --max-time 1 "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited during startup (see $DRILL/server.log)"
    sleep 0.5
  done
  fail "server not ready in 30s"
}

stop_server() {
  kill -TERM "$SERVER_PID" 2>/dev/null || return 0
  for _ in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || return 0
    sleep 0.5
  done
  fail "graceful shutdown timed out"
}

auth() { printf 'Authorization: Bearer %s' "$TOKEN"; }

api() { curl -fsS -H "$(auth)" -H 'Content-Type: application/json' "$@"; }

seed() {
  api -X POST "http://127.0.0.1:$PORT/v3/tenants/acme/collections" -d @- <<JSON >/dev/null
{"name":"docs","fields":[{"name":"embedding","type":"dense","dim":$DIM,
  "index":{"type":"flat"}}]}
JSON
  # One deterministic landmark document plus filler.
  python3 - "$PORT" "$TOKEN" "$N_DOCS" <<'PY'
import json, sys, urllib.request
port, token, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        body = json.load(r)
    if r.status not in (200, 201):
        raise SystemExit(f"{path}: {r.status} {body}")
    return body

B = 100
for off in range(0, n, B):
    docs = []
    for i in range(off, min(off + B, n)):
        bump = 1.0 if i == 42 else 0.0
        vec = [float((i * 7 + j) % 13) / 13.0 + bump for j in range(8)]
        docs.append({"id": i + 1, "vectors": {"embedding": vec},
                     "metadata": {"kind": "landmark" if i == 42 else "filler"}})
    post("/v3/tenants/acme/collections/docs/docs/batch", {"documents": docs})

landmark = [float((42 * 7 + j) % 13) / 13.0 + 1.0 for j in range(8)]
res = post("/v3/tenants/acme/collections/docs/search",
           {"queries": {"embedding": landmark}, "top_k": 1})
if res["documents"][0]["id"] != 43:
    raise SystemExit(f"pre-backup search sanity failed: {res['documents'][0]['id']}")
print(f"seeded {n} docs; landmark query resolves to id 43")
PY
}

assert_restored() {
  python3 - "$PORT" "$TOKEN" "$N_DOCS" <<'PY'
import json, sys, urllib.request
port, token, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
H = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
def call(path, payload=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers=H, method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req) as r:
        return json.load(r)

info = call("/v3/tenants/acme")
assert "docs" in info.get("collections", {}), info
assert info.get("total_documents", 0) >= n, info

coll = call("/v3/tenants/acme/collections/docs")["collection"]
assert coll["name"] == "docs", coll
field = next(f for f in coll["fields"] if f["name"] == "embedding")
assert field["type"] == "dense" and field["dim"] == 8, field
assert field["index"]["type"] == "flat", field

landmark = [float((42 * 7 + j) % 13) / 13.0 + 1.0 for j in range(8)]
res = call("/v3/tenants/acme/collections/docs/search",
           {"queries": {"embedding": landmark}, "top_k": 5})
ids = [d["id"] for d in res["documents"]]
assert ids and ids[0] == 43, ids
meta = res["documents"][0].get("metadata") or {}
assert meta.get("kind") == "landmark", res["documents"][0]

doc = call("/v3/tenants/acme/collections/docs/docs/43")
payload = doc.get("document") or doc.get("doc") or doc
pid = payload.get("id") if isinstance(payload, dict) else None
if pid is None and isinstance(payload, dict):
    inner = payload.get("document") or {}
    pid = inner.get("id")
assert pid == 43, doc

count = len(call("/v3/tenants/acme/collections/docs/search",
                 {"queries": {"embedding": [0.0]*8}, "top_k": n})["documents"])
assert count >= n, f"expected >= {n} documents, saw {count}"
print(f"RESTORE VERIFIED: schema ok, landmark id=43 w/ metadata, {count} docs searchable")
PY
}

rm -rf "$DRILL"
mkdir -p "$STATE_ROOT" "$BACKUP_PARENT"

echo "== build =="
( cd "$ROOT" && GOTOOLCHAIN=auto go build -o deepdata-server ./cmd/deepdata/ )

echo "== seed =="
start_server
seed
stop_server

echo "== offline backup =="
BACKUP_DIR="$BACKUP_PARENT/deepdata-state-$(date +%Y%m%dT%H%M%S)"
[ ! -e "$BACKUP_DIR" ] || fail "backup dir already exists"
cp -a -- "$STATE_ROOT" "$BACKUP_DIR"
printf '%s\n' \
  'VECTORDB_DATA_DIR=local' \
  "VECTORDB_BASE_DIR=$STATE_ROOT" \
  "PRIMARY_DIRECTORY=$STATE_ROOT/local" > "$BACKUP_DIR.manifest"
chmod 0600 "$BACKUP_DIR.manifest"
sync
[ -d "$BACKUP_DIR/local" ] || fail "backup missing primary directory"
echo "backup at $BACKUP_DIR"

echo "== destroy live state =="
rm -rf "$STATE_ROOT"
[ ! -e "$STATE_ROOT" ] || fail "live state survived rm"

echo "== restore =="
cp -a -- "$BACKUP_DIR" "$STATE_ROOT"

echo "== verify restored node =="
start_server
assert_restored
stop_server

echo "DRILL PASSED"
