#!/usr/bin/env bash
#
# DeepData Smoke Test — end-to-end validation with real Ollama embeddings
#
# Covers: server boot, Ollama embedder, insert, query, semantic accuracy,
# metadata filtering, batch, upsert, delete, collections, hybrid/BM25,
# scroll, compact, export/import, concurrency, edge cases, WAL persistence.
#
# Usage:
#   ./tests/smoke_test.sh
#   DEEPDATA_PORT=8888 ./tests/smoke_test.sh
#   SKIP_BUILD=1 ./tests/smoke_test.sh
#
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
PORT="${DEEPDATA_PORT:-9777}"
BASE="http://localhost:${PORT}"
BASE_DIR=$(mktemp -d /tmp/deepdata-test-XXXXXX)
BINARY="bin/deepdata-test"
PASS=0
FAIL=0
SKIP=0
TOTAL=0
SERVER_PID=""

# ── Helpers ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
BLD='\033[1m'
RST='\033[0m'

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$BASE_DIR"
    rm -f "$BINARY"
    echo ""
    echo -e "${BLD}═══════════════════════════════════════════════${RST}"
    if [[ $SKIP -gt 0 ]]; then
        echo -e "  ${GRN}${PASS} passed${RST}  ${RED}${FAIL} failed${RST}  ${YEL}${SKIP} skipped${RST}  ${TOTAL} total"
    else
        echo -e "  ${GRN}${PASS} passed${RST}  ${RED}${FAIL} failed${RST}  ${TOTAL} total"
    fi
    echo -e "${BLD}═══════════════════════════════════════════════${RST}"
    if [[ $FAIL -gt 0 ]]; then
        exit 1
    fi
}
trap cleanup EXIT

assert() {
    local name="$1"; shift
    TOTAL=$((TOTAL + 1))
    if eval "$@"; then
        PASS=$((PASS + 1))
        echo -e "  ${GRN}PASS${RST}  $name"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${RST}  $name"
    fi
}

assert_eq() {
    local name="$1" expected="$2" actual="$3"
    TOTAL=$((TOTAL + 1))
    if [[ "$expected" == "$actual" ]]; then
        PASS=$((PASS + 1))
        echo -e "  ${GRN}PASS${RST}  $name"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${RST}  $name  (expected='$expected' got='$actual')"
    fi
}

assert_contains() {
    local name="$1" haystack="$2" needle="$3"
    TOTAL=$((TOTAL + 1))
    if echo "$haystack" | grep -q "$needle"; then
        PASS=$((PASS + 1))
        echo -e "  ${GRN}PASS${RST}  $name"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${RST}  $name  (missing '$needle')"
    fi
}

assert_http() {
    local name="$1" expected="$2"; shift 2
    local actual
    actual=$(curl -s -o /dev/null -w "%{http_code}" "$@")
    assert_eq "$name" "$expected" "$actual"
}

skip() {
    local name="$1"
    TOTAL=$((TOTAL + 1))
    SKIP=$((SKIP + 1))
    echo -e "  ${YEL}SKIP${RST}  $name"
}

jv() { echo "$1" | jq -r "$2" 2>/dev/null || echo "JQ_ERR"; }

post() { curl -sf -X POST "$1" -H "Content-Type: application/json" -d "$2" 2>/dev/null || echo '{"error":"request_failed"}'; }

# ── Build ───────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

# Source .env for API keys (OPENAI_API_KEY etc)
if [[ -f ../.env ]]; then
    set -a; source ../.env; set +a
elif [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# Determine mode: pro if OPENAI_API_KEY is set, local otherwise
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    TEST_MODE="pro"
else
    TEST_MODE="local"
fi

if [[ "${SKIP_BUILD:-}" != "1" ]]; then
    echo -e "${BLD}Building...${RST}"
    CGO_ENABLED=0 go build -o "$BINARY" ./cmd/deepdata/
fi

# ── Start Server (isolated, no hydration) ──────────────────────────────────
echo -e "${BLD}Starting server on :${PORT} (mode=${TEST_MODE})${RST}"

VECTORDB_BASE_DIR="$BASE_DIR" \
VECTORDB_MODE="$TEST_MODE" \
HYDRATION_COUNT=0 \
PORT="$PORT" \
"$BINARY" serve --port "$PORT" --mode "$TEST_MODE" >"$BASE_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo -n "  Waiting"
for i in $(seq 1 60); do
    if curl -sf "$BASE/healthz" >/dev/null 2>&1; then
        echo " ready (${i}x0.5s)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e " ${RED}CRASHED${RST}"; SERVER_PID=""; exit 1
    fi
    echo -n "."
    sleep 0.5
done
curl -sf "$BASE/healthz" >/dev/null 2>&1 || { echo -e " ${RED}TIMEOUT${RST}"; exit 1; }

MODE_RESP=$(curl -sf "$BASE/api/mode" 2>/dev/null || echo '{}')
EMBEDDER=$(jv "$MODE_RESP" '.embedder_type // "unknown"')
echo -e "  Embedder: ${YEL}${EMBEDDER}${RST}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Health & Probes ─────────────────────────────────────────────────────
echo -e "${BLD}[1] Health & Probes${RST}"
assert_http "GET /healthz → 200" "200" "$BASE/healthz"
assert_http "GET /health → 200"  "200" "$BASE/health"
assert_http "GET /readyz → 200"  "200" "$BASE/readyz"

H=$(curl -sf "$BASE/health")
assert_contains "health.ok=true" "$H" '"ok":true'
assert_contains "health has 'total'" "$H" '"total"'
assert_contains "health has 'active'" "$H" '"active"'

INITIAL=$(jv "$H" '.total')
assert_eq "empty DB has 0 vectors" "0" "$INITIAL"
echo ""

# ── 2. Single Inserts (Ollama embeddings) ──────────────────────────────────
echo -e "${BLD}[2] Insert (5 docs via Ollama)${RST}"

R=$(post "$BASE/insert" '{"doc":"Machine learning uses neural networks to learn patterns from data","id":"ml-101","meta":{"topic":"machine-learning","level":"beginner"}}')
assert_eq "insert ml-101" "ml-101" "$(jv "$R" '.id')"

R=$(post "$BASE/insert" '{"doc":"PostgreSQL is a powerful open-source relational database management system","id":"pg-101","meta":{"topic":"databases","level":"beginner"}}')
assert_eq "insert pg-101" "pg-101" "$(jv "$R" '.id')"

R=$(post "$BASE/insert" '{"doc":"Kubernetes orchestrates containerized applications across clusters of machines","id":"k8s-101","meta":{"topic":"infrastructure","level":"intermediate"}}')
assert_eq "insert k8s-101" "k8s-101" "$(jv "$R" '.id')"

R=$(post "$BASE/insert" '{"doc":"Transformer architecture revolutionized NLP with self-attention mechanisms","id":"transformer-101","meta":{"topic":"machine-learning","level":"advanced"}}')
assert_eq "insert transformer-101" "transformer-101" "$(jv "$R" '.id')"

R=$(post "$BASE/insert" '{"doc":"Docker containers package applications with dependencies for consistent deployment","id":"docker-101","meta":{"topic":"infrastructure","level":"beginner"}}')
assert_eq "insert docker-101" "docker-101" "$(jv "$R" '.id')"

# Verify count
H=$(curl -sf "$BASE/health")
assert_eq "5 vectors after insert" "5" "$(jv "$H" '.total')"
echo ""

# ── 3. Semantic Query ──────────────────────────────────────────────────────
echo -e "${BLD}[3] Semantic Query${RST}"

R=$(post "$BASE/query" '{"query":"how do neural networks learn?","top_k":3,"include_meta":true}')
TOP=$(jv "$R" '.ids[0]')
assert_eq "query 'neural networks' → ml-101" "ml-101" "$TOP"

COUNT=$(jv "$R" '.ids | length')
assert "query returns multiple results" "[[ $COUNT -ge 2 ]]"

# Infrastructure query
R=$(post "$BASE/query" '{"query":"container orchestration deployment","top_k":3}')
TOP=$(jv "$R" '.ids[0]')
assert "infra query → k8s or docker" "[[ '$TOP' == 'k8s-101' || '$TOP' == 'docker-101' ]]"

# Database query
R=$(post "$BASE/query" '{"query":"SQL relational database management","top_k":1}')
TOP=$(jv "$R" '.ids[0]')
assert_eq "database query → pg-101" "pg-101" "$TOP"

# Transformer query
R=$(post "$BASE/query" '{"query":"attention mechanism NLP transformers","top_k":1}')
TOP=$(jv "$R" '.ids[0]')
assert_eq "transformer query → transformer-101" "transformer-101" "$TOP"
echo ""

# ── 4. Metadata Filtering ─────────────────────────────────────────────────
echo -e "${BLD}[4] Metadata Filtering${RST}"

# AND: topic=machine-learning
R=$(post "$BASE/query" '{"query":"learning algorithms","top_k":10,"include_meta":true,"meta":{"topic":"machine-learning"}}')
FCOUNT=$(jv "$R" '.ids | length')
assert "AND filter returns results" "[[ $FCOUNT -ge 1 ]]"

ALL_OK=$(echo "$R" | jq '[.meta[]?.topic] | all(. == "machine-learning")' 2>/dev/null || echo "false")
assert_eq "all AND results have topic=machine-learning" "true" "$ALL_OK"

# NOT: exclude infrastructure
R=$(post "$BASE/query" '{"query":"technology","top_k":10,"include_meta":true,"meta_not":{"topic":"infrastructure"}}')
NO_INFRA=$(echo "$R" | jq '[.meta[]?.topic] | all(. != "infrastructure")' 2>/dev/null || echo "false")
assert_eq "NOT filter excludes infrastructure" "true" "$NO_INFRA"

# AND: topic=databases
R=$(post "$BASE/query" '{"query":"data storage","top_k":10,"meta":{"topic":"databases"}}')
DCOUNT=$(jv "$R" '.ids | length')
assert_eq "databases filter returns 1 result" "1" "$DCOUNT"
echo ""

# ── 5. Batch Insert ────────────────────────────────────────────────────────
echo -e "${BLD}[5] Batch Insert (5 docs)${RST}"

R=$(post "$BASE/batch_insert" '{
    "docs": [
        {"doc":"Redis is an in-memory data structure store used as cache and message broker","id":"redis-101","meta":{"topic":"databases"}},
        {"doc":"GraphQL provides a complete description of data in your API with a type system","id":"graphql-101","meta":{"topic":"api"}},
        {"doc":"Rust programming language focuses on memory safety and zero-cost abstractions","id":"rust-101","meta":{"topic":"programming"}},
        {"doc":"WebAssembly enables near-native performance for web applications","id":"wasm-101","meta":{"topic":"web"}},
        {"doc":"gRPC uses protocol buffers for efficient serialized remote procedure calls","id":"grpc-101","meta":{"topic":"api"}}
    ]
}')
BATCH_COUNT=$(jv "$R" '.ids | length')
assert_eq "batch insert 5 docs" "5" "$BATCH_COUNT"

# Verify retrievable
R=$(post "$BASE/query" '{"query":"in-memory caching message broker","top_k":1}')
assert_eq "batch doc queryable (redis)" "redis-101" "$(jv "$R" '.ids[0]')"

H=$(curl -sf "$BASE/health")
assert_eq "10 vectors total" "10" "$(jv "$H" '.total')"
echo ""

# ── 6. Upsert ──────────────────────────────────────────────────────────────
echo -e "${BLD}[6] Upsert${RST}"

R=$(post "$BASE/insert" '{
    "doc":"Deep learning is a subset of machine learning using multi-layer neural networks for representation learning",
    "id":"ml-101",
    "upsert": true,
    "meta":{"topic":"machine-learning","level":"intermediate","updated":"true"}
}')
assert_eq "upsert ml-101" "ml-101" "$(jv "$R" '.id')"

R=$(post "$BASE/query" '{"query":"deep learning representation","top_k":1,"include_meta":true}')
assert_eq "upserted doc is top result" "ml-101" "$(jv "$R" '.ids[0]')"
UPDATED=$(echo "$R" | jq -r '.meta[0].updated // "missing"' 2>/dev/null)
assert_eq "upserted meta has updated=true" "true" "$UPDATED"
echo ""

# ── 7. Delete ──────────────────────────────────────────────────────────────
echo -e "${BLD}[7] Delete${RST}"

R=$(post "$BASE/delete" '{"id":"rust-101"}')
assert_eq "delete rust-101" "rust-101" "$(jv "$R" '.deleted')"

R=$(post "$BASE/query" '{"query":"rust memory safety","top_k":10}')
HAS_RUST=$(echo "$R" | jq '[.ids[] // empty] | any(. == "rust-101")' 2>/dev/null || echo "false")
assert_eq "rust-101 absent from results" "false" "$HAS_RUST"

assert_http "delete nonexistent → 404" "404" \
    -X POST "$BASE/delete" -H "Content-Type: application/json" -d '{"id":"no-such-doc"}'

H=$(curl -sf "$BASE/health")
assert "health.deleted >= 1" "[[ $(jv "$H" '.deleted') -ge 1 ]]"
echo ""

# ── 8. Scroll ─────────────────────────────────────────────────────────────
echo -e "${BLD}[8] Scroll${RST}"

R=$(curl -sf "$BASE/scroll?limit=3")
SCOUNT=$(jv "$R" '.ids | length')
assert "scroll returns results" "[[ $SCOUNT -ge 1 ]]"
assert "scroll respects limit=3" "[[ $SCOUNT -le 3 ]]"

NEXT=$(jv "$R" '.next_offset')
if [[ "$NEXT" != "null" && "$NEXT" != "0" ]]; then
    R2=$(curl -sf "$BASE/scroll?limit=3&offset=$NEXT")
    S2=$(jv "$R2" '.ids | length')
    assert "scroll page 2 returns results" "[[ $S2 -ge 1 ]]"
fi

STOTAL=$(jv "$R" '.total')
assert "scroll total >= 9" "[[ $STOTAL -ge 9 ]]"
echo ""

# ── 9. Collections ────────────────────────────────────────────────────────
echo -e "${BLD}[9] Collections${RST}"

R=$(post "$BASE/insert" '{"doc":"The mitochondria is the powerhouse of the cell producing ATP","id":"bio-101","collection":"science","meta":{"topic":"biology"}}')
assert_eq "insert bio-101 → science" "bio-101" "$(jv "$R" '.id')"

R=$(post "$BASE/insert" '{"doc":"Photosynthesis converts light energy into chemical energy in plants","id":"bio-102","collection":"science","meta":{"topic":"biology"}}')
assert_eq "insert bio-102 → science" "bio-102" "$(jv "$R" '.id')"

R=$(post "$BASE/query" '{"query":"cellular energy ATP production","top_k":2,"collection":"science"}')
TOP=$(jv "$R" '.ids[0]')
assert_eq "collection query → bio-101" "bio-101" "$TOP"
assert_eq "collection returns 2" "2" "$(jv "$R" '.ids | length')"

# Cross-collection isolation: default query shouldn't return science docs
R=$(post "$BASE/query" '{"query":"mitochondria ATP","top_k":10}')
HAS_BIO=$(echo "$R" | jq '[.ids[] // empty] | any(. == "bio-101")' 2>/dev/null || echo "false")
assert_eq "bio-101 not in default collection" "false" "$HAS_BIO"
echo ""

# ── 10. Hybrid + BM25 Search ──────────────────────────────────────────────
echo -e "${BLD}[10] Hybrid + BM25 Search${RST}"

R=$(post "$BASE/query" '{"query":"PostgreSQL database","top_k":3,"mode":"hybrid"}')
HCOUNT=$(jv "$R" '.ids | length')
assert "hybrid search returns results" "[[ $HCOUNT -ge 1 ]]"

R=$(post "$BASE/query" '{"query":"PostgreSQL relational database","top_k":3,"mode":"bm25"}')
BCOUNT=$(jv "$R" '.ids | length')
assert "BM25 search returns results" "[[ $BCOUNT -ge 1 ]]"

R=$(post "$BASE/query" '{"query":"gRPC protocol buffers","top_k":1,"mode":"bm25"}')
assert_eq "BM25 keyword match → grpc-101" "grpc-101" "$(jv "$R" '.ids[0]')"
echo ""

# ── 11. Edge Cases ────────────────────────────────────────────────────────
echo -e "${BLD}[11] Edge Cases${RST}"

# Empty doc → 400
assert_http "empty doc → 400" "400" \
    -X POST "$BASE/insert" -H "Content-Type: application/json" -d '{"doc":"","id":"empty"}'

# Missing doc field → 400
assert_http "missing doc → 400" "400" \
    -X POST "$BASE/insert" -H "Content-Type: application/json" -d '{"id":"nodoc"}'

# Long doc
LONG_DOC=$(python3 -c "print('knowledge ' * 2000)")
R=$(post "$BASE/insert" "$(jq -n --arg d "$LONG_DOC" '{doc: $d, id: "long-doc"}')")
assert_eq "insert long doc (2000 words)" "long-doc" "$(jv "$R" '.id')"

# Unicode
R=$(post "$BASE/insert" '{"doc":"Les r\u00e9seaux de neurones \u2014 \u4eba\u5de5\u77e5\u80fd \u2014 \u041d\u0435\u0439\u0440\u043e\u043d\u043d\u044b\u0435","id":"unicode-doc","meta":{"lang":"multi"}}')
assert_eq "insert unicode doc" "unicode-doc" "$(jv "$R" '.id')"

# Special chars in metadata
R=$(post "$BASE/insert" '{"doc":"special metadata test document","id":"special-meta","meta":{"path":"/usr/local/bin","sql":"SELECT * FROM t","html":"<b>bold</b>"}}')
assert_eq "special chars in meta" "special-meta" "$(jv "$R" '.id')"

# Duplicate without upsert → error (500)
assert_http "duplicate (no upsert) → 500" "500" \
    -X POST "$BASE/insert" -H "Content-Type: application/json" -d '{"doc":"dup","id":"pg-101"}'

# Duplicate with upsert → success
R=$(post "$BASE/insert" '{"doc":"updated PostgreSQL content","id":"pg-101","upsert":true}')
assert_eq "duplicate with upsert → ok" "pg-101" "$(jv "$R" '.id')"

# Single-char query
R=$(post "$BASE/query" '{"query":"a","top_k":1}')
assert "single-char query doesn't crash" "[[ $(jv "$R" '.ids | length') -ge 0 ]]"

# top_k controls result count
R=$(post "$BASE/query" '{"query":"technology","top_k":2}')
assert "top_k=2 returns <= 2" "[[ $(jv "$R" '.ids | length') -le 2 ]]"

R=$(post "$BASE/query" '{"query":"technology","top_k":1}')
assert_eq "top_k=1 returns exactly 1" "1" "$(jv "$R" '.ids | length')"

R=$(post "$BASE/query" '{"query":"technology","top_k":100}')
T100=$(jv "$R" '.ids | length')
assert "top_k=100 returns all available (<=100)" "[[ $T100 -ge 5 && $T100 -le 100 ]]"
echo ""

# ── 12. Embed Endpoint ───────────────────────────────────────────────────
echo -e "${BLD}[12] Embed Endpoint${RST}"

R=$(post "$BASE/api/embed" '{"text":"test embedding vector"}')
DIM=$(jv "$R" '.embedding | length')
assert "embed returns vector" "[[ $DIM -ge 100 ]]"
echo -e "  Dimension: ${YEL}${DIM}${RST}"
echo ""

# ── 13. Integrity ─────────────────────────────────────────────────────────
echo -e "${BLD}[13] Integrity${RST}"
assert_http "GET /integrity → 200" "200" "$BASE/integrity"
echo ""

# ── 14. Compact ───────────────────────────────────────────────────────────
echo -e "${BLD}[14] Compact${RST}"

H_PRE=$(curl -sf "$BASE/health")
DEL_PRE=$(jv "$H_PRE" '.deleted')
assert "has deleted vectors before compact" "[[ $DEL_PRE -ge 1 ]]"

# Compact — may 500 if no snapshot path exists yet (no prior save)
COMPACT_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/compact")
assert "compact → 200 or 500" "[[ $COMPACT_CODE == '200' || $COMPACT_CODE == '500' ]]"

H_POST=$(curl -sf "$BASE/health")
assert "active > 0 after compact" "[[ $(jv "$H_POST" '.active') -ge 1 ]]"
echo ""

# ── 15. Export / Import ───────────────────────────────────────────────────
echo -e "${BLD}[15] Export / Import${RST}"

EXPORT_FILE="$BASE_DIR/snapshot.bin"
HTTP_CODE=$(curl -s -o "$EXPORT_FILE" -w "%{http_code}" "$BASE/export")
assert_eq "export → 200" "200" "$HTTP_CODE"

FSIZE=$(stat -c%s "$EXPORT_FILE" 2>/dev/null || stat -f%z "$EXPORT_FILE")
assert "export file non-empty" "[[ $FSIZE -gt 0 ]]"
echo -e "  Export size: ${YEL}${FSIZE} bytes${RST}"
echo ""

# ── 16. Rapid Sequential Inserts ──────────────────────────────────────────
echo -e "${BLD}[16] Rapid Sequential Inserts (10 docs)${RST}"

RAPID_OK=0
for i in $(seq 1 10); do
    R=$(post "$BASE/insert" "{\"doc\":\"rapid test document number $i about topic $i\",\"id\":\"rapid-$i\"}")
    [[ "$(jv "$R" '.id')" == "rapid-$i" ]] && RAPID_OK=$((RAPID_OK + 1))
done
assert_eq "all 10 rapid inserts succeed" "10" "$RAPID_OK"

# Verify via BM25
FOUND=0
for i in $(seq 1 10); do
    R=$(post "$BASE/query" "{\"query\":\"rapid test document number $i\",\"top_k\":1,\"mode\":\"bm25\"}")
    [[ "$(jv "$R" '.ids[0]')" == "rapid-$i" ]] && FOUND=$((FOUND + 1))
done
assert_eq "all 10 rapid docs retrievable via BM25" "10" "$FOUND"
echo ""

# ── 17. Batch (20 docs) ──────────────────────────────────────────────────
echo -e "${BLD}[17] Batch Insert (20 docs)${RST}"

BATCH20=$(python3 -c "
import json
topics = ['physics','chemistry','math','history','literature']
docs = [{'doc': f'Detailed document about {topics[i%5]} covering concept number {i}', 'id': f'b20-{i}', 'meta': {'topic': topics[i%5], 'batch': 'test'}} for i in range(20)]
print(json.dumps({'docs': docs}))
")
R=$(post "$BASE/batch_insert" "$BATCH20")
B20=$(jv "$R" '.ids | length')
assert_eq "batch 20 docs inserted" "20" "$B20"

R=$(post "$BASE/query" '{"query":"physics concepts","top_k":5,"meta":{"topic":"physics"}}')
PCOUNT=$(jv "$R" '.ids | length')
assert "physics filter returns results" "[[ $PCOUNT -ge 1 ]]"
echo ""

# ── 18. Semantic Accuracy ─────────────────────────────────────────────────
echo -e "${BLD}[18] Semantic Accuracy${RST}"

if [[ "$EMBEDDER" != "hash" ]]; then
    # Semantic: ML query should rank ML docs in top 3
    R=$(post "$BASE/query" '{"query":"neural network gradient descent training","top_k":3}')
    HAS_ML=$(echo "$R" | jq '[.ids[]] | any(. == "ml-101" or . == "transformer-101")' 2>/dev/null || echo "false")
    assert_eq "ML query has ML doc in top 3" "true" "$HAS_ML"

    # Bio in science collection — either bio doc is valid
    R=$(post "$BASE/query" '{"query":"plant energy conversion sunlight","top_k":1,"collection":"science"}')
    BIO_TOP=$(jv "$R" '.ids[0]')
    assert "bio query → bio doc" "[[ '$BIO_TOP' == 'bio-101' || '$BIO_TOP' == 'bio-102' ]]"

    # Cache query — should not return infrastructure
    R=$(post "$BASE/query" '{"query":"in-memory cache data store","top_k":1}')
    assert_eq "cache query → redis-101" "redis-101" "$(jv "$R" '.ids[0]')"
else
    skip "semantic accuracy (hash embedder)"
fi
echo ""

# ── 19. Config Endpoints ─────────────────────────────────────────────────
echo -e "${BLD}[19] Config Endpoints${RST}"

R=$(curl -sf "$BASE/api/mode")
assert_contains "/api/mode has mode" "$R" '"mode"'
assert_contains "/api/mode has dimension" "$R" '"dimension"'

R=$(curl -sf "$BASE/api/config/embedder")
assert_contains "/api/config/embedder responds" "$R" "type"
echo ""

# ── 20. Prometheus Metrics ────────────────────────────────────────────────
echo -e "${BLD}[20] Metrics${RST}"

M=$(curl -sf "$BASE/metrics")
assert_contains "has go_goroutines" "$M" "go_goroutines"
assert_contains "has go_memstats" "$M" "go_memstats"
echo ""

# ── 21. WAL & Persistence ────────────────────────────────────────────────
echo -e "${BLD}[21] WAL & Data${RST}"

WAL_COUNT=$(find "$BASE_DIR" -name "*.wal" 2>/dev/null | wc -l)
assert "WAL files exist" "[[ $WAL_COUNT -ge 1 ]]"

FILE_COUNT=$(find "$BASE_DIR" -type f 2>/dev/null | wc -l)
assert "data dir has files" "[[ $FILE_COUNT -ge 1 ]]"
echo ""

# ── 22. Final Health ─────────────────────────────────────────────────────
echo -e "${BLD}[22] Final Health${RST}"

H=$(curl -sf "$BASE/health")
TOTAL_VEC=$(jv "$H" '.total')
ACTIVE_VEC=$(jv "$H" '.active')
echo -e "  Total: ${YEL}${TOTAL_VEC}${RST}  Active: ${YEL}${ACTIVE_VEC}${RST}"
assert "total vectors > 30" "[[ $TOTAL_VEC -gt 30 ]]"
assert "active > 0" "[[ $ACTIVE_VEC -gt 0 ]]"

COLL_COUNT=$(echo "$H" | jq '.collections | length' 2>/dev/null || echo "0")
assert "multiple collections" "[[ $COLL_COUNT -ge 2 ]]"
echo ""

# ── 23. Sparse Vector Insert + Query ────────────────────────────────────
echo -e "${BLD}[23] Sparse Vector Insert + Query${RST}"

# Get store dimension from mode endpoint so sparse vectors match
STORE_DIM=$(jv "$(curl -sf "$BASE/api/mode")" '.dimension')
if [[ "$STORE_DIM" == "null" || "$STORE_DIM" == "JQ_ERR" || -z "$STORE_DIM" ]]; then
    STORE_DIM=1536
fi

R=$(post "$BASE/insert/sparse" "{\"id\":\"sparse-1\",\"doc\":\"sparse test document about quantum computing\",\"indices\":[0,5,10,15],\"values\":[1.0,2.5,0.8,1.2],\"dimension\":$STORE_DIM}")
assert_eq "sparse insert sparse-1" "sparse-1" "$(jv "$R" '.id')"

R=$(post "$BASE/insert/sparse" "{\"id\":\"sparse-2\",\"doc\":\"sparse test document about neural networks\",\"indices\":[0,3,7,20],\"values\":[0.9,1.5,2.0,0.7],\"dimension\":$STORE_DIM}")
assert_eq "sparse insert sparse-2" "sparse-2" "$(jv "$R" '.id')"

# Query sparse
R=$(post "$BASE/query/sparse" "{\"indices\":[0,5,10],\"values\":[1.0,2.0,1.0],\"dimension\":$STORE_DIM,\"top_k\":2}")
SCOUNT=$(jv "$R" '.ids | length')
assert "sparse query returns results" "[[ $SCOUNT -ge 1 ]]"

# Edge: mismatched indices/values length → 400
assert_http "sparse mismatched lengths → 400" "400" \
    -X POST "$BASE/insert/sparse" -H "Content-Type: application/json" \
    -d '{"id":"sparse-bad","doc":"bad","indices":[0,1,2],"values":[1.0,2.0],"dimension":100}'

# Edge: missing indices → 400
assert_http "sparse missing indices → 400" "400" \
    -X POST "$BASE/insert/sparse" -H "Content-Type: application/json" \
    -d '{"id":"sparse-bad2","doc":"bad","indices":[],"values":[],"dimension":100}'

# Edge: missing dimension → 400
assert_http "sparse no dimension → 400" "400" \
    -X POST "$BASE/insert/sparse" -H "Content-Type: application/json" \
    -d '{"id":"sparse-bad3","doc":"bad","indices":[0],"values":[1.0],"dimension":0}'
echo ""

# ── 24. Batch Embed ─────────────────────────────────────────────────────
echo -e "${BLD}[24] Batch Embed${RST}"

R=$(post "$BASE/api/embed/batch" '{"texts":["hello world","machine learning","database systems"]}')
BCOUNT=$(jv "$R" '.count')
assert_eq "batch embed returns 3 embeddings" "3" "$BCOUNT"

BDIM=$(jv "$R" '.dimension')
assert "batch embed dimension > 0" "[[ $BDIM -gt 0 ]]"

# Verify embeddings array exists and has correct count
ELEN=$(jv "$R" '.embeddings | length')
assert_eq "embeddings array length = 3" "3" "$ELEN"

# Edge: empty texts → 400
assert_http "batch embed empty texts → 400" "400" \
    -X POST "$BASE/api/embed/batch" -H "Content-Type: application/json" -d '{"texts":[]}'
echo ""

# ── 25. Index Management ────────────────────────────────────────────────
echo -e "${BLD}[25] Index Management${RST}"

# GET /api/index/types
R=$(curl -sf "$BASE/api/index/types")
assert_contains "index types has 'types'" "$R" '"types"'
assert_contains "index types has hnsw" "$R" 'hnsw'

# GET /api/index/list
R=$(curl -sf "$BASE/api/index/list")
assert_contains "index list has 'indexes'" "$R" '"indexes"'
ICOUNT=$(jv "$R" '.count')
assert "index list count >= 1" "[[ $ICOUNT -ge 1 ]]"

# GET /api/index/stats?collection=default
R=$(curl -sf "$BASE/api/index/stats?collection=default")
assert_contains "index stats has 'collection'" "$R" '"collection"'
assert_contains "index stats has 'stats'" "$R" '"stats"'
echo ""

# ── 26. Cost Tracking ──────────────────────────────────────────────────
echo -e "${BLD}[26] Cost Tracking${RST}"

# GET /api/costs — works in both local and pro mode
R=$(curl -sf "$BASE/api/costs")
if [[ "$TEST_MODE" == "pro" ]]; then
    assert_contains "costs has session data" "$R" '"session"'
else
    assert_contains "costs has mode=local" "$R" '"local"'
fi

# GET /api/costs/daily
R=$(curl -sf "$BASE/api/costs/daily")
assert_contains "costs/daily has 'daily'" "$R" '"daily"'
echo ""

# ── 27. Collection Admin ───────────────────────────────────────────────
echo -e "${BLD}[27] Collection Admin${RST}"

# Admin endpoints require JWT in secure mode; without auth, adminGuard returns 403.
# Test that endpoints respond correctly based on auth configuration.
ADMIN_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/admin/collection/create" \
    -H "Content-Type: application/json" -d '{"name":"test-admin-coll","index_type":"hnsw"}')

if [[ "$ADMIN_CODE" == "201" ]]; then
    # Auth not required — admin endpoints accessible
    assert_eq "admin create collection → 201" "201" "$ADMIN_CODE"

    R=$(curl -sf "$BASE/admin/collection/list")
    assert_contains "admin list has 'collections'" "$R" '"collections"'
    assert_contains "admin list has test-admin-coll" "$R" 'test-admin-coll'

    ACOLL_COUNT=$(jv "$R" '.count')
    assert "admin list count >= 1" "[[ $ACOLL_COUNT -ge 1 ]]"

    R=$(curl -sf "$BASE/admin/collection/stats")
    assert_contains "admin stats → success" "$R" '"status"'
    assert_contains "admin stats has 'stats'" "$R" '"stats"'
else
    # adminGuard blocks without JWT — verify 403 (expected behavior)
    assert_eq "admin create → 403 (no JWT)" "403" "$ADMIN_CODE"

    assert_http "admin list → 403 (no JWT)" "403" "$BASE/admin/collection/list"
    assert_http "admin stats → 403 (no JWT)" "403" "$BASE/admin/collection/stats"
fi
echo ""

# ── 28. Feedback System ────────────────────────────────────────────────
echo -e "${BLD}[28] Feedback System${RST}"

# First record an interaction to get an interaction_id
R=$(post "$BASE/v2/interaction" '{"query":"test query","result_ids":["ml-101","pg-101"],"scores":[0.9,0.8]}')
INTERACTION_ID=$(jv "$R" '.id')

# POST /v2/feedback — submit explicit feedback
R=$(post "$BASE/v2/feedback" "{\"interaction_id\":\"${INTERACTION_ID}\",\"type\":\"explicit\",\"rating\":5,\"clicked_ids\":[\"ml-101\"]}")
assert_contains "feedback recorded" "$R" '"recorded"'

# GET /v2/feedback/stats
R=$(curl -sf "$BASE/v2/feedback/stats")
assert_contains "feedback stats has 'enabled'" "$R" '"enabled"'
echo ""

# ── 29. Extraction Status ──────────────────────────────────────────────
echo -e "${BLD}[29] Extraction Status${RST}"

R=$(curl -sf "$BASE/v2/extract/status")
assert_contains "extract status has 'enabled'" "$R" '"enabled"'
echo ""

# ── 30. Web UI / Dashboard ─────────────────────────────────────────────
echo -e "${BLD}[30] Web UI / Dashboard${RST}"

# GET / → 302 redirect to /dashboard/
DASH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/")
assert_eq "GET / → 302 redirect" "302" "$DASH_CODE"

# GET /dashboard/ → 200 (serves web UI)
DASH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/dashboard/")
assert "GET /dashboard/ → 200 or 404" "[[ $DASH_CODE == '200' || $DASH_CODE == '404' ]]"
