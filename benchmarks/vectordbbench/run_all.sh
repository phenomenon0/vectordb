#!/usr/bin/env bash
# Full competitive benchmark: DeepData vs 8 competitors
# Runs all modalities across multiple datasets and scales.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DeepData Comprehensive Vector Database Benchmark           ║"
echo "║  9 databases x multiple datasets x full modality suite      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Start competitor databases ──────────────────────────────────
echo "Step 1: Starting competitor databases..."
podman compose -f docker-compose.all.yml up -d
echo "  Waiting 30s for all services to initialize..."
sleep 30

# Verify health
echo "  Checking service health..."
for svc in qdrant weaviate milvus chromadb pgvector redis elasticsearch; do
    if podman compose -f docker-compose.all.yml ps --status=running | grep -q "$svc"; then
        echo "    ✓ $svc is running"
    else
        echo "    ✗ $svc is NOT running (will be skipped)"
    fi
done

# ── 2. Start DeepData ─────────────────────────────────────────────
echo ""
echo "Step 2: Starting DeepData..."
DEEPDATA_BIN="$(cd "$SCRIPT_DIR/../.." && pwd)/deepdata-server"
if [ ! -f "$DEEPDATA_BIN" ]; then
    echo "  Building DeepData..."
    (cd "$SCRIPT_DIR/../.." && go build -o deepdata-server ./cmd/deepdata/)
    DEEPDATA_BIN="$(cd "$SCRIPT_DIR/../.." && pwd)/deepdata-server"
fi

# Kill any existing DeepData instance
pkill -f "deepdata-server" 2>/dev/null || true
sleep 1

# Start DeepData with benchmark-optimized settings (high rate limits)
API_RPS=100000 TENANT_RPS=100000 TENANT_BURST=100000 \
    HNSW_M=16 HNSW_EFSEARCH=128 SCAN_THRESHOLD=0 EMBEDDER_TYPE=hash \
    VECTORDB_BASE_DIR=/tmp/deepdata-bench PORT=8080 \
    "$DEEPDATA_BIN" &
DEEPDATA_PID=$!
echo "  DeepData PID: $DEEPDATA_PID"
sleep 3

# Verify DeepData
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "  ✓ DeepData is healthy"
else
    echo "  ✗ DeepData failed to start"
fi

# ── 3. Install Python dependencies ────────────────────────────────
echo ""
echo "Step 3: Installing Python dependencies..."
pip install -q httpx numpy tqdm tabulate psycopg2-binary \
    qdrant-client weaviate-client pymilvus chromadb redis lancedb pyarrow 2>&1 | tail -3

# ── 4. Run benchmarks ─────────────────────────────────────────────
echo ""
echo "Step 4: Running benchmarks..."
echo ""

# Quick test (10K, 128d) on all DBs
python3 run_comprehensive.py \
    --dataset random-128d-10k \
    --vdb deepdata qdrant weaviate milvus chromadb pgvector redis elasticsearch lancedb

# Medium test (10K, 768d) - simulates real embeddings
python3 run_comprehensive.py \
    --dataset random-768d-10k \
    --vdb deepdata qdrant weaviate milvus chromadb pgvector redis elasticsearch lancedb

# High-dim test (10K, 1536d) - OpenAI embedding dimension
python3 run_comprehensive.py \
    --dataset random-1536d-10k \
    --vdb deepdata qdrant weaviate milvus chromadb pgvector redis elasticsearch lancedb

# Scale test (100K, 128d) - tests indexing at scale
python3 run_comprehensive.py \
    --dataset random-128d-100k \
    --vdb deepdata qdrant weaviate milvus chromadb pgvector redis elasticsearch lancedb

echo ""
echo "Step 5: Generating final report..."
python3 run_comprehensive.py --report-only

# ── 5. Cleanup ─────────────────────────────────────────────────────
echo ""
echo "Step 6: Cleanup..."
kill $DEEPDATA_PID 2>/dev/null || true
pkill -f "deepdata-server" 2>/dev/null || true
podman compose -f docker-compose.all.yml down -v

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Benchmark Complete!                                        ║"
echo "║                                                             ║"
echo "║  Results: $SCRIPT_DIR/results/comprehensive_results.json    ║"
echo "║  Report:  $SCRIPT_DIR/results/COMPREHENSIVE_REPORT.md       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
