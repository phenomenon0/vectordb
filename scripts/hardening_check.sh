#!/usr/bin/env bash

# Allowlisted, receipt-producing checks for the DeepData production-hardening run.
# This script never edits source files and never evaluates arbitrary command text.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$REPO_ROOT/.deepdata-run/checks"
GO_BUILD_CACHE="$REPO_ROOT/.deepdata-run/go-build-cache"
GO_MODULE_CACHE="${DEEPDATA_GO_MODULE_CACHE:-/tmp/deepdata-go-mod}"
# go.mod requires go >= 1.25.13; the host /usr/lib/golang bin is a custom
# build whose default GOTOOLCHAIN=local is 1.25.5, so pin the toolchain
# explicitly. Override per-invocation via DEEPDATA_GO_TOOLCHAIN.
GO_TOOLCHAIN="${DEEPDATA_GO_TOOLCHAIN:-go1.25.13}"
export GOTOOLCHAIN="$GO_TOOLCHAIN"

list_checks() {
    printf '%s\n' \
        gates-check \
        benchmark-unit \
        go-storage \
        go-persistence \
        go-recovery-snapshot \
        go-recovery-journal \
        go-segmented-envelope \
        go-short \
        go-race \
        go-vet-cgo0 \
        go-apierror \
        go-embed \
        go-mcp \
        go-contract \
        go-usage \
        go-indextypes \
        go-ephemeral \
        go-runtime \
        python-unit \
        python-mypy \
        python-build \
        go-retire \
        go-canonical \
        soak \
        restart-reclaim
}

usage() {
    echo "usage: $0 <check-name> [--force]"
    echo "       $0 --list"
}

if [[ "${1:-}" == "--list" ]]; then
    list_checks
    exit 0
fi

CHECK_NAME="${1:-}"
FORCE="${2:-}"
if [[ -z "$CHECK_NAME" ]] || [[ -n "$FORCE" && "$FORCE" != "--force" ]]; then
    usage >&2
    exit 2
fi

case "$CHECK_NAME" in
    gates-check)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="python3 scripts/gates.py check"
        ;;
    benchmark-unit)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="python -m unittest -q benchmarks/test_mega_bench.py"
        ;;
    go-storage)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=1 go test -short -count=1 -timeout 300s ./internal/storage ./internal/collection ./cmd/deepdata"
        ;;
    go-persistence)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=1 go test -count=1 -p 1 -timeout 900s ./internal/collection ./cmd/deepdata"
        ;;
    go-recovery-snapshot)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=1 go test -count=1 -p 1 -timeout 300s -run 'UnifiedCollectionSnapshot|SnapshotMemory' ./internal/collection"
        ;;
    go-recovery-journal)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=1 go test -count=1 -p 1 -timeout 300s -run 'GeneratedJournal|Corrupt|PartialTail' ./internal/collection"
        ;;
    go-segmented-envelope)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=1 go test -count=1 -p 1 -timeout 300s -run SegmentedColdStart -bench SegmentedColdStart -benchmem -benchtime=3x ./internal/index"
        ;;
    go-short)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=1 go test -short -count=1 -timeout 600s ./..."
        ;;
    go-race)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=1 go test -race -short -count=1 -timeout 900s ./..."
        ;;
    go-vet-cgo0)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE CGO_ENABLED=0 go vet ./..."
        ;;
    go-apierror)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -count=1 -timeout 300s -run Error ./internal/apierror ./internal/collection ./cmd/deepdata ./cmd/deepdata-mcp"
        ;;
    go-embed)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="DEEPDATA_EMBEDDER=hash GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -count=1 -timeout 300s -run 'Embed|Text' ./internal/collection ./cmd/deepdata ./cmd/deepdata-mcp"
        ;;
    go-mcp)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="grep -q deepdata-mcp .github/workflows/ci.yml && GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -count=1 -timeout 300s ./cmd/deepdata-mcp ./api/contract"
        ;;
    go-contract)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -count=1 -timeout 300s -run 'Contract|Operations|Routes|Status|Discovery|ScoreDirection' ./cmd/deepdata && go test -count=1 -timeout 300s ./api/contract ./cmd/deepdata-mcp && go run ./cmd/deepdata routes >/dev/null && (cd sdk/python && python -m pytest tests/test_contract.py -q)"
        ;;
    go-usage)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -count=1 -p 1 -timeout 300s -run 'Usage' ./internal/collection && GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -race -count=1 -p 1 -timeout 300s -run 'Usage' ./internal/collection"
        ;;
    go-indextypes)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -count=1 -p 1 -timeout 300s -run 'IndexTypes' ./internal/collection ./cmd/deepdata && (cd sdk/python && python -m pytest tests/test_contract.py -q)"
        ;;
    go-ephemeral)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go test -count=1 -p 1 -timeout 300s -run 'Ephemeral' ./internal/collection ./cmd/deepdata && (cd sdk/python && python -m pytest tests/test_contract.py -q)"
        ;;
    go-runtime)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go vet ./cmd/deepdata && DEEPDATA_EMBEDDER=hash go test -count=1 -p 1 -timeout 600s ./cmd/deepdata && assert_no_new_vector_store_callers"
        ;;
    python-unit)
        CHECK_CWD="$REPO_ROOT/sdk/python"
        CHECK_DESCRIPTION="python -m pytest tests -q"
        ;;
    python-mypy)
        CHECK_CWD="$REPO_ROOT/sdk/python"
        CHECK_DESCRIPTION="python -m mypy deepdata/"
        ;;
    python-build)
        CHECK_CWD="$REPO_ROOT/sdk/python"
        CHECK_DESCRIPTION="python -m build --outdir <repo>/.deepdata-run/python-dist"
        ;;
    go-retire)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="assert_retired_trees_absent && GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go build ./... && CGO_ENABLED=0 go vet ./... && go mod tidy -diff"
        ;;
    go-canonical)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="assert_canonical_dropped && GOCACHE=$GO_BUILD_CACHE GOMODCACHE=$GO_MODULE_CACHE go build ./... && CGO_ENABLED=0 go vet ./..."
        ;;
    soak)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="SOAK_MINUTES=${SOAK_MINUTES:-25} SOAK_KILLS=${SOAK_KILLS:-5} python3 scripts/rehearsals/soak.py"
        ;;
    restart-reclaim)
        CHECK_CWD="$REPO_ROOT"
        CHECK_DESCRIPTION="DRIFT_MINUTES=${DRIFT_MINUTES:-4} N_HNSW=${N_HNSW:-6000} python3 scripts/rehearsals/restart_reclaim_probe.py"
        ;;
    *)
        echo "unknown check: $CHECK_NAME" >&2
        list_checks >&2
        exit 2
        ;;
esac

# SYS-01: the live server builds its state from serverRuntime, never from the
# legacy engine. NewVectorStore survives only as its own definition and the one
# loadOrInitStore snapshot path the legacy WAL and persistence tests exercise;
# func main() must not name VectorStore at all.
assert_no_new_vector_store_callers() {
    local callers
    callers="$(grep -rn 'NewVectorStore(' cmd internal --include='*.go' \
        | grep -v '_test\.go:' \
        | grep -v '^cmd/deepdata/main\.go:[0-9]*:func NewVectorStore(' \
        | grep -v '^cmd/deepdata/main\.go:[0-9]*:[[:space:]]*vs := NewVectorStore(capacity, dim)$')"
    if [[ -n "$callers" ]]; then
        echo "unexpected NewVectorStore callers outside tests:" >&2
        echo "$callers" >&2
        return 1
    fi
    local in_main
    in_main="$(awk '/^func main\(\) \{$/{inside=1} inside{print} inside && /^\}$/{inside=0}' cmd/deepdata/main.go \
        | grep -n 'VectorStore')"
    if [[ -n "$in_main" ]]; then
        echo "func main() still names VectorStore:" >&2
        echo "$in_main" >&2
        return 1
    fi
    echo "NewVectorStore has no live-server callers and func main() is engine-free"
}

# SYS-03: the retired trees are gone for good. A tree that reappears — or a
# dependency that creeps back into go.mod — silently re-widens the release
# candidate's surface, so the check names every retired path explicitly and
# reads the committed index rather than the working tree.
assert_retired_trees_absent() {
    local retired=(
        client
        cmd/cli
        cmd/deepdata/web-ui
        desktop
        internal/cluster
        internal/cowrieutil
        internal/encoding
        internal/feedback
        internal/obsidian
        internal/wal
        tests/ui
        vdb-test-suite
    )
    local path present=""
    for path in "${retired[@]}"; do
        if [[ -n "$(git -C "$REPO_ROOT" ls-files -- "$path")" ]]; then
            present+="$path"$'\n'
        fi
    done
    if [[ -n "$present" ]]; then
        echo "retired trees are still tracked:" >&2
        printf '%s' "$present" >&2
        return 1
    fi
    local deps
    deps="$(grep -n 'Neumenon/cowrie\|Neumenon/shard\|mattn/go-sqlite3' "$REPO_ROOT/go.mod" "$REPO_ROOT/go.sum")"
    if [[ -n "$deps" ]]; then
        echo "retired dependencies are still required:" >&2
        echo "$deps" >&2
        return 1
    fi
    echo "retired trees are absent and go.mod carries none of their dependencies"
}

assert_canonical_dropped() {
    local hits
    hits="$(grep -rnE '\bCanonical[A-Z]' --include='*.go' --include='*.py' --include='*.proto' \
        "$REPO_ROOT/cmd" "$REPO_ROOT/internal" "$REPO_ROOT/api" "$REPO_ROOT/sdk")"
    if [[ -n "$hits" ]]; then
        echo "Canonical-prefixed identifiers remain:" >&2
        echo "$hits" >&2
        return 1
    fi
    local dir missing="" docs=()
    while IFS= read -r dir; do
        if [[ -f "$dir/doc.go" ]]; then
            docs+=("$dir/doc.go")
        else
            missing+="${dir#"$REPO_ROOT"/}"$'\n'
        fi
    done < <(cd "$REPO_ROOT" && GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" go list -f '{{.Dir}}' ./cmd/... ./internal/...)
    if [[ -n "$missing" ]]; then
        echo "live packages without doc.go:" >&2
        printf '%s' "$missing" >&2
        return 1
    fi
    local unformatted
    unformatted="$(gofmt -l "${docs[@]}")"
    if [[ -n "$unformatted" ]]; then
        echo "doc.go files not gofmt-clean:" >&2
        echo "$unformatted" >&2
        return 1
    fi
    echo "no Canonical-prefixed identifier under cmd, internal, api or sdk; ${#docs[@]} live packages each carry a gofmt-clean doc.go"
}

run_check() {
    case "$CHECK_NAME" in
        gates-check)
            timeout 30s python3 scripts/gates.py check
            ;;
        benchmark-unit)
            timeout 60s python -m unittest -q benchmarks/test_mega_bench.py
            ;;
        go-storage)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" CGO_ENABLED=1 \
                go test -short -count=1 -timeout 300s \
                ./internal/storage ./internal/collection ./cmd/deepdata
            ;;
        go-persistence)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 960s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" CGO_ENABLED=1 \
                go test -count=1 -p 1 -timeout 900s \
                ./internal/collection ./cmd/deepdata
            ;;
        go-recovery-snapshot)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" CGO_ENABLED=1 \
                go test -count=1 -p 1 -timeout 300s \
                -run 'UnifiedCollectionSnapshot|SnapshotMemory' ./internal/collection
            ;;
        go-recovery-journal)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" CGO_ENABLED=1 \
                go test -count=1 -p 1 -timeout 300s \
                -run 'GeneratedJournal|Corrupt|PartialTail' ./internal/collection
            ;;
        go-segmented-envelope)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" CGO_ENABLED=1 \
                go test -count=1 -p 1 -timeout 300s \
                -run SegmentedColdStart -bench SegmentedColdStart -benchmem -benchtime=3x ./internal/index
            ;;
        go-short)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 660s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" CGO_ENABLED=1 \
                go test -short -count=1 -timeout 600s ./...
            ;;
        go-race)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 960s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" CGO_ENABLED=1 \
                go test -race -short -count=1 -timeout 900s ./...
            ;;
        go-vet-cgo0)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                CGO_ENABLED=0 go vet ./...
            ;;
        go-apierror)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -timeout 300s -run Error \
                ./internal/apierror ./internal/collection ./cmd/deepdata ./cmd/deepdata-mcp
            ;;
        go-embed)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env DEEPDATA_EMBEDDER=hash GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -timeout 300s -run 'Embed|Text' \
                ./internal/collection ./cmd/deepdata ./cmd/deepdata-mcp
            ;;
        go-mcp)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            grep -q deepdata-mcp .github/workflows/ci.yml && \
                timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -timeout 300s ./cmd/deepdata-mcp ./api/contract
            ;;
        go-contract)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -timeout 300s \
                -run 'Contract|Operations|Routes|Status|Discovery|ScoreDirection' \
                ./cmd/deepdata && \
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -timeout 300s ./api/contract ./cmd/deepdata-mcp && \
            timeout 180s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go run ./cmd/deepdata routes >/dev/null && \
            (cd "$REPO_ROOT/sdk/python" && timeout 180s python -m pytest tests/test_contract.py -q)
            ;;
        go-usage)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -p 1 -timeout 300s -run 'Usage' ./internal/collection && \
                timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -race -count=1 -p 1 -timeout 300s -run 'Usage' ./internal/collection
            ;;
        go-indextypes)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -p 1 -timeout 300s -run 'IndexTypes' \
                ./internal/collection ./cmd/deepdata && \
            (cd "$REPO_ROOT/sdk/python" && timeout 180s python -m pytest tests/test_contract.py -q)
            ;;
        go-ephemeral)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -p 1 -timeout 300s -run 'Ephemeral' \
                ./internal/collection ./cmd/deepdata && \
            (cd "$REPO_ROOT/sdk/python" && timeout 180s python -m pytest tests/test_contract.py -q)
            ;;
        go-runtime)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go vet ./cmd/deepdata && \
            timeout 660s env DEEPDATA_EMBEDDER=hash GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go test -count=1 -p 1 -timeout 600s ./cmd/deepdata && \
            assert_no_new_vector_store_callers
            ;;
        python-unit)
            timeout 360s python -m pytest tests -q
            ;;
        python-mypy)
            timeout 180s python -m mypy deepdata/
            ;;
        python-build)
            mkdir -p "$REPO_ROOT/.deepdata-run/python-dist"
            timeout 300s python -m build --outdir "$REPO_ROOT/.deepdata-run/python-dist"
            ;;
        go-retire)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            assert_retired_trees_absent && \
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go build ./... && \
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                CGO_ENABLED=0 go vet ./... && \
            timeout 180s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go mod tidy -diff
            ;;
        go-canonical)
            mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            assert_canonical_dropped && \
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                go build ./... && \
            timeout 360s env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                CGO_ENABLED=0 go vet ./...
            ;;
        soak)
            if [[ -n "${SOAK_BIN:-}" && -x "${SOAK_BIN}" ]]; then
                echo "using prebuilt binary $SOAK_BIN"
            else
                mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"
            fi
            timeout "${SOAK_TIMEOUT:-5400}s" env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                python3 scripts/rehearsals/soak.py
            ;;
        restart-reclaim)
            RECLAIM_WORK="$REPO_ROOT/.deepdata-run/rehearsals/memdrift-fix"
            if [[ -n "${BIN:-}" && -x "${BIN}" ]]; then
                echo "using prebuilt binary $BIN"
            else
                mkdir -p "$GO_BUILD_CACHE" "$GO_MODULE_CACHE" "$RECLAIM_WORK"
                env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
                    go build -trimpath -o "$RECLAIM_WORK/deepdata" ./cmd/deepdata || return 1
            fi
            timeout "${RECLAIM_TIMEOUT:-2400}s" python3 scripts/rehearsals/restart_reclaim_probe.py
            ;;
    esac
}

mkdir -p "$RUN_ROOT/$CHECK_NAME"
CHECK_DIR="$RUN_ROOT/$CHECK_NAME"
RECEIPT="$CHECK_DIR/receipt.json"
HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TREE_FINGERPRINT="$(python - "$REPO_ROOT" <<'PY'
import hashlib
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
diff = subprocess.run(
    ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
    check=True,
    stdout=subprocess.PIPE,
).stdout
digest.update(diff)
untracked = subprocess.run(
    ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
    check=True,
    stdout=subprocess.PIPE,
).stdout.split(b"\0")
for encoded_path in sorted(path for path in untracked if path):
    digest.update(encoded_path)
    digest.update(b"\0")
    path = root / encoded_path.decode("utf-8", errors="surrogateescape")
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
print(digest.hexdigest())
PY
)"

if [[ "$FORCE" != "--force" && -f "$RECEIPT" ]]; then
    if python - "$RECEIPT" "$HEAD_COMMIT" "$TREE_FINGERPRINT" <<'PY'
import json
import sys

path, commit, fingerprint = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

if (
    receipt.get("status") == "passed"
    and receipt.get("git_commit") == commit
    and receipt.get("tree_fingerprint") == fingerprint
):
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "cached pass: $CHECK_NAME ($HEAD_COMMIT)"
        exit 0
    fi
fi

STARTED_AT="$(date -Iseconds)"
LOG_STAMP="$(date '+%Y%m%dT%H%M%S%z')"
LOG_PATH="$CHECK_DIR/$LOG_STAMP.log"

echo "check: $CHECK_NAME"
echo "commit: $HEAD_COMMIT"
echo "cwd: $CHECK_CWD"
echo "command: $CHECK_DESCRIPTION"

set +e
(
    cd -- "$CHECK_CWD" || exit 125
    run_check
) 2>&1 | tee "$LOG_PATH"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

FINISHED_AT="$(date -Iseconds)"
STATUS="failed"
if [[ "$EXIT_CODE" -eq 0 ]]; then
    STATUS="passed"
elif [[ "$EXIT_CODE" -eq 124 ]]; then
    STATUS="timed_out"
fi

RECEIPT_TMP="$CHECK_DIR/.receipt.$$.tmp"
python - \
    "$RECEIPT_TMP" \
    "$RECEIPT" \
    "$CHECK_NAME" \
    "$STATUS" \
    "$EXIT_CODE" \
    "$STARTED_AT" \
    "$FINISHED_AT" \
    "$HEAD_COMMIT" \
    "$TREE_FINGERPRINT" \
    "$CHECK_CWD" \
    "$CHECK_DESCRIPTION" \
    "$LOG_PATH" <<'PY'
import json
import os
import sys

(
    temporary,
    destination,
    name,
    status,
    exit_code,
    started_at,
    finished_at,
    commit,
    fingerprint,
    working_directory,
    command,
    log_path,
) = sys.argv[1:]

def host_facts():
    """Record where the check ran.

    Memory and timing evidence (soak, restart-reclaim, cold-start envelopes) is
    only interpretable against the machine that produced it, and a receipt is
    otherwise indistinguishable between hosts.
    """
    facts = {}
    try:
        uname = os.uname()
        facts["hostname"] = uname.nodename
        facts["kernel"] = uname.release
        facts["arch"] = uname.machine
    except (AttributeError, OSError):
        pass
    try:
        facts["cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        facts["cpus"] = os.cpu_count()
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    facts["mem_total_kb"] = int(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        pass
    return facts


receipt = {
    "schema_version": 2,
    "check": name,
    "host": host_facts(),
    "status": status,
    "exit_code": int(exit_code),
    "started_at": started_at,
    "finished_at": finished_at,
    "git_commit": commit,
    "tree_fingerprint": fingerprint,
    "working_directory": working_directory,
    "command": command,
    "log_path": log_path,
}

with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
directory_fd = os.open(os.path.dirname(destination), os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY

echo "receipt: $RECEIPT"
exit "$EXIT_CODE"
