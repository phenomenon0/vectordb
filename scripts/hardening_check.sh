#!/usr/bin/env bash

# Allowlisted, receipt-producing checks for the DeepData production-hardening run.
# This script never edits source files and never evaluates arbitrary command text.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$REPO_ROOT/.deepdata-run/checks"
GO_BUILD_CACHE="$REPO_ROOT/.deepdata-run/go-build-cache"
GO_MODULE_CACHE="${DEEPDATA_GO_MODULE_CACHE:-/tmp/deepdata-go-mod}"
# go.mod requires go >= 1.25.12; the host /usr/lib/golang bin is a custom
# build whose default GOTOOLCHAIN=local is 1.25.5, so pin the toolchain
# explicitly. Override per-invocation via DEEPDATA_GO_TOOLCHAIN.
GO_TOOLCHAIN="${DEEPDATA_GO_TOOLCHAIN:-go1.25.12}"
export GOTOOLCHAIN="$GO_TOOLCHAIN"

list_checks() {
    printf '%s\n' \
        gates-check \
        benchmark-unit \
        go-storage \
        go-short \
        go-race \
        go-vet-cgo0 \
        go-apierror \
        go-embed \
        go-mcp \
        python-unit \
        python-mypy \
        python-build \
        ui-build
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
    ui-build)
        CHECK_CWD="$REPO_ROOT/cmd/deepdata/web-ui"
        CHECK_DESCRIPTION="npm run build"
        ;;
    *)
        echo "unknown check: $CHECK_NAME" >&2
        list_checks >&2
        exit 2
        ;;
esac

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
        ui-build)
            timeout 600s npm run build
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

receipt = {
    "schema_version": 1,
    "check": name,
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
