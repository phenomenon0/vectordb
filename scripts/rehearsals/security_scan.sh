#!/usr/bin/env bash
# SEC-01: gitleaks, govulncheck, gosec and trivy scans are clean on the frozen
# tree.
#
# "Clean" needs a written bar or the gate means nothing, so each scanner runs
# with its threshold spelled out here rather than in a reviewer's head:
#
#   gitleaks     no secret at all, over full history, minus .gitleaksignore
#   govulncheck  no vulnerability reachable from this module's code
#   gosec        no HIGH-severity, HIGH-confidence finding; annotated #nosec
#                sites must carry a written reason
#   trivy        no fixable HIGH/CRITICAL in the image. Unfixed base-image CVEs
#                are excluded because no action exists for them, and a bar
#                nobody can clear gets waived instead of enforced.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${SECURITY_SCAN_IMAGE:-deepdata-rc:frozen}"
RUNTIME="${CONTAINER_RUNTIME:-podman}"
OUT="${SECURITY_SCAN_DIR:-$ROOT/.deepdata-run/security}"
GO_BUILD_CACHE="$ROOT/.deepdata-run/go-build-cache"
GO_MODULE_CACHE="${DEEPDATA_GO_MODULE_CACHE:-/tmp/deepdata-go-mod}"
export GOTOOLCHAIN="${DEEPDATA_GO_TOOLCHAIN:-go1.25.13}"

fail() { echo "security-scan failed: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || { echo "$1 is required for SEC-01" >&2; exit 127; }; }

cd -- "$ROOT"
PATH="$PATH:$(go env GOPATH)/bin"
for tool in gitleaks govulncheck gosec trivy "$RUNTIME" jq; do require "$tool"; done

# The statement says "on the frozen tree": a scan of uncommitted edits proves
# nothing about the candidate.
[[ -z "$(git status --porcelain)" ]] || fail "worktree is dirty; scan only a frozen tree"
COMMIT="$(git rev-parse --short HEAD)"
mkdir -p "$OUT" "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"

# trivy exports the whole image through TMPDIR to scan it. The default /tmp
# here is a 16G tmpfs shared with everything else on the box, and an image
# export is large enough to hit "disk quota exceeded" mid-scan -- which trivy
# reports as an analysis failure, not as a full disk. Keep the scratch beside
# the other run artifacts, on real disk.
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
echo "frozen at $COMMIT"

echo "--- gitleaks (full history)"
gitleaks detect --no-banner --redact --report-format json \
  --report-path "$OUT/gitleaks-$COMMIT.json" \
  || fail "gitleaks reported secrets; see $OUT/gitleaks-$COMMIT.json"

echo "--- govulncheck (reachable only)"
env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
  govulncheck ./... >"$OUT/govulncheck-$COMMIT.txt" 2>&1 \
  || { tail -n 40 "$OUT/govulncheck-$COMMIT.txt" >&2; fail "govulncheck found reachable vulnerabilities"; }

echo "--- gosec (HIGH severity, HIGH confidence)"
# gosec exits non-zero when it reports findings, so the bar is enforced by the
# filter flags, not by a grep over the output.
# -quiet suppresses the -out report as well as the console summary, so it is
# deliberately absent: without the report there is nothing to audit and the
# load-failure guard below has nothing to read.
env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
  gosec -severity high -confidence high -fmt json \
  -out "$OUT/gosec-$COMMIT.json" ./... >"$OUT/gosec-$COMMIT.log" 2>&1 \
  || { tail -n 20 "$OUT/gosec-$COMMIT.log" >&2; fail "gosec reported HIGH/HIGH findings; see $OUT/gosec-$COMMIT.json"; }
# gosec reports "0 issues" for a package it could not load, so a load failure
# has to be a failure and not a clean bill of health -- a wrong GOTOOLCHAIN
# alone produces "found: 0" over a tree that never compiled. The one benign
# case is the amd64 assembly stubs: gosec's type-checker cannot see a body that
# lives in a .s file, and it scans those files anyway.
jq -r '(.["Golang errors"] // {}) | to_entries[] | .value[].error' "$OUT/gosec-$COMMIT.json" \
  | grep -v '^# command-line-arguments$' \
  | grep -v 'missing function body$' \
  | grep -v '^[[:space:]]*$' > "$OUT/gosec-load-errors.txt" || true
if [ -s "$OUT/gosec-load-errors.txt" ]; then
  cat "$OUT/gosec-load-errors.txt" >&2
  fail "gosec could not load one or more packages; its 0-finding result is meaningless"
fi
SCANNED="$(jq -r '.Stats.files' "$OUT/gosec-$COMMIT.json")"
[ "$SCANNED" -gt 0 ] || fail "gosec scanned no files"
NOSEC="$(jq -r '.Stats.nosec' "$OUT/gosec-$COMMIT.json")"
echo "gosec: 0 HIGH/HIGH findings over $SCANNED files, $NOSEC annotated #nosec sites"

echo "--- trivy image $IMAGE (fixable HIGH/CRITICAL)"
"$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1 \
  || fail "image $IMAGE is absent; run scripts/rehearsals/build_artifacts.sh first"
trivy image --quiet --ignore-unfixed --severity HIGH,CRITICAL --exit-code 1 \
  --format json --output "$OUT/trivy-$COMMIT.json" "$IMAGE" \
  || fail "trivy found fixable HIGH/CRITICAL vulnerabilities; see $OUT/trivy-$COMMIT.json"

echo "RESULT: PASS  gitleaks, govulncheck, gosec and trivy clean at $COMMIT"
