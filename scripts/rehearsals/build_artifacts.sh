#!/usr/bin/env bash
# PKG-02: Linux binaries, container image and SBOM build reproducibly from the
# frozen tree.
#
# "Reproducibly" is the load-bearing word, so this builds the server binary
# twice into separate output trees and compares digests. A single build proves
# only that the compiler ran; two identical builds prove the output does not
# depend on build path, timestamp, or module cache state.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${CONTAINER_RUNTIME:-podman}"
IMAGE="${BUILD_ARTIFACTS_IMAGE:-deepdata-rc:frozen}"
OUT="${BUILD_ARTIFACTS_DIR:-$ROOT/.deepdata-run/artifacts}"
GO_BUILD_CACHE="$ROOT/.deepdata-run/go-build-cache"
GO_MODULE_CACHE="${DEEPDATA_GO_MODULE_CACHE:-/tmp/deepdata-go-mod}"
export GOTOOLCHAIN="${DEEPDATA_GO_TOOLCHAIN:-go1.25.13}"

fail() { echo "build-artifacts failed: $*" >&2; exit 1; }

cd -- "$ROOT"

# The statement says "from the frozen tree". A dirty tree is not frozen, and an
# artifact built from one cannot be re-derived by anyone else.
[[ -z "$(git status --porcelain)" ]] || fail "worktree is dirty; artifacts must build from a frozen tree"
COMMIT="$(git rev-parse HEAD)"
echo "frozen at $COMMIT"

rm -rf -- "$OUT"
mkdir -p "$OUT/a" "$OUT/b" "$GO_BUILD_CACHE" "$GO_MODULE_CACHE"

build_once() {
  # -trimpath is what makes the build path-independent; without it the two
  # output directories alone would produce different binaries.
  env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
      CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
      go build -trimpath -ldflags="-s -w" -o "$1/deepdata" ./cmd/deepdata/ \
    || fail "go build into $1 failed"
}

echo "--- build 1/2"; build_once "$OUT/a"
echo "--- build 2/2"; build_once "$OUT/b"

SUM_A="$(sha256sum "$OUT/a/deepdata" | cut -d' ' -f1)"
SUM_B="$(sha256sum "$OUT/b/deepdata" | cut -d' ' -f1)"
echo "build a: $SUM_A"
echo "build b: $SUM_B"
[[ "$SUM_A" == "$SUM_B" ]] || fail "binary is not reproducible: $SUM_A != $SUM_B"

# A binary that does not run is not an artifact. `routes` is the check: it is
# the one subcommand that needs no data directory, no environment and no
# server, so it exercises the real binary without standing anything up. There
# is deliberately no --version assertion -- the CLI has no such flag, and
# inventing one at RC freeze would widen the canonical surface to satisfy a
# test. The version claim is REL-04's, and it is made against the tracked
# files, not against the binary.
"$OUT/a/deepdata" routes >"$OUT/routes.txt" 2>&1 || {
  tail -n 20 "$OUT/routes.txt" >&2
  fail "built binary will not run: deepdata routes"
}
ROUTES="$(grep -c . "$OUT/routes.txt")"
[ "$ROUTES" -gt 0 ] || fail "deepdata routes printed nothing"
echo "binary runs: routes listed $ROUTES lines"

cp -- "$OUT/a/deepdata" "$OUT/deepdata-linux-amd64"
rm -rf -- "$OUT/a" "$OUT/b"
printf '%s  deepdata-linux-amd64\n' "$SUM_A" > "$OUT/SHA256SUMS"

command -v "$RUNTIME" >/dev/null || fail "$RUNTIME not found"
echo "--- container image ($RUNTIME)"
"$RUNTIME" build -t "$IMAGE" -f Dockerfile . >"$OUT/image-build.log" 2>&1 \
  || { tail -n 40 "$OUT/image-build.log" >&2; fail "image build failed"; }
IMAGE_ID="$("$RUNTIME" image inspect --format '{{.Id}}' "$IMAGE")"
echo "image: $IMAGE $IMAGE_ID"

# The image must carry the same binary the reproducibility check just pinned;
# otherwise the SBOM describes something the SHA256SUMS file does not cover.
IN_IMAGE="$("$RUNTIME" run --rm --entrypoint sha256sum "$IMAGE" /usr/local/bin/deepdata | cut -d' ' -f1)"
[[ "$IN_IMAGE" == "$SUM_A" ]] || fail "image binary $IN_IMAGE != reproducible build $SUM_A"
echo "image binary matches the reproducible build"

command -v trivy >/dev/null || fail "trivy not found (needed for the SBOM)"
echo "--- SBOM"
trivy image --quiet --format cyclonedx --output "$OUT/sbom.cdx.json" "$IMAGE" \
  || fail "trivy SBOM generation failed"
COMPONENTS="$(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1])).get("components") or []))' "$OUT/sbom.cdx.json")"
[[ "$COMPONENTS" -gt 0 ]] || fail "SBOM lists no components"
echo "sbom: $COMPONENTS components"

{
  echo "commit=$COMMIT"
  echo "binary_sha256=$SUM_A"
  echo "image=$IMAGE"
  echo "image_id=$IMAGE_ID"
  echo "sbom_components=$COMPONENTS"
} > "$OUT/MANIFEST"

echo "RESULT: PASS  artifacts in $OUT"
