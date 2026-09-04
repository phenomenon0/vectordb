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

# The released binary must name the commit it came from. Go stamps this from
# git automatically, so the assertion is that the stamp is present, points at
# the frozen commit, and does not say the tree was modified.
VCS_REV="$(go version -m "$OUT/a/deepdata" | awk '$1=="build" && $2=="vcs.revision"{print $3}')"
VCS_MOD="$(go version -m "$OUT/a/deepdata" | awk '$1=="build" && $2=="vcs.modified"{print $3}')"
[ -n "$VCS_REV" ] || fail "released binary carries no vcs.revision stamp; it cannot be traced to a commit"
[ "$VCS_REV" = "$COMMIT" ] || fail "binary stamped $VCS_REV but the frozen commit is $COMMIT"
[ "$VCS_MOD" = "false" ] || fail "binary stamped vcs.modified=$VCS_MOD; it was built from a dirty tree"
echo "binary stamped at $VCS_REV (vcs.modified=false)"

cp -- "$OUT/a/deepdata" "$OUT/deepdata-linux-amd64"
printf '%s  deepdata-linux-amd64\n' "$SUM_A" > "$OUT/SHA256SUMS"

# The Dockerfile COPYs api, cmd and internal but not .git, so the in-image
# build has no repository to stamp from and Go silently drops the VCS fields.
# That is one differing input, not a differing source tree: comparing the image
# against the stamped build would fail forever and prove nothing. The honest
# comparison is against a build given the same inputs the image had, which is
# what -buildvcs=false reproduces. Both claims are then real -- the shipped
# binary is traceable to the commit, and the image is byte-reproducible.
echo "--- image-equivalent build (-buildvcs=false, matching the Dockerfile context)"
env GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MODULE_CACHE" \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -buildvcs=false -ldflags="-s -w" -o "$OUT/b/deepdata" ./cmd/deepdata/ \
  || fail "go build -buildvcs=false failed"
SUM_IMG_EXPECTED="$(sha256sum "$OUT/b/deepdata" | cut -d' ' -f1)"
echo "image-equivalent: $SUM_IMG_EXPECTED"
rm -rf -- "$OUT/a" "$OUT/b"

command -v "$RUNTIME" >/dev/null || fail "$RUNTIME not found"
echo "--- container image ($RUNTIME)"
"$RUNTIME" build -t "$IMAGE" -f Dockerfile . >"$OUT/image-build.log" 2>&1 \
  || { tail -n 40 "$OUT/image-build.log" >&2; fail "image build failed"; }
IMAGE_ID="$("$RUNTIME" image inspect --format '{{.Id}}' "$IMAGE")"
echo "image: $IMAGE $IMAGE_ID"

# The image must carry a binary this machine can rebuild bit for bit;
# otherwise the SBOM describes something no tracked source produces.
IN_IMAGE="$("$RUNTIME" run --rm --entrypoint sha256sum "$IMAGE" /usr/local/bin/deepdata | cut -d' ' -f1)"
[[ "$IN_IMAGE" == "$SUM_IMG_EXPECTED" ]] \
  || fail "image binary $IN_IMAGE != image-equivalent build $SUM_IMG_EXPECTED"
echo "image binary reproduces bit for bit"

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
  echo "image_binary_sha256=$SUM_IMG_EXPECTED"
  echo "vcs_revision=$VCS_REV"
  echo "image=$IMAGE"
  echo "image_id=$IMAGE_ID"
  echo "sbom_components=$COMPONENTS"
} > "$OUT/MANIFEST"

echo "RESULT: PASS  artifacts in $OUT"
