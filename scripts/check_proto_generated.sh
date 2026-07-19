#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT_DIR"

scripts/generate_proto.sh
if ! git diff --exit-code -- api/gen/deepdata/v1 api/gen/deepdata/v3; then
  echo "protobuf Go output is stale; run scripts/generate_proto.sh" >&2
  exit 1
fi
