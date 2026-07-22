#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOC_BIN="${PROTOC:-protoc}"
PROTOC_VERSION="${PROTOC_VERSION:-31.1}"
PROTOC_GEN_GO_VERSION="${PROTOC_GEN_GO_VERSION:-1.36.11}"
PROTOC_GEN_GO_GRPC_VERSION="${PROTOC_GEN_GO_GRPC_VERSION:-1.6.1}"

require_version() {
  local command_name="$1"
  local expected="$2"
  local actual
  if ! command -v -- "$command_name" >/dev/null 2>&1; then
    echo "required protobuf tool is missing: $command_name" >&2
    exit 127
  fi
  actual="$($command_name --version)"
  if [[ "$actual" != "$expected" ]]; then
    echo "$command_name version mismatch: got '$actual', require $expected" >&2
    exit 2
  fi
}

if ! command -v -- "$PROTOC_BIN" >/dev/null 2>&1; then
  echo "required protobuf compiler is missing: $PROTOC_BIN" >&2
  exit 127
fi

protoc_actual="$($PROTOC_BIN --version)"
if [[ "$protoc_actual" != "libprotoc $PROTOC_VERSION" ]]; then
  echo "protoc version mismatch: got '$protoc_actual', require libprotoc $PROTOC_VERSION" >&2
  exit 2
fi
require_version protoc-gen-go "protoc-gen-go v$PROTOC_GEN_GO_VERSION"
require_version protoc-gen-go-grpc "protoc-gen-go-grpc $PROTOC_GEN_GO_GRPC_VERSION"

if [[ -n "${PROTOC_INCLUDE:-}" ]]; then
  include_dir="$PROTOC_INCLUDE"
else
  protoc_path="$(command -v -- "$PROTOC_BIN")"
  include_dir="$(cd -- "$(dirname -- "$protoc_path")/../include" 2>/dev/null && pwd || true)"
fi
if [[ -z "$include_dir" || ! -f "$include_dir/google/protobuf/struct.proto" ]]; then
  echo "google/protobuf/struct.proto not found; set PROTOC_INCLUDE to protoc's include directory" >&2
  exit 2
fi

cd -- "$ROOT_DIR"
"$PROTOC_BIN" \
  -I api/proto \
  -I "$include_dir" \
  --go_out=api/gen \
  --go_opt=paths=source_relative \
  --go-grpc_out=api/gen \
  --go-grpc_opt=paths=source_relative \
  api/proto/deepdata/v1/deepdata.proto \
  api/proto/deepdata/v3/deepdata.proto
