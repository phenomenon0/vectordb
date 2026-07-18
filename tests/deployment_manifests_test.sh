#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chart="$repo_root/deploy/helm/deepdata"

if ! command -v helm >/dev/null 2>&1; then
  echo "helm is required to validate deployment manifests" >&2
  exit 127
fi

assert_contains() {
  local file=$1
  local expected=$2
  if ! grep -Fq -- "$expected" "$file"; then
    echo "missing expected deployment contract in $file: $expected" >&2
    exit 1
  fi
}

assert_rendered_env() {
  local file=$1
  local name=$2
  local expected=$3
  if ! awk -v name="$name" -v expected="$expected" '
    $0 ~ "- name: " name {
      if ((getline) > 0) {
        line=$0
        sub(/^[[:space:]]*value:[[:space:]]*/, "", line)
        gsub(/^"|"$/, "", line)
        if (line == expected) found=1
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$file"; then
    echo "rendered environment $name does not equal $expected" >&2
    exit 1
  fi
}

rendered="$(mktemp)"
ephemeral_rendered="$(mktemp)"
trap 'rm -f "$rendered" "$ephemeral_rendered"' EXIT

helm lint "$chart" --strict
helm template deepdata-contract "$chart" >"$rendered"
helm template deepdata-contract "$chart" \
  --set persistence.enabled=false >"$ephemeral_rendered"

for expected in \
  "runAsNonRoot: true" \
  "runAsUser: 10001" \
  "runAsGroup: 10001" \
  "fsGroup: 10001" \
  "fsGroupChangePolicy: OnRootMismatch" \
  "type: RuntimeDefault" \
  "allowPrivilegeEscalation: false" \
  "readOnlyRootFilesystem: true" \
  "- ALL" \
  "name: VECTORDB_BASE_DIR" \
  "name: VECTORDB_DATA_DIR" \
  "name: GRPC_PORT" \
  "containerPort: 50051" \
  "targetPort: grpc" \
  "path: /livez" \
  "path: /readyz" \
  "mountPath: /data" \
  "mountPath: /tmp"; do
  assert_contains "$rendered" "$expected"
done

assert_contains "$rendered" "claimName: deepdata-contract-deepdata-data"
assert_rendered_env "$rendered" VECTORDB_BASE_DIR /data
assert_rendered_env "$rendered" VECTORDB_DATA_DIR local
assert_rendered_env "$rendered" GRPC_PORT 50051
assert_contains "$ephemeral_rendered" "emptyDir: {}"
if grep -Fq -- "claimName: deepdata-contract-deepdata-data" "$ephemeral_rendered"; then
  echo "persistence-disabled rendering unexpectedly references a PVC" >&2
  exit 1
fi

assert_contains "$repo_root/Dockerfile" "USER 10001:10001"
assert_contains "$repo_root/Dockerfile" "ENV GRPC_PORT=50051"
assert_contains "$repo_root/Dockerfile" "EXPOSE 8080 50051"
assert_contains "$repo_root/Dockerfile" 'http://localhost:${PORT}/livez'
assert_contains "$repo_root/Dockerfile" "ENV VECTORDB_BASE_DIR=/data"
assert_contains "$repo_root/Dockerfile" "ENV VECTORDB_DATA_DIR=local"
assert_contains "$repo_root/Dockerfile" "COPY api ./api"
assert_contains "$repo_root/Dockerfile" "COPY cmd ./cmd"
assert_contains "$repo_root/Dockerfile" "COPY internal ./internal"
assert_contains "$repo_root/.dockerignore" ".deepdata-run"
assert_contains "$repo_root/.dockerignore" "node_modules/"
assert_contains "$repo_root/.dockerignore" "tools/"
if grep -Eq '^COPY[[:space:]]+\.[[:space:]]+\.$' "$repo_root/Dockerfile"; then
  echo "Dockerfile must keep an allowlisted production source copy" >&2
  exit 1
fi
assert_contains "$repo_root/docker-compose.yml" 'image: "${DEEPDATA_IMAGE:-deepdata:local}"'
assert_contains "$repo_root/docker-compose.yml" '${DEEPDATA_GRPC_PORT:-50051}:50051'
assert_contains "$repo_root/docker-compose.yml" "- GRPC_PORT=50051"
assert_contains "$repo_root/docker-compose.yml" "http://localhost:8080/livez"
assert_contains "$repo_root/docker-compose.yml" "- VECTORDB_BASE_DIR=/data"
assert_contains "$repo_root/docker-compose.yml" "- VECTORDB_DATA_DIR=local"
assert_contains "$repo_root/docs/kubernetes.md" "runAsUser: 10001"
assert_contains "$repo_root/docs/kubernetes.md" "fsGroup: 10001"
assert_contains "$repo_root/docs/kubernetes.md" "readOnlyRootFilesystem: true"

for file in \
  "$repo_root/Dockerfile" \
  "$repo_root/docker-compose.yml" \
  "$chart/values.yaml" \
  "$chart/templates/deployment.yaml"; do
  if grep -Eq 'USE_HASH_EMBEDDER|VECTOR_CAPACITY|HYDRATION_COUNT' "$file"; then
    echo "unsupported server-managed embedding setting remains in $file" >&2
    exit 1
  fi
done

echo "deployment manifest contracts passed"
