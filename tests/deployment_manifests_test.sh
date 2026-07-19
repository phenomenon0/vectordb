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

assert_template_fails() {
  local name=$1
  shift
  if helm template "deepdata-invalid-$name" "$chart" "$@" >/dev/null 2>&1; then
    echo "Helm unexpectedly accepted invalid RC configuration: $name" >&2
    exit 1
  fi
}

rendered="$(mktemp)"
existing_claim_rendered="$(mktemp)"
jwt_rendered="$(mktemp)"
trap 'rm -f "$rendered" "$existing_claim_rendered" "$jwt_rendered"' EXIT

digest="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
rc_args=(
  --set-string "image.digest=$digest"
  --set persistence.verifiedPOSIXSemantics=true
)

helm lint "$chart" --strict "${rc_args[@]}"
helm template deepdata-contract "$chart" "${rc_args[@]}" >"$rendered"
helm template deepdata-contract "$chart" "${rc_args[@]}" \
  --set persistence.existingClaim=deepdata-preprovisioned >"$existing_claim_rendered"
helm template deepdata-contract "$chart" "${rc_args[@]}" \
  --set auth.existingSecret=deepdata-jwt \
  --set auth.existingSecretType=jwtSecret \
  --set auth.existingSecretKey=jwt-secret >"$jwt_rendered"

assert_template_fails missing-digest \
  --set persistence.verifiedPOSIXSemantics=true
assert_template_fails mutable-image \
  --set-string image.digest=sha256:not-a-digest \
  --set persistence.verifiedPOSIXSemantics=true
assert_template_fails multiple-replicas "${rc_args[@]}" --set replicaCount=2
assert_template_fails provider-mode "${rc_args[@]}" --set config.mode=pro
assert_template_fails ephemeral-storage "${rc_args[@]}" --set persistence.enabled=false
assert_template_fails unverified-storage --set-string "image.digest=$digest"
assert_template_fails missing-secret-name "${rc_args[@]}" --set auth.existingSecret=
assert_template_fails missing-secret-key "${rc_args[@]}" --set auth.existingSecretKey=
assert_template_fails non-linux "${rc_args[@]}" --set nodeSelector.kubernetes\\.io/os=windows
assert_template_fails non-amd64 "${rc_args[@]}" --set nodeSelector.kubernetes\\.io/arch=arm64

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
  "name: TENANT_RPS" \
  "name: TENANT_BURST" \
  "name: AUTH_FAILURE_RPS" \
  "name: AUTH_FAILURE_BURST" \
  "name: MAX_RATE_LIMIT_KEYS" \
  "name: REQUIRE_AUTH" \
  "name: API_TOKEN" \
  "name: deepdata-auth" \
  "kubernetes.io/os: linux" \
  "kubernetes.io/arch: amd64" \
  "replicas: 1" \
  "automountServiceAccountToken: false" \
  "terminationGracePeriodSeconds: 90" \
  "startupProbe:" \
  "failureThreshold: 60" \
  "helm.sh/resource-policy: keep" \
  "image: \"deepdata@$digest\"" \
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
assert_rendered_env "$rendered" TENANT_RPS 100
assert_rendered_env "$rendered" TENANT_BURST 100
assert_rendered_env "$rendered" AUTH_FAILURE_RPS 1
assert_rendered_env "$rendered" AUTH_FAILURE_BURST 5
assert_rendered_env "$rendered" MAX_RATE_LIMIT_KEYS 100000
assert_rendered_env "$rendered" REQUIRE_AUTH 1
assert_contains "$existing_claim_rendered" "claimName: deepdata-preprovisioned"
if grep -Fq -- "kind: PersistentVolumeClaim" "$existing_claim_rendered"; then
  echo "existing-claim rendering unexpectedly creates a PVC" >&2
  exit 1
fi
assert_contains "$jwt_rendered" "name: JWT_SECRET"
if grep -Fq -- "name: API_TOKEN" "$jwt_rendered"; then
  echo "JWT rendering unexpectedly configures API_TOKEN" >&2
  exit 1
fi

assert_contains "$repo_root/Dockerfile" "USER 10001:10001"
assert_contains "$repo_root/Dockerfile" "golang:1.25.5-bookworm@sha256:"
assert_contains "$repo_root/Dockerfile" "debian:bookworm-slim@sha256:"
assert_contains "$repo_root/Dockerfile" "ENV GRPC_PORT=50051"
assert_contains "$repo_root/Dockerfile" "EXPOSE 8080 50051"
assert_contains "$repo_root/Dockerfile" 'http://localhost:${PORT}/livez'
assert_contains "$repo_root/Dockerfile" "ENV VECTORDB_BASE_DIR=/data"
assert_contains "$repo_root/Dockerfile" "ENV VECTORDB_DATA_DIR=local"
assert_contains "$repo_root/Dockerfile" "COPY api ./api"
assert_contains "$repo_root/Dockerfile" "COPY cmd ./cmd"
assert_contains "$repo_root/Dockerfile" "COPY internal ./internal"
assert_contains "$repo_root/Dockerfile" "CGO_ENABLED=0 GOOS=linux GOARCH=amd64"
assert_contains "$repo_root/.dockerignore" ".deepdata-run"
assert_contains "$repo_root/.dockerignore" "node_modules/"
assert_contains "$repo_root/.dockerignore" "tools/"
if grep -Eq '^COPY[[:space:]]+\.[[:space:]]+\.$' "$repo_root/Dockerfile"; then
  echo "Dockerfile must keep an allowlisted production source copy" >&2
  exit 1
fi
assert_contains "$repo_root/docker-compose.yml" 'image: "${DEEPDATA_IMAGE:-deepdata:local}"'
assert_contains "$repo_root/docker-compose.yml" "platform: linux/amd64"
assert_contains "$repo_root/docker-compose.yml" 'DEEPDATA_BIND_HOST:-127.0.0.1'
assert_contains "$repo_root/docker-compose.yml" '${DEEPDATA_PORT-8080}:8080'
assert_contains "$repo_root/docker-compose.yml" '${DEEPDATA_GRPC_PORT-50051}:50051'
assert_contains "$repo_root/docker-compose.yml" "- GRPC_PORT=50051"
assert_contains "$repo_root/docker-compose.yml" "http://localhost:8080/livez"
assert_contains "$repo_root/docker-compose.yml" "- VECTORDB_BASE_DIR=/data"
assert_contains "$repo_root/docker-compose.yml" "- VECTORDB_DATA_DIR=local"
assert_contains "$repo_root/docker-compose.yml" "- TENANT_RPS=100"
assert_contains "$repo_root/docker-compose.yml" "- TENANT_BURST=100"
assert_contains "$repo_root/docker-compose.yml" "- AUTH_FAILURE_RPS=1"
assert_contains "$repo_root/docker-compose.yml" "- AUTH_FAILURE_BURST=5"
assert_contains "$repo_root/docker-compose.yml" "- MAX_RATE_LIMIT_KEYS=100000"
assert_contains "$repo_root/docker-compose.yml" "- REQUIRE_AUTH=1"
assert_contains "$repo_root/docker-compose.yml" "DEEPDATA_API_TOKEN:?"
assert_contains "$repo_root/docker-compose.yml" 'user: "10001:10001"'
assert_contains "$repo_root/docker-compose.yml" "read_only: true"
assert_contains "$repo_root/docker-compose.yml" "/tmp:rw,nosuid,nodev,noexec"
assert_contains "$repo_root/docker-compose.yml" "no-new-privileges:true"
assert_contains "$repo_root/docker-compose.yml" "stop_grace_period: 90s"
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

if grep -Eq '(^|[^A-Z_])API_RPS([^A-Z_]|$)' \
  "$repo_root/docker-compose.yml" \
  "$chart/values.yaml" \
  "$chart/templates/deployment.yaml"; then
  echo "legacy API_RPS remains in a canonical deployment manifest" >&2
  exit 1
fi

echo "deployment manifest contracts passed"
