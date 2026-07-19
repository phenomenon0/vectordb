#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
compose_file="$repo_root/docker-compose.yml"
runtime=${CONTAINER_RUNTIME:-docker}
project="deepdata-compose-contract-$PPID-$$"
api_token=compose-contract-test-token-strong-credential
tenant=compose-contract
collection=compose_docs
document_id=301
work_dir=""
grpc_probe=""
http_port=""
grpc_port=""
base_url=""
grpc_address=""
current_container_id=""
first_container_id=""
compose=()

fail() {
  echo "Compose contract failed: $*" >&2
  return 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "$1 is required to validate the Compose contract" >&2
    exit 127
  }
}

require_command "$runtime"
require_command curl
require_command go
require_command jq

case "$runtime" in
  docker)
    if ! docker compose version >/dev/null 2>&1; then
      echo "Docker Compose is required when CONTAINER_RUNTIME=docker" >&2
      exit 127
    fi
    compose=(docker compose --project-name "$project" --file "$compose_file")
    ;;
  podman)
    require_command podman-compose
    compose=(podman-compose --project-name "$project" --file "$compose_file")
    ;;
  *)
    fail "unsupported CONTAINER_RUNTIME=$runtime; expected docker or podman"
    ;;
esac

: "${DEEPDATA_IMAGE:?set DEEPDATA_IMAGE to the prebuilt container image}"
export DEEPDATA_IMAGE
export DEEPDATA_API_TOKEN="$api_token"
export DEEPDATA_BIND_HOST=127.0.0.1
export DEEPDATA_PORT=
export DEEPDATA_GRPC_PORT=
export COMPOSE_PROJECT_NAME="$project"

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/deepdata-compose-contract.XXXXXX")
grpc_probe="$work_dir/grpc-probe"

cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  if ((rc != 0)); then
    echo "logs from failed Compose project $project:" >&2
    "${compose[@]}" logs deepdata >&2 || true
    if [[ -n "$current_container_id" ]]; then
      "$runtime" logs "$current_container_id" >&2 || true
    fi
  fi
  # The project name is process-unique; this is the only volume deletion in
  # the test and cannot target another Compose project.
  "${compose[@]}" down --volumes --remove-orphans --timeout 30 >/dev/null 2>&1 || true
  rm -rf -- "$work_dir"
  exit "$rc"
}
trap cleanup EXIT

"$runtime" image inspect "$DEEPDATA_IMAGE" >/dev/null 2>&1 || \
  fail "prebuilt image is unavailable: $DEEPDATA_IMAGE"

(cd -- "$repo_root" && go build -trimpath -o "$grpc_probe" ./tests/container_grpc_probe)
"${compose[@]}" config >/dev/null

service_container_id() {
  local id
  # podman-compose does not accept a service argument for `ps`; this isolated
  # Compose file contains exactly one service, so the project-wide result is
  # unambiguous for both implementations.
  id=$("${compose[@]}" ps -q | sed -n '1p')
  if [[ -z "$id" ]]; then
    fail "Compose did not report a deepdata service container"
    return 1
  fi
  printf '%s\n' "$id"
}

resolve_ports() {
  # podman-compose's `port` command does not report an automatically assigned
  # host port. Querying the selected runtime by the Compose-owned container ID
  # is portable across Docker and Podman and returns the effective mapping.
  http_port=$("$runtime" port "$current_container_id" 8080/tcp | sed -n '1s/.*://p')
  grpc_port=$("$runtime" port "$current_container_id" 50051/tcp | sed -n '1s/.*://p')
  if [[ ! "$http_port" =~ ^[1-9][0-9]*$ ]]; then
    fail "invalid HTTP port mapping: $http_port"
    return 1
  fi
  if [[ ! "$grpc_port" =~ ^[1-9][0-9]*$ ]]; then
    fail "invalid gRPC port mapping: $grpc_port"
    return 1
  fi
  base_url="http://127.0.0.1:$http_port"
  grpc_address="127.0.0.1:$grpc_port"
}

wait_ready() {
  local ready=false
  for _ in $(seq 1 240); do
    if [[ "$($runtime inspect --format '{{.State.Running}}' "$current_container_id" 2>/dev/null)" != true ]]; then
      "$runtime" logs "$current_container_id" >&2 || true
      fail "Compose service exited before readiness"
      return 1
    fi
    if curl -fsS --max-time 1 "$base_url/readyz" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 0.25
  done
  if [[ "$ready" != true ]]; then
    fail "Compose service did not become ready"
    return 1
  fi
  curl -fsS --max-time 2 "$base_url/livez" >/dev/null
}

start_service() {
  "${compose[@]}" up -d --no-build deepdata >/dev/null
  current_container_id=$(service_container_id) || return 1
  resolve_ports || return 1
  wait_ready || return 1
}

stop_service() {
  local exit_code
  [[ -n "$current_container_id" ]] || fail "no Compose service container to stop"
  "${compose[@]}" stop --timeout 30 deepdata >/dev/null
  exit_code="$($runtime inspect --format '{{.State.ExitCode}}' "$current_container_id")"
  if [[ "$exit_code" != "0" ]]; then
    "$runtime" logs "$current_container_id" >&2 || true
    fail "Compose service exited with status $exit_code after SIGTERM"
  fi
}

api() {
  local method=$1 path=$2 body=${3-}
  if [[ -n "$body" ]]; then
    curl -fsS --max-time 10 -X "$method" "$base_url$path" \
      -H "Authorization: Bearer $api_token" \
      -H 'Content-Type: application/json' \
      --data-binary "$body"
  else
    curl -fsS --max-time 10 -X "$method" "$base_url$path" \
      -H "Authorization: Bearer $api_token"
  fi
}

assert_json() {
  local name=$1 json=$2 filter=$3
  if ! jq -e "$filter" <<<"$json" >/dev/null; then
    echo "$json" >&2
    fail "$name"
    return 1
  fi
  echo "PASS: $name"
}

create_body='{
  "name":"compose_docs",
  "fields":[
    {"name":"embedding","type":"dense","dim":3,"index":{"type":"flat"}}
  ]
}'

echo "starting isolated Compose project $project"
start_service
first_container_id=$current_container_id

response=$(api POST "/v3/tenants/$tenant/collections" "$create_body")
assert_json "Compose HTTP V3 create collection" "$response" '.status == "success"'

response=$(api POST "/v3/tenants/$tenant/collections/$collection/docs" '{
  "id":301,
  "vectors":{"embedding":[1,0,0]},
  "metadata":{"source":"compose-contract"}
}')
assert_json "Compose HTTP V3 insert" "$response" '.id == 301'

response=$(api POST "/v3/tenants/$tenant/collections/$collection/search" '{
  "queries":{"embedding":[1,0,0]},
  "top_k":10
}')
assert_json "Compose HTTP V3 search" "$response" \
  '([.documents[].id] | index(301)) != null'

"$grpc_probe" "$grpc_address" "$api_token" "$tenant" "$collection" "$document_id"

echo "stopping Compose service with SIGTERM"
stop_service
"$runtime" rm "$current_container_id" >/dev/null
current_container_id=""

echo "recreating Compose service on the same project volume"
start_service
[[ "$current_container_id" != "$first_container_id" ]] || \
  fail "Compose did not replace the stopped service container"

response=$(api GET "/v3/tenants/$tenant/collections/$collection")
assert_json "Compose collection survived replacement" "$response" \
  '.collection.Name == "compose_docs" and .collection.DocCount == 1'

response=$(api POST "/v3/tenants/$tenant/collections/$collection/search" '{
  "queries":{"embedding":[1,0,0]},
  "top_k":10
}')
assert_json "Compose document survived replacement" "$response" \
  '([.documents[].id] | index(301)) != null'

"$grpc_probe" "$grpc_address" "$api_token" "$tenant" "$collection" "$document_id"

response=$(api DELETE "/v3/tenants/$tenant/collections/$collection/docs" '{"doc_id":301}')
assert_json "Compose HTTP V3 delete document" "$response" '.status == "success"'

response=$(api DELETE "/v3/tenants/$tenant/collections/$collection")
assert_json "Compose HTTP V3 delete collection" "$response" '.status == "success"'

response=$(api GET "/v3/tenants/$tenant/collections")
assert_json "Compose contract collections cleaned up" "$response" '.count == 0'

echo "asserting final graceful Compose shutdown"
stop_service

echo "Compose authenticated HTTP/gRPC, replacement persistence, cleanup, and graceful shutdown passed"
