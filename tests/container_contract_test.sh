#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${1:-deepdata-contract:test}
runtime=${CONTAINER_RUNTIME:-docker}
api_token=container-contract-test-token-strong-credential
tenant=container-contract
collection=http_docs
document_id=101
volume="deepdata-contract-$PPID-$$"
first_container="${volume}-first"
restart_container="${volume}-restart"
work_dir=""
grpc_probe=""
http_port=""
grpc_port=""
base_url=""
grpc_address=""

fail() {
  echo "container contract failed: $*" >&2
  return 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "$1 is required to validate the container contract" >&2
    exit 127
  }
}

require_command "$runtime"
require_command curl
require_command go
require_command jq

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/deepdata-container-contract.XXXXXX")
grpc_probe="$work_dir/grpc-probe"

cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  if ((rc != 0)); then
    for candidate in "$first_container" "$restart_container"; do
      if "$runtime" inspect "$candidate" >/dev/null 2>&1; then
        echo "logs from failed container $candidate:" >&2
        "$runtime" logs "$candidate" >&2 || true
      fi
    done
  fi
  "$runtime" rm -f "$first_container" "$restart_container" >/dev/null 2>&1 || true
  "$runtime" volume rm -f "$volume" >/dev/null 2>&1 || true
  rm -rf -- "$work_dir"
  exit "$rc"
}
trap cleanup EXIT

(cd -- "$repo_root" && go build -trimpath -o "$grpc_probe" ./tests/container_grpc_probe)

identity="$($runtime run --rm --entrypoint /usr/bin/id "$image" -u):$($runtime run --rm --entrypoint /usr/bin/id "$image" -g)"
if [[ "$identity" != "10001:10001" ]]; then
  fail "unexpected DeepData container identity: $identity"
fi

"$runtime" volume create "$volume" >/dev/null
"$runtime" run --rm \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,noexec \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  -e API_TOKEN="$api_token" \
  -e REQUIRE_AUTH=1 \
  -v "$volume:/data" \
  --entrypoint /bin/sh \
  "$image" -euc '
    test "$(id -u):$(id -g)" = "10001:10001"
    test "$GRPC_PORT" = 50051
    test "$VECTORDB_BASE_DIR" = /data
    test "$VECTORDB_DATA_DIR" = local
    test -w /data
    test -w /tmp
    test "$(sed -n "s/^CapBnd:[[:space:]]*//p" /proc/self/status)" = 0000000000000000
    test "$(sed -n "s/^NoNewPrivs:[[:space:]]*//p" /proc/self/status)" = 1
    root_options=$(awk "\$2 == \"/\" {print \$4; exit}" /proc/mounts)
    case ",$root_options," in
      *,ro,*) ;;
      *) echo "container root filesystem is not read-only" >&2; exit 1 ;;
    esac
    : > /data/container-contract
    : > /tmp/container-contract
  '

resolve_ports() {
  local name=$1
  http_port="$($runtime port "$name" 8080/tcp | sed -n '1s/.*://p')"
  grpc_port="$($runtime port "$name" 50051/tcp | sed -n '1s/.*://p')"
  [[ "$http_port" =~ ^[1-9][0-9]*$ ]] || fail "invalid HTTP port mapping: $http_port"
  [[ "$grpc_port" =~ ^[1-9][0-9]*$ ]] || fail "invalid gRPC port mapping: $grpc_port"
  base_url="http://127.0.0.1:$http_port"
  grpc_address="127.0.0.1:$grpc_port"
}

wait_ready() {
  local name=$1 ready=false
  for _ in $(seq 1 240); do
    if [[ "$($runtime inspect --format '{{.State.Running}}' "$name" 2>/dev/null)" != true ]]; then
      "$runtime" logs "$name" >&2 || true
      fail "$name exited before readiness"
    fi
    if curl -fsS --max-time 1 "$base_url/readyz" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 0.25
  done
  [[ "$ready" == true ]] || fail "$name did not become ready"
  curl -fsS --max-time 2 "$base_url/livez" >/dev/null
}

start_container() {
  local name=$1
  "$runtime" run -d \
    --name "$name" \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,noexec \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    -e API_TOKEN="$api_token" \
    -e REQUIRE_AUTH=1 \
    -v "$volume:/data" \
    -p 127.0.0.1::8080 \
    -p 127.0.0.1::50051 \
    "$image" >/dev/null
  resolve_ports "$name"
  wait_ready "$name"
}

stop_and_remove() {
  local name=$1 exit_code
  "$runtime" stop --time 30 "$name" >/dev/null
  exit_code="$($runtime inspect --format '{{.State.ExitCode}}' "$name")"
  if [[ "$exit_code" != "0" ]]; then
    "$runtime" logs "$name" >&2 || true
    fail "$name exited with status $exit_code after SIGTERM"
  fi
  "$runtime" rm "$name" >/dev/null
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
  fi
  echo "PASS: $name"
}

create_body='{
  "name":"http_docs",
  "fields":[
    {"name":"embedding","type":"dense","dim":3,"index":{"type":"flat"}}
  ]
}'

echo "starting first hardened container"
start_container "$first_container"

response=$(api POST "/v3/tenants/$tenant/collections" "$create_body")
assert_json "HTTP V3 create collection" "$response" '.status == "success"'

response=$(api POST "/v3/tenants/$tenant/collections/$collection/docs" '{
  "id":101,
  "vectors":{"embedding":[1,0,0]},
  "metadata":{"source":"container-contract"}
}')
assert_json "HTTP V3 insert" "$response" '.id == 101'

response=$(api POST "/v3/tenants/$tenant/collections/$collection/search" '{
  "queries":{"embedding":[1,0,0]},
  "top_k":10
}')
assert_json "HTTP V3 search" "$response" '([.documents[].id] | index(101)) != null'

"$grpc_probe" "$grpc_address" "$api_token" "$tenant" "$collection" "$document_id"

echo "stopping and removing first container"
stop_and_remove "$first_container"

echo "starting a new container on the same named volume"
start_container "$restart_container"

response=$(api GET "/v3/tenants/$tenant/collections/$collection")
assert_json "HTTP collection survived container replacement" "$response" \
  '.collection.Name == "http_docs" and .collection.DocCount == 1'

response=$(api POST "/v3/tenants/$tenant/collections/$collection/search" '{
  "queries":{"embedding":[1,0,0]},
  "top_k":10
}')
assert_json "HTTP document survived container replacement" "$response" \
  '([.documents[].id] | index(101)) != null'

"$grpc_probe" "$grpc_address" "$api_token" "$tenant" "$collection" "$document_id"

response=$(api DELETE "/v3/tenants/$tenant/collections/$collection/docs" '{"doc_id":101}')
assert_json "HTTP V3 delete document" "$response" '.status == "success"'

response=$(api POST "/v3/tenants/$tenant/collections/$collection/search" '{
  "queries":{"embedding":[1,0,0]},
  "top_k":10
}')
assert_json "deleted document is absent" "$response" \
  '([.documents[].id] | index(101)) == null'

response=$(api DELETE "/v3/tenants/$tenant/collections/$collection")
assert_json "HTTP V3 delete collection" "$response" '.status == "success"'

response=$(api GET "/v3/tenants/$tenant/collections")
assert_json "container contract collections cleaned up" "$response" '.count == 0'

echo "asserting successful graceful SIGTERM shutdown"
stop_and_remove "$restart_container"

echo "container identity, hardened filesystem, probes, authenticated HTTP/gRPC, restart persistence, cleanup, and graceful shutdown passed"
