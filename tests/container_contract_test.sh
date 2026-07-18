#!/usr/bin/env bash
set -euo pipefail

image=${1:-deepdata-contract:test}
runtime=${CONTAINER_RUNTIME:-docker}
volume="deepdata-contract-$PPID-$$"
container="${volume}-server"

if ! command -v "$runtime" >/dev/null 2>&1; then
  echo "$runtime is required to validate the container contract" >&2
  exit 127
fi

cleanup() {
  "$runtime" rm -f "$container" >/dev/null 2>&1 || true
  "$runtime" volume rm -f "$volume" >/dev/null 2>&1 || true
}
trap cleanup EXIT

identity="$($runtime run --rm --entrypoint /usr/bin/id "$image" -u):$($runtime run --rm --entrypoint /usr/bin/id "$image" -g)"
if [[ "$identity" != "10001:10001" ]]; then
  echo "unexpected DeepData container identity: $identity" >&2
  exit 1
fi

"$runtime" volume create "$volume" >/dev/null
"$runtime" run --rm \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,noexec \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  -v "$volume:/data" \
  --entrypoint /bin/sh \
  "$image" -euc '
    test "$(id -u):$(id -g)" = "10001:10001"
    test "$GRPC_PORT" = 50051
    test "$VECTORDB_BASE_DIR" = /data
    test "$VECTORDB_DATA_DIR" = local
    test -w /data
    test -w /tmp
    : > /data/container-contract
    : > /tmp/container-contract
  '

"$runtime" run -d \
  --name "$container" \
  --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,noexec \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  -v "$volume:/data" \
  -p 127.0.0.1::8080 \
  -p 127.0.0.1::50051 \
  "$image" >/dev/null

http_port="$($runtime port "$container" 8080/tcp | sed -n '1s/.*://p')"
grpc_port="$($runtime port "$container" 50051/tcp | sed -n '1s/.*://p')"
[[ "$http_port" =~ ^[1-9][0-9]*$ ]]
[[ "$grpc_port" =~ ^[1-9][0-9]*$ ]]

ready=false
for _ in $(seq 1 60); do
  if [[ "$($runtime inspect --format '{{.State.Running}}' "$container")" != true ]]; then
    "$runtime" logs "$container" >&2 || true
    echo "DeepData container exited before readiness" >&2
    exit 1
  fi
  if curl -fsS --max-time 1 "http://127.0.0.1:$http_port/readyz" >/dev/null; then
    ready=true
    break
  fi
  sleep 1
done
[[ "$ready" == true ]]
curl -fsS --max-time 2 "http://127.0.0.1:$http_port/livez" >/dev/null
timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$grpc_port"

"$runtime" stop --time 15 "$container" >/dev/null
"$runtime" rm "$container" >/dev/null

echo "container identity, fresh-volume, probes, and protocol contracts passed"
