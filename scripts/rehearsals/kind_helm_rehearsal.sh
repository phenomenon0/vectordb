#!/usr/bin/env bash
# OPS-03: the container, Compose and kind+Helm lifecycle contracts, run against
# one image built from the frozen tree.
#
# The three legs are one rehearsal on purpose. Each proves a different half of
# the deployment claim -- the image runs and keeps its state (container), the
# published Compose file wires it correctly (Compose), and the chart schedules
# it with a real digest and survives losing its pod (kind+Helm) -- and all three
# must hold for the same bytes, so they share a single build.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
runtime=${CONTAINER_RUNTIME:-podman}
image=${DEEPDATA_IMAGE:-deepdata-rc:rehearsal}
cluster=${KIND_CLUSTER:-deepdata-rehearsal}
registry_name=${KIND_REGISTRY_NAME:-deepdata-rehearsal-registry}
registry_port=${KIND_REGISTRY_PORT:-5001}
registry_image=${KIND_REGISTRY_IMAGE:-docker.io/library/registry:2}
release=deepdata-rehearsal
namespace=deepdata-rehearsal
api_token=kind-helm-rehearsal-token-strong-credential
tenant=kind-rehearsal
collection=kind_docs
document_id=701
work_dir=""
forward_pid=""
cluster_created=0

log() { printf '[kind-helm] %s\n' "$*"; }
fail() { echo "kind+Helm rehearsal failed: $*" >&2; exit 1; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "$1 is required for the deployment lifecycle rehearsal" >&2
    exit 127
  }
}

for tool in "$runtime" kind helm kubectl curl jq go podman-compose; do
  require_command "$tool"
done

# The rehearsal binds evidence to a commit, so it must not run over edits that
# commit does not contain.
if ! git -C "$repo_root" diff --quiet HEAD -- || [ -n "$(git -C "$repo_root" ls-files --others --exclude-standard)" ]; then
  fail "the working tree is dirty; rehearse only a committed tree"
fi

cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  if [ -n "$forward_pid" ]; then kill "$forward_pid" >/dev/null 2>&1; wait "$forward_pid" 2>/dev/null; fi
  if [ "$cluster_created" -eq 1 ]; then
    KIND_EXPERIMENTAL_PROVIDER="$runtime" kind delete cluster --name "$cluster" >/dev/null 2>&1
  fi
  "$runtime" rm -f "$registry_name" >/dev/null 2>&1
  [ -n "$work_dir" ] && rm -rf "$work_dir"
  exit "$rc"
}
trap cleanup EXIT

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/deepdata-kind-helm.XXXXXX")

log "building $image from $(git -C "$repo_root" rev-parse --short HEAD)"
"$runtime" build -t "$image" "$repo_root" >"$work_dir/build.log" 2>&1 ||
  { cat "$work_dir/build.log" >&2; fail "image build"; }

log "leg 1/3: container lifecycle contract"
CONTAINER_RUNTIME="$runtime" "$repo_root/tests/container_contract_test.sh" "$image" ||
  fail "container contract"

log "leg 2/3: Compose lifecycle contract"
CONTAINER_RUNTIME="$runtime" DEEPDATA_IMAGE="$image" "$repo_root/tests/compose_contract_test.sh" ||
  fail "Compose contract"

log "leg 3/3: kind + Helm lifecycle contract"

# The cluster is created before the registry because the registry has to be a
# member of kind's network from the moment it starts. Rootless podman puts a
# port-publishing container under pasta, and `podman network connect` rejects a
# pasta container outright ("pasta" is not supported: invalid network mode), so
# a registry started first can never be attached afterwards. Nothing about that
# is visible at the time -- it surfaces minutes later as ImagePullBackOff,
# which reads like a bad digest.
cat >"$work_dir/kind.yaml" <<KIND
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
  - |-
    [plugins."io.containerd.grpc.v1.cri".registry]
      config_path = "/etc/containerd/certs.d"
KIND

KIND_EXPERIMENTAL_PROVIDER="$runtime" kind create cluster \
  --name "$cluster" --config "$work_dir/kind.yaml" --wait 120s ||
  fail "kind create cluster (provider=$runtime)"
cluster_created=1
node="${cluster}-control-plane"
kubecfg="$work_dir/kubeconfig"
KIND_EXPERIMENTAL_PROVIDER="$runtime" kind get kubeconfig --name "$cluster" >"$kubecfg"
export KUBECONFIG="$kubecfg"

# The chart refuses a tag-only image, so the rehearsal has to mint a real
# manifest digest, and a throwaway registry is the only way to get one without
# publishing anything. --network puts it where the node can reach it; -p still
# publishes to the host, which is what the push below goes through.
"$runtime" rm -f "$registry_name" >/dev/null 2>&1 || true
"$runtime" run -d --name "$registry_name" --network kind \
  -p "127.0.0.1:${registry_port}:5000" "$registry_image" >/dev/null ||
  fail "starting the rehearsal registry on kind's network"
registry_ref="localhost:${registry_port}/deepdata"
for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:${registry_port}/v2/" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:${registry_port}/v2/" >/dev/null || fail "local registry did not start"

"$runtime" push --tls-verify=false --digestfile "$work_dir/digest" \
  "$image" "${registry_ref}:rehearsal" >"$work_dir/push.log" 2>&1 ||
  { cat "$work_dir/push.log" >&2; fail "push to the rehearsal registry"; }
digest=$(cat "$work_dir/digest")
[[ "$digest" =~ ^sha256:[a-f0-9]{64}$ ]] || fail "unusable manifest digest: $digest"
log "manifest digest $digest"

# The node pulls by the same localhost:PORT reference the digest was minted
# under, so containerd has to be told where that name really lives. Address the
# registry by IP and not by container name: the node resolves DNS through
# podman's resolver, which does not reliably answer for it ("lookup
# deepdata-rehearsal-registry ... server misbehaving").
registry_ip=$("$runtime" inspect -f \
  '{{ (index .NetworkSettings.Networks "kind").IPAddress }}' "$registry_name" 2>/dev/null || true)
[[ "$registry_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || fail "registry has no IPv4 address on the kind network: '${registry_ip:-<empty>}'"
log "registry reachable at $registry_ip:5000 from the node"

"$runtime" exec "$node" mkdir -p "/etc/containerd/certs.d/localhost:${registry_port}"
"$runtime" exec "$node" sh -c \
  "printf '[host.\"http://%s:5000\"]\n  capabilities = [\"pull\", \"resolve\"]\n' '$registry_ip' \
   > '/etc/containerd/certs.d/localhost:${registry_port}/hosts.toml'"
"$runtime" exec "$node" systemctl restart containerd
kubectl wait --for=condition=Ready "node/$node" --timeout=120s >/dev/null

# Prove the node can actually reach the registry before asking Helm to pull
# through it; otherwise a network fault surfaces 5 minutes later as an install
# timeout with no cause attached. /dev/tcp rather than curl: the node image is
# not required to carry an HTTP client, but it does run bash.
"$runtime" exec "$node" bash -c \
  "exec 3<>/dev/tcp/${registry_ip}/5000 && printf 'GET /v2/ HTTP/1.0\r\n\r\n' >&3 && head -n 1 <&3" \
  2>/dev/null | grep -q ' 200 ' \
  || fail "the kind node cannot reach the rehearsal registry at $registry_ip:5000"

kubectl create namespace "$namespace" >/dev/null
kubectl create secret generic deepdata-auth \
  --namespace "$namespace" --from-literal=api-token="$api_token" >/dev/null

# kind's local-path volumes are ext4 directories on the node, which is where
# flock, fsync and same-directory rename actually hold.
helm install "$release" "$repo_root/deploy/helm/deepdata" \
  --namespace "$namespace" \
  --set image.repository="$registry_ref" \
  --set image.digest="$digest" \
  --set image.pullPolicy=IfNotPresent \
  --set persistence.verifiedPOSIXSemantics=true \
  --wait --timeout 5m >"$work_dir/helm-install.log" 2>&1 ||
  {
    cat "$work_dir/helm-install.log" >&2
    kubectl -n "$namespace" describe pods >&2 || true
    fail "helm install"
  }

kubectl -n "$namespace" rollout status "deployment/${release}-deepdata" --timeout=180s >/dev/null ||
  fail "deployment did not roll out"

local_port=""
start_forward() {
  local_port=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
  kubectl -n "$namespace" port-forward "service/${release}-deepdata" "${local_port}:8080" \
    >"$work_dir/forward.log" 2>&1 &
  forward_pid=$!
  for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:${local_port}/readyz" >/dev/null 2>&1 && return 0
    sleep 1
  done
  cat "$work_dir/forward.log" >&2
  fail "the deployed pod never became ready"
}
stop_forward() {
  # wait on a process this function just killed returns 143, and under set -e
  # that status would abort the rehearsal mid-way as if a check had failed.
  if [ -n "$forward_pid" ]; then
    kill "$forward_pid" >/dev/null 2>&1 || true
    wait "$forward_pid" 2>/dev/null || true
  fi
  forward_pid=""
}

api() {
  local method=$1 path=$2 payload=${3:-}
  if [ -n "$payload" ]; then
    curl -fsS -X "$method" -H "Authorization: Bearer $api_token" \
      -H 'Content-Type: application/json' -d "$payload" "http://127.0.0.1:${local_port}${path}"
  else
    curl -fsS -X "$method" -H "Authorization: Bearer $api_token" \
      "http://127.0.0.1:${local_port}${path}"
  fi
}

start_forward
collections_path="/v3/tenants/${tenant}/collections"
api POST "$collections_path" \
  "{\"name\":\"${collection}\",\"fields\":[{\"name\":\"embedding\",\"type\":\"dense\",\"dim\":2,\"index\":{\"type\":\"flat\"}}]}" \
  >/dev/null || fail "create collection through the chart's service"
api POST "${collections_path}/${collection}/docs" \
  "{\"id\":${document_id},\"vectors\":{\"embedding\":[1,0]},\"metadata\":{\"leg\":\"kind\"}}" \
  >/dev/null || fail "insert through the chart's service"

# Losing the pod is the lifecycle event the chart exists to survive: the PVC,
# not the pod, is where the state lives.
log "deleting the pod to prove the claim outlives it"
old_pod=$(kubectl -n "$namespace" get pods -l app.kubernetes.io/instance="$release" \
  -o jsonpath='{.items[0].metadata.name}')
stop_forward
kubectl -n "$namespace" delete "pod/$old_pod" --wait=true --timeout=180s >/dev/null
kubectl -n "$namespace" rollout status "deployment/${release}-deepdata" --timeout=300s >/dev/null ||
  fail "deployment did not recover after pod deletion"
new_pod=$(kubectl -n "$namespace" get pods -l app.kubernetes.io/instance="$release" \
  -o jsonpath='{.items[0].metadata.name}')
[ "$new_pod" != "$old_pod" ] || fail "pod $old_pod was never replaced"

start_forward
count=$(api GET "${collections_path}/${collection}" | jq -r '.collection.doc_count')
[ "$count" = "1" ] || fail "rescheduled pod reports doc_count=$count, want 1"
found=$(api POST "${collections_path}/${collection}/search" \
  '{"queries":{"embedding":[1,0]},"top_k":1}' | jq -r '.documents[0].id')
[ "$found" = "$document_id" ] || fail "rescheduled pod returned document $found, want $document_id"
log "state survived pod $old_pod -> $new_pod"
stop_forward

helm uninstall "$release" --namespace "$namespace" --wait --timeout 3m >/dev/null ||
  fail "helm uninstall"
if kubectl -n "$namespace" get "deployment/${release}-deepdata" >/dev/null 2>&1; then
  fail "helm uninstall left the deployment behind"
fi

log "container, Compose and kind+Helm lifecycle contracts all passed on $digest"
