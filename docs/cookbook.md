# DeepData RC Cookbook

This cookbook describes the supported release-candidate surface: one persistent
DeepData process on Linux, tenant-aware HTTP V3, and the matching unary gRPC
service. Clients provide every dense and sparse vector.

Supported mutations are create collection, delete collection, insert one,
batch insert, upsert one, and delete document. Supported reads are tenant
information, collection get/list, document get-by-ID, and dense, sparse, or
hybrid search. Dense fields use HNSW or Flat; sparse fields use the inverted
index.

## Canonical Python client

```python
from deepdata import DeepDataClient

with DeepDataClient(
    "http://localhost:8080",
    api_token="replace-with-token",
) as client:
    tenant = client.tenant("org-123")
    tenant.create_collection(
        "docs",
        fields=[
            {
                "name": "embedding",
                "type": "dense",
                "dim": 3,
                "index": {"type": "hnsw"},
            },
            {
                "name": "keywords",
                "type": "sparse",
                "dim": 10000,
                "index": {"type": "inverted"},
            },
        ],
    )

    first = tenant.insert(
        "docs",
        vectors={
            "embedding": [0.1, 0.2, 0.3],
            "keywords": {"indices": [4, 9], "values": [0.8, 0.4], "dim": 10000},
        },
        metadata={"source": "example"},
    )

    tenant.batch_insert(
        "docs",
        [
            {"vectors": {
                "embedding": [0.3, 0.2, 0.1],
                "keywords": {"indices": [2], "values": [1.0], "dim": 10000},
            }},
            {"id": 1002, "vectors": {
                "embedding": [0.2, 0.3, 0.1],
                "keywords": {"indices": [7], "values": [1.0], "dim": 10000},
            }},
        ],
    )

    results = tenant.search(
        "docs",
        queries={"embedding": [0.1, 0.2, 0.3]},
        top_k=5,
    )
    tenant.upsert(
        "docs",
        id=1002,
        vectors={"embedding": [0.2, 0.3, 0.1]},
        metadata={"source": "example", "state": "revised"},
    )
    one = tenant.get_document("docs", 1002)
    tenant.delete_document("docs", first.id)
```

Generate query vectors with the same client-side model and normalization used
for inserts. DeepData does not transform text into vectors.

## Raw HTTP V3

All canonical routes are below
`/v3/tenants/{tenant}/collections`. Send credentials in the `Authorization`
header, never in a URL.

```bash
BASE_URL=http://localhost:8080
TENANT=org-123
TOKEN=replace-with-token

# Create a collection.
curl -fsS -X POST "$BASE_URL/v3/tenants/$TENANT/collections" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"docs",
    "fields":[
      {"name":"embedding","type":"dense","dim":3,"index":{"type":"flat"}},
      {"name":"keywords","type":"sparse","dim":64,"index":{"type":"inverted"}}
    ]
  }'

# Insert one document with a caller-selected ID.
curl -fsS -X POST "$BASE_URL/v3/tenants/$TENANT/collections/docs/docs" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "id":101,
    "vectors":{
      "embedding":[1,0,0],
      "keywords":{"indices":[1,3],"values":[0.8,0.4],"dim":64}
    },
    "metadata":{"kind":"example"}
  }'

# Batch insert is all-or-nothing.
curl -fsS -X POST "$BASE_URL/v3/tenants/$TENANT/collections/docs/docs/batch" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"documents":[
    {"vectors":{
      "embedding":[0,1,0],
      "keywords":{"indices":[2],"values":[1],"dim":64}
    }},
    {"id":103,"vectors":{
      "embedding":[0,0,1],
      "keywords":{"indices":[3],"values":[1],"dim":64}
    }}
  ]}'

# List, get, and search.
curl -fsS "$BASE_URL/v3/tenants/$TENANT/collections" \
  -H "Authorization: Bearer $TOKEN"
curl -fsS "$BASE_URL/v3/tenants/$TENANT/collections/docs" \
  -H "Authorization: Bearer $TOKEN"
curl -fsS -X POST "$BASE_URL/v3/tenants/$TENANT/collections/docs/search" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"queries":{"embedding":[1,0,0]},"top_k":5}'

# Upsert creates-or-replaces a caller-addressed document.
curl -fsS -X PUT "$BASE_URL/v3/tenants/$TENANT/collections/docs/docs/103" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"vectors":{"embedding":[1,0,0]},"metadata":{"kind":"revised"}}'

# Get a single document by ID.
curl -fsS "$BASE_URL/v3/tenants/$TENANT/collections/docs/docs/103" \
  -H "Authorization: Bearer $TOKEN"

# Delete one document, then delete the collection.
curl -fsS -X DELETE "$BASE_URL/v3/tenants/$TENANT/collections/docs/docs" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"doc_id":103}'
curl -fsS -X DELETE "$BASE_URL/v3/tenants/$TENANT/collections/docs" \
  -H "Authorization: Bearer $TOKEN"
```

For hybrid search, send exactly two query fields and an explicit fusion policy:

```json
{
  "queries": {
    "embedding": [1, 0, 0],
    "keywords": {"indices": [1, 3], "values": [0.8, 0.4], "dim": 64}
  },
  "top_k": 10,
  "hybrid_params": {
    "strategy": "weighted",
    "weights": {"embedding": 0.7, "keywords": 0.3}
  }
}
```

## gRPC contract

The gRPC service mirrors the tenant and operation model. It exposes eleven
unary methods; there are no streaming RPCs and server reflection is not part of
the RC contract.

| Method | Operation |
|---|---|
| `GetTenantInfo` | Tenant and collection counts |
| `ListCollections` | List tenant collections |
| `GetCollection` | Read one collection schema |
| `CreateCollection` | Create collection |
| `DeleteCollection` | Delete collection |
| `Insert` | Insert one document |
| `BatchInsert` | Atomic batch insert |
| `Search` | Dense, sparse, or hybrid search |
| `Upsert` | Insert-or-replace one caller-addressed document |
| `GetDoc` | Read one document by ID |
| `DeleteDoc` | Delete one document |

Use the checked-in schema at `api/proto/deepdata/v3/deepdata.proto`. A minimal
Go request looks like this:

```go
conn, err := grpc.NewClient(
    "localhost:50051",
    grpc.WithTransportCredentials(insecure.NewCredentials()),
)
if err != nil {
    return err
}
defer conn.Close()

client := deepdatav3.NewDeepDataClient(conn)
ctx := metadata.AppendToOutgoingContext(
    context.Background(),
    "authorization", "Bearer "+token,
)
response, err := client.Search(ctx, &deepdatav3.SearchRequest{
    TenantId:  "org-123",
    Collection: "docs",
    Queries: map[string]*deepdatav3.VectorData{
        "embedding": {
            Data: &deepdatav3.VectorData_Dense{
                Dense: &deepdatav3.DenseVector{Values: []float32{1, 0, 0}},
            },
        },
    },
    TopK: 5,
})
```

Production deployments terminate TLS at a trusted proxy or load balancer. The
plaintext credentials above are suitable only for a trusted local connection.

## Offline backup

Official systemd deployments use:

```ini
Environment=VECTORDB_MODE=local
Environment=VECTORDB_BASE_DIR=/var/lib/deepdata
Environment=VECTORDB_DATA_DIR=local
```

The exact primary directory is `/var/lib/deepdata/local`; the backup boundary
is the whole `/var/lib/deepdata` state root. Stop DeepData before copying it.

```bash
#!/usr/bin/env bash
set -euo pipefail

wait_for_deepdata() {
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    sudo systemctl is-active --quiet deepdata || return 1
    if curl -fsS --max-time 1 http://localhost:8080/readyz >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

verify_process_contract() {
  local main_pid process_env
  main_pid=$(sudo systemctl show deepdata --property=MainPID --value)
  [[ "$main_pid" =~ ^[1-9][0-9]*$ ]]
  process_env=$(sudo cat -- "/proc/$main_pid/environ" | tr '\0' '\n')
  grep -Fqx -- "VECTORDB_MODE=local" <<<"$process_env"
  grep -Fqx -- "VECTORDB_BASE_DIR=$STATE_ROOT" <<<"$process_env"
  grep -Fqx -- "VECTORDB_DATA_DIR=local" <<<"$process_env"
}

STATE_ROOT=$(realpath -e -- /var/lib/deepdata)
BACKUP_PARENT=$(realpath -e -- /backup)
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-300}
BACKUP_DIR="$BACKUP_PARENT/deepdata-state-$(date +%Y%m%dT%H%M%S)"
MANIFEST="$BACKUP_DIR.manifest"

[[ "$READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]
[[ "$STATE_ROOT" != / && -d "$STATE_ROOT/local" ]]
[[ ! -e "$BACKUP_DIR" && ! -e "$MANIFEST" ]]
case "$BACKUP_DIR/" in
  "$STATE_ROOT/"*) echo "backup destination is inside state root" >&2; exit 1 ;;
esac

wait_for_deepdata
verify_process_contract
sudo systemctl stop deepdata
! sudo systemctl is-active --quiet deepdata

sudo cp -a -- "$STATE_ROOT" "$BACKUP_DIR"
printf '%s\n' \
  'VECTORDB_MODE=local' \
  "VECTORDB_BASE_DIR=$STATE_ROOT" \
  'VECTORDB_DATA_DIR=local' \
  "PRIMARY_DIRECTORY=$STATE_ROOT/local" | sudo tee "$MANIFEST" >/dev/null
sudo chmod 0600 "$MANIFEST"
sync
sudo test -d "$BACKUP_DIR/local"

sudo systemctl start deepdata
wait_for_deepdata
verify_process_contract
```

If any precondition or copy fails, leave DeepData stopped and preserve both the
source and partial destination for diagnosis.

## Offline restore with rollback

Restore only a verified whole-root backup. Never merge it into existing state.
`RESTORE_ASSERT_SCRIPT` is mandatory and must fail unless the expected V3
tenant, collection schema, meaningful document count, representative query,
and gRPC result are all correct. It receives the HTTP base URL and gRPC address;
credentials can be passed through its environment.

```bash
#!/usr/bin/env bash
set -euo pipefail

wait_for_deepdata() {
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    sudo systemctl is-active --quiet deepdata || return 1
    if curl -fsS --max-time 1 http://localhost:8080/readyz >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

verify_process_contract() {
  local main_pid process_env
  main_pid=$(sudo systemctl show deepdata --property=MainPID --value)
  [[ "$main_pid" =~ ^[1-9][0-9]*$ ]]
  process_env=$(sudo cat -- "/proc/$main_pid/environ" | tr '\0' '\n')
  grep -Fqx -- "VECTORDB_MODE=local" <<<"$process_env"
  grep -Fqx -- "VECTORDB_BASE_DIR=$STATE_ROOT" <<<"$process_env"
  grep -Fqx -- "VECTORDB_DATA_DIR=local" <<<"$process_env"
}

STATE_ROOT=$(realpath -e -- /var/lib/deepdata)
RESTORE_SRC=$(realpath -e -- /backup/deepdata-state-20260718T020000)
: "${RESTORE_ASSERT_SCRIPT:?set an executable V3/gRPC assertion script}"
RESTORE_ASSERT_SCRIPT=$(realpath -e -- "$RESTORE_ASSERT_SCRIPT")
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-300}
TAG=$(date +%Y%m%dT%H%M%S)
STATE_PARENT=$(dirname -- "$STATE_ROOT")
RESTORE_STAGE="$STATE_PARENT/.deepdata-restore-$TAG"
ROLLBACK_ROOT="$STATE_PARENT/deepdata-before-restore-$TAG"
FAILED_ROOT="$STATE_PARENT/deepdata-failed-restore-$TAG"

[[ "$READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]
[[ -x "$RESTORE_ASSERT_SCRIPT" ]]
[[ "$STATE_ROOT" != / && "$RESTORE_SRC" != / ]]
[[ -d "$STATE_ROOT/local" && -d "$RESTORE_SRC/local" ]]
[[ ! -e "$RESTORE_STAGE" && ! -e "$ROLLBACK_ROOT" && ! -e "$FAILED_ROOT" ]]
case "$RESTORE_SRC/" in
  "$STATE_ROOT/"*) echo "restore source is inside live state" >&2; exit 1 ;;
esac
case "$STATE_ROOT/" in
  "$RESTORE_SRC/"*) echo "live state is inside restore source" >&2; exit 1 ;;
esac

wait_for_deepdata
verify_process_contract
sudo cp -a -- "$RESTORE_SRC" "$RESTORE_STAGE"
sudo test -d "$RESTORE_STAGE/local"
sync

sudo systemctl stop deepdata
! sudo systemctl is-active --quiet deepdata
sudo mv -- "$STATE_ROOT" "$ROLLBACK_ROOT"
if ! sudo mv -- "$RESTORE_STAGE" "$STATE_ROOT"; then
  sudo mv -- "$ROLLBACK_ROOT" "$STATE_ROOT"
  echo "restore cutover failed; original state reinstated" >&2
  exit 1
fi
sync

if sudo systemctl start deepdata &&
   wait_for_deepdata &&
   verify_process_contract &&
   "$RESTORE_ASSERT_SCRIPT" http://localhost:8080 localhost:50051
then
  echo "restore passed readiness plus V3/gRPC data assertions"
  echo "retain rollback state at $ROLLBACK_ROOT until the change is accepted"
else
  sudo systemctl stop deepdata || true
  if sudo systemctl is-active --quiet deepdata; then
    echo "restored server would not stop; manual recovery required" >&2
    exit 1
  fi
  sudo mv -- "$STATE_ROOT" "$FAILED_ROOT"
  sudo mv -- "$ROLLBACK_ROOT" "$STATE_ROOT"
  sync
  if sudo systemctl start deepdata && wait_for_deepdata && verify_process_contract; then
    echo "restore validation failed; original state reinstated and ready" >&2
  else
    echo "original state was reinstated but did not become ready" >&2
  fi
  exit 1
fi
```

Rehearse the same procedure and assertion script on an isolated Linux host
before relying on a backup. Readiness without semantic V3 and gRPC checks is not
restore evidence.
