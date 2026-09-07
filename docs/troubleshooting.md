# DeepData RC Troubleshooting

These procedures apply to the Linux amd64, persistent, single-node release
candidate and its tenant-aware V3/gRPC contract.

## Startup failures

### Authentication configuration rejected

Normal startup requires exactly one of `API_TOKEN` or `JWT_SECRET`. If neither
is set, configure one credential. If both are set, remove the unintended one.
The process rejects either condition before opening the data directory.

`DEEPDATA_INSECURE_DEV_MODE=1` permits credentialless startup only for an
explicit disposable local development process. Do not use it to bypass a
production startup failure, with persistent data, or on a network-accessible
listener.

### HTTP or gRPC address already in use

DeepData requires the complete configured listener set. If either bind fails,
startup fails rather than serving a partial API.

```bash
ss -ltnp | grep -E ':(8080|50051)\b'
PORT=8081 GRPC_PORT=50052 ./deepdata serve
```

### State directory permission denied

The systemd layout must be writable only by the service account:

```bash
sudo install -d -o deepdata -g deepdata -m 0750 /var/lib/deepdata
sudo -u deepdata test -w /var/lib/deepdata
sudo systemctl show deepdata --property=User,Group,WorkingDirectory
```

With `VECTORDB_BASE_DIR=/var/lib/deepdata` and
`VECTORDB_DATA_DIR=local`, the exact primary directory is
`/var/lib/deepdata/local`.

The official container identity is `10001:10001`, and its exact primary
directory is `/data/local` on the whole `/data` state volume.

### Another process owns the state lock

Only one process may open a given tenant's persistent store: each tenant under
a data directory owns its own lock file at
`<data-dir>/index.gob.tenants/<tenant>.lock` (`internal/collection/store_set.go`).
Confirm that a previous process or container is not still running. Do not
delete the lock file to work around a live owner.

```bash
systemctl status deepdata --no-pager
ps -ef | grep '[d]eepdata'
docker compose ps
```

### Existing legacy state blocks canonical startup

Canonical startup fails closed when it finds older root or unscoped collection
state. Preserve the entire stopped state root. Do not rename, delete, or merge
individual files. Migrate with a separately tested offline tool before starting
the RC against that path.

## Connectivity

### HTTP readiness

```bash
curl -fsS http://localhost:8080/livez
curl -fsS http://localhost:8080/readyz
```

Liveness proves the process can answer. Readiness also includes canonical
persistence health; neither proves that a particular collection contains the
expected data.

### gRPC connectivity without reflection

The RC does not require server reflection. Point `grpcurl` at the checked-in
schema and call a concrete unary method:

```bash
grpcurl -plaintext \
  -import-path api/proto \
  -proto deepdata/v3/deepdata.proto \
  -H 'authorization: Bearer replace-with-token' \
  -d '{"tenant_id":"org-123"}' \
  localhost:50051 deepdata.v3.DeepData/ListCollections
```

If HTTP is ready but this call fails, verify `GRPC_PORT`, the Service or load
balancer port, and protocol routing.

## Authentication and authorization

Set exactly one static token or JWT secret. Authentication is required by
normal startup; shipped deployments also set `REQUIRE_AUTH=1` as defense in
depth:

```bash
export API_TOKEN='replace-with-a-long-random-token'
REQUIRE_AUTH=1 ./deepdata serve
```

Send the token in a header:

```bash
curl -fsS \
  -H "Authorization: Bearer ${API_TOKEN}" \
  http://localhost:8080/v3/tenants/org-123/collections
```

- `401` means credentials are absent or invalid.
- `403` means the authenticated tenant, permission, collection scope, or
  administrative role does not allow the operation.
- Collection create/delete and tenant-wide list/info require administrative
  authorization.
- Search/get require read permission; insert/batch/delete-document require
  write permission.

Never put a bearer token in a URL.

## Insert and search errors

### Invalid field or dimension

Every supplied vector must match a declared field and dimension. Dense fields
accept finite numeric arrays. Sparse fields accept equal-length `indices` and
`values`, an integer `dim`, and in-range unsigned indices.

Dense fields support only HNSW or Flat. Sparse fields use the inverted index.

### Empty or low-quality results

- Confirm the tenant and collection path are correct.
- Confirm the same client-side vector model and normalization were used for
  inserts and queries.
- Confirm metadata filters match stored metadata.
- Reduce `top_k` while diagnosing and increase `ef_search` only when using an
  HNSW field.
- For hybrid search, send exactly two declared query fields and explicit
  weights for those field names.

### Batch request rejected

Canonical batch insert is atomic. One malformed document rejects the whole
batch; the server never reports partial success. Correct the failing item and
retry with the same intended IDs.

## Persistence and recovery

### Corrupt or truncated journal/snapshot

The only automatic repair is a structurally valid, terminal EOF-short frame in
the active journal. That frame was never fully appended or acknowledged;
startup truncates it back to the last completely verified record, synchronizes
the repair, reparses the journal, and resumes at the next sequence number.

Every other recovery error fails closed: a partial frozen journal, complete
frame with a bad checksum, complete frame whose length field was corrupted
past end of file, unknown version/store ID, sequence gap, non-prefix junk, or
corrupt/truncated snapshot. Keep the server stopped and preserve a
copy of the whole configured state root. Do not edit or delete snapshot,
journal, or lock artifacts, because doing so can discard acknowledged writes
or destroy diagnostic evidence.

A replay error reading `invalid sparse indices: expected []uint32-compatible
value, got <nil>` is not corruption: a document with no sparse terms was
journaled with null index and value arrays, which binaries before e910a6b
refused. Upgrade the binary and start again; do not edit the journal.

Restore a previously verified whole-root backup with the
[offline procedure](cookbook.md#offline-restore-with-rollback). Restore is not
successful until its mandatory assertion checks expected V3 tenant/schema,
document counts, representative search results, and gRPC access.

### Disk full

Stop writes, free space outside the DeepData state root, and inspect filesystem
and volume capacity. Do not manually remove files beneath `/var/lib/deepdata`
or `/data`. Expand the filesystem/PVC when the state itself consumes the
available capacity.

### Readiness changes to 503 after startup

Treat this as loss of canonical persistence health. Stop routing traffic,
preserve the complete state root, and inspect server logs. Restart only after
the cause is understood; repeated restarts are not a repair procedure.

## Docker and Compose

### Container exits immediately

```bash
docker compose ps
docker compose logs --no-color deepdata
docker inspect --format '{{.Config.User}}' "$(docker compose images -q deepdata)"
```

Compose requires `DEEPDATA_API_TOKEN` while resolving the file. If Compose
reports that the variable is missing, set it in the protected operator
environment before `pull`, `config`, or `up`:

```bash
export DEEPDATA_API_TOKEN='replace-with-a-long-random-token'
docker compose config
docker compose up -d --no-build deepdata
```

Do not also inject `JWT_SECRET`; the server rejects both authentication modes
being configured at once.

The candidate image must run as `10001:10001`, mount the named state volume at
`/data`, and set:

```text
VECTORDB_MODE=local
VECTORDB_BASE_DIR=/data
VECTORDB_DATA_DIR=local
```

### Existing volume has the wrong numeric owner

Treat ownership migration as an offline upgrade:

1. Record and retain the exact previous image ID.
2. Stop Compose and prove no container still mounts the state volume.
3. Capture and verify a read-only archive or storage snapshot of the whole
   volume.
4. Verify the candidate image runs as `10001:10001` without mounting state.
5. Change ownership with a short-lived root maintenance container, not the
   DeepData container.
6. Start the candidate without rebuilding.
7. Require readiness plus the same V3/gRPC assertion used for restore.

If validation fails, restore the verified archive into a new empty volume and
run the recorded previous image. Never unpack a backup over the modified source
volume.

## Kubernetes

### Pod pending or volume not writable

```bash
kubectl describe pod deepdata-0
kubectl describe pvc data-deepdata-0
kubectl get node -L kubernetes.io/os
```

The pod must schedule to Linux and the storage driver must honor `fsGroup:
10001`, or ownership must be prepared before the application starts. Do not use
privileged mode as a permission workaround.

### OOMKilled

Increase the memory request and limit or reduce the active dataset. HNSW keeps
vectors and graph data in memory; budget at least vector bytes plus graph,
metadata, process, and transient indexing overhead.

### Restore candidate cannot attach

The example StatefulSet uses `volumeClaimTemplates`; ordinal zero remains bound
to its original claim. Use a rehearsed CSI/operator cutover or a manifest with
an explicit existing-claim setting. Retain the original PVC and snapshot until
semantic V3/gRPC validation passes.

## Evidence to collect

For an unresolved incident, preserve:

- exact binary/image digest and source commit;
- sanitized environment names and resolved state paths;
- server logs around the first failure;
- filesystem type, free space, and mount options;
- stopped whole-state copy or volume snapshot; and
- the exact request shape and HTTP/gRPC status without credentials.
