# Canonical collection API (release candidate)

This document describes the supported collection contract for the DeepData
release candidate. The production surface is deliberately small:

- persistent, single-node operation on Linux only;
- tenant-aware V3 HTTP plus the equivalent unary gRPC service;
- caller-supplied dense and sparse vectors;
- HNSW or Flat for dense fields and Inverted/BM25 for sparse fields;
- one-field search or two-field hybrid search; and
- five mutations: create/delete collection, insert, atomic batch insert, and
  delete document.

Historical V2/root routes and advanced source packages are not part of this
contract.

## Runtime and authentication

The server owns one data directory and holds a lifetime filesystem lock. A
second writer is rejected. Persistent startup on Windows and macOS is rejected
because their filesystem semantics are not release-qualified.

For production, configure either a static bearer token or JWT verification and
require authentication:

```bash
export API_TOKEN='replace-with-a-long-random-token'
export REQUIRE_AUTH=1
./deepdata serve
```

Send the credential on HTTP requests as:

```http
Authorization: Bearer replace-with-a-long-random-token
```

A static token is an administrative credential. JWTs can carry a `tenant_id`,
`read`, `write`, or `admin` permissions, and an optional collection allowlist.
See [the security guide](../../docs/security.md) for the permission mapping and
network boundary.

## HTTP routes

All canonical data routes are below `/v3/tenants/{tenant_id}`.

| Method | Path | Required permission | Operation |
|---|---|---|---|
| `GET` | `/v3/tenants/{tenant_id}` | `admin` | Tenant counters |
| `GET` | `/v3/tenants/{tenant_id}/collections` | `admin` | List collections |
| `POST` | `/v3/tenants/{tenant_id}/collections` | `admin` | Create collection |
| `GET` | `/v3/tenants/{tenant_id}/collections/{name}` | `read` | Get collection |
| `DELETE` | `/v3/tenants/{tenant_id}/collections/{name}` | `admin` | Delete collection |
| `POST` | `/v3/tenants/{tenant_id}/collections/{name}/docs` | `write` | Insert document |
| `POST` | `/v3/tenants/{tenant_id}/collections/{name}/docs/batch` | `write` | Atomic batch insert |
| `DELETE` | `/v3/tenants/{tenant_id}/collections/{name}/docs` | `write` | Delete document |
| `POST` | `/v3/tenants/{tenant_id}/collections/{name}/search` | `read` | Search |

Tenant and collection path identifiers must contain 1–64 ASCII letters,
digits, hyphens, or underscores. JSON request bodies are strict; unknown fields
are rejected.

The unauthenticated operational routes are `GET /livez`, `GET /healthz`,
`GET /readyz`, and `GET /metrics`. Restrict them at the network boundary.

### Create a collection

```http
POST /v3/tenants/acme/collections
Authorization: Bearer TOKEN
Content-Type: application/json

{
  "name": "products",
  "fields": [
    {
      "name": "embedding",
      "type": "dense",
      "dim": 3,
      "index": {
        "type": "hnsw",
        "params": {"m": 16, "ef_construction": 200}
      }
    },
    {
      "name": "keywords",
      "type": "sparse",
      "dim": 10000,
      "index": {
        "type": "inverted",
        "params": {"k1": 1.2, "b": 0.75}
      }
    }
  ],
  "description": "Dense and sparse product retrieval"
}
```

Supported field/index combinations are:

| Vector field | Index | Notes |
|---|---|---|
| Dense | `hnsw` | Approximate cosine search |
| Dense | `flat` | Exact cosine by default; `metric` may be `cosine` or `euclidean` |
| Sparse | `inverted` | Sparse/BM25-style scoring; optional `k1` and `b` |

IVF, DiskANN, binary vectors, quantization, and CUDA are rejected by the
canonical persistence boundary.

### Insert one document

```http
POST /v3/tenants/acme/collections/products/docs
Authorization: Bearer TOKEN
Content-Type: application/json

{
  "id": 1001,
  "vectors": {
    "embedding": [0.1, 0.2, 0.3],
    "keywords": {
      "indices": [4, 19],
      "values": [0.8, 0.4],
      "dim": 10000
    }
  },
  "metadata": {"category": "audio"}
}
```

An explicitly supplied ID must be a positive `uint64`. Omit `id` to have the
server assign one. V3 has no upsert or metadata-update mutation.

### Insert a batch

```http
POST /v3/tenants/acme/collections/products/docs/batch
Authorization: Bearer TOKEN
Content-Type: application/json

{
  "documents": [
    {
      "id": 1002,
      "vectors": {"embedding": [0.2, 0.1, 0.4]},
      "metadata": {"category": "audio"}
    },
    {
      "id": 1003,
      "vectors": {"embedding": [0.8, 0.1, 0.1]}
    }
  ]
}
```

A batch contains at most 10,000 documents and is all-or-nothing. There is no
partial-success or continue-on-error mode.

### Delete a document

```http
DELETE /v3/tenants/acme/collections/products/docs
Authorization: Bearer TOKEN
Content-Type: application/json

{"doc_id": 1003}
```

### Search

Dense search supplies the query vector for the named field:

```http
POST /v3/tenants/acme/collections/products/search
Authorization: Bearer TOKEN
Content-Type: application/json

{
  "queries": {"embedding": [0.1, 0.2, 0.3]},
  "top_k": 10,
  "ef_search": 128,
  "filters": {"category": {"$eq": "audio"}},
  "include_vectors": false
}
```

Two-field hybrid search requires explicit fusion parameters:

```json
{
  "queries": {
    "embedding": [0.1, 0.2, 0.3],
    "keywords": {"indices": [4, 19], "values": [0.8, 0.4], "dim": 10000}
  },
  "top_k": 10,
  "hybrid_params": {
    "strategy": "weighted",
    "weights": {"embedding": 0.8, "keywords": 0.2}
  }
}
```

Search accepts at most two query fields and `top_k` must be between 1 and
1,000. Supported fusion strategies are `rrf`, `weighted`, and `linear`.

## gRPC mirror

The canonical protobuf is
[`api/proto/deepdata/v3/deepdata.proto`](../../api/proto/deepdata/v3/deepdata.proto).
`deepdata.v3.DeepData` exposes exactly nine unary RPCs:

1. `GetTenantInfo`
2. `ListCollections`
3. `GetCollection`
4. `CreateCollection`
5. `DeleteCollection`
6. `Insert`
7. `BatchInsert`
8. `Search`
9. `DeleteDoc`

Pass the same bearer credential in gRPC `authorization` metadata. The gRPC
methods use the same tenant manager, authorization decisions, validation, and
durable mutation journal as HTTP. There are no streaming RPCs in the RC.

## Durability behavior

Each accepted mutation is validated, appended and synchronized to the journal,
and then applied before success is returned. Startup replays complete journal
records after the latest checksummed snapshot. A terminal EOF-short frame in
the active journal is treated as an unacknowledged torn append: recovery
truncates to the last fully verified frame, synchronizes that repair, and
reparses before accepting traffic. Frozen or otherwise malformed/corrupt
records still fail closed. A journal/apply fault latches
the store unhealthy: canonical HTTP mutations and reads fail closed and
`/readyz` returns `503` until an operator restarts after addressing the cause.

Graceful shutdown checkpoints the store after handlers drain. Crash recovery
does not depend on graceful shutdown.

## Explicitly outside the RC

- V2 and root writes, rename, metadata mutation, upsert, bulk-specialized
  imports, document fetch/scan, and destructive “drop all” operations;
- server-managed embeddings or provider hot-swapping;
- GraphRAG, extraction, recommendation, discovery, and feedback APIs;
- replication, clustering, follower restore, and snapshot streaming;
- IVF, DiskANN, binary/PQ quantization, and CUDA; and
- the web UI, desktop wrapper, and legacy broad SDK methods as supported
  production surfaces.
