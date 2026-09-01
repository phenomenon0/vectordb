# Canonical collection API (release candidate)

This document describes the supported collection contract for the DeepData
release candidate. The production surface is deliberately small:

- persistent, single-node operation on Linux only;
- tenant-aware V3 HTTP plus the equivalent unary gRPC service;
- caller-supplied dense and sparse vectors;
- HNSW or Flat for dense fields and Inverted/BM25 for sparse fields;
- one-field search or two-field hybrid search, plus single-document fetch by
  caller-supplied ID; and
- six mutations: create collection, delete collection, insert, atomic batch
  insert, delete document, and upsert (`CreateCollection`, `DeleteCollection`,
  `Insert`, `BatchInsert`, `DeleteDoc`, `Upsert`).

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
| `PUT` | `/v3/tenants/{tenant_id}/collections/{name}/docs/{doc_id}` | `write` | Upsert document |
| `GET` | `/v3/tenants/{tenant_id}/collections/{name}/docs/{doc_id}` | `read` | Get document |
| `POST` | `/v3/tenants/{tenant_id}/collections/{name}/search` | `read` | Search |

Tenant and collection path identifiers must contain 1–64 ASCII letters,
digits, hyphens, or underscores. JSON request bodies are strict; unknown fields
are rejected.

`GET /livez`, `GET /healthz`, and `GET /readyz` are unauthenticated probes.
`GET /metrics` sits behind the same authentication guard as the API routes
when authentication is required (cmd/deepdata/server.go:1503-1507,
`TestMetricsEndpointRequiresAuthWhenEnabled`). Restrict all four at the
network boundary.

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
server assign one. To replace a document under a caller-supplied ID, send
`PUT .../docs/{doc_id}` with a body of `vectors` and optional `metadata` (the ID
comes from the path and must be a non-zero `uint64`); `GET .../docs/{doc_id}`
returns the stored `vectors` and `metadata`, or `404` when absent. There is no
partial metadata-update mutation.

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

Three optional agent-retrieval request fields (bde4f94) refine a search:

- `score_floor` — confidence filter on raw scores in the field's metric
  direction: a maximum distance on dense fields, a minimum score on sparse
  and fused scores; `0` disables it (internal/collection/types.go:385-393).
- `fallback` — `{"primary": "...", "secondary": "...", "threshold": 0.0}`
  searches the secondary field when the primary yields no confident hit
  (zero hits, or best score worse than `threshold` when set). Both named
  fields must be present in `queries` and differ; `fallback` is mutually
  exclusive with `hybrid_params` (types.go:395-398, 422-426).
- `usage_boost` — blends tenant usage frecency into the ordering; `0`
  disables it and values `>= 1` are rejected. Reported scores stay raw
  (types.go:400-404; collection.go:533).

The response adds `best_score` (best raw score among returned hits, `0` when
none), `weak_match` (`true` when `score_floor` is set and nothing survived
it), and `fell_back_to` (the secondary field name when the ladder fired,
omitted otherwise) — types.go:466-478 and the HTTP struct in
cmd/deepdata/collection_http.go:357-367.

## Errors

Every HTTP and gRPC error is an `internal/apierror` value (the envelope struct
at `internal/apierror/apierror.go:43-53`, the code table at :63-75). Engine
errors reach it through `FromEngine` (:92), one `errors.Is` table over the
sentinels in limits.go (`ErrInvalidArgument`, `ErrInvalidSearchArgument`,
`ErrCollectionNotFound`, `ErrDocumentNotFound`, `ErrCollectionExists`,
`ErrDocumentExists`, `ErrTenantLimitExceeded`, `ErrCollectionLimitExceeded`);
an unrecognised error is `internal`. HTTP writes the envelope as JSON with the
table's status and sets `Retry-After` when `retry_after_ms` is set
(`WriteHTTP`, :129). gRPC returns the table's status code and message with an
`ErrorInfo` detail (`reason` = code, `domain` = `deepdata`, metadata `hint`,
`field`, `request_id`, `docs`) and a `RetryInfo` detail when retryable
(:147). `request_id` is the caller's `X-Request-ID` header or `x-request-id`
metadata, else one the server mints; both transports echo it back
(cmd/deepdata/server.go:3375, cmd/deepdata/main.go:3950-3965).

```json
{
  "code": "not_found",
  "message": "collection not found: missing for tenant acme",
  "hint": "list the tenant's collections to see what exists; document ids are the ones you inserted",
  "request_id": "agent-req-7",
  "retryable": false,
  "docs": "internal/collection/API.md#errors"
}
```

`field` names the offending request field when known and is omitted otherwise.

| code | HTTP | gRPC | retryable | when |
|---|---|---|---|---|
| `invalid_argument` | 400 | `InvalidArgument` | no | malformed body or identifier; `top_k`, `ef_search`, `score_floor`, `usage_boost`, `hybrid_params`/`fallback` shape out of range |
| `not_found` | 404 | `NotFound` | no | unknown route, collection or document |
| `already_exists` | 409 | `AlreadyExists` | no | create of an existing collection; insert of an existing document id (`PUT` upserts instead) |
| `unauthenticated` | 401 | `Unauthenticated` | no | missing or invalid credential |
| `permission_denied` | 403 | `PermissionDenied` | no | the token lacks the permission or collection scope |
| `quota_exceeded` | 409 | `FailedPrecondition` | no | tenant or collection limit; fixed for the process lifetime, so retrying cannot help |
| `payload_too_large` | 413 | `ResourceExhausted` | no | request or response above the size limits |
| `rate_limited` | 429 | `ResourceExhausted` | yes; `retry_after_ms` 1000, `Retry-After: 1` | per-tenant or authentication-failure limiter |
| `unavailable` | 503 | `Unavailable` | yes | persistence fault (see Durability behavior) or shutdown |
| `method_not_allowed` | 405 | `Unimplemented` | no | known path, wrong method |
| `internal` | 500 | `Internal` | no | unexpected fault; report the `request_id` |

Tests: `cmd/deepdata/apierror_transport_test.go` (HTTP envelope, request id
echo, 429 `Retry-After`, gRPC `ErrorInfo`/`RetryInfo`, interceptor plumbing),
`internal/apierror/apierror_test.go` (the table), and
TestSearchErrorsAreTypedSentinels in `agent_retrieval_test.go` (engine sentinels).

## gRPC mirror

The canonical protobuf is
[`api/proto/deepdata/v3/deepdata.proto`](../../api/proto/deepdata/v3/deepdata.proto).

<!-- generated:grpc-rpcs -->
`deepdata.v3.DeepData` exposes 11 unary RPCs: `GetTenantInfo`, `ListCollections`, `GetCollection`, `CreateCollection`, `DeleteCollection`, `Insert`, `BatchInsert`, `Search`, `DeleteDoc`, `Upsert`, `GetDoc`.
<!-- /generated -->

HTTP route table — generated once the routes subcommand exists (gate DOC-03):
<!-- generated:http-routes -->
<!-- /generated -->

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

- V2 and root writes, rename, metadata mutation, bulk-specialized imports,
  document scan, and destructive “drop all” operations;
- server-managed embeddings or provider hot-swapping;
- GraphRAG, extraction, recommendation, discovery, and feedback APIs;
- replication, clustering, follower restore, and snapshot streaming;
- IVF, DiskANN, binary/PQ quantization, and CUDA; and
- the web UI, desktop wrapper, and legacy broad SDK methods as supported
  production surfaces.
