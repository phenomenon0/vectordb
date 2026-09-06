# Canonical collection API (release candidate)

This document describes the supported collection contract for the DeepData
release candidate. The production surface is deliberately small:

- persistent, single-node operation on Linux and macOS (`store_lock_unix.go`,
  build tag `linux || darwin`); Linux amd64 is the only gated release target;
- tenant-aware V3 HTTP plus the equivalent unary gRPC service;
- caller-supplied dense and sparse vectors, or `texts` for fields that bind an
  `embedding`, embedded server-side by the process embedder (`DEEPDATA_EMBEDDER`);
- HNSW or Flat for dense fields and Inverted/BM25 for sparse fields;
- one-field search or two-field hybrid search, plus single-document fetch by
  caller-supplied ID; and
- nine mutations: create collection, delete collection, insert, atomic batch
  insert, delete document, upsert, create tenant, update tenant, and delete
  tenant (`CreateCollection`, `DeleteCollection`, `Insert`, `BatchInsert`,
  `DeleteDoc`, `Upsert`, `CreateTenant`, `UpdateTenant`, `DeleteTenant`).

Historical V2/root routes and advanced source packages are not part of this
contract.

## Runtime and authentication

The server owns one data directory and holds a lifetime filesystem lock. A
second writer is rejected. Persistent startup is rejected on Windows and every
other non-POSIX target (`store_lock_other.go`): only `linux || darwin` build a
store lock. macOS runs the same `flock(2)` implementation as Linux, so it starts
persistently; it is supported code on a platform that is not a gated release
target, which is not the same thing as a qualified deployment.

Inside that directory the store keeps one file prefix and writes its artifacts
beside it: `<prefix>.journal` and `<prefix>.snapshot` hold canonical state
(durability class A, corruption fails the store closed), and
`<prefix>.usage.json` holds the per-collection `UsageTracker` records that
`usage_boost` reads (durability class B). The sidecar is rewritten atomically
with every snapshot and on graceful close, and read once at open. It is a
ranking hint, not data: no sidecar is a normal open, and one that is
unreadable, corrupt, or of an unknown version is logged as an error and
discarded whole — the collection stays up answering by similarity with an
empty tracker, and no fault is latched. See the durability classes in
[the architecture map](../../docs/ARCHITECTURE.md#durability-classes-journal-52).

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

All canonical data routes are below `/v3/tenants/{tenant}`. The table below is
rendered from [`api/contract/v3/operations.json`](../../api/contract/v3/operations.json)
by `go run ./cmd/deepdata routes`; the same file answers `GET /v3/status` and is
checked against the dispatcher and the proto service by
`cmd/deepdata/contract_test.go`.

<!-- generated:http-routes -->
| method | path | permission | grpc_rpc |
|---|---|---|---|
| POST | /v3/tenants | admin | CreateTenant |
| GET | /v3/tenants | admin | ListTenants |
| GET | /v3/tenants/{tenant} | admin | GetTenantInfo |
| PUT | /v3/tenants/{tenant} | admin | UpdateTenant |
| DELETE | /v3/tenants/{tenant} | admin | DeleteTenant |
| GET | /v3/tenants/{tenant}/collections | read | ListCollections |
| POST | /v3/tenants/{tenant}/collections | admin | CreateCollection |
| GET | /v3/tenants/{tenant}/collections/{collection} | read | GetCollection |
| DELETE | /v3/tenants/{tenant}/collections/{collection} | admin | DeleteCollection |
| POST | /v3/tenants/{tenant}/collections/{collection}/docs | write | Insert |
| DELETE | /v3/tenants/{tenant}/collections/{collection}/docs | write | DeleteDoc |
| POST | /v3/tenants/{tenant}/collections/{collection}/docs/batch | write | BatchInsert |
| PUT | /v3/tenants/{tenant}/collections/{collection}/docs/{doc_id} | write | Upsert |
| GET | /v3/tenants/{tenant}/collections/{collection}/docs/{doc_id} | read | GetDoc |
| POST | /v3/tenants/{tenant}/collections/{collection}/search | read | Search |
| GET | /v3/status | read |  |
<!-- /generated -->

`GET /v3/status` is the server's self-description: build version, the operation
list above, the embedder it will use, its limits, its capabilities and its
accreted-signal state (cmd/deepdata/status.go:41). `signals.usage.loaded` is
always present and is false only when a usage sidecar existed and was
discarded, so an operator can tell "these ranking hints were lost" from "this
store never had any".

Tenant and collection path identifiers must contain 1–64 ASCII letters,
digits, hyphens, or underscores. JSON request bodies are strict; unknown fields
are rejected.

`GET /livez`, `GET /healthz`, and `GET /readyz` are unauthenticated probes.
`GET /metrics` exposes every tenant's usage, so it requires the
server-administrator credential, not just any authenticated caller
(cmd/deepdata/server.go:1320-1324, `TestMetricsRequiresServerAdmin`). Restrict
all four at the network boundary.

### Tenant lifecycle

`POST /v3/tenants`, `GET /v3/tenants`, `PUT /v3/tenants/{tenant}`, and
`DELETE /v3/tenants/{tenant}` provision, list, update and remove tenant
records. All four require the `server_admin` claim (or the static server
token, which is always server-admin); a tenant-scoped `admin` JWT gets
`403 permission_denied` even for its own tenant, since tenant lifecycle
crosses tenant boundaries by nature.

`PUT` upserts: updating a tenant that does not exist creates it, so replay
after a restart needs only this one path. `status` is `active` or
`suspended`; a suspended tenant's data-plane routes (collections, documents,
search) return `403 permission_denied` with a hint pointing back at
`PUT /v3/tenants/{tenant}` to reactivate. Quota fields (`max_documents`,
`max_bytes`, `max_collections`) are administrator-set ceilings — a zero field
means server default, the `MAX_TENANT_*` environment values. `DELETE` removes
the tenant record and every collection it owns; deleting an unknown tenant is
`404 not_found`.

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

A field may bind an `embedding` so requests can send `texts` for it instead of
vectors. Dense fields bind the server's embedder (`{"provider": "ollama",
"model": "nomic-embed-text"}`; `model` and `dim` may be omitted and are filled
from the server). Sparse fields bind `{"provider": "bm25"}`, a deterministic
term hash (`TextToSparse`, `internal/collection/migration.go:211`) that needs no
embedder. The binding is journaled with the schema (`types.go:207`). A dense
binding whose provider or model differs from the server's `DEEPDATA_EMBEDDER`
is `409 embedding_mismatch`; a binding on a server without an embedder is
`503 embedder_unavailable`; a `dim` that disagrees with the embedder is
`400 invalid_argument` (`cmd/deepdata/embed_text.go:101`). `GET` on the
collection returns the resolved `dim` and `embedding`.

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

For bound fields send `texts` (field name → text) instead of `vectors`; every
schema field must appear in exactly one of the two. Had `embedding` bound to the
server's embedder and `keywords` to `bm25`, the same document could be sent as:

```json
{"id": 1001, "texts": {"embedding": "wireless studio headphones", "keywords": "wireless studio headphones"}, "metadata": {"text": "wireless studio headphones"}}
```

The server stores text only where told: put it in `metadata` to read it back.
A field in both `texts` and `vectors` is `400 invalid_argument` with
`field: "texts.<name>"`; `texts` for an unbound field is `400` with the hint to
bind an embedding or send a vector (`cmd/deepdata/embed_text.go:132`). Upsert
and batch documents take `texts` the same way.

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

`texts` (field name → query text) replaces `queries` for bound fields and may
be mixed with it; together they name at most two fields. Each text is embedded
with the embedder's query path, and the response reports `embedded_by` (field →
`provider:model`, e.g. `{"embedding": "ollama:nomic-embed-text", "keywords": "bm25"}`),
omitted when every query was a vector.

Three optional agent-retrieval request fields (bde4f94) refine a search:

- `score_floor` — confidence filter on raw scores in the field's metric
  direction: a maximum distance on dense fields, a minimum score on sparse
  and fused scores; `0` disables it (internal/collection/types.go:474-482).
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
(cmd/deepdata/server.go:2504-2514, cmd/deepdata/main.go:3704-3717).

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
| `not_found` | 404 | `NotFound` | no | unknown route, tenant, collection or document |
| `already_exists` | 409 | `AlreadyExists` | no | create of an existing tenant or collection; insert of an existing document id (`PUT` upserts instead) |
| `unauthenticated` | 401 | `Unauthenticated` | no | missing or invalid credential |
| `permission_denied` | 403 | `PermissionDenied` | no | the token lacks the permission, collection scope, or `server_admin` claim a tenant lifecycle route requires; also a suspended tenant's data-plane writes, with a hint to reactivate via `PUT /v3/tenants/{tenant}` |
| `quota_exceeded` | 409 | `FailedPrecondition` | no | tenant or collection limit; fixed for the process lifetime, so retrying cannot help |
| `embedding_mismatch` | 409 | `FailedPrecondition` | no | the field binds an embedding provider or model other than the one this server runs (`embedder` in `/readyz`); send a vector or recreate the collection with the server's `provider:model` |
| `payload_too_large` | 413 | `ResourceExhausted` | no | request or response above the size limits |
| `rate_limited` | 429 | `ResourceExhausted` | yes; `retry_after_ms` 1000, `Retry-After: 1` | per-tenant or authentication-failure limiter |
| `unavailable` | 503 | `Unavailable` | yes | persistence fault (see Durability behavior) or shutdown |
| `embedder_unavailable` | 503 | `Unavailable` | yes | `texts` sent to a server started with `DEEPDATA_EMBEDDER=none`, or the embedder failed on the request; send vectors or start the server with an embedder |
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
`deepdata.v3.DeepData` exposes 15 unary RPCs: `GetTenantInfo`, `CreateTenant`, `ListTenants`, `UpdateTenant`, `DeleteTenant`, `ListCollections`, `GetCollection`, `CreateCollection`, `DeleteCollection`, `Insert`, `BatchInsert`, `Search`, `DeleteDoc`, `Upsert`, `GetDoc`.
<!-- /generated -->

Pass the same bearer credential in gRPC `authorization` metadata. The gRPC
methods use the same tenant manager, authorization decisions, validation, and
durable mutation journal as HTTP. There are no streaming RPCs in the RC.
`texts` and `embedded_by` are the same-named map fields on the proto messages
(`InsertRequest.texts = 7`, batch documents `texts = 5`, `SearchRequest.texts = 12`,
`UpsertRequest.texts = 7`, `SearchResponse.embedded_by = 6`) and the binding is
`VectorFieldConfig.embedding = 6`.

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

Tenant records (status and quota) are journaled and snapshotted the same way
collection and document mutations are, so a tenant's lifecycle state survives
a restart and a journal replay exactly like a collection does.

## Explicitly outside the RC

- V2 and root writes, rename, metadata mutation, bulk-specialized imports,
  document scan, and destructive “drop all” operations;
- switching the embedder at runtime (one per process, chosen by `DEEPDATA_EMBEDDER`);
- GraphRAG, extraction, recommendation, discovery, and feedback APIs;
- replication, clustering, follower restore, and snapshot streaming;
- IVF, DiskANN, binary/PQ quantization, and CUDA; and
- the web UI, desktop wrapper, and legacy broad SDK methods as supported
  production surfaces.
