# Why DeepData RC

DeepData's release candidate is a focused, self-hosted retrieval server for
teams that can operate one persistent Linux node and want a small contract they
can test end to end.

## What the RC is

- A Linux-only, single-node persistent service with one process holding the
  data-directory lock.
- A tenant-aware V3 HTTP API and eleven equivalent unary gRPC methods.
- Caller-supplied vectors, or texts for fields that bind an embedding when the
  server runs one (`DEEPDATA_EMBEDDER`; the default `none` leaves embedding to
  the application).
- Dense search with HNSW or exact Flat indexes.
- Sparse Inverted/BM25 search and explicit two-field hybrid fusion.
- Six durable mutations: create/delete collection, insert, atomic batch
  insert, delete document, and upsert by caller-supplied ID.
- Static bearer-token or tenant-scoped JWT authentication.
- Readiness that fails closed when the checksummed snapshot, mutation journal,
  or lifetime lock is unhealthy.

The deliberate constraint is the product: HTTP and gRPC share one tenant
manager, authorization model, and append-before-apply durability boundary.

## When it is a good fit

| Need | Fit |
|---|---|
| Self-hosted vector retrieval on one Linux node | Good |
| Application already owns embedding generation | Good |
| Dense, sparse, or two-field hybrid retrieval | Good |
| Tenant and collection isolation with static/JWT auth | Good |
| A narrow API that can be crash-tested and restored as a unit | Good |

## When it is not a good fit

- A managed service or a control plane that hides infrastructure operations.
- Built-in replication, automatic failover, clustering, or multi-node scale.
- Persistent Windows or macOS deployment.
- Switching the embedder at runtime: one per process, chosen at startup.
- GraphRAG, extraction, recommendations, discovery, or feedback loops.
- DiskANN, IVF, binary/PQ quantization, or CUDA acceleration.
- Rename, partial metadata update, document scan, or “drop all.” Upsert and
  single-document fetch by ID are in the contract (a99fe53); the remaining
  omissions keep the surface small enough to crash-test and restore as a unit.
- A supported web dashboard, desktop wrapper, or broad multi-language SDK
  surface.

## Architecture at a glance

```text
caller embedding / sparse pipeline
              |
              v
   V3 HTTP :8080   gRPC :50051
              \     /
       tenant authorization
              |
       canonical collection engine
       HNSW / Flat / Inverted
              |
   checksummed snapshot + strict journal
              |
       one locked Linux data directory
```

TLS termination and disk encryption belong outside the process, at the reverse
proxy/service-mesh and filesystem/volume layers. That boundary keeps transport
and key management explicit instead of implying unqualified built-in security
features.

## Release proof

A credible deployment should prove more than a successful query:

1. HTTP and gRPC enforce the same tenant and collection scopes.
2. An acknowledged insert survives process termination and restart.
3. Corrupt or incompatible persistence causes startup/readiness failure rather
   than silent reinitialization.
4. A stopped backup can be restored and passes application-specific semantic
   queries before traffic is admitted.
5. Unsupported routes and index types remain unavailable.

See the [installation guide](installation.md), [security guide](security.md),
and [canonical collection API](../internal/collection/API.md) for the bounded
operational contract.
