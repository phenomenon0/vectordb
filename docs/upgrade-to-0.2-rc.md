# Upgrade and migration to DeepData 0.2.0-rc.1

This candidate deliberately does not auto-upgrade the older root/V2 runtime
into the canonical tenant-aware store. An automatic guess could put documents
in the wrong tenant, lose index semantics, or overwrite recoverable data.

## Protocol changes

- Canonical HTTP uses only `/v3/tenants/{tenant_id}/...` routes.
- Canonical gRPC is `deepdata.v3.DeepData`. The historical `deepdata.v1`
  descriptor remains frozen for source/wire compatibility but its service is
  not registered by the RC server.
- Every request uses explicit tenant identity. Static bearer credentials are
  server administrators; scoped JWTs remain bounded by tenant, permission, and
  optional collection claims.
- Root/V2 writes, recommendation/discovery, server-managed embeddings, and
  advanced mutation paths are unavailable.

Regenerate gRPC clients from `api/proto/deepdata/v3/deepdata.proto` and replace
old message types rather than pointing a V1 stub at the new server.

## Existing data

Before starting the candidate, stop the old process and take a verified,
whole-root backup. Raw legacy `.manager`/`.tenants` artifacts and unified
snapshots containing V2 collections are refused before serving traffic; raw
files are left byte-for-byte unchanged and no migration marker/snapshot is
created.

Migration is therefore an explicit offline export/import operation:

1. Run the last compatible legacy binary against a copied state root.
2. Export collection schemas, stable document IDs, metadata, and every caller
   vector without modifying the original root.
3. Create explicit V3 tenants/collections using only HNSW, Flat, or inverted
   indexes and insert documents through the canonical mutations.
4. Stop and restart the RC server, then verify tenant/collection counts,
   representative IDs and metadata, dense/sparse/hybrid search, and an
   authenticated V3 gRPC read.
5. Retain both the original backup and the sanitized migration evidence until
   rollback is no longer required.

There is no in-place rollback after canonical writes. Roll back by stopping
the candidate and restoring the complete pre-migration root to the compatible
legacy binary. Never mix files from the two state roots.

## Migrating to per-tenant stores

The canonical server keeps one durable store per tenant under
`<data-dir>/index.gob.tenants/`. A data directory still on the earlier 0.1
single-store layout (`<data-dir>/index.gob.collections.*`) refuses to serve
and names this section in its error.

Run, once, against the data directory (not a copy -- it never rewrites the
old files):

```
deepdata migrate-tenants <data-dir>
```

It refuses with a one-line reason and does nothing if the directory is
already migrated, still holds raw pre-0.1 legacy artifacts, or has no
single-store state to migrate. Otherwise it exports every tenant into its own
store, reopens the result to verify collection names and document counts
match, and prints a report:

```
tenant  collections  documents
acme    1            2
beta    1            0            (1 ephemeral collection(s) reset to 0 docs)
```

Ephemeral collections (ADR 0009) are schema-only and always reset to zero
documents across a restart, migration included -- the report says so per
tenant rather than treating it as a mismatch. Per-tenant usage/quota counters
are not migrated either; they re-accrete from zero under the new layout.

Rollback is implicit: migrate-tenants never checkpoints or deletes the old
`index.gob.collections.*` prefix, so a failed or unwanted migration leaves it
exactly as it was. Start `deepdata serve` once you are satisfied, then delete
the old prefix whenever you like.

## Canonical journal compatibility

The candidate writes durable mutation envelope V2. It retains frozen replay
semantics for acknowledged canonical V1 journal records, including batches
that predate current request-admission limits. This compatibility applies to
the canonical collection journal only; it is not an implicit V2/root data
migration.
