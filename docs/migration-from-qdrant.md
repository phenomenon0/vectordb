# Migrating from Qdrant to DeepData V3

The release candidate accepts caller-supplied vectors through tenant-aware HTTP
V3 or the matching unary gRPC service. Migration therefore reads points from
Qdrant and writes their vectors and payloads into an explicitly created
DeepData collection.

## Mapping

| Qdrant | DeepData RC |
|---|---|
| Collection | Tenant collection |
| Point vector or named vectors | Dense vector fields |
| Sparse vector | Sparse field with inverted index |
| Payload | Document metadata |
| Integer or UUID point ID | Positive `uint64` document ID |
| Search | Dense, sparse, or two-field hybrid search |

DeepData document IDs are positive `uint64` values. The example below assigns
new sequential IDs and stores the original Qdrant ID in metadata, so integer,
UUID, and string source IDs are handled without collisions.

Run the example against a newly created, empty target collection. If it stops
after committing earlier batches, delete and recreate that target before
retrying so the sequential ID mapping remains deterministic.

## 1. Create the target schema

Choose a tenant and create fields that match the source vector names and
dimensions. Dense fields support `hnsw` or `flat`; sparse fields use
`inverted`.

```bash
DEEPDATA_URL=http://localhost:8080
DEEPDATA_TENANT=migration
DEEPDATA_TOKEN=replace-with-token

curl -fsS -X POST \
  "$DEEPDATA_URL/v3/tenants/$DEEPDATA_TENANT/collections" \
  -H "Authorization: Bearer $DEEPDATA_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"docs",
    "fields":[
      {"name":"embedding","type":"dense","dim":384,"index":{"type":"hnsw"}}
    ]
  }'
```

Do not change a vector dimension during transfer. If a new vector model is
required, generate the replacement vectors in the migration client and create
a separate collection with the new dimension.

## 2. Stream points into atomic batches

This example uses the Qdrant Python client and the DeepData HTTP contract. It
preserves payload values as metadata and sends source vectors unchanged.

```python
from __future__ import annotations

import os
from typing import Any

import requests
from qdrant_client import QdrantClient

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]
DEEPDATA_URL = os.environ.get("DEEPDATA_URL", "http://localhost:8080")
DEEPDATA_TENANT = os.environ["DEEPDATA_TENANT"]
DEEPDATA_COLLECTION = os.environ["DEEPDATA_COLLECTION"]
DEEPDATA_TOKEN = os.environ["DEEPDATA_TOKEN"]
BATCH_SIZE = 500

qdrant = QdrantClient(url=QDRANT_URL)
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {DEEPDATA_TOKEN}",
    "Content-Type": "application/json",
})
batch_url = (
    f"{DEEPDATA_URL}/v3/tenants/{DEEPDATA_TENANT}"
    f"/collections/{DEEPDATA_COLLECTION}/docs/batch"
)


def canonical_vectors(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {name: list(values) for name, values in raw.items()}
    return {"embedding": list(raw)}


def send(documents: list[dict[str, Any]]) -> None:
    response = session.post(
        batch_url,
        json={"documents": documents},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("inserted") != len(documents):
        raise RuntimeError(f"unexpected batch acknowledgement: {body}")


offset = None
next_id = 1
pending: list[dict[str, Any]] = []

while True:
    points, offset = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=BATCH_SIZE,
        offset=offset,
        with_payload=True,
        with_vectors=True,
    )
    for point in points:
        if point.vector is None:
            raise RuntimeError(f"point {point.id} has no vector")
        metadata = dict(point.payload or {})
        metadata["qdrant_id"] = str(point.id)
        pending.append({
            "id": next_id,
            "vectors": canonical_vectors(point.vector),
            "metadata": metadata,
        })
        next_id += 1

        if len(pending) == BATCH_SIZE:
            send(pending)
            pending = []

    if offset is None:
        break

if pending:
    send(pending)

print(f"migrated {next_id - 1} points")
```

For named vectors, every Qdrant name must match a DeepData field. If the source
contains a vector type the target schema does not declare, stop rather than
silently dropping it.

## 3. Verify before cutover

Run verification against both systems before moving traffic:

1. Compare source point count with DeepData tenant and collection counts.
2. Sample original IDs from the `qdrant_id` metadata mapping.
3. Run representative query vectors and inspect expected neighbors.
4. Restart DeepData and repeat the count and query checks.
5. Exercise the same collection through gRPC if that client surface will be
   used in production.

Keep Qdrant read-only and available for rollback until the migrated collection
passes these checks under the exact DeepData candidate build.
