# Migrating from Pinecone to DeepData RC

This guide covers a dense-vector migration from a Pinecone index into the
canonical tenant-aware V3 API. DeepData is self-hosted and single-node in this
release candidate; it is not a managed or highly available replacement.

## Contract differences to decide first

| Pinecone concept | DeepData RC |
|---|---|
| Index / namespace | Tenant collection; define an explicit namespace map |
| String vector ID | Positive `uint64`; preserve the source ID in metadata |
| Upsert | Not supported; import into an empty collection |
| Query vector | Caller-supplied query vector |
| Fetch by ID | Not supported |
| List/paginated scan | Not supported |
| Delete by ID | Delete one numeric document ID |
| Metadata filter | Supported subset; validate each application filter |
| Managed replication/HA | Not provided by the RC |

The destination supports HNSW cosine search and Flat cosine or Euclidean
search. A Pinecone dot-product index is not a direct semantic match. Normalize
stored and query vectors and validate rankings, or defer that migration.

## 1. Freeze writes and export values

Use Pinecone's list and fetch operations while the source is still available.
The example creates a positive DeepData ID per namespace and preserves the
original ID as `_pinecone_id`.

```python
import json
from pathlib import Path

from pinecone import Pinecone

pc = Pinecone(api_key="replace-me")
index = pc.Index("my-index")

NAMESPACE_MAP = {
    "": "default",
    "customer-a": "customer_a",
}

for namespace, collection in NAMESPACE_MAP.items():
    source_ids = []
    for page in index.list(namespace=namespace):
        source_ids.extend(page)

    output = Path(f"pinecone-{collection}.jsonl")
    deepdata_id = 1
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(source_ids), 100):
            fetched = index.fetch(
                ids=source_ids[start : start + 100],
                namespace=namespace,
            )
            for source_id in source_ids[start : start + 100]:
                source = fetched.vectors.get(source_id)
                if source is None:
                    raise RuntimeError(f"missing fetched vector {source_id}")
                metadata = dict(source.metadata or {})
                metadata["_pinecone_id"] = str(source_id)
                record = {
                    "id": deepdata_id,
                    "vector": list(source.values),
                    "metadata": metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                deepdata_id += 1

    print(namespace or "<default>", "->", collection, deepdata_id - 1, "records")
```

Pinecone SDK pagination shapes can differ by client version. Test the exporter
against a copy, confirm that no IDs are skipped, and validate every vector's
dimension before deleting any managed data.

## 2. Create and import a destination collection

This example uses HNSW for a cosine source index. For an Euclidean source,
choose `{"type": "flat", "params": {"metric": "euclidean"}}` and validate
the operational cost of exact search.

```python
import json
import os
from pathlib import Path

from deepdata import DeepDataClient

TENANT = "acme"
COLLECTION = "default"
BATCH_SIZE = 500
path = Path(f"pinecone-{COLLECTION}.jsonl")

records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
if not records:
    raise RuntimeError("empty namespaces need an explicitly configured dimension")
dimension = len(records[0]["vector"])
if any(len(record["vector"]) != dimension for record in records):
    raise RuntimeError("inconsistent vector dimensions")

with DeepDataClient(
    os.environ.get("DEEPDATA_URL", "http://localhost:8080"),
    api_token=os.environ["DEEPDATA_TOKEN"],
) as client:
    tenant = client.tenant(TENANT)
    tenant.create_collection(
        COLLECTION,
        fields=[{
            "name": "embedding",
            "type": "dense",
            "dim": dimension,
            "index": {"type": "hnsw"},
        }],
    )

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        tenant.batch_insert(
            COLLECTION,
            [{
                "id": record["id"],
                "vectors": {"embedding": record["vector"]},
                "metadata": record["metadata"],
            } for record in batch],
        )
```

Import into a new empty collection. V3 batch insert is atomic, but the entire
multi-batch migration is not one transaction and there is no upsert/resume
mode.

The importer credential needs `admin` permission to create the collection and
`write` permission to insert. Use narrower tenant JWTs for application traffic
after provisioning.

## 3. Change application queries

Continue generating query vectors in the application and pass them directly:

```python
results = tenant.search(
    "default",
    queries={"embedding": query_vector},
    top_k=10,
    filters={"source": {"$eq": "web"}},
)
```

Pinecone sparse values can be mapped only through an explicitly configured
DeepData sparse field using `{indices, values, dim}`. This guide does not claim
equivalent sparse scoring; validate it as a separate migration.

## 4. Verify before cutover

- Compare namespace counts with each destination collection's `doc_count`.
- Compare neighbors for a fixed corpus of query vectors and filters.
- Confirm score/ranking behavior for the source metric.
- Retain the JSONL ID map because DeepData has no fetch-by-ID or document scan.
- Exercise stopped backup and recovery for the Linux data directory.
- Keep Pinecone writes frozen until validation and application cutover finish.

If the application requires upsert, fetch, list, managed replication, or
online migration with concurrent writes, the current RC is not a compatible
target. Delete then insert is not an atomic replacement operation.
