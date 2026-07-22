# Migrating from Chroma to DeepData RC

This guide covers a dense-vector migration into the canonical tenant-aware V3
API. DeepData does not embed text in the release candidate: export Chroma's
stored embeddings, or generate vectors with your chosen embedding pipeline
before import.

## Contract differences to decide first

| Chroma concept | DeepData RC |
|---|---|
| Collection | Tenant collection |
| String document ID | Positive `uint64`; preserve the source ID in metadata |
| Document text | No dedicated text field; store it in metadata or your source-of-truth system |
| Embedding function | Caller-owned; pass vectors on insert and search |
| Add | Insert or atomic batch insert |
| Update/upsert | Not supported |
| Get/fetch document by ID | Not supported |
| List/scan documents | Not supported |
| Delete by ID | Delete one numeric document ID |
| Metadata filters | Available, but not a one-to-one translation of every Chroma operator |

Tenant and collection names must contain 1–64 ASCII letters, digits, hyphens,
or underscores. Create and retain an explicit name map when source names do not
meet that rule.

Check the Chroma collection's distance configuration before import. DeepData
HNSW is cosine; Flat supports cosine or Euclidean. Inner-product rankings are
not a direct match unless the vector normalization and expected ordering are
validated explicitly.

## 1. Freeze writes and export

The example assigns stable positive DeepData IDs in export order and preserves
the Chroma ID as `_chroma_id`. Keep the generated JSONL as the migration
manifest. For a large collection, page with `limit` and `offset` as shown
instead of loading the collection at once.

```python
import json
from pathlib import Path

import chromadb

PAGE_SIZE = 500
COLLECTION_MAP = {
    "source-docs": "source_docs",
}

client = chromadb.PersistentClient(path="./chroma-data")

for listed in client.list_collections():
    source_name = listed.name if hasattr(listed, "name") else str(listed)
    target_name = COLLECTION_MAP[source_name]
    collection = client.get_collection(source_name)
    output = Path(f"chroma-{target_name}.jsonl")

    offset = 0
    deepdata_id = 1
    with output.open("w", encoding="utf-8") as handle:
        while True:
            page = collection.get(
                limit=PAGE_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = page.get("ids") or []
            if not ids:
                break

            documents = page.get("documents") or [None] * len(ids)
            metadatas = page.get("metadatas") or [None] * len(ids)
            embeddings = page.get("embeddings")
            if embeddings is None:
                raise RuntimeError(f"{source_name}: export did not return embeddings")

            for source_id, document, metadata, embedding in zip(
                ids, documents, metadatas, embeddings
            ):
                vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                target_metadata = dict(metadata or {})
                target_metadata["_chroma_id"] = str(source_id)
                if document is not None:
                    target_metadata["_document"] = document
                record = {
                    "id": deepdata_id,
                    "vector": vector,
                    "metadata": target_metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                deepdata_id += 1

            offset += len(ids)

    print(source_name, "->", target_name, deepdata_id - 1, "records")
```

Chroma client return shapes have varied across releases. Run this export
against a copy first and verify that every line has a non-empty vector of the
expected dimension.

## 2. Create an empty V3 collection and import

Use a fresh destination collection. Replaying this script into a populated
collection is not an upsert and is not a supported resume mechanism.

```python
import json
import os
from pathlib import Path

from deepdata import DeepDataClient

TENANT = "acme"
COLLECTION = "source_docs"
BATCH_SIZE = 500
path = Path(f"chroma-{COLLECTION}.jsonl")

records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
if not records:
    raise RuntimeError("empty collections need an explicitly configured dimension")
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

The credential used by this importer must have `admin` permission to create
the collection and `write` permission to insert. A configured static server
token supplies both; application JWTs should be narrower after provisioning.

V3 batches are all-or-nothing and contain at most 10,000 documents. Smaller
batches make a failed migration easier to diagnose.

## 3. Change application queries

Generate the query vector with the same model and preprocessing used for the
stored vectors, then pass it to V3:

```python
results = tenant.search(
    "source_docs",
    queries={"embedding": query_vector},
    top_k=10,
    filters={"source": {"$eq": "web"}},
)
```

Do not translate `query_texts` into text sent to DeepData; there is no
server-managed embedding endpoint in the RC.

## 4. Verify before cutover

- Compare the source count with `tenant.get_collection(...).collection.doc_count`.
- Search a fixed set of source query vectors and compare expected neighbors.
- Verify metadata/filter behavior with representative records.
- Retain the JSONL ID map because V3 has no fetch-by-ID or document scan.
- Keep source writes frozen until validation and application cutover finish.

If an application depends on Chroma update/upsert, fetch-by-ID, document scan,
or `where_document`, redesign that dependency or defer migration. Delete then
insert is two separate mutations and is not an atomic replacement operation.
