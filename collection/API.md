# Multi-Vector Collection API (v2)

This document describes the new v2 API for multi-vector collections, which supports hybrid search with dense and sparse vectors.

## Overview

The v2 API enables:
- **Multi-vector collections**: Store multiple vector types (dense + sparse) per document
- **Hybrid search**: Combine dense semantic search with sparse keyword search
- **Flexible schema**: Define multiple vector fields with different index types per collection
- **RRF fusion**: Reciprocal Rank Fusion for combining search results

## API Endpoints

All v2 endpoints are prefixed with `/v2/`.

### Collection Management

#### Create Collection

```http
POST /v2/collections
Content-Type: application/json

{
  "name": "products",
  "fields": [
    {
      "name": "embedding",
      "type": 0,
      "dim": 384,
      "index": {
        "type": 0,
        "params": {
          "m": 16,
          "ef_construction": 200
        }
      }
    },
    {
      "name": "keywords",
      "type": 1,
      "dim": 10000,
      "index": {
        "type": 4,
        "params": {
          "k1": 1.2,
          "b": 0.75
        }
      }
    }
  ],
  "description": "Product catalog with semantic + keyword search"
}
```

**Vector Types:**
- `0`: Dense (embeddings)
- `1`: Sparse (BM25, SPLADE)
- `2`: Binary (future)

**Index Types:**
- `0`: HNSW (graph-based, best for dense vectors)
- `1`: IVF (clustering-based)
- `2`: FLAT (brute-force exact search)
- `3`: DiskANN (disk-backed hybrid)
- `4`: Inverted (best for sparse vectors)

**Response:**
```json
{
  "status": "success",
  "message": "collection \"products\" created"
}
```

#### List Collections

```http
GET /v2/collections
```

**Response:**
```json
{
  "status": "success",
  "count": 2,
  "collections": [
    {
      "name": "products",
      "fields": [...],
      "description": "Product catalog",
      "metadata": {},
      "doc_count": 1500
    }
  ]
}
```

#### Get Collection Info

```http
GET /v2/collections/{name}
```

**Response:**
```json
{
  "status": "success",
  "collection": {
    "name": "products",
    "fields": [
      {
        "name": "embedding",
        "type": 0,
        "dim": 384,
        "index": {
          "type": 0,
          "params": {"m": 16, "ef_construction": 200}
        }
      },
      {
        "name": "keywords",
        "type": 1,
        "dim": 10000,
        "index": {
          "type": 4,
          "params": {"k1": 1.2, "b": 0.75}
        }
      }
    ],
    "description": "Product catalog",
    "doc_count": 1500
  }
}
```

#### Get Collection Statistics

```http
GET /v2/collections/{name}/stats
```

**Response:**
```json
{
  "status": "success",
  "name": "products",
  "doc_count": 1500,
  "manager_stats": {
    "collection_count": 3,
    "total_documents": 4200,
    "collections": {
      "products": {
        "name": "products",
        "doc_count": 1500,
        "field_count": 2
      }
    }
  }
}
```

#### Delete Collection

```http
DELETE /v2/collections/{name}
```

**Response:**
```json
{
  "status": "success",
  "message": "collection \"products\" deleted"
}
```

### Document Operations

#### Insert Document

```http
POST /v2/insert
Content-Type: application/json

{
  "collection": "products",
  "doc": "High-quality wireless headphones with noise cancellation",
  "vectors": {
    "embedding": [0.1, 0.2, 0.3, ...],  // Dense vector (384 dims)
    "keywords": {
      "indices": [42, 157, 389, 1024, 2056],
      "values": [2.5, 1.8, 3.2, 1.0, 2.1],
      "dim": 10000
    }
  },
  "metadata": {
    "category": "electronics",
    "price": "299.99",
    "brand": "AudioTech"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "id": 42,
  "message": "document added"
}
```

#### Search (Dense-only)

```http
POST /v2/search
Content-Type: application/json

{
  "collection": "products",
  "queries": {
    "embedding": [0.15, 0.22, 0.31, ...]
  },
  "top_k": 10
}
```

**Response:**
```json
{
  "status": "success",
  "documents": [
    {
      "id": 42,
      "vectors": {...},
      "metadata": {
        "category": "electronics",
        "price": "299.99"
      }
    }
  ],
  "scores": [0.95, 0.89, 0.87, ...],
  "candidates_examined": 120
}
```

#### Hybrid Search (Dense + Sparse with RRF)

```http
POST /v2/search
Content-Type: application/json

{
  "collection": "products",
  "queries": {
    "embedding": [0.15, 0.22, 0.31, ...],  // Semantic query
    "keywords": {                           // Keyword query
      "indices": [42, 157],
      "values": [1.0, 0.8],
      "dim": 10000
    }
  },
  "top_k": 10,
  "hybrid_params": {
    "strategy": "rrf",
    "weights": {
      "dense": 0.7,
      "sparse": 0.3
    },
    "rrf_constant": 60.0
  }
}
```

**Fusion Strategies:**
- `"rrf"`: Reciprocal Rank Fusion (default, parameter-free)
- `"weighted"`: Weighted sum of scores
- `"linear"`: Linear combination

**Response:**
```json
{
  "status": "success",
  "documents": [...],
  "scores": [0.98, 0.93, 0.89, ...],
  "candidates_examined": 240
}
```

#### Delete Document

```http
POST /v2/delete
Content-Type: application/json

{
  "collection": "products",
  "doc_id": 42
}
```

**Response:**
```json
{
  "status": "success",
  "message": "document 42 deleted from collection products"
}
```

## Usage Examples

### Python Example

```python
import requests
import numpy as np

API_URL = "http://localhost:8080"

# 1. Create collection
schema = {
    "name": "articles",
    "fields": [
        {
            "name": "embedding",
            "type": 0,  # Dense
            "dim": 768,
            "index": {
                "type": 0,  # HNSW
                "params": {"m": 16, "ef_construction": 200}
            }
        },
        {
            "name": "keywords",
            "type": 1,  # Sparse
            "dim": 50000,
            "index": {
                "type": 4,  # Inverted
                "params": {"k1": 1.2, "b": 0.75}
            }
        }
    ],
    "description": "News articles with hybrid search"
}

resp = requests.post(f"{API_URL}/v2/collections", json=schema)
print(f"Collection created: {resp.json()}")

# 2. Insert document with dense + sparse vectors
doc = {
    "collection": "articles",
    "doc": "AI breakthrough in natural language processing",
    "vectors": {
        "embedding": np.random.randn(768).tolist(),  # Dense embedding
        "keywords": {
            "indices": [42, 157, 389, 1024],
            "values": [2.5, 1.8, 3.2, 1.0],
            "dim": 50000
        }
    },
    "metadata": {
        "category": "technology",
        "published": "2025-12-10"
    }
}

resp = requests.post(f"{API_URL}/v2/insert", json=doc)
print(f"Document added: {resp.json()}")

# 3. Hybrid search
search_req = {
    "collection": "articles",
    "queries": {
        "embedding": np.random.randn(768).tolist(),  # Semantic query
        "keywords": {
            "indices": [42, 157],
            "values": [1.0, 0.8],
            "dim": 50000
        }
    },
    "top_k": 5,
    "hybrid_params": {
        "strategy": "rrf",
        "weights": {"dense": 0.7, "sparse": 0.3},
        "rrf_constant": 60.0
    }
}

resp = requests.post(f"{API_URL}/v2/search", json=search_req)
results = resp.json()
print(f"Found {len(results['documents'])} results")
for doc, score in zip(results['documents'], results['scores']):
    print(f"  Score {score:.3f}: {doc['metadata']}")
```

### cURL Examples

**Create collection:**
```bash
curl -X POST http://localhost:8080/v2/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "fields": [
      {
        "name": "embedding",
        "type": 0,
        "dim": 384,
        "index": {"type": 0, "params": {"m": 16}}
      }
    ]
  }'
```

**Insert document:**
```bash
curl -X POST http://localhost:8080/v2/insert \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "products",
    "doc": "Wireless headphones",
    "vectors": {
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  }'
```

**Search:**
```bash
curl -X POST http://localhost:8080/v2/search \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "products",
    "queries": {
      "embedding": [0.15, 0.22, ...]
    },
    "top_k": 5
  }'
```

## Migration from v1 to v2

**Key Differences:**

| Feature | v1 (Legacy) | v2 (Multi-Vector) |
|---------|-------------|-------------------|
| Endpoint prefix | `/admin/collection/` | `/v2/collections` |
| Collection schema | Single index type | Multiple vector fields |
| Vectors per document | One | Multiple (dense + sparse) |
| Hybrid search | Not supported | Full support with RRF |
| Index types | HNSW, IVF, FLAT | + Inverted for sparse |

**Migration Steps:**

1. Create new v2 collection with multi-vector schema
2. Export documents from v1 collection
3. Re-encode documents with sparse vectors (e.g., BM25, SPLADE)
4. Insert into v2 collection with both dense and sparse vectors
5. Test hybrid search performance
6. Switch application to use v2 endpoints
7. Delete old v1 collection

## Performance Considerations

- **Dense-only search**: Similar performance to v1 (HNSW index)
- **Sparse-only search**: Fast with inverted index (BM25 scoring)
- **Hybrid search**: ~2x candidates examined, but much better relevance
- **Memory usage**: +5-10% per collection due to multiple indexes
- **SJSON encoding**: 48% smaller for dense, 94% smaller for sparse vectors

## Best Practices

1. **Use hybrid search for production RAG**: Combines semantic understanding with exact keyword matching
2. **Tune fusion weights**: Start with `dense: 0.7, sparse: 0.3`, adjust based on evaluation
3. **Index configuration**:
   - Dense: `m=16, ef_construction=200` for good recall/speed balance
   - Sparse: `k1=1.2, b=0.75` (standard BM25 parameters)
4. **Sparse vector generation**: Use BM25 tokenization or SPLADE model
5. **Field naming**: Use descriptive names like "embedding", "keywords", "title_sparse"

## Troubleshooting

**"collection already exists"**: Use GET /v2/collections to list existing collections

**"dimension mismatch"**: Verify vector dimensions match schema definition

**"invalid sparse vector format"**: Ensure sparse vectors have `indices`, `values`, and `dim` fields

**"hybrid search requires 2 fields"**: Current implementation requires exactly 1 dense + 1 sparse field

## References

- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [SJSON Format](../storage/README.md)
