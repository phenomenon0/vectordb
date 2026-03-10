"""Phase 1b: Embed chunks and generate queries using OpenAI text-embedding-3-small.

Reads collection JSONL files (without embeddings), embeds all chunks,
generates 50 realistic code-search queries, and saves everything.
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent / "data"

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100  # OpenAI batch limit for embeddings

# 50 realistic code-search queries for a vector database codebase
QUERIES = [
    "HNSW index search algorithm implementation",
    "how does batch insert work",
    "cosine similarity distance calculation",
    "rate limiting middleware",
    "JWT authentication token validation",
    "collection management create delete",
    "metadata filtering during search",
    "sparse vector insertion",
    "knowledge graph entity extraction",
    "hybrid search combining dense and sparse",
    "RRF reciprocal rank fusion",
    "WAL write ahead log persistence",
    "index compaction and garbage collection",
    "memory mapped file storage",
    "vector quantization product quantization",
    "concurrent read write locking",
    "HTTP server endpoint handler",
    "reranking search results",
    "BM25 text scoring",
    "multi-tenant isolation",
    "collection schema field types",
    "scroll pagination cursor",
    "health check endpoint",
    "metrics prometheus instrumentation",
    "embedding dimension validation",
    "nearest neighbor graph construction",
    "delete document by ID",
    "backup and restore snapshot",
    "error handling retry logic",
    "connection pooling HTTP client",
    "Go test benchmark performance",
    "HNSW ef construction parameter tuning",
    "index build progress tracking",
    "vector normalization L2 norm",
    "upsert document update existing",
    "Docker container deployment configuration",
    "API versioning v2 endpoints",
    "document metadata key value pairs",
    "search result scoring and ranking",
    "file import obsidian markdown",
    "CLI command line interface arguments",
    "security input sanitization",
    "telemetry usage tracking",
    "graph weighted fusion scoring",
    "flat brute force exact search",
    "batch concurrent processing goroutines",
    "config environment variable parsing",
    "sparse index inverted posting list",
    "collection statistics count size",
    "web UI frontend serving static files",
]


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file")
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(input=batch, model=MODEL)
        # Sort by index to preserve order
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])
    return all_embeddings


def process_collection(client: OpenAI, input_path: Path, output_path: Path) -> int:
    """Read chunks, embed them, write back with embeddings."""
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print(f"  No records in {input_path.name}, skipping")
        return 0

    texts = [r["text"] for r in records]
    print(f"  Embedding {len(texts)} chunks from {input_path.name}...")
    embeddings = embed_texts(client, texts)

    with open(output_path, "w") as f:
        for rec, emb in zip(records, embeddings):
            rec["embedding"] = emb
            f.write(json.dumps(rec) + "\n")

    print(f"  Wrote {len(records)} embedded chunks to {output_path.name}")
    return len(records)


def generate_queries(client: OpenAI, output_path: Path) -> int:
    """Embed query strings and save to JSONL."""
    print(f"  Embedding {len(QUERIES)} queries...")
    embeddings = embed_texts(client, QUERIES)

    with open(output_path, "w") as f:
        for i, (query, emb) in enumerate(zip(QUERIES, embeddings)):
            rec = {
                "id": f"q_{i:03d}",
                "text": query,
                "embedding": emb,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"  Wrote {len(QUERIES)} queries to {output_path.name}")
    return len(QUERIES)


def main():
    print("Phase 1b: Embedding data with OpenAI text-embedding-3-small")

    client = get_client()

    # Check that prepare_data.py has been run
    coll_a_input = DATA_DIR / "collection_a.jsonl"
    coll_b_input = DATA_DIR / "collection_b.jsonl"

    if not coll_a_input.exists():
        raise FileNotFoundError(
            f"{coll_a_input} not found. Run prepare_data.py first."
        )

    # We overwrite in-place (adding embedding field)
    n_a = process_collection(client, coll_a_input, coll_a_input)
    n_b = process_collection(client, coll_b_input, coll_b_input)
    n_q = generate_queries(client, DATA_DIR / "queries.jsonl")

    # Estimate cost
    total_tokens_est = (n_a + n_b) * 200 + n_q * 20  # rough estimate
    cost_est = total_tokens_est / 1_000_000 * 0.02
    print(f"\nDone. Estimated cost: ~${cost_est:.4f}")
    print(f"  Collection A: {n_a} chunks")
    print(f"  Collection B: {n_b} chunks")
    print(f"  Queries: {n_q}")


if __name__ == "__main__":
    main()
