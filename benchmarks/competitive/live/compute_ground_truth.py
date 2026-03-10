"""Phase 1c: Compute brute-force ground truth for recall evaluation.

Loads all embeddings, computes exact cosine similarity for each query,
and saves sorted top-100 neighbor IDs as ground truth.
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all query-corpus pairs.

    Args:
        queries: (n_queries, dim)
        corpus: (n_corpus, dim)

    Returns:
        (n_queries, n_corpus) similarity matrix
    """
    # Normalize
    q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    c_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    return q_norm @ c_norm.T


def compute_ground_truth(
    queries: list[dict],
    corpus: list[dict],
    top_k: int = 100,
    meta_filter: dict[str, str] | None = None,
) -> list[dict]:
    """Compute brute-force top-k nearest neighbors for each query."""
    # Filter corpus if needed
    if meta_filter:
        filtered_corpus = [
            r for r in corpus
            if all(r["meta"].get(k) == v for k, v in meta_filter.items())
        ]
    else:
        filtered_corpus = corpus

    if not filtered_corpus:
        print(f"  Warning: no corpus records match filter {meta_filter}")
        return []

    corpus_ids = [r["id"] for r in filtered_corpus]
    corpus_embs = np.array(
        [r["embedding"] for r in filtered_corpus], dtype=np.float32
    )
    query_embs = np.array(
        [q["embedding"] for q in queries], dtype=np.float32
    )

    print(f"  Computing {len(queries)} queries × {len(filtered_corpus)} corpus...")
    sims = cosine_similarity_matrix(query_embs, corpus_embs)

    results = []
    for i, query in enumerate(tqdm(queries, desc="  Ground truth")):
        # Get top-k indices sorted by descending similarity
        top_indices = np.argsort(sims[i])[::-1][:top_k]
        neighbors = [
            {"id": corpus_ids[idx], "score": float(sims[i][idx])}
            for idx in top_indices
        ]
        results.append({
            "query_id": query["id"],
            "query_text": query["text"],
            "neighbors": neighbors,
        })

    return results


def main():
    print("Phase 1c: Computing brute-force ground truth")

    # Load data
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl(DATA_DIR / "collection_a.jsonl")

    if not queries:
        raise FileNotFoundError("queries.jsonl not found or empty. Run embed_data.py first.")
    if not corpus:
        raise FileNotFoundError("collection_a.jsonl not found or empty. Run embed_data.py first.")

    # Verify embeddings exist
    if "embedding" not in corpus[0]:
        raise ValueError("Corpus records missing 'embedding' field. Run embed_data.py first.")

    # Unfiltered ground truth
    print("\nComputing unfiltered ground truth...")
    gt = compute_ground_truth(queries, corpus, top_k=100)
    gt_path = DATA_DIR / "ground_truth.jsonl"
    with open(gt_path, "w") as f:
        for rec in gt:
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(gt)} query results to {gt_path.name}")

    # Filtered ground truth: package=index (~20% selectivity)
    print("\nComputing filtered ground truth (package=index)...")
    gt_filtered = compute_ground_truth(
        queries, corpus, top_k=100,
        meta_filter={"package": "index"}
    )
    if gt_filtered:
        gt_filt_path = DATA_DIR / "ground_truth_filtered.jsonl"
        with open(gt_filt_path, "w") as f:
            for rec in gt_filtered:
                f.write(json.dumps(rec) + "\n")
        print(f"  Wrote {len(gt_filtered)} query results to {gt_filt_path.name}")
    else:
        # Try a different package if index doesn't have enough records
        packages = {}
        for r in corpus:
            pkg = r["meta"].get("package", "unknown")
            packages[pkg] = packages.get(pkg, 0) + 1
        print(f"  Available packages: {packages}")

        # Pick the package with ~20% of records
        target = len(corpus) * 0.2
        best_pkg = min(packages, key=lambda p: abs(packages[p] - target))
        print(f"  Using package={best_pkg} ({packages[best_pkg]} records) for filtered ground truth")

        gt_filtered = compute_ground_truth(
            queries, corpus, top_k=100,
            meta_filter={"package": best_pkg}
        )
        gt_filt_path = DATA_DIR / "ground_truth_filtered.jsonl"
        with open(gt_filt_path, "w") as f:
            for rec in gt_filtered:
                f.write(json.dumps(rec) + "\n")

        # Save the filter used so benchmark.py knows
        filter_meta_path = DATA_DIR / "filter_meta.json"
        with open(filter_meta_path, "w") as f:
            json.dump({"package": best_pkg}, f)
        print(f"  Wrote {len(gt_filtered)} filtered results, filter saved to filter_meta.json")

    # Sanity check: self-recall should be 1.0 for first query
    if gt:
        top1_score = gt[0]["neighbors"][0]["score"]
        print(f"\n  Sanity: query 0 top-1 score = {top1_score:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
