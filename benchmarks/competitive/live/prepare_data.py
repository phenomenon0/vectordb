"""Phase 1a: Chunk DeepData .go source files into benchmark collections.

Walks the DeepData repo collecting .go files and chunks them by
function/type boundaries (~500 tokens, 50-token overlap).
Outputs collection_a (all files) and collection_b (internal/ only).
"""

import hashlib
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parent / "data"

# Approximate tokens as words split by whitespace (close enough for Go code)
TARGET_TOKENS = 500
OVERLAP_TOKENS = 50

# Regex to split Go code at function/type boundaries
GO_BOUNDARY = re.compile(
    r"^(?:func |type |var |const |// ----)", re.MULTILINE
)


def collect_go_files(root: Path, subdir: str | None = None) -> list[Path]:
    """Collect all .go files under root, optionally restricted to subdir."""
    base = root / subdir if subdir else root
    files = sorted(base.rglob("*.go"))
    # Exclude vendor, testdata, and generated files
    return [
        f for f in files
        if "vendor" not in f.parts
        and "testdata" not in f.parts
        and not f.name.endswith("_generated.go")
    ]


def extract_metadata(filepath: Path, repo_root: Path) -> dict:
    """Extract metadata from a Go file path."""
    rel = filepath.relative_to(repo_root)
    parts = rel.parts
    package = parts[-2] if len(parts) > 1 else "main"
    has_test = filepath.name.endswith("_test.go")
    loc = sum(1 for _ in filepath.open("r", errors="replace"))
    return {
        "file_path": str(rel),
        "package": package,
        "language": "go",
        "has_test": str(has_test).lower(),
        "loc": str(loc),
    }


def chunk_go_file(text: str, target_tokens: int = TARGET_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Split Go source into chunks at function/type boundaries.

    Falls back to token-window chunking if no boundaries found.
    """
    # Find boundary positions
    boundaries = [m.start() for m in GO_BOUNDARY.finditer(text)]

    if not boundaries:
        # No function boundaries — fall back to token-window
        return _chunk_by_tokens(text, target_tokens, overlap)

    # Always start from 0
    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    # Create segments between boundaries
    segments = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)

    # Merge small segments, split large ones
    chunks = []
    current = ""
    for seg in segments:
        tokens_current = len(current.split())
        tokens_seg = len(seg.split())

        if tokens_current + tokens_seg <= target_tokens * 1.3:
            current = (current + "\n\n" + seg).strip() if current else seg
        else:
            if current:
                chunks.append(current)
            if tokens_seg > target_tokens * 1.5:
                chunks.extend(_chunk_by_tokens(seg, target_tokens, overlap))
            else:
                current = seg
                continue
            current = ""

    if current.strip():
        chunks.append(current.strip())

    # Add overlap: prepend last N tokens of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = chunks[i - 1].split()
            overlap_text = " ".join(prev_tokens[-overlap:])
            overlapped.append(overlap_text + "\n" + chunks[i])
        chunks = overlapped

    return chunks if chunks else [text]


def _chunk_by_tokens(text: str, target: int, overlap: int) -> list[str]:
    """Simple token-window chunking."""
    words = text.split()
    if len(words) <= target:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end < len(words) else end
    return chunks


def make_id(file_path: str, chunk_index: int) -> str:
    """Deterministic chunk ID."""
    raw = f"{file_path}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def process_collection(
    name: str, go_files: list[Path], repo_root: Path, output_path: Path
) -> int:
    """Process a set of Go files into a JSONL collection."""
    records = []
    for fpath in go_files:
        try:
            text = fpath.read_text(errors="replace")
        except Exception as e:
            print(f"  Skipping {fpath}: {e}")
            continue

        meta = extract_metadata(fpath, repo_root)
        chunks = chunk_go_file(text)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            chunk_meta = {**meta, "chunk_index": str(i)}
            records.append({
                "id": make_id(meta["file_path"], i),
                "text": chunk,
                "meta": chunk_meta,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"  {name}: {len(go_files)} files → {len(records)} chunks → {output_path.name}")
    return len(records)


def main():
    print("Phase 1a: Preparing benchmark data")
    print(f"  Repo root: {REPO_ROOT}")

    # Collection A: all .go files
    all_files = collect_go_files(REPO_ROOT)
    print(f"  Found {len(all_files)} .go files total")
    count_a = process_collection(
        "collection_a", all_files, REPO_ROOT,
        DATA_DIR / "collection_a.jsonl"
    )

    # Collection B: internal/ only
    internal_files = collect_go_files(REPO_ROOT, "internal")
    print(f"  Found {len(internal_files)} .go files in internal/")
    count_b = process_collection(
        "collection_b", internal_files, REPO_ROOT,
        DATA_DIR / "collection_b.jsonl"
    )

    print(f"\nDone. Collection A: {count_a} chunks, Collection B: {count_b} chunks")
    print(f"Output: {DATA_DIR}")


if __name__ == "__main__":
    main()
