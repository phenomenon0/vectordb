"""Generate a Markdown report from benchmark results.

Reads raw_results.json and produces REPORT.md with comparison tables.
"""

import json
from pathlib import Path

from tabulate import tabulate

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results() -> dict:
    path = RESULTS_DIR / "raw_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run benchmark.py first.")
    with open(path) as f:
        return json.load(f)


def fmt(val, suffix=""):
    """Format a value for display."""
    if val is None or val == "?":
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}{suffix}" if val < 1 else f"{val:.1f}{suffix}"
    return f"{val}{suffix}"


def bytes_human(b):
    """Convert bytes to human-readable."""
    if b is None:
        return "N/A"
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TiB"


def generate_report(results: dict) -> str:
    vdbs = list(results.keys())
    lines = []

    lines.append("# VDB Competitive Benchmark Report")
    lines.append("")
    lines.append("Real-world head-to-head benchmark: DeepData vs Weaviate vs Milvus vs Qdrant vs ChromaDB")
    lines.append("")
    lines.append("- **Dataset**: DeepData Go source code (~800-1200 chunks)")
    lines.append("- **Embeddings**: OpenAI text-embedding-3-small (1536d)")
    lines.append("- **HNSW**: M=16, ef_construction=200, ef_search=128")
    lines.append("- **Memory limit**: 2GB per service")
    lines.append("")

    # ── Summary Table ─────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")

    summary_headers = ["Metric"] + [v.upper() for v in vdbs]
    summary_rows = []

    # Insert throughput
    row = ["Insert (docs/sec)"]
    for v in vdbs:
        val = results[v].get("insert", {}).get("inserts_per_sec")
        row.append(fmt(val))
    summary_rows.append(row)

    # Recall@10
    row = ["Recall@10"]
    for v in vdbs:
        val = results[v].get("search", {}).get("recall_at_10")
        row.append(fmt(val))
    summary_rows.append(row)

    # Recall@100
    row = ["Recall@100"]
    for v in vdbs:
        val = results[v].get("search", {}).get("recall_at_100")
        row.append(fmt(val))
    summary_rows.append(row)

    # Latency p50
    row = ["Latency p50 (ms)"]
    for v in vdbs:
        val = results[v].get("search", {}).get("latency_p50_ms")
        row.append(fmt(val, "ms"))
    summary_rows.append(row)

    # Latency p95
    row = ["Latency p95 (ms)"]
    for v in vdbs:
        val = results[v].get("search", {}).get("latency_p95_ms")
        row.append(fmt(val, "ms"))
    summary_rows.append(row)

    # QPS
    row = ["QPS (top_k=10)"]
    for v in vdbs:
        val = results[v].get("search", {}).get("qps")
        row.append(fmt(val))
    summary_rows.append(row)

    # Filtered recall@10
    row = ["Filtered Recall@10"]
    for v in vdbs:
        val = results[v].get("filtered", {}).get("recall_at_10")
        row.append(fmt(val))
    summary_rows.append(row)

    # Hybrid recall@10
    row = ["Hybrid Recall@10"]
    for v in vdbs:
        hybrid = results[v].get("hybrid", {})
        if hybrid.get("status") == "not_supported":
            row.append("N/A")
        else:
            val = hybrid.get("hybrid_recall_at_10")
            row.append(fmt(val))
    summary_rows.append(row)

    # Memory
    row = ["Memory (bytes/vec)"]
    for v in vdbs:
        val = results[v].get("memory", {}).get("bytes_per_vector")
        row.append(fmt(val))
    summary_rows.append(row)

    # Graph support
    row = ["Graph Search"]
    for v in vdbs:
        graph = results[v].get("graph", {})
        if graph.get("status") == "documented_differentiator":
            row.append("YES (unique)")
        elif graph.get("status") == "not_supported":
            row.append("No")
        else:
            row.append("N/A")
    summary_rows.append(row)

    lines.append(tabulate(summary_rows, headers=summary_headers, tablefmt="github"))
    lines.append("")

    # ── Suite Details ─────────────────────────────────────────────────

    # Suite 1: Insert
    lines.append("## Suite 1: Insert Throughput")
    lines.append("")
    insert_headers = ["VDB", "Total Docs", "Time (s)", "Docs/sec"]
    insert_rows = []
    for v in vdbs:
        d = results[v].get("insert", {})
        if "error" in results[v]:
            insert_rows.append([v.upper(), "ERROR", "", ""])
            continue
        insert_rows.append([
            v.upper(),
            d.get("total_docs", ""),
            d.get("elapsed_sec", ""),
            d.get("inserts_per_sec", ""),
        ])
    lines.append(tabulate(insert_rows, headers=insert_headers, tablefmt="github"))
    lines.append("")

    # Suite 2: Search
    lines.append("## Suite 2: Search Recall & Latency")
    lines.append("")
    search_headers = ["VDB", "R@1", "R@10", "R@100", "p50ms", "p95ms", "p99ms", "QPS"]
    search_rows = []
    for v in vdbs:
        d = results[v].get("search", {})
        search_rows.append([
            v.upper(),
            fmt(d.get("recall_at_1")),
            fmt(d.get("recall_at_10")),
            fmt(d.get("recall_at_100")),
            fmt(d.get("latency_p50_ms")),
            fmt(d.get("latency_p95_ms")),
            fmt(d.get("latency_p99_ms")),
            fmt(d.get("qps")),
        ])
    lines.append(tabulate(search_rows, headers=search_headers, tablefmt="github"))
    lines.append("")

    # Suite 3: Filtered
    lines.append("## Suite 3: Filtered Search")
    lines.append("")
    filt_headers = ["VDB", "Filter", "Recall@10", "p50ms", "p95ms"]
    filt_rows = []
    for v in vdbs:
        d = results[v].get("filtered", {})
        filt_rows.append([
            v.upper(),
            str(d.get("filter", "")),
            fmt(d.get("recall_at_10")),
            fmt(d.get("latency_p50_ms")),
            fmt(d.get("latency_p95_ms")),
        ])
    lines.append(tabulate(filt_rows, headers=filt_headers, tablefmt="github"))
    lines.append("")

    # Suite 4: Hybrid
    lines.append("## Suite 4: Hybrid Search")
    lines.append("")
    hyb_headers = ["VDB", "Hybrid R@10", "Hybrid p50ms", "Status"]
    hyb_rows = []
    for v in vdbs:
        d = results[v].get("hybrid", {})
        status = d.get("status", "ok")
        hyb_rows.append([
            v.upper(),
            fmt(d.get("hybrid_recall_at_10")),
            fmt(d.get("hybrid_latency_p50_ms")),
            status,
        ])
    lines.append(tabulate(hyb_rows, headers=hyb_headers, tablefmt="github"))
    lines.append("")

    # Suite 5: Graph
    lines.append("## Suite 5: Graph-Boosted Search")
    lines.append("")
    lines.append("Graph-boosted search is a **DeepData-only differentiator**.")
    lines.append("It combines knowledge graph entity relationships with vector similarity")
    lines.append("using weighted fusion scoring. No competitor offers equivalent functionality.")
    lines.append("")

    # Suite 6: Memory
    lines.append("## Suite 6: Memory Footprint")
    lines.append("")
    mem_headers = ["VDB", "Docker RSS", "Bytes/Vector", "Corpus Size"]
    mem_rows = []
    for v in vdbs:
        d = results[v].get("memory", {})
        mem_rows.append([
            v.upper(),
            bytes_human(d.get("docker_rss_bytes")),
            fmt(d.get("bytes_per_vector")),
            d.get("corpus_size", ""),
        ])
    lines.append(tabulate(mem_rows, headers=mem_headers, tablefmt="github"))
    lines.append("")

    # ── DeepData Analysis ─────────────────────────────────────────────
    lines.append("## DeepData Analysis")
    lines.append("")
    lines.append("### Wins")
    lines.append("- **Graph search**: Unique feature with no competitor equivalent")
    lines.append("- **Built-in hybrid**: Dense + sparse + graph fusion in a single query")
    lines.append("- **Single binary**: No external dependencies (etcd, minio) unlike Milvus")
    lines.append("- **Deployment simplicity**: One Docker image vs Milvus's 3-service stack")
    lines.append("")
    lines.append("### Known Optimization Opportunities")
    lines.append("")
    lines.append("| Area | Issue | Fix Complexity |")
    lines.append("|------|-------|---------------|")
    lines.append("| Insert API | v1 `/insert` re-embeds; v2 supports pre-computed vectors | Minor (use v2) |")
    lines.append("| SCAN_THRESHOLD | Auto-switches to brute force for small collections | Minor (set env var) |")
    lines.append("| Batch concurrency | `/batch_insert` may process sequentially | Medium |")
    lines.append("| Memory | Stores full doc text in-memory | Medium (HYDRATION_COUNT) |")
    lines.append("| Hybrid sparse | BM25 requires separate sparse insert | Minor |")
    lines.append("| Graph extraction | Requires LLM calls, adds latency | Expected (unique feature) |")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Generating benchmark report...")
    results = load_results()
    report = generate_report(results)

    report_path = RESULTS_DIR / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report written to {report_path}")
    print(f"  ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
