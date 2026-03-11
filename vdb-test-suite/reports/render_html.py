#!/usr/bin/env python3
"""Render a run report as a simple HTML page."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from reports.thresholds import check_threshold


def render_html(report: dict) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>VDB Benchmark Report</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: left; }",
        "th { background: #f5f5f5; }",
        ".pass { color: #2d7d2d; font-weight: bold; }",
        ".fail { color: #d32f2f; font-weight: bold; }",
        ".violation { color: #d32f2f; }",
        "</style></head><body>",
        "<h1>VDB Benchmark Report</h1>",
        f"<p><strong>Timestamp:</strong> {report.get('timestamp_utc', 'unknown')}</p>",
    ]

    env = report.get("config", {}).get("env", {})
    if env:
        parts.append(f"<p><strong>Env:</strong> {env.get('cpu', '?')} | "
                      f"{env.get('ram_gb', '?')}GB | Python {env.get('python_version', '?')} | "
                      f"git <code>{env.get('git_commit', '?')}</code></p>")

    for result in report.get("results", []):
        scenario = result.get("scenario", "?")
        dataset = result.get("dataset", "?")
        vdb = result.get("vdb", "?")
        success = result.get("success", False)
        cls = "pass" if success else "fail"
        status = "PASS" if success else "FAIL"

        parts.append(f"<h2>{scenario} / {dataset} ({vdb}) "
                      f"<span class='{cls}'>[{status}]</span></h2>")

        if result.get("error"):
            parts.append(f"<p class='fail'>Error: {result['error']}</p>")

        metrics = result.get("metrics", {})
        if metrics:
            parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            for k, v in sorted(metrics.items()):
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                parts.append(f"<tr><td>{k}</td><td>{val}</td></tr>")
            parts.append("</table>")

        scenario_key = f"{scenario}/{dataset}/{vdb}"
        violations = check_threshold(scenario_key, metrics)
        if violations:
            parts.append("<p class='violation'><strong>Threshold violations:</strong></p><ul>")
            for v in violations:
                parts.append(f"<li class='violation'>{v}</li>")
            parts.append("</ul>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render report as HTML")
    parser.add_argument("report", help="Path to report JSON")
    parser.add_argument("--output", required=True, help="Output HTML file")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    html = render_html(report)
    Path(args.output).write_text(html)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
