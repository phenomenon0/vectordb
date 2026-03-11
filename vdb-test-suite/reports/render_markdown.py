#!/usr/bin/env python3
"""Render a run report as a markdown summary (suitable for PR comments)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from reports.thresholds import check_threshold


def render(report: dict) -> str:
    lines = []
    lines.append("## VDB Benchmark Report")
    lines.append(f"**Timestamp:** {report.get('timestamp_utc', 'unknown')}")
    lines.append("")

    config = report.get("config", {})
    env = config.get("env", {})
    if env:
        lines.append(f"**Env:** {env.get('cpu', '?')} | {env.get('ram_gb', '?')}GB | "
                      f"Python {env.get('python_version', '?')} | git `{env.get('git_commit', '?')}`")
        lines.append("")

    for result in report.get("results", []):
        scenario = result.get("scenario", "?")
        dataset = result.get("dataset", "?")
        vdb = result.get("vdb", "?")
        success = result.get("success", False)
        status = "PASS" if success else "FAIL"

        lines.append(f"### {scenario} — {dataset} ({vdb}) [{status}]")

        if result.get("error"):
            lines.append(f"> Error: {result['error']}")

        metrics = result.get("metrics", {})
        if metrics:
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |")
                else:
                    lines.append(f"| {k} | {v} |")

        # Check thresholds
        scenario_key = f"{scenario}/{dataset}/{vdb}"
        violations = check_threshold(scenario_key, metrics)
        if violations:
            lines.append("")
            lines.append("**Threshold violations:**")
            for v in violations:
                lines.append(f"- {v}")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render report as markdown")
    parser.add_argument("report", help="Path to report JSON")
    parser.add_argument("--output", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    md = render(report)

    if args.output:
        Path(args.output).write_text(md)
        print(f"Written to {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
