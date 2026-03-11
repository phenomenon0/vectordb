#!/usr/bin/env python3
"""Compare two run reports and print deltas."""
from __future__ import annotations
import argparse
import json
import sys


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare_metrics(
    baseline: dict[str, float],
    current: dict[str, float],
) -> dict[str, tuple[float, float, float]]:
    """Returns {metric: (baseline_val, current_val, pct_change)}."""
    deltas = {}
    all_keys = sorted(set(baseline) | set(current))
    for key in all_keys:
        b = baseline.get(key, 0)
        c = current.get(key, 0)
        pct = ((c - b) / b * 100) if b != 0 else 0
        deltas[key] = (b, c, pct)
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two VDB benchmark runs")
    parser.add_argument("baseline", help="Path to baseline report JSON")
    parser.add_argument("current", help="Path to current report JSON")
    args = parser.parse_args()

    base = load_report(args.baseline)
    curr = load_report(args.current)

    base_results = {(r["scenario"], r["dataset"]): r for r in base["results"]}
    curr_results = {(r["scenario"], r["dataset"]): r for r in curr["results"]}

    all_keys = sorted(set(base_results) | set(curr_results))

    for key in all_keys:
        scenario, dataset = key
        print(f"\n{'='*60}")
        print(f"{scenario} / {dataset}")
        print(f"{'='*60}")

        b_metrics = base_results.get(key, {}).get("metrics", {})
        c_metrics = curr_results.get(key, {}).get("metrics", {})

        if not b_metrics and not c_metrics:
            print("  (no metrics)")
            continue

        deltas = compare_metrics(b_metrics, c_metrics)
        print(f"  {'Metric':<25} {'Baseline':>12} {'Current':>12} {'Change':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for metric, (b, c, pct) in deltas.items():
            sign = "+" if pct > 0 else ""
            flag = " ***" if abs(pct) > 5 else ""
            print(f"  {metric:<25} {b:>12.4f} {c:>12.4f} {sign}{pct:>8.1f}%{flag}")


if __name__ == "__main__":
    main()
