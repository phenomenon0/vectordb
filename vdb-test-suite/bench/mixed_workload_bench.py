#!/usr/bin/env python3
"""Mixed workload benchmark: concurrent search + insert + delete."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from framework.clients.deepdata import DeepDataClient
from framework.config import RunConfig
from framework.envinfo import collect_env_info
from framework.report import make_run_report
from framework.scenarios.mixed import run_mixed_workload_scenario
from framework.server.deepdata_process import DeepDataProcess
from framework.workload import MixedWorkloadConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed workload benchmark")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=128, type=int)
    p.add_argument("--n-seed", default=10000, type=int)
    p.add_argument("--duration", default=30, type=float, help="Duration in seconds")
    p.add_argument("--concurrency", default=1, type=int)
    p.add_argument("--search-pct", default=70.0, type=float)
    p.add_argument("--insert-pct", default=20.0, type=float)
    p.add_argument("--delete-pct", default=10.0, type=float)
    p.add_argument("--json", default=None)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    server = DeepDataProcess(port=args.port)
    if not args.skip_build:
        server.build()
    server.start(clean=True)

    client = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
    try:
        mix = MixedWorkloadConfig(
            search_pct=args.search_pct,
            insert_pct=args.insert_pct,
            delete_pct=args.delete_pct,
            dim=args.dim,
        )
        result = run_mixed_workload_scenario(
            client=client,
            collection="mixed_bench",
            dim=args.dim,
            n_seed=args.n_seed,
            duration_s=args.duration,
            concurrency=args.concurrency,
            mix=mix,
        )

        m = result.metrics
        print(f"Mixed workload ({args.duration}s, concurrency={args.concurrency}):")
        print(f"  Total ops:    {m['total_ops']:.0f} ({m['ops_per_sec']:.0f} ops/s)")
        print(f"  Search:       {m['search_count']:.0f} ops, p50={m['search_p50_ms']:.2f}ms, p95={m['search_p95_ms']:.2f}ms")
        print(f"  Insert:       {m['insert_count']:.0f} ops, p95={m['insert_p95_ms']:.2f}ms")
        print(f"  Delete:       {m['delete_count']:.0f} ops, p95={m['delete_p95_ms']:.2f}ms")
        print(f"  Errors:       {m['error_count']:.0f}")

        if args.json:
            env = collect_env_info()
            report = make_run_report([result], {"env": env.to_dict()})
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
    finally:
        client.close()
        server.stop()


if __name__ == "__main__":
    main()
