#!/usr/bin/env python3
"""Realistic application benchmark: high-dim embeddings, mixed top_k, query repetition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from framework.clients.deepdata import DeepDataClient
from framework.envinfo import collect_env_info
from framework.report import make_run_report
from framework.scenarios.realistic import run_realistic_app_scenario
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realistic app workload benchmark")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=1536, type=int)
    p.add_argument("--n-base", default=50000, type=int)
    p.add_argument("--n-queries", default=500, type=int)
    p.add_argument("--top-k", default="10,20,50", help="Comma-separated top_k values")
    p.add_argument("--ef-search", default=200, type=int)
    p.add_argument("--json", default=None)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = [int(x) for x in args.top_k.split(",")]

    server = DeepDataProcess(port=args.port)
    if not args.skip_build:
        server.build()
    server.start(clean=True)

    client = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
    try:
        result = run_realistic_app_scenario(
            client=client,
            collection="realistic_bench",
            dim=args.dim,
            n_base=args.n_base,
            n_queries=args.n_queries,
            top_k_values=top_k_values,
            ef_search=args.ef_search,
        )

        m = result.metrics
        print(f"Realistic app benchmark ({args.dim}d, {args.n_base} vectors):")
        print(f"  Ingest:       {m.get('ingest_qps', 0):.0f} vec/s")
        for k in top_k_values:
            p50 = m.get(f"top{k}_p50_ms", 0)
            p95 = m.get(f"top{k}_p95_ms", 0)
            cnt = m.get(f"top{k}_count", 0)
            print(f"  top_k={k:>3}:     {cnt:.0f} queries, p50={p50:.2f}ms, p95={p95:.2f}ms")

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
