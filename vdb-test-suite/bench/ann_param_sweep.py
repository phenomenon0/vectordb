#!/usr/bin/env python3
"""HNSW parameter sweep: find optimal M, ef_construction, ef_search tradeoffs."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from framework.clients.deepdata import DeepDataClient
from framework.config import RunConfig
from framework.datasets import DatasetManifest, load_dataset
from framework.envinfo import collect_env_info
from framework.report import ScenarioResult, make_run_report
from framework.scenarios.ann import run_param_sweep_scenario
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNSW parameter sweep benchmark")
    p.add_argument("--base", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--dim", required=True, type=int)
    p.add_argument("--n-base", default=0, type=int)
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--m-values", default="8,16,24,32", help="Comma-separated M values")
    p.add_argument("--efc-values", default="100,200,300,500", help="ef_construction values")
    p.add_argument("--efs-values", default="32,64,100,200,400", help="ef_search values")
    p.add_argument("--n-search", default=100, type=int)
    p.add_argument("--json", default=None)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    m_vals = [int(x) for x in args.m_values.split(",")]
    efc_vals = [int(x) for x in args.efc_values.split(",")]
    efs_vals = [int(x) for x in args.efs_values.split(",")]

    manifest = DatasetManifest(
        name=args.dataset_name,
        dim=args.dim,
        base_path=Path(args.base),
        query_path=Path(args.queries),
        gt_path=Path(args.gt),
        n_base=args.n_base,
    )
    ds = load_dataset(manifest)

    server = DeepDataProcess(port=args.port)
    if not args.skip_build:
        server.build()
    server.start(clean=True)

    client = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
    try:
        results = run_param_sweep_scenario(
            client=client,
            dataset=ds,
            server_builder=server,
            m_values=m_vals,
            efc_values=efc_vals,
            efs_values=efs_vals,
            n_search=args.n_search,
        )

        print(f"\n{'M':>3} {'EfC':>5} {'EfS':>5} {'R@10':>8} {'P95ms':>8} {'Build':>8} {'RSS_MB':>8}")
        print("-" * 56)
        for r in results:
            m = r.metrics
            print(f"{m['M']:3.0f} {m['ef_construction']:5.0f} {m['ef_search']:5.0f} "
                  f"{m['recall_at_10']:8.4f} {m['search_p95_ms']:8.2f} "
                  f"{m['build_time_s']:8.2f} {m['rss_mb']:8.1f}")

        if args.json:
            env = collect_env_info()
            report = make_run_report(results, {"sweep": True, "env": env.to_dict()})
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nSaved to {args.json}")
    finally:
        client.close()
        server.stop()


if __name__ == "__main__":
    main()
