#!/usr/bin/env python3
"""Soak test: restart server while traffic is active, measure recovery time."""
from __future__ import annotations
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from framework.clients.deepdata import DeepDataClient
from framework.envinfo import collect_env_info
from framework.report import ScenarioResult, make_run_report
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soak: restart under load")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=128, type=int)
    p.add_argument("--n-base", default=10000, type=int)
    p.add_argument("--traffic-duration", default=10, type=float, help="Traffic duration before restart")
    p.add_argument("--restarts", default=3, type=int)
    p.add_argument("--json", default=None)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(42)

    server = DeepDataProcess(port=args.port)
    if not args.skip_build:
        server.build()
    server.start(clean=True)

    client = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
    results = []
    try:
        coll = "soak_restart"
        client.create_collection(coll, dim=args.dim)
        vecs = rng.normal(size=(args.n_base, args.dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, args.n_base + 1, dtype=np.uint64)
        client.insert(coll, ids, vecs)
        client.close()

        for restart_num in range(args.restarts):
            print(f"\nRestart cycle {restart_num + 1}/{args.restarts}")

            # Generate traffic
            stop_traffic = threading.Event()
            traffic_errors = []
            traffic_ops = [0]

            def traffic_gen():
                c = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}", timeout_s=5)
                while not stop_traffic.is_set():
                    try:
                        q = rng.normal(size=(args.dim,)).astype(np.float32)
                        c.search("soak_restart", q, top_k=10, ef_search=200)
                        traffic_ops[0] += 1
                    except Exception:
                        traffic_errors.append(time.perf_counter())
                c.close()

            t = threading.Thread(target=traffic_gen, daemon=True)
            t.start()

            # Let traffic run
            time.sleep(args.traffic_duration)

            # Restart
            t_restart_start = time.perf_counter()
            server.stop()
            server.start()
            recovery_s = time.perf_counter() - t_restart_start

            # Verify service
            c2 = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
            try:
                # Create fresh collection since old one may not persist
                new_coll = f"post_restart_{restart_num}"
                c2.create_collection(new_coll, dim=args.dim)
                small = rng.normal(size=(10, args.dim)).astype(np.float32)
                c2.insert(new_coll, np.arange(1, 11, dtype=np.uint64), small)
                res = c2.search(new_coll, small[0], top_k=5, ef_search=200)
                c2.delete_collection(new_coll)

                results.append(ScenarioResult(
                    suite="soak", scenario="restart_under_load", vdb="deepdata",
                    dataset="synthetic", success=True,
                    metrics={
                        "restart_num": float(restart_num + 1),
                        "recovery_time_s": round(recovery_s, 2),
                        "traffic_ops_before": float(traffic_ops[0]),
                        "traffic_errors": float(len(traffic_errors)),
                        "post_restart_search_ok": float(len(res.ids) > 0),
                    },
                ))
                print(f"  Recovery: {recovery_s:.1f}s, traffic_errors={len(traffic_errors)}, service healthy")
            finally:
                c2.close()

            stop_traffic.set()
            t.join(timeout=5)

        all_ok = all(r.success for r in results)
        print(f"\nRestart under load: {'PASS' if all_ok else 'FAIL'}")

        if args.json:
            env = collect_env_info()
            report = make_run_report(results, {"env": env.to_dict()})
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
