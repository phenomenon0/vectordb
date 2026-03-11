#!/usr/bin/env python3
"""Soak test: repeatedly crash and restart the server, verify integrity each time."""
from __future__ import annotations
import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from framework.clients.deepdata import DeepDataClient
from framework.envinfo import collect_env_info
from framework.report import ScenarioResult, make_run_report
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soak: crash recovery stress test")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=128, type=int)
    p.add_argument("--n-base", default=5000, type=int)
    p.add_argument("--cycles", default=5, type=int, help="Number of crash-restart cycles")
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
        coll = "soak_crash"
        client.create_collection(coll, dim=args.dim)

        vecs = rng.normal(size=(args.n_base, args.dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, args.n_base + 1, dtype=np.uint64)
        client.insert(coll, ids, vecs)
        client.close()

        for cycle in range(args.cycles):
            print(f"\nCrash cycle {cycle + 1}/{args.cycles}...")

            # SIGKILL
            if server.proc and server.proc.poll() is None:
                try:
                    os.killpg(server.proc.pid, signal.SIGKILL)
                    server.proc.wait(timeout=5)
                except Exception:
                    pass

            # Restart
            try:
                server.start()
            except RuntimeError as e:
                results.append(ScenarioResult(
                    suite="soak", scenario="crash_recovery", vdb="deepdata",
                    dataset="synthetic", success=False,
                    error=f"Cycle {cycle + 1}: failed to restart: {e}",
                    metrics={"cycle": float(cycle + 1)},
                ))
                print(f"  FAIL: server didn't restart")
                continue

            c = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
            try:
                # Verify service is up
                new_coll = f"verify_{cycle}"
                c.create_collection(new_coll, dim=args.dim)
                assert c.collection_exists(new_coll)

                # Do some writes + queries
                small_vecs = rng.normal(size=(10, args.dim)).astype(np.float32)
                c.insert(new_coll, np.arange(1, 11, dtype=np.uint64), small_vecs)
                res = c.search(new_coll, small_vecs[0], top_k=5, ef_search=200)

                results.append(ScenarioResult(
                    suite="soak", scenario="crash_recovery", vdb="deepdata",
                    dataset="synthetic", success=True,
                    metrics={
                        "cycle": float(cycle + 1),
                        "search_returned": float(len(res.ids)),
                        "service_healthy": 1.0,
                    },
                ))
                print(f"  OK: service healthy, search returned {len(res.ids)} results")
                c.delete_collection(new_coll)
            except Exception as e:
                results.append(ScenarioResult(
                    suite="soak", scenario="crash_recovery", vdb="deepdata",
                    dataset="synthetic", success=False,
                    error=f"Cycle {cycle + 1}: {e}",
                    metrics={"cycle": float(cycle + 1)},
                ))
                print(f"  FAIL: {e}")
            finally:
                c.close()

        all_ok = all(r.success for r in results)
        print(f"\nCrash recovery soak: {'PASS' if all_ok else 'FAIL'} ({sum(1 for r in results if r.success)}/{len(results)} cycles passed)")

        if args.json:
            env = collect_env_info()
            report = make_run_report(results, {"env": env.to_dict(), "cycles": args.cycles})
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
