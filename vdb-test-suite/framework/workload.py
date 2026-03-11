from __future__ import annotations
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from framework.clients.base import VDBClient


class OpType(Enum):
    SEARCH = "search"
    INSERT = "insert"
    DELETE = "delete"
    UPSERT = "upsert"


@dataclass(slots=True)
class OpResult:
    op: OpType
    latency_ms: float
    success: bool
    error: str = ""


@dataclass(slots=True)
class MixedWorkloadConfig:
    search_pct: float = 70.0
    insert_pct: float = 20.0
    delete_pct: float = 10.0
    upsert_pct: float = 0.0
    top_k: int = 10
    ef_search: int = 200
    dim: int = 128


@dataclass
class WorkloadDriver:
    client: Any  # VDBClient
    collection: str
    config: MixedWorkloadConfig
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    _next_id: int = 1
    _live_ids: list[int] = field(default_factory=list)

    def seed_ids(self, ids: list[int]) -> None:
        self._live_ids = list(ids)
        self._next_id = max(ids) + 1 if ids else 1

    def _pick_op(self) -> OpType:
        r = random.random() * 100
        cumul = 0.0
        for op, pct in [
            (OpType.SEARCH, self.config.search_pct),
            (OpType.INSERT, self.config.insert_pct),
            (OpType.DELETE, self.config.delete_pct),
            (OpType.UPSERT, self.config.upsert_pct),
        ]:
            cumul += pct
            if r < cumul:
                return op
        return OpType.SEARCH

    def run_one(self) -> OpResult:
        op = self._pick_op()
        t0 = time.perf_counter()
        try:
            if op == OpType.SEARCH:
                q = self.rng.normal(size=(self.config.dim,)).astype(np.float32)
                self.client.search(
                    self.collection, q,
                    top_k=self.config.top_k,
                    ef_search=self.config.ef_search,
                )
            elif op == OpType.INSERT:
                vec = self.rng.normal(size=(1, self.config.dim)).astype(np.float32)
                doc_id = self._next_id
                self._next_id += 1
                self.client.insert(
                    self.collection,
                    np.array([doc_id], dtype=np.uint64),
                    vec,
                )
                self._live_ids.append(doc_id)
            elif op == OpType.DELETE:
                if self._live_ids:
                    idx = random.randrange(len(self._live_ids))
                    doc_id = self._live_ids.pop(idx)
                    self.client.delete_ids(self.collection, [doc_id])
            elif op == OpType.UPSERT:
                if self._live_ids:
                    doc_id = random.choice(self._live_ids)
                    vec = self.rng.normal(size=(1, self.config.dim)).astype(np.float32)
                    self.client.upsert(
                        self.collection,
                        np.array([doc_id], dtype=np.uint64),
                        vec,
                    )
            elapsed = (time.perf_counter() - t0) * 1000.0
            return OpResult(op=op, latency_ms=elapsed, success=True)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return OpResult(op=op, latency_ms=elapsed, success=False, error=str(e))

    def run_duration(
        self,
        duration_s: float,
        concurrency: int = 1,
    ) -> list[OpResult]:
        results: list[OpResult] = []
        deadline = time.perf_counter() + duration_s

        if concurrency <= 1:
            while time.perf_counter() < deadline:
                results.append(self.run_one())
            return results

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            while time.perf_counter() < deadline:
                if len(futures) >= concurrency * 2:
                    done = [f for f in futures if f.done()]
                    for f in done:
                        results.append(f.result())
                        futures.remove(f)
                futures.append(pool.submit(self.run_one))
            for f in as_completed(futures):
                results.append(f.result())

        return results
