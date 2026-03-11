from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ScenarioResult:
    suite: str
    scenario: str
    vdb: str
    dataset: str
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(slots=True)
class RunReport:
    timestamp_utc: str
    results: list[ScenarioResult]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "results": [asdict(r) for r in self.results],
            "config": self.config,
        }


def make_run_report(results: list[ScenarioResult], config: dict[str, Any]) -> RunReport:
    return RunReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        results=results,
        config=config,
    )
