from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

from framework.report import RunReport, ScenarioResult, make_run_report

ScenarioFn = Callable[[], ScenarioResult]


class ScenarioRunner:
    def __init__(self, config: dict):
        self.config = config
        self.results: list[ScenarioResult] = []

    def run(self, scenario_fn: ScenarioFn) -> ScenarioResult:
        result = scenario_fn()
        self.results.append(result)
        return result

    def build_report(self) -> RunReport:
        return make_run_report(self.results, self.config)

    def save_json(self, path: str | Path) -> None:
        report = self.build_report()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
