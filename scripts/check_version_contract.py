#!/usr/bin/env python3
"""Fail when shipped artifact metadata drifts from the version source."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER = (ROOT / "internal/releaseinfo/version.txt").read_text().strip()
match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-rc\.(\d+)", SEMVER)
if match is None:
    raise SystemExit(f"invalid canonical RC SemVer: {SEMVER!r}")
PYTHON_VERSION = ".".join(match.groups()[:3]) + f"rc{match.group(4)}"


def require_equal(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label} is {actual!r}; expected {expected!r}")


pyproject = tomllib.loads((ROOT / "sdk/python/pyproject.toml").read_text())
require_equal("Python project version", pyproject["project"]["version"], PYTHON_VERSION)

python_init = (ROOT / "sdk/python/deepdata/__init__.py").read_text()
python_match = re.search(r'^__version__ = "([^"]+)"$', python_init, re.MULTILINE)
if python_match is None:
    raise SystemExit("Python __version__ assignment is missing")
require_equal("Python __version__", python_match.group(1), PYTHON_VERSION)

chart = (ROOT / "deploy/helm/deepdata/Chart.yaml").read_text()
chart_version = re.search(r"^version:\s*(\S+)\s*$", chart, re.MULTILINE)
app_version = re.search(r'^appVersion:\s*"([^"]+)"\s*$', chart, re.MULTILINE)
if chart_version is None or app_version is None:
    raise SystemExit("Helm chart version fields are missing")
require_equal("Helm chart version", chart_version.group(1), SEMVER)
require_equal("Helm appVersion", app_version.group(1), SEMVER)

values = (ROOT / "deploy/helm/deepdata/values.yaml").read_text()
tag = re.search(r"^\s{2}tag:\s*(\S+)\s*$", values, re.MULTILINE)
if tag is None:
    raise SystemExit("Helm image tag is missing")
require_equal("Helm image tag", tag.group(1), SEMVER)

dockerfile = (ROOT / "Dockerfile").read_text()
docker_version = re.search(
    r"^ARG DEEPDATA_VERSION=(\S+)\s*$", dockerfile, re.MULTILINE
)
if docker_version is None:
    raise SystemExit("Docker DEEPDATA_VERSION argument is missing")
require_equal("Docker image version", docker_version.group(1), SEMVER)

print(f"release version contract passed: {SEMVER} (Python {PYTHON_VERSION})")
