from __future__ import annotations
import os
import platform
import subprocess
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class EnvInfo:
    git_commit: str
    hostname: str
    os: str
    cpu: str
    ram_gb: float
    python_version: str

    def to_dict(self) -> dict:
        return {
            "git_commit": self.git_commit,
            "hostname": self.hostname,
            "os": self.os,
            "cpu": self.cpu,
            "ram_gb": self.ram_gb,
            "python_version": self.python_version,
        }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        except Exception:
            return 0.0


def collect_env_info() -> EnvInfo:
    return EnvInfo(
        git_commit=_git_commit(),
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        cpu=_cpu_model(),
        ram_gb=_ram_gb(),
        python_version=sys.version.split()[0],
    )
