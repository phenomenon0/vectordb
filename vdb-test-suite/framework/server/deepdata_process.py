from __future__ import annotations
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = PROJECT_ROOT / "deepdata-server"
DEFAULT_DATA_DIR = Path("/tmp/deepdata-bench")


@dataclass(slots=True)
class DeepDataProcess:
    proc: subprocess.Popen | None = None
    binary_path: Path = DEFAULT_BINARY
    data_dir: Path = DEFAULT_DATA_DIR
    port: int = 8080

    def build(self) -> None:
        result = subprocess.run(
            ["go", "build", "-o", str(self.binary_path), "./cmd/deepdata/"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Build failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def start(
        self,
        hnsw_m: int = 16,
        ef_construction: int = 300,
        ef_search: int = 200,
        startup_timeout_s: float = 20.0,
        clean: bool = False,
    ) -> None:
        if clean and self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        env = {
            **os.environ,
            "API_RPS": "100000",
            "TENANT_RPS": "100000",
            "TENANT_BURST": "100000",
            "SCAN_THRESHOLD": "0",
            "VECTORDB_BASE_DIR": str(self.data_dir),
            "PORT": str(self.port),
            "HNSW_M": str(hnsw_m),
            "HNSW_EF_CONSTRUCTION": str(ef_construction),
            "HNSW_EFSEARCH": str(ef_search),
        }

        self.proc = subprocess.Popen(
            [str(self.binary_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

        deadline = time.perf_counter() + startup_timeout_s
        base_url = f"http://127.0.0.1:{self.port}"
        with httpx.Client(base_url=base_url, timeout=2.0) as client:
            while time.perf_counter() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("deepdata process exited during startup")
                try:
                    resp = client.get("/health")
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                time.sleep(0.25)

        self.stop()
        raise RuntimeError("deepdata server failed to become healthy")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue
