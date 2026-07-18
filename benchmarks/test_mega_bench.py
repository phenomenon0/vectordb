import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mega_bench


def sample_result(vdb="milvus", qps=10.0, dataset="tiny", ef_search=16):
    return {
        "vdb": vdb,
        "dataset": dataset,
        "ef_search": ef_search,
        "recall_at_10": 1.0,
        "recall_at_100": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
        "qps": qps,
        "insert_qps": 10.0,
    }


class CheckpointTests(unittest.TestCase):
    def test_normalize_uses_latest_result_and_clears_stale_failure(self):
        key = "milvus|tiny|ef=16"
        cp = {
            "results": [sample_result(qps=10.0), sample_result(qps=20.0)],
            "completed": [key, key, "missing|tiny|ef=16"],
            "failures": [
                {"key": key, "error": "old failure"},
                {"key": "other|tiny|ef=16", "error": "current failure"},
            ],
        }

        mega_bench.normalize_checkpoint(cp)

        self.assertEqual(1, len(cp["results"]))
        self.assertEqual(20.0, cp["results"][0]["qps"])
        self.assertEqual([key], cp["completed"])
        self.assertEqual(["other|tiny|ef=16"], [f["key"] for f in cp["failures"]])

    def test_mark_completed_atomically_upserts(self):
        key = "milvus|tiny|ef=16"
        cp = {
            "results": [sample_result(qps=10.0)],
            "completed": [key],
            "failures": [{"key": key, "error": "transient"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            checkpoint = results_dir / "checkpoint.json"
            with mock.patch.object(mega_bench, "RESULTS_DIR", results_dir), \
                    mock.patch.object(mega_bench, "CHECKPOINT_FILE", checkpoint):
                mega_bench.mark_completed(cp, key, sample_result(qps=30.0))
                stored = json.loads(checkpoint.read_text())

        self.assertEqual(30.0, stored["results"][0]["qps"])
        self.assertEqual([key], stored["completed"])
        self.assertEqual([], stored["failures"])

    def test_rerun_schedule_preserves_last_good_result(self):
        milvus_key = "milvus|tiny|ef=16"
        chroma_key = "chromadb|tiny|ef=16"
        cp = {
            "results": [sample_result(), sample_result("chromadb")],
            "completed": [milvus_key, chroma_key],
            "failures": [{"key": milvus_key, "error": "old"}],
        }

        scheduled = mega_bench.schedule_vdb_rerun(
            cp, ["milvus"], dataset_names=["tiny"], ef_values=[16])

        self.assertEqual(1, scheduled)
        self.assertEqual({"milvus", "chromadb"}, {r["vdb"] for r in cp["results"]})
        self.assertEqual([milvus_key, chroma_key], cp["completed"])
        self.assertEqual([milvus_key], cp["pending_reruns"])
        self.assertFalse(mega_bench.is_completed(cp, milvus_key))
        self.assertEqual([], cp["failures"])

    def test_rerun_schedule_is_scoped_to_selected_dataset_and_ef(self):
        selected = sample_result(dataset="sift", ef_search=16)
        other_dataset = sample_result(dataset="glove", ef_search=16)
        other_ef = sample_result(dataset="sift", ef_search=32)
        cp = {
            "results": [selected, other_dataset, other_ef],
            "completed": [
                mega_bench.result_key(selected),
                mega_bench.result_key(other_dataset),
                mega_bench.result_key(other_ef),
            ],
            "failures": [],
        }

        scheduled = mega_bench.schedule_vdb_rerun(
            cp, ["milvus"], dataset_names=["sift"], ef_values=[16])

        self.assertEqual(1, scheduled)
        self.assertEqual(
            {mega_bench.result_key(selected)}, set(cp["pending_reruns"]),
        )
        self.assertEqual(3, len(cp["results"]))

    def test_manifest_rejects_relevant_dependency_drift(self):
        cp = {
            "results": [],
            "completed": [],
            "failures": [],
            "run_manifest": {
                "schema_version": mega_bench.CHECKPOINT_SCHEMA_VERSION,
                "metric_protocol": mega_bench.METRIC_PROTOCOL,
                "n_search": 100,
                "datasets": {},
                "packages_at_manifest_creation": {"numpy": "0.0.invalid"},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "dependency changed"):
            mega_bench.ensure_run_manifest(cp, {}, [], 100, ["milvus"], [16])

    def test_report_coverage_uses_explicit_cells_not_axis_product(self):
        first = sample_result("qdrant", dataset="sift", ef_search=16)
        second = sample_result("milvus", dataset="glove", ef_search=32)
        cp = {
            "results": [first, second],
            "completed": [mega_bench.result_key(first), mega_bench.result_key(second)],
            "failures": [],
            "run_manifest": {
                "intended_cells": [
                    mega_bench.result_key(first), mega_bench.result_key(second),
                ],
                "matrix": {
                    "vdbs": ["qdrant", "milvus"],
                    "datasets": ["sift", "glove"],
                    "ef_search": [16, 32],
                },
            },
        }

        report = mega_bench.generate_report(cp)

        self.assertIn("**Coverage:** 2/2 matrix cells", report)

    def test_existing_checkpoint_requires_resume_or_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            checkpoint = results_dir / "checkpoint.json"
            checkpoint.write_text("sentinel")
            with mock.patch.object(mega_bench, "RESULTS_DIR", results_dir), \
                    mock.patch.object(mega_bench, "CHECKPOINT_FILE", checkpoint), \
                    mock.patch.object(sys, "argv", ["mega_bench.py"]):
                with self.assertRaisesRegex(SystemExit, "checkpoint exists"):
                    mega_bench.main()
            self.assertEqual("sentinel", checkpoint.read_text())


if __name__ == "__main__":
    unittest.main()
