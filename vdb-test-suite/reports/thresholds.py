"""CI gate thresholds — pass/fail boundaries for automated regression detection."""
from __future__ import annotations

THRESHOLDS = {
    # ANN recall gates
    "ann/sift-100k/deepdata": {
        "recall_at_10_min": 0.995,
        "recall_at_100_min": 0.95,
        "search_p95_ms_max": 5.0,
        "search_qps_regression_max_pct": 5.0,
        "p95_regression_max_pct": 10.0,
    },
    "ann/glove-100d-full/deepdata": {
        "recall_at_10_min": 0.95,
        "recall_at_100_min": 0.85,
        "search_p95_ms_max": 10.0,
    },
    # Persistence gates
    "persistence/deepdata": {
        "count_must_match": True,
        "post_restart_recall_drop_max": 0.005,
    },
    # CRUD gates
    "crud/deepdata": {
        "count_must_be_exact": True,
        "deleted_ids_must_not_leak": True,
    },
    # Soak gates
    "soak/mixed_workload/deepdata": {
        "zero_crashes": True,
        "error_count_max": 0,
        "latency_drift_pct_max": 25.0,
        "rss_drift_pct_max": 20.0,
    },
}


def check_threshold(
    scenario_key: str,
    metrics: dict[str, float],
) -> list[str]:
    """Return list of threshold violations (empty = pass)."""
    violations = []
    thresholds = THRESHOLDS.get(scenario_key, {})

    for key, limit in thresholds.items():
        if key.endswith("_min"):
            metric_name = key[:-4]
            val = metrics.get(metric_name, 0)
            if val < limit:
                violations.append(f"{metric_name}={val:.4f} < min {limit}")
        elif key.endswith("_max"):
            metric_name = key[:-4]
            val = metrics.get(metric_name, 0)
            if val > limit:
                violations.append(f"{metric_name}={val:.4f} > max {limit}")
        elif key.endswith("_max_pct"):
            metric_name = key[:-8]
            val = metrics.get(metric_name, 0)
            if val > limit:
                violations.append(f"{metric_name}={val:.1f}% > max {limit}%")

    return violations
