import argparse
import json
from pathlib import Path

import pytest

from orchard_inference.compare_mlx import compare_artifacts, improvement_percent


def test_improvement_percent_handles_higher_and_lower_is_better() -> None:
    orchard = {
        "generated_tokens_per_second": 150.0,
        "latency_p95_seconds": 0.8,
    }
    baseline = {
        "generated_tokens_per_second": 100.0,
        "latency_p95_seconds": 1.0,
    }

    assert improvement_percent(orchard, baseline, "generated_tokens_per_second") == 50.0
    assert improvement_percent(orchard, baseline, "latency_p95_seconds") == pytest.approx(20.0)


def test_compare_artifacts_matches_concurrency_and_writes_deltas(tmp_path: Path) -> None:
    orchard = {
        "runs": [
            {
                "concurrency": 1,
                "summary": {
                    "successful_requests_per_second": 2.0,
                    "latency_p95_seconds": 0.5,
                    "ttft_p95_seconds": 0.2,
                    "inter_token_p95_seconds": 0.01,
                },
            }
        ]
    }
    baseline = {
        "metadata": {"runner": "direct_mlx_lm_generate"},
        "runs": [
            {
                "concurrency": 1,
                "summary": {
                    "successful_requests_per_second": 1.0,
                    "latency_p95_seconds": 1.0,
                    "ttft_p95_seconds": 0.4,
                    "inter_token_p95_seconds": 0.02,
                },
            }
        ],
    }
    orchard_path = tmp_path / "orchard.json"
    baseline_path = tmp_path / "baseline.json"
    orchard_path.write_text(json.dumps(orchard))
    baseline_path.write_text(json.dumps(baseline))

    result = compare_artifacts(
        argparse.Namespace(
            orchard=str(orchard_path),
            baseline=str(baseline_path),
            metrics=[
                "successful_requests_per_second",
                "latency_p95_seconds",
                "ttft_p95_seconds",
                "inter_token_p95_seconds",
            ],
        )
    )

    assert result["comparisons"] == [
        {
            "concurrency": 1,
            "orchard_successful_requests_per_second": 2.0,
            "baseline_successful_requests_per_second": 1.0,
            "successful_requests_per_second_improvement_percent": 100.0,
            "orchard_latency_p95_seconds": 0.5,
            "baseline_latency_p95_seconds": 1.0,
            "latency_p95_seconds_improvement_percent": 50.0,
            "orchard_ttft_p95_seconds": 0.2,
            "baseline_ttft_p95_seconds": 0.4,
            "ttft_p95_seconds_improvement_percent": 50.0,
            "orchard_inter_token_p95_seconds": 0.01,
            "baseline_inter_token_p95_seconds": 0.02,
            "inter_token_p95_seconds_improvement_percent": 50.0,
        }
    ]
