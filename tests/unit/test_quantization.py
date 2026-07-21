import argparse
import json

import pytest

from orchard_inference.quantization import _metric_value, compare


def test_metric_value_reads_exact_prometheus_series() -> None:
    text = "orchard_model_load_seconds_count 1.0\norchard_model_load_seconds_sum 0.25\n"
    assert _metric_value(text, "orchard_model_load_seconds_sum") == 0.25
    assert _metric_value(text, "missing") is None


def test_comparison_rejects_different_architectures(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(str(tmp_path))
    first = directory / "first.json"
    second = directory / "second.json"
    first.write_text(json.dumps({"metadata": {"architecture": "a", "quantization": "4bit"}}))
    second.write_text(json.dumps({"metadata": {"architecture": "b", "quantization": "8bit"}}))
    with pytest.raises(SystemExit, match="mixed-architecture"):
        compare(
            argparse.Namespace(
                artifacts=[str(first), str(second)], output=str(directory / "comparison")
            )
        )
