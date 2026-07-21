import importlib.util
import json
from pathlib import Path


def _load_module() -> object:
    script = Path("scripts/generate-benchmark-charts.py")
    spec = importlib.util.spec_from_file_location("chart_generator", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chart_generator_writes_svg_from_benchmark_json(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "bench.json"
    output_dir = tmp_path / "charts"
    artifact.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "concurrency": 1,
                        "summary": {
                            "successful_requests_per_second": 10,
                            "generated_tokens_per_second": 30,
                            "latency_p95_seconds": 0.1,
                        },
                    },
                    {
                        "concurrency": 2,
                        "summary": {
                            "successful_requests_per_second": 18,
                            "generated_tokens_per_second": 50,
                            "latency_p95_seconds": 0.2,
                        },
                    },
                ]
            }
        )
    )

    runs = module._load_runs(artifact)  # type: ignore[attr-defined]
    chart = module._chart(  # type: ignore[attr-defined]
        "successful_requests_per_second",
        "Successful requests/s",
        module._points(runs, "successful_requests_per_second"),  # type: ignore[attr-defined]
    )
    output_dir.mkdir()
    path = output_dir / "chart.svg"
    path.write_text(chart)

    assert path.read_text().startswith("<svg")
    assert "Successful requests/s" in path.read_text()
