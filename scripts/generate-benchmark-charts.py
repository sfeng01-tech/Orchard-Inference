#!/usr/bin/env python3
"""Generate simple SVG benchmark charts from real orchard-bench JSON artifacts."""

import argparse
import json
from pathlib import Path
from typing import Any

SERIES = (
    ("successful_requests_per_second", "Successful requests/s"),
    ("generated_tokens_per_second", "Generated tokens/s"),
    ("latency_p95_seconds", "P95 latency (s)"),
    ("queue_p95_seconds", "P95 queue (s)"),
)


def _load_runs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit(f"{path} does not contain any benchmark runs")
    return runs


def _points(runs: list[dict[str, Any]], metric: str) -> list[tuple[int, float]]:
    points = []
    for run in runs:
        summary = run.get("summary", {})
        value = summary.get(metric)
        if value is None:
            continue
        points.append((int(run["concurrency"]), float(value)))
    return points


def _polyline(points: list[tuple[int, float]], width: int, height: int) -> str:
    if not points:
        return ""
    padding = 48
    x_values = [x for x, _value in points]
    y_values = [value for _x, value in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.0, max(y_values)
    x_span = max(1, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    rendered = []
    for x_value, y_value in points:
        x = padding + (x_value - x_min) / x_span * (width - padding * 2)
        y = height - padding - (y_value - y_min) / y_span * (height - padding * 2)
        rendered.append(f"{x:.1f},{y:.1f}")
    return " ".join(rendered)


def _chart(metric: str, title: str, points: list[tuple[int, float]]) -> str:
    width, height = 720, 360
    line = _polyline(points, width, height)
    view_box = f"0 0 {width} {height}"
    labels = "".join(
        f'<text x="52" y="{84 + index * 22}" font-size="13">c={concurrency}: {value:.4g}</text>'
        for index, (concurrency, value) in enumerate(points)
    )
    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="{view_box}">'
    )
    return f"""{opening}
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="48" y="36" font-size="20" font-family="Arial, sans-serif">{title}</text>
  <line x1="48" y1="312" x2="672" y2="312" stroke="#222" stroke-width="1"/>
  <line x1="48" y1="48" x2="48" y2="312" stroke="#222" stroke-width="1"/>
  <polyline points="{line}" fill="none" stroke="#2563eb" stroke-width="3"/>
  {labels}
  <text x="310" y="344" font-size="13" font-family="Arial, sans-serif">Concurrency</text>
  <metadata>{metric}</metadata>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="generate-benchmark-charts.py")
    parser.add_argument("input", type=Path, help="orchard-bench JSON artifact")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/charts"))
    args = parser.parse_args()
    runs = _load_runs(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for metric, title in SERIES:
        points = _points(runs, metric)
        if not points:
            continue
        path = args.output_dir / f"{metric}.svg"
        path.write_text(_chart(metric, title, points))
        generated.append(str(path))
    if not generated:
        raise SystemExit("no supported metrics found in benchmark artifact")
    print(json.dumps({"charts": generated}, indent=2))


if __name__ == "__main__":
    main()
