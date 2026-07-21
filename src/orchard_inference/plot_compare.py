"""Render comparison plots from orchard-vs-mlx benchmark JSON files."""

import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_COLORS = {"orchard": "#2196F3", "baseline": "#FF9800"}
_BAR_W = 0.4
_PLOTS = [
    ("successful_requests_per_second", "Requests/sec", "higher = better", True),
    ("generated_tokens_per_second", "Generated tokens/sec", "higher = better", True),
    ("latency_p50_seconds", "Latency p50", "lower = better", False),
    ("latency_p95_seconds", "Latency p95", "lower = better", False),
    ("ttft_p50_seconds", "TTFT p50", "lower = better", False),
    ("ttft_p95_seconds", "TTFT p95", "lower = better", False),
    ("inter_token_p50_seconds", "ITL / TPOT p50", "lower = better", False),
    ("inter_token_p95_seconds", "ITL / TPOT p95", "lower = better", False),
]


def _load(path: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text()))


def _available_plots(comparisons: list[dict[str, Any]]) -> list[tuple[str, str, str, bool]]:
    available = []
    for metric, title, subtitle, higher_is_better in _PLOTS:
        if all(
            row.get(f"orchard_{metric}") is not None
            and row.get(f"baseline_{metric}") is not None
            for row in comparisons
        ):
            available.append((metric, title, subtitle, higher_is_better))
    return available


def _render_panel(
    axis: Any,
    comparisons: list[dict[str, Any]],
    metric: str,
    title: str,
    subtitle: str,
    higher_is_better: bool,
) -> None:
    concurrencies = [row["concurrency"] for row in comparisons]
    orchard = [row[f"orchard_{metric}"] for row in comparisons]
    baseline = [row[f"baseline_{metric}"] for row in comparisons]
    shifted = [value + _BAR_W * 0.5 for value in concurrencies]

    axis.bar(
        concurrencies,
        orchard,
        color=_COLORS["orchard"],
        width=_BAR_W,
        alpha=0.85,
        label="Orchard",
    )
    axis.bar(
        shifted,
        baseline,
        color=_COLORS["baseline"],
        width=_BAR_W,
        alpha=0.85,
        label="Direct MLX",
    )
    for index, row in enumerate(comparisons):
        improvement = row.get(f"{metric}_improvement_percent")
        if improvement is None:
            continue
        x_mid = concurrencies[index] + _BAR_W * 0.25
        y_top = max(orchard[index], baseline[index])
        prefix = "+" if improvement > 0 else ""
        axis.text(
            x_mid,
            y_top,
            f"{prefix}{improvement:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        gap = (
            orchard[index] - baseline[index]
            if higher_is_better
            else baseline[index] - orchard[index]
        )
        if gap > 0:
            y_start = min(orchard[index], baseline[index])
            axis.arrow(
                x_mid,
                y_start,
                0,
                abs(gap),
                head_width=0,
                head_length=0,
                fc="#4CAF50",
                ec="none",
                alpha=0.28,
            )
    axis.set_xlabel("Concurrency")
    axis.set_title(f"{title} ({subtitle})")
    axis.set_xticks(concurrencies)
    axis.legend(loc="best", fontsize=8)
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.2f}"))
    max_value = max(max(orchard), max(baseline), 0.001)
    axis.set_ylim(0, max_value * 1.18)


def _render(artifact: str, output_dir: str | None = None) -> Path:
    data = _load(artifact)
    comparisons = data["comparisons"]
    plots = _available_plots(comparisons)
    if not plots:
        raise SystemExit("No comparable numeric metrics found in artifact")

    columns = 2
    rows = math.ceil(len(plots) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, max(4, rows * 3.4)))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    fig.suptitle(
        "Llama-3.2-3B-Instruct (4bit) - Orchard vs Direct MLX",
        fontsize=14,
        fontweight="bold",
    )

    for axis, (metric, title, subtitle, higher_is_better) in zip(flat_axes, plots, strict=False):
        _render_panel(axis, comparisons, metric, title, subtitle, higher_is_better)
    for axis in flat_axes[len(plots) :]:
        axis.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(output_dir or ".") / "orchard-vs-mlx-llama32.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <comparison.json> [output_dir]", file=sys.stderr)
        sys.exit(1)
    artifact = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/results"
    path = _render(artifact, output_dir)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
