"""Smoke tests for interactive_viz artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from visualization.interactive_viz.io import load_json, load_npz
from visualization.interactive_viz.metrics import compute_summary_metrics


def main(run_dir: str) -> None:
    run_path = Path(run_dir)
    summary = load_json(run_path / "analysis" / "summary.json")
    if not summary:
        raise RuntimeError("analysis/summary.json not found.")
    layers = list(summary.get("layers", {}).keys())
    if not layers:
        raise RuntimeError("No layers in analysis summary.")

    layer = layers[0]
    orig = load_json(run_path / "original" / "layerwise_summary.json")
    record = orig.get("records", {}).get(layer, [])[0]
    raw_path = record.get("raw_attn_file")
    if not raw_path:
        raise RuntimeError("raw_attn_file not found in summary.")
    data = load_npz(Path(raw_path))
    attn = data.get("attn")
    if attn is None or attn.size == 0:
        raise RuntimeError("Empty attention tensor.")

    # Compare tensor with itself to ensure metrics are finite.
    metrics = compute_summary_metrics(attn, attn)
    for name, val in metrics.items():
        if not np.isfinite(val).all():
            raise RuntimeError(f"Non-finite metric: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m visualization.interactive_viz.tests.smoke_test <run_dir>"
        )
    main(sys.argv[1])

