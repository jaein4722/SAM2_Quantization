#!/usr/bin/env python3
"""CLI for sanitizing existing min-max QAT checkpoints for SAM2 inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from projects.minmax_qat.utils import export_inference_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to the original checkpoint.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <checkpoint>_sam2.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_path = export_inference_checkpoint(args.checkpoint, args.output)
    print(f"Sanitized checkpoint written to {export_path}")


if __name__ == "__main__":
    main()
