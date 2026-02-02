"""Utilities for exporting min-max QAT checkpoints for vanilla SAM2 inference."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch

_SKIP_TOKENS = (".weight_quant", ".act_in", ".act_out", "._last_quantized_weight")


def _strip_prefix(key: str) -> str:
    if key.startswith("model."):
        return key[6:]
    return key


def _should_skip(key: str) -> bool:
    return any(token in key for token in _SKIP_TOKENS)


def sanitize_state_dict(raw_state: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Remove quantization-only entries and SAM2Train prefixes."""

    sanitized = OrderedDict()
    for key, tensor in raw_state.items():
        stripped = _strip_prefix(key)
        if _should_skip(stripped):
            continue
        sanitized[stripped] = tensor
    return sanitized


def export_inference_checkpoint(src: Path, dst: Path | None = None) -> Path:
    """Export a min-max QAT checkpoint to a SAM2-compatible weights file."""

    src = Path(src)
    if dst is None:
        dst = src.with_name(src.stem + "_sam2.pt")
    state = torch.load(src, map_location="cpu")
    if "model" not in state:
        raise ValueError(f"Checkpoint at {src} does not contain a 'model' entry")
    sanitized = sanitize_state_dict(state["model"])
    export_state = {"model": sanitized}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_state, dst)
    return dst
