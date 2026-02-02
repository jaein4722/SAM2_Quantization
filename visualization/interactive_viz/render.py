"""Rendering helpers for heatmaps."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image


def normalize_map(
    arr: np.ndarray,
    mode: str = "shared",
    ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mode == "per-model":
        vmin, vmax = float(arr.min()), float(arr.max())
    elif mode == "fp-anchored" and ref is not None:
        vmin, vmax = float(ref.min()), float(ref.max())
    else:
        if ref is not None:
            vmin = float(min(arr.min(), ref.min()))
            vmax = float(max(arr.max(), ref.max()))
        else:
            vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def to_heatmap(
    grid: np.ndarray,
    cmap: str = "viridis",
    size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        grid_u8 = (grid * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(grid_u8, mode="L")
        return img.resize(size, Image.NEAREST) if size else img

    cmap_obj = plt.get_cmap(cmap)
    rgba = cmap_obj(grid)
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if size:
        img = img.resize(size, Image.BILINEAR)
    return img

