"""IO helpers for attention analysis artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_layer_range(spec: str, candidates: List[str]) -> List[str]:
    """Resolve layer specs like image_encoder_trunk_blocks_0:23 to real keys."""
    if not spec:
        return candidates
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    selected: List[str] = []
    sanitized = {k: k.replace(".", "_") for k in candidates}
    for part in parts:
        if ":" in part:
            start = 0
            end = -1
            base = part.strip()
            match = re.match(r"(.+)_([0-9]+):([0-9]+)$", part)
            if match:
                base = match.group(1)
                start = int(match.group(2))
                end = int(match.group(3))
            else:
                base, rng = part.split(":", 1)
                base = base.strip().rstrip("_")
                try:
                    start = int(rng)
                except ValueError:
                    start = 0
            for key, key_safe in sanitized.items():
                if not key_safe.startswith(base):
                    continue
                # Extract last integer
                nums = [int(s) for s in key_safe.split("_") if s.isdigit()]
                if not nums:
                    continue
                idx = nums[-1]
                if end < 0:
                    if idx >= start:
                        selected.append(key)
                elif start <= idx <= end:
                    selected.append(key)
        else:
            for key, key_safe in sanitized.items():
                if part == key or part == key_safe:
                    selected.append(key)
    if not selected:
        return candidates
    return sorted(set(selected), key=candidates.index)


def infer_token_grid(token_count: int) -> Optional[Tuple[int, int]]:
    side = int(round(token_count ** 0.5))
    if side * side == token_count:
        return side, side
    return None

