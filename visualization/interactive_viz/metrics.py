"""Metrics for attention/logits comparison."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _safe_normalize(arr: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return arr / denom


def entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p_safe = np.clip(p, eps, 1.0)
    return -(p_safe * np.log(p_safe)).sum(axis=-1)


def topk_mass(p: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(p.shape[:-1], dtype=p.dtype)
    k = min(k, p.shape[-1])
    part = np.partition(p, -k, axis=-1)[..., -k:]
    return part.sum(axis=-1)


def cosine_similarity(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    a_n = _safe_normalize(a, axis=axis)
    b_n = _safe_normalize(b, axis=axis)
    return np.sum(a_n * b_n, axis=axis)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Simple rank with average ties; last axis only."""
    flat = a.reshape(-1, a.shape[-1])
    ranks = np.zeros_like(flat, dtype=np.float32)
    for i, row in enumerate(flat):
        order = np.argsort(row)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        sorted_row = row[order]
        diffs = np.diff(sorted_row)
        tie_starts = np.where(diffs != 0)[0] + 1
        splits = np.split(np.arange(order.size), tie_starts)
        for group in splits:
            if group.size == 0:
                continue
            avg_rank = group.mean()
            ranks[i, order[group]] = avg_rank
    return ranks.reshape(a.shape)


def spearmanr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = _rankdata(a)
    rb = _rankdata(b)
    return cosine_similarity(ra, rb, axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    m = 0.5 * (p_safe + q_safe)
    kl_pm = (p_safe * (np.log(p_safe) - np.log(m))).sum(axis=-1)
    kl_qm = (q_safe * (np.log(q_safe) - np.log(m))).sum(axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def compute_summary_metrics(
    attn_fp: np.ndarray,
    attn_q: np.ndarray,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Compute per-query metrics for FP vs Quant attention.

    attn_fp/attn_q: shape [H, Q, K]
    """
    metrics = {}
    js = js_divergence(attn_fp, attn_q)
    cosine = cosine_similarity(attn_fp, attn_q)
    spearman = spearmanr(attn_fp, attn_q)
    ent_fp = entropy(attn_fp)
    ent_q = entropy(attn_q)
    metrics["js_divergence"] = js
    metrics["cosine"] = cosine
    metrics["spearman"] = spearman
    metrics["entropy_fp"] = ent_fp
    metrics["entropy_quant"] = ent_q
    metrics["entropy_delta"] = ent_q - ent_fp
    for k in topk:
        metrics[f"topk_mass_{k}_fp"] = topk_mass(attn_fp, k)
        metrics[f"topk_mass_{k}_quant"] = topk_mass(attn_q, k)
    return metrics

