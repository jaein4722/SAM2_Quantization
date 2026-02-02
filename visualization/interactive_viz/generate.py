"""Generate analysis artifacts for interactive visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from visualization.interactive_viz.io import infer_token_grid, load_json, load_npz, save_json, parse_layer_range
from visualization.interactive_viz.metrics import compute_summary_metrics
from visualization.interactive_viz.render import normalize_map, to_heatmap


def _find_raw_file(record: Dict, key: str) -> Optional[Path]:
    path = record.get(key)
    if not path:
        return None
    return Path(path)


def _load_layer_attn(run_dir: Path, layer: str, kind: str) -> Optional[np.ndarray]:
    summary = load_json(run_dir / "layerwise_summary.json")
    recs = summary.get("records", {}).get(layer, [])
    if not recs:
        return None
    record = recs[0]
    key = "raw_attn_file" if kind == "attn" else "raw_logits_file"
    raw_path = _find_raw_file(record, key)
    if raw_path is None or not raw_path.exists():
        return None
    data = load_npz(raw_path)
    if kind == "attn" and "attn" in data:
        return data["attn"].astype(np.float32)
    if kind == "logits" and "logits" in data:
        return data["logits"].astype(np.float32)
    return None


def _load_qtoken_meta(run_dir: Path, layer: str) -> Dict:
    summary = load_json(run_dir / "layerwise_summary.json")
    recs = summary.get("records", {}).get(layer, [])
    if not recs:
        return {}
    record = recs[0]
    return {
        "qtoken_indices": record.get("qtoken_indices"),
        "q_grid": record.get("q_grid"),
        "k_grid": record.get("k_grid"),
        "q_tokens": record.get("q_tokens"),
        "k_tokens": record.get("k_tokens"),
    }


def _compute_output_impact(run_dir: Path, out_dir: Path, reference_mask: Optional[Path]) -> Dict:
    orig_mask = run_dir / "original" / "prediction_images" / "mask_0.png"
    quant_mask = run_dir / "quantized" / "prediction_images" / "mask_0.png"
    if not orig_mask.exists() or not quant_mask.exists():
        return {}
    orig = np.asarray(Image.open(orig_mask).convert("L"), dtype=np.float32) / 255.0
    quant = np.asarray(Image.open(quant_mask).convert("L"), dtype=np.float32) / 255.0
    diff = quant - orig
    abs_diff = np.abs(diff)
    gx = np.gradient(orig, axis=1)
    gy = np.gradient(orig, axis=0)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    edge = normalize_map(edge)
    edge_weighted = abs_diff * edge

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "orig_mask": str(orig_mask),
        "quant_mask": str(quant_mask),
    }
    for name, arr in [
        ("diff", diff),
        ("abs_diff", abs_diff),
        ("edge_weighted", edge_weighted),
    ]:
        norm = normalize_map(arr)
        img = to_heatmap(norm, cmap="magma")
        out_path = out_dir / f"{name}.png"
        img.save(out_path)
        paths[name] = str(out_path)

    if reference_mask and reference_mask.exists():
        ref = np.asarray(Image.open(reference_mask).convert("L"), dtype=np.float32) > 0.5
        orig_bin = orig > 0.5
        quant_bin = quant > 0.5
        iou_orig = float((orig_bin & ref).sum() / max((orig_bin | ref).sum(), 1))
        iou_quant = float((quant_bin & ref).sum() / max((quant_bin | ref).sum(), 1))
        paths["iou_reference"] = {
            "orig": iou_orig,
            "quant": iou_quant,
        }
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interactive analysis artifacts.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="", help="Layer range spec, e.g. image_encoder_trunk_blocks_0:23")
    parser.add_argument("--heads", type=str, default="all", help="Head list or 'all'.")
    parser.add_argument("--topk", type=str, default="1,5,10", help="Top-k list for mass metrics.")
    parser.add_argument("--reference-mask", type=str, default=None)
    parser.add_argument("--save-raw", action="store_true", help="Save selected raw slices into analysis dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    orig_dir = run_dir / "original"
    quant_dir = run_dir / "quantized"
    analysis_dir = run_dir / "analysis"

    orig_summary = load_json(orig_dir / "layerwise_summary.json")
    quant_summary = load_json(quant_dir / "layerwise_summary.json")
    orig_layers = list(orig_summary.get("records", {}).keys())
    quant_layers = list(quant_summary.get("records", {}).keys())
    common_layers = [l for l in orig_layers if l in quant_layers]
    selected_layers = parse_layer_range(args.layers, common_layers)

    head_list: Optional[List[int]] = None
    if args.heads != "all":
        head_list = [int(h.strip()) for h in args.heads.split(",") if h.strip()]

    topk_vals = [int(v.strip()) for v in args.topk.split(",") if v.strip()]
    summary_out: Dict[str, Dict] = {"layers": {}, "output_impact": {}}

    for layer in selected_layers:
        attn_fp = _load_layer_attn(orig_dir, layer, "attn")
        attn_q = _load_layer_attn(quant_dir, layer, "attn")
        if attn_fp is None or attn_q is None:
            continue
        meta = _load_qtoken_meta(orig_dir, layer)
        q_indices = meta.get("qtoken_indices") or list(range(attn_fp.shape[1]))
        if head_list is None:
            head_list = list(range(attn_fp.shape[0]))
        attn_fp_sel = attn_fp[head_list]
        attn_q_sel = attn_q[head_list]

        metrics = compute_summary_metrics(attn_fp_sel, attn_q_sel, topk=tuple(topk_vals))
        metrics_mean = {k: v.mean(axis=-1).tolist() for k, v in metrics.items()}

        logits_fp = _load_layer_attn(orig_dir, layer, "logits")
        logits_q = _load_layer_attn(quant_dir, layer, "logits")
        logits_metrics = {}
        if logits_fp is not None and logits_q is not None:
            logits_fp_sel = logits_fp[head_list]
            logits_q_sel = logits_q[head_list]
            diff = logits_q_sel - logits_fp_sel
            logits_metrics = {
                "mean_abs_diff": np.mean(np.abs(diff), axis=-1).tolist(),
                "mean_diff": np.mean(diff, axis=-1).tolist(),
            }

        summary_out["layers"][layer] = {
            "heads": head_list,
            "qtoken_indices": q_indices,
            "q_grid": meta.get("q_grid"),
            "k_grid": meta.get("k_grid"),
            "metrics_mean": metrics_mean,
            "metrics_per_q": {k: v.tolist() for k, v in metrics.items()},
            "logits_metrics": logits_metrics,
        }

        if args.save_raw:
            raw_dir = analysis_dir / "raw_slices"
            raw_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_dir / f"{layer.replace('.', '_')}_attn_fp.npz",
                attn=attn_fp_sel.astype(np.float16),
                q_indices=np.asarray(q_indices, dtype=np.int32),
            )
            np.savez_compressed(
                raw_dir / f"{layer.replace('.', '_')}_attn_quant.npz",
                attn=attn_q_sel.astype(np.float16),
                q_indices=np.asarray(q_indices, dtype=np.int32),
            )

    reference = Path(args.reference_mask) if args.reference_mask else None
    summary_out["output_impact"] = _compute_output_impact(
        run_dir,
        analysis_dir / "output_impact",
        reference,
    )

    save_json(analysis_dir / "summary.json", summary_out)
    meta = {
        "run_dir": str(run_dir),
        "layers": selected_layers,
        "heads": args.heads,
        "topk": topk_vals,
    }
    save_json(analysis_dir / "meta.json", meta)


if __name__ == "__main__":
    main()

