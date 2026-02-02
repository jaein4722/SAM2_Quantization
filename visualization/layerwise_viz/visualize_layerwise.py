#!/usr/bin/env python
"""
Layer-wise visualization utility for SAM2.

- Hooks Attention/RoPEAttention modules to record q/k/v shapes and attention maps.
- Records outputs of TwoWayAttentionBlock and MemoryAttentionLayer.
- Saves predictions and a JSON summary for downstream inspection.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import (
    build_sam2,
    build_sam2_hf,
    build_sam2_video_predictor,
)
from sam2.modeling.backbones.hieradet import MultiScaleAttention, do_pool
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import Attention, RoPEAttention, TwoWayAttentionBlock
from sam2.sam2_image_predictor import SAM2ImagePredictor


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class HTMLReportGenerator:
    """Generates an HTML report from layerwise visualization results."""
    
    def __init__(self, summary: Dict, output_dir: Path):
        self.summary = summary
        self.output_dir = output_dir
        self.report_path = output_dir / "report.html"

    def generate(self):
        records = self.summary.get("records", {})
        prediction_images = self.summary.get("prediction_images", [])
        
        # Sort keys to ensure consistent order: Encoder -> Decoder
        sorted_keys = sorted(records.keys(), key=lambda k: (
            0 if "image_encoder" in k else 1,  # Encoder first
            int(k.split('.')[3]) if "blocks" in k and k.split('.')[3].isdigit() else 999, # Block index
            k
        ))

        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>SAM2 Layerwise Visualization Report</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; background: #f0f0f0; }",
            ".container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "h1, h2, h3 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }",
            ".section { margin-bottom: 40px; }",
            ".layer-block { margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 4px; }",
            ".layer-header { display: flex; justify_content: space-between; align-items: center; background: #f8f9fa; padding: 10px; margin: -15px -15px 15px -15px; border-bottom: 1px solid #ddd; }",
            ".layer-title { font-weight: bold; font-size: 1.1em; color: #444; }",
            ".layer-meta { font-size: 0.85em; color: #666; font-family: monospace; }",
            ".gallery { display: flex; flex-wrap: wrap; gap: 10px; }",
            ".gallery-item { text-align: center; }",
            ".gallery-item img { max-width: 300px; border: 1px solid #ccc; border-radius: 4px; transition: transform 0.2s; }",
            ".gallery-item img:hover { transform: scale(1.05); border-color: #007bff; }",
            ".gallery-caption { font-size: 0.8em; color: #555; margin-top: 5px; }",
            ".prediction-section { display: flex; gap: 20px; flex-wrap: wrap; }",
            ".nav-links { position: sticky; top: 0; background: white; padding: 10px 0; border-bottom: 1px solid #ddd; z-index: 100; }",
            ".nav-links a { margin-right: 15px; text-decoration: none; color: #007bff; font-weight: bold; }",
            ".nav-links a:hover { text-decoration: underline; }",
            # Lightbox Styles
            ".lightbox { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); }",
            ".lightbox-content { margin: 5% auto; display: block; max-width: 90%; max-height: 90%; }",
            ".close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }",
            ".close:hover, .close:focus { color: #bbb; text-decoration: none; cursor: pointer; }",
            ".img-container { position: relative; display: inline-block; }",
            ".hist-icon { position: absolute; top: 5px; right: 5px; background: rgba(255,255,255,0.8); border-radius: 50%; padding: 5px; cursor: pointer; font-size: 12px; }",
            "</style>",
            "<script>",
            "function openLightbox(src) {",
            "  var lightbox = document.getElementById('lightbox');",
            "  var lightboxImg = document.getElementById('lightbox-img');",
            "  lightbox.style.display = 'block';",
            "  lightboxImg.src = src;",
            "}",
            "function closeLightbox() {",
            "  document.getElementById('lightbox').style.display = 'none';",
            "}",
            "</script>",
            "</head>",
            "<body>",
            "<div id='lightbox' class='lightbox' onclick='closeLightbox()'>",
            "  <span class='close'>&times;</span>",
            "  <img class='lightbox-content' id='lightbox-img'>",
            "</div>",
            "<div class='container'>",
            "<h1>SAM2 Layerwise Visualization Report</h1>",
            "<div class='nav-links'>",
            "<a href='#predictions'>Predictions</a>",
            "<a href='#encoder'>Image Encoder</a>",
            "<a href='#decoder'>Mask Decoder</a>",
            "</div>"
        ]

        # Predictions Section
        html_content.append("<div id='predictions' class='section'>")
        html_content.append("<h2>Predictions</h2>")
        html_content.append("<div class='prediction-section'>")
        for idx, pred in enumerate(prediction_images):
            # Paths in JSON might be absolute or relative. Resolve them.
            try:
                mask_p = Path(pred['mask']).resolve()
                out_p = self.output_dir.resolve()
                if str(mask_p).startswith(str(out_p)):
                    mask_path = mask_p.relative_to(out_p)
                else:
                    mask_path = Path(pred['mask']).name
            except Exception:
                mask_path = Path(pred['mask']).name

            try:
                if pred.get('overlay'):
                    ov_p = Path(pred['overlay']).resolve()
                    out_p = self.output_dir.resolve()
                    if str(ov_p).startswith(str(out_p)):
                        overlay_path = ov_p.relative_to(out_p)
                    else:
                        overlay_path = Path(pred['overlay']).name
                else:
                    overlay_path = None
            except Exception:
                overlay_path = Path(pred['overlay']).name if pred.get('overlay') else None

            try:
                if pred.get('histogram'):
                    hist_p = Path(pred['histogram']).resolve()
                    out_p = self.output_dir.resolve()
                    if str(hist_p).startswith(str(out_p)):
                        hist_path = hist_p.relative_to(out_p)
                    else:
                        hist_path = Path(pred['histogram']).name
                else:
                    hist_path = None
            except Exception:
                hist_path = None
            
            html_content.append(f"<div class='gallery-item'><img src='{mask_path}' alt='Mask {idx}'><div class='gallery-caption'>Mask {idx}</div></div>")
            if overlay_path:
                hist_attr = f"onclick=\"openLightbox('{hist_path}')\" style='cursor:pointer'" if hist_path else ""
                caption_extra = " (Click for Hist)" if hist_path else ""
                html_content.append(f"<div class='gallery-item'><img src='{overlay_path}' alt='Mask Overlay {idx}' {hist_attr}><div class='gallery-caption'>Mask Overlay {idx}{caption_extra}</div></div>")
        html_content.append("</div></div>")

        # Encoder Section
        html_content.append("<div id='encoder' class='section'>")
        html_content.append("<h2>Image Encoder Layers</h2>")
        
        encoder_records = [k for k in sorted_keys if "image_encoder" in k]
        self._append_layers(html_content, encoder_records, records)
        html_content.append("</div>")

        # Decoder Section
        html_content.append("<div id='decoder' class='section'>")
        html_content.append("<h2>Mask Decoder Layers</h2>")
        decoder_records = [k for k in sorted_keys if "sam_mask_decoder" in k]
        self._append_layers(html_content, decoder_records, records)
        html_content.append("</div>")

        html_content.append("</div></body></html>")
        
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))
        logging.info("Saved HTML report to %s", self.report_path)

    def _append_layers(self, html, layer_keys, records):
        for key in layer_keys:
            layer_data_list = records[key]
            # Assuming we visualize the first call (call_idx 0)
            if not layer_data_list: continue
            
            # Handle multiple calls if necessary, currently taking first
            data = layer_data_list[0] 
            
            html.append(f"<div class='layer-block' id='{key}'>")
            html.append(f"<div class='layer-header'>")
            html.append(f"<span class='layer-title'>{key}</span>")
            
            # Metadata string
            meta_str = f"Kind: {data.get('kind', 'Unknown')} | Heads: {data.get('num_heads', '-')} | "
            if 'q_shape' in data:
                meta_str += f"Q: {data['q_shape']}"
            html.append(f"<span class='layer-meta'>{meta_str}</span>")
            html.append("</div>")

            # Images
            html.append("<div class='gallery'>")
            
            # Check for overlay images
            overlays = data.get('attn_overlay_images_all', [])
            histograms = data.get('attn_histograms_all', [])
            heatmaps = data.get('attn_images_all', [])
            
            found_images = False
            for i, img_path in enumerate(overlays):
                try:
                    p = Path(img_path).resolve()
                    out = self.output_dir.resolve()
                    rel_path = p.relative_to(out)
                except ValueError:
                    rel_path = Path(img_path).name
                
                # Resolve histogram path
                hist_rel_path = None
                if i < len(histograms):
                    try:
                        hp = Path(histograms[i]).resolve()
                        out = self.output_dir.resolve()
                        hist_rel_path = hp.relative_to(out)
                    except ValueError:
                        hist_rel_path = Path(histograms[i]).name

                hist_attr = f"onclick=\"openLightbox('{hist_rel_path}')\" style='cursor:pointer'" if hist_rel_path else ""
                caption_extra = " (Click for Hist)" if hist_rel_path else ""
                
                html.append(f"<div class='gallery-item'><img src='{rel_path}' loading='lazy' {hist_attr}><div class='gallery-caption'>Overlay Q{i}{caption_extra}</div></div>")
                found_images = True
            
            for i, img_path in enumerate(heatmaps):
                # Avoid duplicates if overlay only mode
                if not overlays: 
                    try:
                        p = Path(img_path).resolve()
                        out = self.output_dir.resolve()
                        rel_path = p.relative_to(out)
                    except ValueError:
                        rel_path = Path(img_path).name

                    html.append(f"<div class='gallery-item'><img src='{rel_path}' loading='lazy'><div class='gallery-caption'>Heatmap Q{i}</div></div>")
                    found_images = True
            
            if not found_images:
                html.append("<div class='gallery-caption'>No visualization images generated.</div>")
                
            html.append("</div></div>")


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _infer_token_grid(token_count: int) -> Optional[Tuple[int, int]]:
    side = int(round(token_count ** 0.5))
    if side * side == token_count:
        return side, side
    return None


def _select_default_qtokens(token_count: int) -> List[int]:
    grid = _infer_token_grid(token_count)
    if grid is None:
        return [0] if token_count > 0 else []
    h, w = grid
    idxs = [
        0,
        w - 1,
        (h - 1) * w,
        h * w - 1,
        (h // 2) * w + (w // 2),
    ]
    return sorted({i for i in idxs if 0 <= i < token_count})


def _maybe_truncate_tokens(x: torch.Tensor, max_tokens: Optional[int]) -> torch.Tensor:
    if max_tokens is None or max_tokens < 0:
        return x
    q_lim = min(max_tokens, x.shape[-2])
    k_lim = min(max_tokens, x.shape[-1]) if x.dim() >= 3 else None
    if k_lim is None:
        return x[..., :q_lim, :]
    return x[..., :q_lim, :k_lim]


class LayerwiseRecorder:
    def __init__(
        self,
        save_dir: Path,
        save_attn: bool = False,
        save_qkv: bool = False,
        max_tokens: Optional[int] = 512,
        render_attn_images: bool = False,
        render_attn_overlay: bool = False,
        attn_overlay_only: bool = False,
        max_attn_queries: int = 2,
        overlay_image: Optional[Image.Image] = None,
        overlay_alpha: float = 0.85,
        cmap: str = "viridis",
        topk_attn: int = 0,
        topk_symlink: bool = False,
        save_prediction_images: bool = True,
        save_raw_attn: bool = False,
        save_raw_logits: bool = False,
        save_full_attn: bool = False,
        raw_qtokens: Optional[List[int]] = None,
        raw_dtype: str = "float16",
    ) -> None:
        self.save_dir = save_dir
        self.attn_dir = save_dir / "attn_tensors"
        self.attn_all_dir = self.attn_dir / "all"
        # self.attn_layer_dir = self.attn_dir / "by_layer" # Removed by request
        self.attn_img_dir = save_dir / "attn_images"
        self.attn_img_all_dir = self.attn_img_dir / "all"
        # self.attn_img_layer_dir = self.attn_img_dir / "by_layer" # Removed by request
        self.attn_overlay_dir = save_dir / "attn_overlays"
        self.attn_overlay_all_dir = self.attn_overlay_dir / "all"
        # self.attn_overlay_layer_dir = self.attn_overlay_dir / "by_layer" # Removed by request
        self.attn_topk_dir = save_dir / "attn_topk"
        self.prediction_dir = save_dir / "prediction_images"
        self.save_attn = save_attn
        self.save_qkv = save_qkv
        self.max_tokens = max_tokens
        self.render_attn_images = render_attn_images
        self.render_attn_overlay = render_attn_overlay
        self.attn_overlay_only = attn_overlay_only
        self.max_attn_queries = max_attn_queries
        self.overlay_image = overlay_image.convert("RGBA") if overlay_image else None
        self.overlay_alpha = overlay_alpha
        self.cmap_name = cmap
        self.cmap = None
        self.topk_attn = topk_attn
        self.topk_symlink = topk_symlink
        self.save_prediction_images_enabled = save_prediction_images
        self.save_raw_attn = save_raw_attn
        self.save_raw_logits = save_raw_logits
        self.save_full_attn = save_full_attn
        self.raw_qtokens = raw_qtokens
        self.raw_dtype = raw_dtype
        self.raw_dir = save_dir / "raw_attn"
        self.records: Dict[str, List[Dict]] = {}
        self.rank_items: List[Dict] = []
        self.prediction_images: List[Dict] = []
        self.input_image_path: Optional[str] = None
        self.image_size: Optional[Tuple[int, int]] = None
        
        if self.save_attn or self.save_qkv:
            self.attn_all_dir.mkdir(parents=True, exist_ok=True)
            # self.attn_layer_dir.mkdir(parents=True, exist_ok=True)
            
        if self.render_attn_images and not self.attn_overlay_only:
            self.attn_img_all_dir.mkdir(parents=True, exist_ok=True)
            # self.attn_img_layer_dir.mkdir(parents=True, exist_ok=True)
            
        if self.render_attn_overlay and self.overlay_image is not None:
            self.attn_overlay_all_dir.mkdir(parents=True, exist_ok=True)
            # self.attn_overlay_layer_dir.mkdir(parents=True, exist_ok=True)
            
        if (
            self.render_attn_images
            or self.render_attn_overlay
            or self.save_prediction_images_enabled
        ):
            from matplotlib import cm

            self.cmap = cm.get_cmap(cmap)
        if self.save_prediction_images_enabled:
            self.prediction_dir.mkdir(parents=True, exist_ok=True)
        if self.save_raw_attn or self.save_raw_logits:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

    def register(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (Attention, RoPEAttention)):
                module.register_forward_pre_hook(
                    self._make_attention_hook(name), with_kwargs=True
                )
            if isinstance(module, TwoWayAttentionBlock):
                module.register_forward_hook(
                    self._make_two_way_hook(name), with_kwargs=True
                )
            if isinstance(module, MemoryAttentionLayer):
                module.register_forward_hook(
                    self._make_memory_hook(name), with_kwargs=True
                )
            if isinstance(module, MultiScaleAttention):
                module.register_forward_pre_hook(
                    self._make_msa_hook(name), with_kwargs=True
                )
            if isinstance(module, MemoryAttention):
                module.register_forward_hook(
                    self._make_memory_stack_hook(name), with_kwargs=True
                )

    def _append_record(self, name: str, payload: Dict) -> int:
        self.records.setdefault(name, [])
        self.records[name].append(payload)
        return len(self.records[name]) - 1

    def _layer_dir(self, base_dir: Path, module_name: str) -> Path:
        # NOTE: This method is deprecated as we removed by_layer dirs
        layer_dir = base_dir / _safe_name(module_name)
        layer_dir.mkdir(parents=True, exist_ok=True)
        return layer_dir

    def _normalize_grid(self, grid: torch.Tensor) -> torch.Tensor:
        gmin, gmax = grid.min(), grid.max()
        if (gmax - gmin) > 1e-6:
            return (grid - gmin) / (gmax - gmin)
        return torch.zeros_like(grid)

    def _grid_to_rgb(self, grid: torch.Tensor) -> np.ndarray:
        if self.cmap is None:
            raise RuntimeError("Colormap is not initialized for attention visualization.")
        grid = self._normalize_grid(grid)
        grid_np = grid.detach().cpu().numpy()
        rgba = self.cmap(grid_np)
        rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
        return rgb

    def _record_rank_item(
        self, kind: str, path: Path, score: float, meta: Dict, query_idx: int
    ) -> None:
        path_abs = Path(path).resolve()
        self.rank_items.append(
            {
                "kind": kind,
                "path": str(path_abs),
                "score": float(score),
                "module": meta.get("module"),
                "call_idx": meta.get("call_idx"),
                "query_idx": int(query_idx),
            }
        )

    def _write_topk(self) -> Dict[str, List[Dict]]:
        if self.topk_attn <= 0 or not self.rank_items:
            return {}
        topk_dir = self.attn_topk_dir
        topk_dir.mkdir(parents=True, exist_ok=True)
        by_kind: Dict[str, List[Dict]] = {}
        for item in self.rank_items:
            by_kind.setdefault(item["kind"], []).append(item)
        summary: Dict[str, List[Dict]] = {}
        for kind, items in by_kind.items():
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            top_items = items_sorted[: self.topk_attn]
            summary[kind] = top_items
            list_path = topk_dir / f"topk_{kind}.txt"
            with list_path.open("w", encoding="utf-8") as f:
                for rank, item in enumerate(top_items, start=1):
                    f.write(
                        f"{rank}\t{item['score']:.6f}\t{item['path']}\t"
                        f"{item['module']}\t{item['call_idx']}\t{item['query_idx']}\n"
                    )
            if self.topk_symlink:
                kind_dir = topk_dir / kind
                kind_dir.mkdir(parents=True, exist_ok=True)
                for rank, item in enumerate(top_items, start=1):
                    src = Path(item["path"]).resolve()
                    if not src.exists():
                        continue
                    score_tag = f"{item['score']:.4f}".replace(".", "_")
                    link_name = f"{rank:02d}_{score_tag}_{src.name}"
                    link_path = kind_dir / link_name
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(src)
        return summary

    def _make_attention_hook(self, name: str):
        def hook(module, args, kwargs):
            # Support both positional and kwarg calls
            q = kwargs.get("q") if kwargs else None
            k = kwargs.get("k") if kwargs else None
            v = kwargs.get("v") if kwargs else None
            if q is None or k is None or v is None:
                if len(args) >= 3:
                    q, k, v = args[:3]
                else:
                    return
            num_k_exclude_rope = kwargs.get("num_k_exclude_rope", 0) if kwargs else 0
            with torch.no_grad():
                q_proj = module.q_proj(q)
                k_proj = module.k_proj(k)
                v_proj = module.v_proj(v)

                q_head = module._separate_heads(q_proj, module.num_heads)
                k_head = module._separate_heads(k_proj, module.num_heads)
                v_head = module._separate_heads(v_proj, module.num_heads)

                scores = torch.matmul(
                    q_head, k_head.transpose(-2, -1)
                ) / math.sqrt(q_head.shape[-1])
                attn_weights = scores.softmax(dim=-1)

                meta = {
                    "module": name,
                    "kind": module.__class__.__name__,
                    "q_shape": list(q.shape),
                    "k_shape": list(k.shape),
                    "v_shape": list(v.shape),
                    "attn_shape": list(attn_weights.shape),
                    "num_heads": module.num_heads,
                    "num_k_exclude_rope": int(num_k_exclude_rope),
                }
                call_idx = self._append_record(name, meta)
                meta["call_idx"] = call_idx

                save_name = f"{_safe_name(name)}_{call_idx}"
                if self.save_attn:
                    attn_to_save = _maybe_truncate_tokens(
                        attn_weights, self.max_tokens
                    ).cpu()
                    attn_path_all = self.attn_all_dir / f"{save_name}_attn.pt"
                    # attn_path_layer = (
                    #     self._layer_dir(self.attn_layer_dir, name)
                    #     / f"{save_name}_attn.pt"
                    # )
                    torch.save(attn_to_save, attn_path_all)
                    # torch.save(attn_to_save, attn_path_layer)
                    meta["attn_file_all"] = str(attn_path_all)
                    # meta["attn_file_layer"] = str(attn_path_layer)
                    meta["attn_file"] = str(attn_path_all)

                if self.save_qkv:
                    q_save = _maybe_truncate_tokens(q_head, self.max_tokens).cpu()
                    k_save = _maybe_truncate_tokens(k_head, self.max_tokens).cpu()
                    v_save = _maybe_truncate_tokens(v_head, self.max_tokens).cpu()
                    qkv_path_all = self.attn_all_dir / f"{save_name}_qkv.pt"
                    # qkv_path_layer = (
                    #     self._layer_dir(self.attn_layer_dir, name)
                    #     / f"{save_name}_qkv.pt"
                    # )
                    torch.save({"q": q_save, "k": k_save, "v": v_save}, qkv_path_all)
                    # torch.save(
                    #     {"q": q_save, "k": k_save, "v": v_save}, qkv_path_layer
                    # )
                    meta["qkv_file_all"] = str(qkv_path_all)
                    # meta["qkv_file_layer"] = str(qkv_path_layer)
                    meta["qkv_file"] = str(qkv_path_all)

                if self.render_attn_images and not self.attn_overlay_only:
                    self._save_attn_images(attn_weights, save_name, meta)
                if self.render_attn_overlay and self.overlay_image is not None:
                    self._save_attn_overlay(attn_weights, save_name, meta)
                self._maybe_save_raw(scores, attn_weights, save_name, meta)

        return hook

    def _make_msa_hook(self, name: str):
        def hook(module, args, kwargs):
            if not args:
                return
            x = args[0]
            with torch.no_grad():
                B, H, W, _ = x.shape
                qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
                q, k, v = torch.unbind(qkv, 2)
                if module.q_pool:
                    q = do_pool(q.reshape(B, H, W, -1), module.q_pool)
                    H, W = q.shape[1:3]
                    q = q.reshape(B, H * W, module.num_heads, -1)
                qh = q.transpose(1, 2)
                kh = k.transpose(1, 2)
                vh = v.transpose(1, 2)
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(
                    qh.shape[-1]
                )
                attn_weights = scores.softmax(dim=-1)

                meta = {
                    "module": name,
                    "kind": module.__class__.__name__,
                    "q_shape": list(qh.shape),  # [B, heads, Q, C]
                    "k_shape": list(kh.shape),
                    "v_shape": list(vh.shape),
                    "attn_shape": list(attn_weights.shape),
                    "num_heads": module.num_heads,
                    "q_tokens": qh.shape[-2],
                    "k_tokens": kh.shape[-2],
                }
                call_idx = self._append_record(name, meta)
                meta["call_idx"] = call_idx
                save_name = f"{_safe_name(name)}_{call_idx}"

                if self.save_attn:
                    attn_to_save = _maybe_truncate_tokens(
                        attn_weights, self.max_tokens
                    ).cpu()
                    attn_path_all = self.attn_all_dir / f"{save_name}_attn.pt"
                    # attn_path_layer = (
                    #     self._layer_dir(self.attn_layer_dir, name)
                    #     / f"{save_name}_attn.pt"
                    # )
                    torch.save(attn_to_save, attn_path_all)
                    # torch.save(attn_to_save, attn_path_layer)
                    meta["attn_file_all"] = str(attn_path_all)
                    # meta["attn_file_layer"] = str(attn_path_layer)
                    meta["attn_file"] = str(attn_path_all)

                if self.save_qkv:
                    q_save = _maybe_truncate_tokens(qh, self.max_tokens).cpu()
                    k_save = _maybe_truncate_tokens(kh, self.max_tokens).cpu()
                    v_save = _maybe_truncate_tokens(vh, self.max_tokens).cpu()
                    qkv_path_all = self.attn_all_dir / f"{save_name}_qkv.pt"
                    # qkv_path_layer = (
                    #     self._layer_dir(self.attn_layer_dir, name)
                    #     / f"{save_name}_qkv.pt"
                    # )
                    torch.save({"q": q_save, "k": k_save, "v": v_save}, qkv_path_all)
                    # torch.save(
                    #     {"q": q_save, "k": k_save, "v": v_save}, qkv_path_layer
                    # )
                    meta["qkv_file_all"] = str(qkv_path_all)
                    # meta["qkv_file_layer"] = str(qkv_path_layer)
                    meta["qkv_file"] = str(qkv_path_all)

                if self.render_attn_images and not self.attn_overlay_only:
                    self._save_attn_images(attn_weights, save_name, meta)
                if self.render_attn_overlay and self.overlay_image is not None:
                    self._save_attn_overlay(attn_weights, save_name, meta)
                self._maybe_save_raw(scores, attn_weights, save_name, meta)

        return hook

    def _maybe_save_raw(
        self,
        logits: torch.Tensor,
        attn_weights: torch.Tensor,
        save_name: str,
        meta: Dict,
    ) -> None:
        if not (self.save_raw_attn or self.save_raw_logits):
            return
        if attn_weights.dim() != 4:
            return
        attn = attn_weights[0].detach()
        logit = logits[0].detach()
        q_tokens = attn.shape[-2]
        k_tokens = attn.shape[-1]
        q_grid = _infer_token_grid(q_tokens)
        k_grid = _infer_token_grid(k_tokens)
        q_indices = self.raw_qtokens or _select_default_qtokens(q_tokens)
        if self.save_full_attn:
            q_indices = list(range(q_tokens))
        if not q_indices:
            return
        q_idx = torch.as_tensor(q_indices, device=attn.device)
        attn_slice = attn.index_select(dim=-2, index=q_idx)
        logit_slice = logit.index_select(dim=-2, index=q_idx)
        dtype = torch.float16 if self.raw_dtype == "float16" else torch.float32
        base_payload = {"q_indices": q_idx.detach().cpu().numpy()}
        if q_grid:
            base_payload["q_grid"] = np.asarray(q_grid, dtype=np.int32)
        if k_grid:
            base_payload["k_grid"] = np.asarray(k_grid, dtype=np.int32)
        if self.save_raw_attn:
            payload = dict(base_payload)
            payload["attn"] = attn_slice.to(dtype=dtype).cpu().numpy()
            attn_path = self.raw_dir / f"{save_name}_attn.npz"
            np.savez_compressed(attn_path, **payload)
            meta["raw_attn_file"] = str(attn_path)
        if self.save_raw_logits:
            payload = dict(base_payload)
            payload["logits"] = logit_slice.to(dtype=dtype).cpu().numpy()
            logits_path = self.raw_dir / f"{save_name}_logits.npz"
            np.savez_compressed(logits_path, **payload)
            meta["raw_logits_file"] = str(logits_path)
        meta["q_tokens"] = int(q_tokens)
        meta["k_tokens"] = int(k_tokens)
        meta["qtoken_indices"] = [int(i) for i in q_indices]
        if q_grid:
            meta["q_grid"] = list(q_grid)
        if k_grid:
            meta["k_grid"] = list(k_grid)

    def _save_attn_images(
        self, attn_weights: torch.Tensor, save_name: str, meta: Dict
    ) -> None:
        """Save per-query attention heatmaps as PNG."""
        with torch.no_grad():
            attn_mean = attn_weights.mean(dim=1)  # [B, Q, K]
            bsz, qn, kn = attn_mean.shape
            max_q = min(self.max_attn_queries, qn)
            module_name = meta.get("module", "unknown")
            all_dir = self.attn_img_all_dir
            # layer_dir = self._layer_dir(self.attn_img_layer_dir, module_name)
            imgs = []
            for qi in range(max_q):
                arr = attn_mean[0, qi]  # [K]
                score = float(arr.max().item())
                side = int(math.isqrt(kn))
                if side * side == kn:
                    grid = arr[: side * side].reshape(side, side)
                else:
                    grid = arr.unsqueeze(0)  # 1 x K strip
                grid_rgb = self._grid_to_rgb(grid)
                imgs.append((qi, grid_rgb, score))

            for qi, grid_rgb, score in imgs:
                out_path_all = all_dir / f"{save_name}_q{qi}.png"
                # out_path_layer = layer_dir / f"{save_name}_q{qi}.png"
                Image.fromarray(grid_rgb, mode="RGB").save(out_path_all)
                # Image.fromarray(grid_rgb, mode="RGB").save(out_path_layer)
                self._record_rank_item("image", out_path_all, score, meta, qi)

            if imgs:
                meta["attn_images_all"] = [
                    str(all_dir / f"{save_name}_q{qi}.png") for qi, _, _ in imgs
                ]
                # meta["attn_images_layer"] = [
                #     str(layer_dir / f"{save_name}_q{qi}.png") for qi, _, _ in imgs
                # ]
                meta["attn_images"] = meta["attn_images_all"]

    def _save_histogram(self, data: np.ndarray, save_path: Path, title: str = "Histogram"):
        """Save histogram of data."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Flatten and filter stats
        flat_data = data.flatten()
        mean_val = flat_data.mean()
        max_val = flat_data.max()
        min_val = flat_data.min()
        std_val = flat_data.std()
        
        ax.hist(flat_data, bins=50, log=True, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"{title}\nMin:{min_val:.4f} Max:{max_val:.4f} Mean:{mean_val:.4f}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count (Log Scale)")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='png')
        plt.close(fig)

    def _save_image_with_prompts(self, base_img: Image.Image, overlay_img: Image.Image, save_path: Path, alpha: float = 0.5):
        """Save overlay image with prompts using matplotlib."""
        try:
            import matplotlib
            matplotlib.use('Agg') # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            # Fallback to PIL blend if matplotlib is missing
            logging.warning("Matplotlib not found, saving without prompts visualization.")
            Image.blend(base_img, overlay_img, alpha).save(save_path)
            return

        # Setup figure
        dpi = 100
        w, h = base_img.size
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Show base image
        ax.imshow(base_img)
        
        # Show overlay (heatmap) with alpha
        ax.imshow(overlay_img, alpha=alpha)

        # Plot prompts
        if hasattr(self, 'prompts_points') and self.prompts_points is not None:
            coords, labels = self.prompts_points
            if coords is not None:
                for (x, y), label in zip(coords, labels):
                    # label 1 = foreground (lime star), 0 = background (red star)
                    color = 'lime' if label == 1 else 'red'
                    marker = '*'
                    ax.scatter([x], [y], color=color, marker=marker, s=400, edgecolors='white', linewidth=1.5)
        
        if hasattr(self, 'prompts_box') and self.prompts_box is not None:
            import matplotlib.patches as patches
            x0, y0, x1, y1 = self.prompts_box
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=3, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        # Save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='png')
        plt.close(fig)

    def _save_attn_overlay(
        self, attn_weights: torch.Tensor, save_name: str, meta: Dict
    ) -> None:
        """Save attention heatmaps overlaid on the original image."""
        if self.overlay_image is None:
            return
        base_rgb = self.overlay_image.convert("RGB")
        with torch.no_grad():
            attn_mean = attn_weights.mean(dim=1)  # [B, Q, K]
            bsz, qn, kn = attn_mean.shape
            max_q = min(self.max_attn_queries, qn)
            module_name = meta.get("module", "unknown")
            all_dir = self.attn_overlay_all_dir
            # layer_dir = self._layer_dir(self.attn_overlay_layer_dir, module_name)
            
            # Use 'jet' cmap by default if not specified or viridis
            cmap = self.cmap
            if self.cmap_name == "viridis": 
                try:
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap("jet")
                except Exception:
                    pass

            overlay_paths_all = []
            # overlay_paths_layer = []
            hist_paths_all = [] # New: Store histogram paths

            for qi in range(max_q):
                arr = attn_mean[0, qi]  # [K]
                score = float(arr.max().item())
                side = int(math.isqrt(kn))
                if side * side == kn:
                    grid = arr[: side * side].reshape(side, side)
                else:
                    grid = arr.unsqueeze(0)
                
                grid_norm = self._normalize_grid(grid)
                grid_norm_np = grid_norm.detach().cpu().numpy()
                
                # Save Histogram (Original values, not normalized for distribution check)
                # But visualization uses normalized. Let's save histogram of the raw attention weights (arr)
                # to see the dominance.
                arr_np = arr.detach().cpu().numpy()
                hist_path_all = all_dir / f"{save_name}_q{qi}_hist.png"
                # hist_path_layer = layer_dir / f"{save_name}_q{qi}_hist.png"
                self._save_histogram(arr_np, hist_path_all, title=f"Attn Hist: {save_name} Q{qi}")
                
                # import shutil
                # if hist_path_all.exists():
                #     shutil.copy(hist_path_all, hist_path_layer)
                
                hist_paths_all.append(str(hist_path_all))

                # Apply colormap
                heat_rgba = cmap(grid_norm_np)
                heat_rgb = (heat_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
                
                # Resize heatmap to match base image
                heat_img = Image.fromarray(heat_rgb, mode="RGB").resize(base_rgb.size, Image.BILINEAR)
                
                out_path_all = all_dir / f"{save_name}_q{qi}.png"
                # out_path_layer = layer_dir / f"{save_name}_q{qi}.png"
                
                alpha = self.overlay_alpha
                if alpha > 0.9: alpha = 0.5 

                self._save_image_with_prompts(base_rgb, heat_img, out_path_all, alpha=alpha)
                # Copy to layer dir
                # if out_path_all.exists():
                #     shutil.copy(out_path_all, out_path_layer)

                overlay_paths_all.append(str(out_path_all))
                # overlay_paths_layer.append(str(out_path_layer))
                self._record_rank_item("overlay", out_path_all, score, meta, qi)

            if overlay_paths_all:
                meta["attn_overlay_images_all"] = overlay_paths_all
                # meta["attn_overlay_images_layer"] = overlay_paths_layer
                meta["attn_overlay_images"] = overlay_paths_all
                meta["attn_histograms_all"] = hist_paths_all # Record hist paths

    def save_prediction_images(self, masks: np.ndarray) -> None:
        if not self.save_prediction_images_enabled or masks is None:
            return
        
        # Use 'jet' cmap by default if not specified or viridis
        cmap = self.cmap
        if self.cmap_name == "viridis": 
            try:
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap("jet")
            except Exception:
                pass

        base_rgb = self.overlay_image.convert("RGB") if self.overlay_image is not None else None
        
        for idx, mask in enumerate(masks):
            mask_arr = np.asarray(mask, dtype=np.float32)
            if mask_arr.ndim > 2:
                mask_arr = np.squeeze(mask_arr)
            
            # Sigmoid if raw logits
            if mask_arr.min() < 0 or mask_arr.max() > 1:
                mask_arr = 1 / (1 + np.exp(-mask_arr))
                
            mmin, mmax = float(mask_arr.min()), float(mask_arr.max())
            if mmax > mmin:
                mask_norm = (mask_arr - mmin) / (mmax - mmin)
            else:
                mask_norm = np.zeros_like(mask_arr)

            mask_path = self.prediction_dir / f"mask_{idx}.png"
            mask_img = (mask_norm * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(mask_img, mode="L").save(mask_path)

            overlay_path = None
            if base_rgb is not None:
                # Apply colormap (Jet)
                heat_rgba = cmap(mask_norm)
                heat_rgb = (heat_rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
                
                # Resize heatmap
                if base_rgb.size != (mask_norm.shape[1], mask_norm.shape[0]):
                     heat_img = Image.fromarray(heat_rgb, mode="RGB").resize(base_rgb.size, Image.BILINEAR)
                else:
                     heat_img = Image.fromarray(heat_rgb, mode="RGB")

                overlay_path = self.prediction_dir / f"mask_{idx}_overlay.png"
                
                alpha = self.overlay_alpha
                if alpha > 0.9: alpha = 0.5 
                
                self._save_image_with_prompts(base_rgb, heat_img, overlay_path, alpha=alpha)

            # Save Histogram for mask
            hist_path = self.prediction_dir / f"mask_{idx}_hist.png"
            self._save_histogram(mask_arr, hist_path, title=f"Mask {idx} Logits/Probs")

            self.prediction_images.append(
                {
                    "mask": str(mask_path.resolve()),
                    "overlay": str(overlay_path.resolve()) if overlay_path else None,
                    "histogram": str(hist_path.resolve()),
                }
            )

    def _make_two_way_hook(self, name: str):
        def hook(module, args, kwargs, output):
            if output is None:
                return
            queries_out, keys_out = output
            meta = {
                "module": name,
                "kind": module.__class__.__name__,
                "queries_shape": list(queries_out.shape),
                "keys_shape": list(keys_out.shape),
            }
            self._append_record(name, meta)

        return hook

    def _make_memory_hook(self, name: str):
        def hook(module, args, kwargs, output):
            if not args or output is None:
                return
            tgt, memory = args[:2]
            num_obj_ptr_tokens = 0
            if len(args) >= 5:
                num_obj_ptr_tokens = int(args[4])
            meta = {
                "module": name,
                "kind": module.__class__.__name__,
                "tgt_shape": list(tgt.shape),
                "memory_shape": list(memory.shape),
                "output_shape": list(output.shape),
                "num_obj_ptr_tokens": num_obj_ptr_tokens,
            }
            self._append_record(name, meta)

        return hook

    def _make_memory_stack_hook(self, name: str):
        def hook(module, args, kwargs, output):
            if output is None:
                return
            meta = {
                "module": name,
                "kind": module.__class__.__name__,
                "output_shape": list(output.shape) if hasattr(output, "shape") else None,
                "num_layers": getattr(module, "num_layers", None),
            }
            self._append_record(name, meta)

        return hook

    def dump_summary(self, path: Path) -> None:
        topk_summary = self._write_topk()
        summary = {
            "records": self.records,
            "topk": topk_summary,
            "cmap": self.cmap_name,
            "prediction_images": self.prediction_images,
            "input_image": self.input_image_path,
            "image_size": list(self.image_size) if self.image_size else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info("Saved layerwise summary to %s", path)


def _parse_points(raw: Optional[List[float]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if raw is None:
        return None, None
    if len(raw) % 3 != 0:
        raise ValueError("points must be provided as triples: x y label ...")
    coords: List[List[float]] = []
    labels: List[int] = []
    for i in range(0, len(raw), 3):
        x, y, label = raw[i : i + 3]
        coords.append([x, y])
        labels.append(int(label))
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _parse_box(raw: Optional[List[float]]) -> Optional[np.ndarray]:
    if raw is None:
        return None
    if len(raw) != 4:
        raise ValueError("box must have four values: x0 y0 x1 y1")
    return np.asarray(raw, dtype=np.float32)


def _box_from_mask(mask_path: str) -> Optional[np.ndarray]:
    """Compute tight box from a binary mask file."""
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _build_model(args) -> torch.nn.Module:
    if args.hf_model_id:
        logging.info("Loading model from HF id: %s", args.hf_model_id)
        return build_sam2_hf(
            args.hf_model_id, device=args.device, mode="eval", apply_postprocessing=True
        )
    if args.checkpoint is None:
        raise ValueError("checkpoint is required when hf_model_id is not set.")
    logging.info("Loading model from config=%s checkpoint=%s", args.config, args.checkpoint)
    return build_sam2(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
        mode="eval",
        apply_postprocessing=True,
    )


def _build_video_model(args) -> torch.nn.Module:
    logging.info(
        "Loading video model from config=%s checkpoint=%s", args.config, args.checkpoint
    )
    return build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
        mode="eval",
        apply_postprocessing=True,
    )


def _save_predictions(
    masks: np.ndarray, ious: np.ndarray, low_res_masks: np.ndarray, out_dir: Path
) -> None:
    # out_dir.mkdir(parents=True, exist_ok=True)
    # save_path = out_dir / "predictions.pt"
    # User requested to remove tensor files.
    # torch.save(
    #     {
    #         "masks": torch.from_numpy(masks),
    #         "ious": torch.from_numpy(ious),
    #         "low_res_masks": torch.from_numpy(low_res_masks),
    #     },
    #     save_path,
    # )
    # logging.info("Saved predictions to %s", save_path)
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layerwise visualization for SAM2.")
    parser.add_argument("--config", type=str, default="configs/sam2/sam2_hiera_t.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM2 checkpoint.")
    parser.add_argument("--hf-model-id", type=str, default=None, help="Optional HF repo id.")
    parser.add_argument(
        "--image", type=str, default=None, help="Input RGB image path (for single image mode)."
    )
    parser.add_argument(
        "--points",
        type=float,
        nargs="+",
        default=None,
        help="Point prompts as repeated triples: x y label (label in {0,1}).",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs="+",
        default=None,
        help="Optional box prompt as x0 y0 x1 y1 (pixel coordinates).",
    )
    parser.add_argument(
        "--annotation-mask",
        type=str,
        default=None,
        help="Optional binary mask file to derive a box (first frame).",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="If set, return 3 masks (matches SAM behaviour).",
    )
    parser.add_argument(
        "--save-attn",
        action="store_true",
        help="Store attention maps (can be large).",
    )
    parser.add_argument(
        "--save-qkv",
        action="store_true",
        help="Store projected q/k/v (can be large).",
    )
    parser.add_argument(
        "--save-raw-attn",
        action="store_true",
        help="Store raw attention tensors for analysis (light mode by default).",
    )
    parser.add_argument(
        "--save-raw-logits",
        action="store_true",
        help="Store raw logits (pre-softmax) tensors for analysis.",
    )
    parser.add_argument(
        "--save-full-attn",
        action="store_true",
        help="Save full attention/logits for all query tokens (larger).",
    )
    parser.add_argument(
        "--raw-qtokens",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of query token indices to save (light mode).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Limit tokens saved per head for attn/qkv (-1 for no limit).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sam2_logs/layerwise_viz",
        help="Directory to save outputs.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument(
        "--render-attn-images",
        action="store_true",
        help="Save attention heatmaps as PNG (head-mean, limited queries).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for attention visualizations.",
    )
    parser.add_argument(
        "--max-attn-queries",
        type=int,
        default=2,
        help="Max queries per attention call to render as image.",
    )
    parser.add_argument(
        "--render-attn-overlay",
        action="store_true",
        help="Overlay attention on the original image and save PNG to attn_overlays/.",
    )
    parser.add_argument(
        "--attn-overlay-only",
        action="store_true",
        help="Skip plain heatmap PNGs; save only overlays (attn tensors still saved).",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.85,
        help="Alpha multiplier for overlay (0~1, applied after normalization).",
    )
    parser.add_argument(
        "--topk-attn",
        type=int,
        default=0,
        help="Collect top-k attention visualizations by peak intensity.",
    )
    parser.add_argument(
        "--topk-symlink",
        action="store_true",
        help="Create symlinks for top-k attention items (also writes a text list).",
    )
    # Video options
    parser.add_argument(
        "--video-frames-dir",
        type=str,
        default=None,
        help="Directory of ordered video frames (png/jpg). Enables video/temporal path.",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=None,
        help="Track at most N frames (default: all frames in dir).",
    )
    parser.add_argument(
        "--video-start-frame",
        type=int,
        default=0,
        help="Frame index to seed prompts (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_tokens = None if args.max_tokens is None or args.max_tokens < 0 else args.max_tokens

    # Resolve overlay image (first frame for video, given image otherwise)
    if args.video_frames_dir:
        frame_paths = sorted(
            list(Path(args.video_frames_dir).glob("*.png"))
            + list(Path(args.video_frames_dir).glob("*.jpg"))
        )
        if not frame_paths:
            raise FileNotFoundError(
                f"No frames (*.png|*.jpg) found under {args.video_frames_dir}"
            )
        overlay_image = Image.open(frame_paths[0]).convert("RGB")
        model = _build_video_model(args)
    else:
        if args.image is None:
            raise ValueError("Provide --image for single-image mode or --video-frames-dir for video.")
        overlay_image = Image.open(args.image).convert("RGB")
        model = _build_model(args)


    recorder = LayerwiseRecorder(
        save_dir=Path(args.output_dir),
        save_attn=args.save_attn,
        save_qkv=args.save_qkv,
        max_tokens=max_tokens,
        render_attn_images=args.render_attn_images,
        render_attn_overlay=args.render_attn_overlay,
        attn_overlay_only=args.attn_overlay_only,
        max_attn_queries=args.max_attn_queries,
        overlay_image=overlay_image,
        overlay_alpha=args.overlay_alpha,
        cmap=args.cmap,
        topk_attn=args.topk_attn,
        topk_symlink=args.topk_symlink,
        save_raw_attn=args.save_raw_attn,
        save_raw_logits=args.save_raw_logits,
        save_full_attn=args.save_full_attn,
        raw_qtokens=args.raw_qtokens,
    )
    recorder.input_image_path = args.image if args.image else args.video_frames_dir
    recorder.image_size = overlay_image.size
    recorder.register(model)

    point_coords, point_labels = _parse_points(args.points)
    box = _parse_box(args.box)
    if box is None and args.annotation_mask:
        box_from_mask = _box_from_mask(args.annotation_mask)
        if box_from_mask is not None:
            box = box_from_mask
            logging.info("Using box derived from annotation mask: %s", args.annotation_mask)
        else:
            logging.warning("Annotation mask is empty; box not set: %s", args.annotation_mask)

    # Store prompts in recorder for visualization
    recorder.prompts_points = (point_coords, point_labels) if point_coords is not None else None
    recorder.prompts_box = box

    if args.video_frames_dir:
        if point_coords is None and box is None:
            raise ValueError("Provide --points or --box for the first frame in video mode.")
        video_predictor = model  # SAM2VideoPredictor
        with torch.no_grad():
            state = video_predictor.init_state(video_path=args.video_frames_dir)
            video_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=args.video_start_frame,
                obj_id=0,
                points=point_coords,
                labels=point_labels,
                box=box,
                normalize_coords=True,
            )
            for _ in video_predictor.propagate_in_video(
                inference_state=state,
                start_frame_idx=args.video_start_frame,
                max_frame_num_to_track=args.video_max_frames,
                reverse=False,
            ):
                pass
        out_dir = Path(args.output_dir)
        recorder.dump_summary(out_dir / "layerwise_summary.json")

        # Generate HTML Report
        try:
            import json
            with open(out_dir / "layerwise_summary.json", "r") as f:
                summary_data = json.load(f)
            HTMLReportGenerator(summary_data, out_dir).generate()
        except Exception as e:
            logging.error(f"Failed to generate HTML report: {e}")

        logging.info("Done video pass.")
    else:
        predictor = SAM2ImagePredictor(model)
        with torch.no_grad():
            predictor.set_image(overlay_image)
            masks, ious, low_res_masks = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=args.multimask_output,
                return_logits=True,
                normalize_coords=True,
            )
        out_dir = Path(args.output_dir)
        _save_predictions(masks, ious, low_res_masks, out_dir)
        recorder.save_prediction_images(masks)
        recorder.dump_summary(out_dir / "layerwise_summary.json")

        # Generate HTML Report
        try:
            import json
            with open(out_dir / "layerwise_summary.json", "r") as f:
                summary_data = json.load(f)
            HTMLReportGenerator(summary_data, out_dir).generate()
        except Exception as e:
            logging.error(f"Failed to generate HTML report: {e}")

        logging.info("Done image pass.")


if __name__ == "__main__":
    main()
