#!/usr/bin/env python
"""
Layer-wise visualization utility for quantized SAM2 (adaptive QAT).

Loads student weights plus adaptive QAT quantization state, applies activation
quantization hooks, and then records layerwise outputs and attention maps.
Now includes HTML report generation for easier visualization.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from omegaconf import OmegaConf

# Register resolvers for OmegaConf
try:
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.utils.checkpoint_utils import load_state_dict_into_model

from visualize_layerwise import (
    LayerwiseRecorder,
    _box_from_mask,
    _parse_box,
    _parse_points,
    _save_predictions,
)


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
            "</style>",
            "</head>",
            "<body>",
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
            # And compute relative path to output_dir (HTML location)
            
            try:
                mask_p = Path(pred['mask']).resolve()
                out_p = self.output_dir.resolve()
                if str(mask_p).startswith(str(out_p)):
                    mask_path = mask_p.relative_to(out_p)
                else:
                    # Fallback or symlink needed if outside? 
                    # For now assume inside as per standard flow.
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
            
            html_content.append(f"<div class='gallery-item'><img src='{mask_path}' alt='Mask {idx}'><div class='gallery-caption'>Mask {idx}</div></div>")
            if overlay_path:
                html_content.append(f"<div class='gallery-item'><img src='{overlay_path}' alt='Mask Overlay {idx}'><div class='gallery-caption'>Mask Overlay {idx}</div></div>")
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
            heatmaps = data.get('attn_images_all', [])
            
            # If no images found but render was requested, maybe check if file exists
            # (LayerwiseRecorder saves paths in meta)
            
            found_images = False
            for i, img_path in enumerate(overlays):
                try:
                    p = Path(img_path).resolve()
                    out = self.output_dir.resolve()
                    rel_path = p.relative_to(out)
                except ValueError:
                    rel_path = Path(img_path).name
                
                html.append(f"<div class='gallery-item'><img src='{rel_path}' loading='lazy'><div class='gallery-caption'>Overlay Q{i}</div></div>")
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


def _load_config(config_path: str):
    """
    Load yaml config.
    If it's an Adaptive QAT config, extract the SAM2 model config and quantization config.
    """
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    
    # Check if this is a standard SAM2 config or Adaptive QAT config
    # Adaptive QAT config typically has 'trainer' or 'model' with 'quantization'
    quant_cfg = None
    feature_layers = None
    
    # Heuristic to detect Adaptive QAT config structure
    is_qat_config = False
    
    if "trainer" in cfg and "model" in cfg.trainer and "quantization" in cfg.trainer.model:
        # Case 1: Full training config
        quant_cfg = cfg.trainer.model.quantization
        feature_layers = cfg.trainer.model.get("feature_layers")
        # Try to find SAM2 model config path or name. 
        # Usually it's passed via CLI in training, but here we need to deduce or it is embedded.
        # If not present, we fall back to a default or require it.
        # Ideally, the qat config might refer to the model config.
        # BUT, the user prompt implies we want to merge.
        # Let's assume the user passes the QAT config as --config.
        # And we need to figure out which SAM2 model config to use.
        # If the QAT config doesn't specify, we might default or error.
        is_qat_config = True
        
    elif "model" in cfg and "quantization" in cfg.model:
        # Case 2: Model-only config
        quant_cfg = cfg.model.quantization
        feature_layers = cfg.model.get("feature_layers")
        is_qat_config = True

    # If it is NOT a QAT config, then maybe the user passed a SAM2 config directly?
    # But this script is specifically for quantized visualization.
    # So we assume the user provides the QAT config.
    
    if not is_qat_config:
        raise ValueError(
            f"The provided config {config_path} does not appear to be an Adaptive QAT config "
            "(missing 'quantization' section). For standard SAM2 visualization, use visualize_layerwise.py."
        )

    # Convert to container
    quant_cfg_dict = OmegaConf.to_container(quant_cfg, resolve=True)
    if isinstance(quant_cfg_dict, dict):
        quant_cfg_dict.pop("_target_", None)
        
    # Determine SAM2 config file to use.
    # Often QAT configs are applied ON TOP of a base model.
    # We need to map the QAT config to a SAM2 model config.
    # 1. Check if 'model_type' or similar exists in QAT config.
    # 2. Or infer from filename (e.g. 'base_plus' -> 'sam2_hiera_b+.yaml')
    
    # Simple mapping based on filename or explicit field if we add one.
    sam2_config_file = "configs/sam2/sam2_hiera_t.yaml" # Default
    
    # Map common names in filename to config files
    # Note: paths are relative to sam2/ usually
    fname = Path(config_path).name
    if "base_plus" in fname or "b+" in fname:
        sam2_config_file = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif "large" in fname or "_l." in fname:
        sam2_config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif "small" in fname or "_s." in fname:
        sam2_config_file = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif "tiny" in fname or "_t." in fname:
        sam2_config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"
        
    logging.info(f"Inferred SAM2 config: {sam2_config_file} from QAT config: {config_path}")
    
    return sam2_config_file, quant_cfg_dict, feature_layers


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> Dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        student_state = checkpoint["model"]
        aux_state = checkpoint.get("model_aux") or {}
    else:
        student_state = checkpoint
        aux_state = {}

    if not isinstance(student_state, dict):
        raise ValueError("Checkpoint does not contain a model state_dict.")

    load_state_dict_into_model(student_state, model, strict=True)
    if not aux_state:
        logging.warning("No model_aux found; quantization params will use config defaults.")
    return aux_state


def _apply_adaptive_qat_quantization(
    model: torch.nn.Module,
    quant_cfg_dict: Dict,
    feature_layers: Optional[List[str]],
    aux_state: Dict,
) -> None:
    from projects.adaptive_qat.models.system import AdaptiveQATQuantConfig
    from projects.adaptive_qat.utils import (
        ActivationQuantizer,
        BitRange,
        LayerBitController,
        load_importance_config,
        resolve_map,
    )
    from projects.adaptive_qat.utils.module_utils import resolve_module

    def _filter_module_names(
        root: torch.nn.Module, names: List[str]
    ) -> Dict[str, torch.nn.Module]:
        resolved = {}
        for name in names:
            try:
                module = resolve_module(root, name)
            except (AttributeError, IndexError, KeyError):
                continue
            resolved[name] = module
        return resolved

    quant_cfg = AdaptiveQATQuantConfig(**quant_cfg_dict)

    file_importance: Dict[str, float] = {}
    if quant_cfg.importance_path:
        file_importance = load_importance_config(quant_cfg.importance_path)

    candidate_layers = list(feature_layers) if feature_layers else (
        list(quant_cfg.layers) if quant_cfg.layers is not None else list(file_importance.keys())
    )
    if not candidate_layers:
        raise ValueError("No quantization layers found from adaptive QAT config.")

    student_module_map = _filter_module_names(model, candidate_layers)
    layer_names = sorted(student_module_map.keys())
    if not layer_names:
        raise ValueError("No quantization layers matched the current model.")

    if quant_cfg.init_bits is None:
        raise ValueError("quantization.init_bits must be provided as a scalar value.")
    if isinstance(quant_cfg.init_bits, dict):
        raise ValueError("quantization.init_bits must be a scalar value, not a mapping.")
    try:
        init_bits_value = float(quant_cfg.init_bits)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantization.init_bits must be convertible to float.") from exc

    init_bits_map = {name: init_bits_value for name in layer_names}
    importance_map = resolve_map(
        layer_names,
        explicit=quant_cfg.importance,
        default_map=file_importance,
        default_value=1.0,
    )

    bit_controller = LayerBitController(
        layer_names=layer_names,
        init_bits=init_bits_map,
        bit_range=BitRange(quant_cfg.min_bits, quant_cfg.max_bits),
        requires_grad=quant_cfg.requires_grad,
        smoothing=quant_cfg.smoothing,
        importance=importance_map,
        smoothing_end_ratio=quant_cfg.smoothing_end_ratio,
        smoothing_importance_scale=quant_cfg.smoothing_importance_scale,
    )

    bit_state = {
        key[len("bit_controller.") :]: value
        for key, value in aux_state.items()
        if isinstance(key, str) and key.startswith("bit_controller.")
    }
    if bit_state:
        bit_controller.load_state_dict(bit_state, strict=False)
    bit_controller.clamp_()

    quantizer = ActivationQuantizer(
        student_module_map,
        bit_controller,
        allow_bit_grad=quant_cfg.allow_bit_grad,
        act_config=quant_cfg.act,
    )

    quant_state = {
        key[len("student_quantizer.") :]: value
        for key, value in aux_state.items()
        if isinstance(key, str) and key.startswith("student_quantizer.")
    }
    if quant_state:
        quantizer.load_state_dict(quant_state, strict=False)

    # Move quantization modules to the same device as the model
    device = next(model.parameters()).device
    bit_controller.to(device)
    quantizer.to(device)

    model._adaptive_qat_bit_controller = bit_controller  # type: ignore[attr-defined]
    model._adaptive_qat_quantizer = quantizer  # type: ignore[attr-defined]
    quantizer.eval()


def _export_quant_stats(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    bit_controller = getattr(model, "_adaptive_qat_bit_controller", None)
    quantizer = getattr(model, "_adaptive_qat_quantizer", None)
    if bit_controller is not None:
        stats["bits"] = {
            name: float(val)
            for name, val in bit_controller.export_state().items()
        }
    if quantizer is not None and hasattr(quantizer, "get_stats"):
        stats["activation"] = quantizer.get_stats()
    if quantizer is not None and hasattr(quantizer, "ema_scales"):
        stats["ema_scales"] = {
            name: float(quantizer.ema_scales[idx].item())
            for idx, name in enumerate(getattr(quantizer, "_states", {}).keys())
        }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layerwise visualization for quantized SAM2 (adaptive QAT)."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Adaptive QAT config YAML (contains quantization params). Model type inferred from filename."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Adaptive QAT checkpoint containing student weights.",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Input RGB image path (single image)."
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


def main() -> None:
    args = parse_args()
    max_tokens = None if args.max_tokens is None or args.max_tokens < 0 else args.max_tokens

    # Load Config & Infer Model Type
    sam2_config_file, quant_cfg_dict, feature_layers = _load_config(args.config)

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
        model = build_sam2_video_predictor(
            config_file=sam2_config_file,
            ckpt_path=None,
            device=args.device,
            mode="eval",
            apply_postprocessing=True,
        )
    else:
        if args.image is None:
            raise ValueError("Provide --image for single-image mode or --video-frames-dir for video.")
        overlay_image = Image.open(args.image).convert("RGB")
        model = build_sam2(
            config_file=sam2_config_file,
            ckpt_path=None,
            device=args.device,
            mode="eval",
            apply_postprocessing=True,
        )

    aux_state = _load_checkpoint(model, args.checkpoint)
    _apply_adaptive_qat_quantization(model, quant_cfg_dict, feature_layers, aux_state)
    model.eval()

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
        video_predictor = model
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
        report_gen = HTMLReportGenerator(recorder.dump_summary(out_dir / "layerwise_summary.json"), out_dir) # dump_summary returns None currently, need fix in Recorder or read file
        # Actually LayerwiseRecorder.dump_summary writes to file. 
        # We can construct dictionary directly or modify dump_summary.
        # Let's read the file back or just use internal state if accessible.
        # But recorder.dump_summary returns None.
        # We will fix this by creating the summary dict manually here or reading the json.
        
        with open(out_dir / "layerwise_summary.json", "r") as f:
            import json
            summary_data = json.load(f)
        HTMLReportGenerator(summary_data, out_dir).generate()
        quant_stats = _export_quant_stats(model)
        if quant_stats:
            with open(out_dir / "quant_stats.json", "w", encoding="utf-8") as f:
                json.dump(quant_stats, f, indent=2)
        
        logging.info("Done video pass.")
        return

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

    quant_stats = _export_quant_stats(model)
    if quant_stats:
        with open(out_dir / "quant_stats.json", "w", encoding="utf-8") as f:
            import json
            json.dump(quant_stats, f, indent=2)

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
