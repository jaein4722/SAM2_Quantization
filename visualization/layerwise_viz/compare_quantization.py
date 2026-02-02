#!/usr/bin/env python
"""
Compare visualization results between Original and Quantized SAM2 models.

Runs visualize_layerwise.py and visualize_layerwise_quantized.py, then merges the results
into a side-by-side HTML report.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Original vs Quantized SAM2 visualization.")
    parser.add_argument("--config-quantized", type=str, required=True, help="Adaptive QAT Config for quantized model.")
    parser.add_argument("--config-original", type=str, default=None, help="Optional SAM2 Config for original model. If not set, inferred from QAT config.")
    parser.add_argument("--checkpoint-original", type=str, required=True, help="Path to Original SAM2 checkpoint.")
    parser.add_argument("--checkpoint-quantized", type=str, required=True, help="Path to Quantized SAM2 checkpoint (Adaptive QAT).")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--points", type=float, nargs="+", default=None, help="Points: x y label ...")
    parser.add_argument("--box", type=float, nargs="+", default=None, help="Box: x0 y0 x1 y1")
    parser.add_argument("--output-dir", type=str, default="sam2_logs/comparison_viz", help="Root output directory.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-original", type=str, default=None, help="Specific device for original model (e.g. cuda:1)")
    parser.add_argument("--device-quantized", type=str, default=None, help="Specific device for quantized model (e.g. cuda:2)")
    parser.add_argument("--save-raw", action="store_true", help="Save raw attn/logits tensors for analysis.")
    parser.add_argument("--save-full-attn", action="store_true", help="Save full attention/logits for all query tokens.")
    parser.add_argument("--raw-qtokens", type=int, nargs="+", default=None, help="Optional list of query token indices to save.")
    return parser.parse_args()


def start_script(script_name: str, args: List[str], output_dir: Path) -> subprocess.Popen:
    # Removed --save-attn to prevent saving large tensor files
    cmd = [sys.executable, script_name] + args + ["--output-dir", str(output_dir), "--render-attn-overlay"]
    logging.info(f"Starting {script_name}...")
    return subprocess.Popen(cmd)


class ComparisonHTMLGenerator:
    def __init__(self, original_dir: Path, quantized_dir: Path, output_dir: Path):
        self.original_dir = original_dir
        self.quantized_dir = quantized_dir
        self.output_dir = output_dir
        self.report_path = output_dir / "comparison_report.html"

    def load_summary(self, dir_path: Path) -> Dict:
        path = dir_path / "layerwise_summary.json"
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def generate(self):
        orig_summary = self.load_summary(self.original_dir)
        quant_summary = self.load_summary(self.quantized_dir)
        
        orig_records = orig_summary.get("records", {})
        quant_records = quant_summary.get("records", {})
        
        # Merge keys
        all_keys = set(orig_records.keys()) | set(quant_records.keys())
        sorted_keys = sorted(list(all_keys), key=lambda k: (
            0 if "image_encoder" in k else 1,
            int(k.split('.')[3]) if "blocks" in k and k.split('.')[3].isdigit() else 999,
            k
        ))

        html = [
            "<!DOCTYPE html>", "<html>", "<head>",
            "<title>SAM2 Quantization Comparison</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }",
            ".container { max-width: 1800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }",
            "h1, h2 { text-align: center; color: #333; }",
            ".row { display: flex; margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }",
            ".col { flex: 1; padding: 15px; text-align: center; }",
            ".col-orig { background-color: #f0f7ff; border-right: 2px solid #ddd; }",
            ".col-quant { background-color: #fff0e6; }",
            ".col img { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".col h3 { margin-top: 0; font-size: 1.1em; color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 15px; }",
            ".meta { font-size: 0.85em; color: #777; font-family: monospace; margin-bottom: 10px; background: #fff; padding: 5px; display: inline-block; border-radius: 3px; }",
            ".layer-title { font-weight: bold; font-size: 1.3em; margin-bottom: 15px; display: block; background: #333; color: #fff; padding: 8px 15px; border-radius: 4px; }",
            ".diff-note { color: red; font-weight: bold; }",
            # Lightbox Styles
            ".lightbox { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); }",
            ".lightbox-content { margin: 5% auto; display: block; max-width: 90%; max-height: 90%; }",
            ".close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }",
            ".close:hover, .close:focus { color: #bbb; text-decoration: none; cursor: pointer; }",
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
            "</head>", "<body>",
            "<div id='lightbox' class='lightbox' onclick='closeLightbox()'>",
            "  <span class='close'>&times;</span>",
            "  <img class='lightbox-content' id='lightbox-img'>",
            "</div>",
            "<div class='container'>",
            "<h1>SAM2 Quantization Comparison</h1>",
            f"<div style='text-align:center; margin-bottom: 30px;'><span style='background:#f0f7ff; padding:5px 10px; border:1px solid #ccc; margin-right:10px;'>Original</span> <span style='background:#fff0e6; padding:5px 10px; border:1px solid #ccc;'>Quantized</span></div>"
        ]

        # Predictions Comparison
        html.append("<h2>Predictions</h2><div class='row'>")
        self._add_prediction_col(html, orig_summary, self.original_dir, "Original", "col-orig")
        self._add_prediction_col(html, quant_summary, self.quantized_dir, "Quantized", "col-quant")
        html.append("</div>")

        # Layerwise Comparison
        html.append("<h2>Layerwise Attention</h2>")
        for key in sorted_keys:
            html.append(f"<div><span class='layer-title'>{key}</span></div>")
            html.append("<div class='row'>")
            
            orig_data = orig_records.get(key, [{}])[0]
            quant_data = quant_records.get(key, [{}])[0]
            
            self._add_layer_col(html, orig_data, self.original_dir, "Original", "col-orig")
            self._add_layer_col(html, quant_data, self.quantized_dir, "Quantized", "col-quant")
            
            html.append("</div>")

        html.append("</div></body></html>")
        
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        logging.info(f"Saved comparison report to {self.report_path}")

    def _get_rel_path(self, abs_path_str: str, base_dir: Path) -> str:
        if not abs_path_str: return ""
        try:
            # We want path relative to output_dir (where html is)
            # abs_path_str is usually absolute path from recorder.
            p = Path(abs_path_str).resolve()
            root = self.output_dir.resolve()
            return str(p.relative_to(root))
        except ValueError:
            # Fallback
            return Path(abs_path_str).name

    def _add_prediction_col(self, html, summary, base_dir, title, css_class=""):
        preds = summary.get("prediction_images", [])
        html.append(f"<div class='col {css_class}'>")
        html.append(f"<h3>{title}</h3>")
        if not preds:
            html.append("<p>No predictions</p>")
        else:
            for i, pred in enumerate(preds):
                ov_path = pred.get('overlay')
                hist_path = pred.get('histogram')
                
                # If overlay is missing, try mask
                img_path = ov_path if ov_path else pred.get('mask')
                
                if img_path:
                    rel = self._get_rel_path(img_path, base_dir)
                    hist_rel = self._get_rel_path(hist_path, base_dir) if hist_path else None
                    
                    hist_attr = f"onclick=\"openLightbox('{hist_rel}')\" style='cursor:pointer'" if hist_rel else ""
                    caption_extra = " (Click for Hist)" if hist_rel else ""
                    
                    html.append(f"<div><img src='{rel}' {hist_attr}><p>Mask {i}{caption_extra}</p></div>")
                else:
                    html.append("<p>Image not found</p>")
        html.append("</div>")

    def _add_layer_col(self, html, data, base_dir, title, css_class=""):
        html.append(f"<div class='col {css_class}'>")
        # html.append(f"<h3>{title}</h3>")
        if not data:
            html.append("<p>N/A</p></div>")
            return

        # Metadata
        q_shape = data.get('q_shape', '-')
        heads = data.get('num_heads', '-')
        html.append(f"<div class='meta'>Heads: {heads} | Q: {q_shape}</div>")
        
        imgs = data.get('attn_overlay_images_all', [])
        hists = data.get('attn_histograms_all', [])
        
        if not imgs:
            html.append("<p>No images</p>")
        else:
            # Show first 2 queries max to save space
            for i, img_path in enumerate(imgs[:2]):
                rel = self._get_rel_path(img_path, base_dir)
                hist_rel = None
                if i < len(hists):
                    hist_rel = self._get_rel_path(hists[i], base_dir)
                
                hist_attr = f"onclick=\"openLightbox('{hist_rel}')\" style='cursor:pointer'" if hist_rel else ""
                caption_extra = " (Click for Hist)" if hist_rel else ""

                html.append(f"<div><img src='{rel}' {hist_attr}><p>Query {i}{caption_extra}</p></div>")
        html.append("</div>")


def main():
    args = parse_args()
    
    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    
    orig_dir = root_dir / "original"
    quant_dir = root_dir / "quantized"
    
    # Common arguments
    common_args = [
        "--image", args.image,
    ]
    if args.points:
        common_args += ["--points"] + [str(p) for p in args.points]
    if args.box:
        common_args += ["--box"] + [str(b) for b in args.box]

    # Resolve Original Config
    sam2_config = args.config_original
    if not sam2_config:
        # Infer from QAT config name if not provided
        fname = Path(args.config_quantized).name
        sam2_config = "configs/sam2/sam2_hiera_t.yaml" # Default fallback
        if "base_plus" in fname or "b+" in fname:
            sam2_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "large" in fname or "_l." in fname:
            sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "small" in fname or "_s." in fname:
            sam2_config = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "tiny" in fname or "_t." in fname:
            sam2_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    logging.info(f"Using Original Config: {sam2_config}")
    logging.info(f"Using Quantized Config: {args.config_quantized}")

    # Determine devices
    if args.device_original:
        dev_orig = args.device_original
    else:
        # Auto-assign if base device is cuda
        if "cuda" in args.device:
            dev_orig = "cuda:1"
        else:
            dev_orig = args.device
            
    if args.device_quantized:
        dev_quant = args.device_quantized
    else:
        # Auto-assign if base device is cuda
        if "cuda" in args.device:
            dev_quant = "cuda:2"
        else:
            dev_quant = args.device

    logging.info(f"Devices - Original: {dev_orig}, Quantized: {dev_quant}")

    orig_args = common_args + [
        "--config", sam2_config,
        "--checkpoint", args.checkpoint_original,
        "--device", dev_orig
    ]
    
    quant_args = common_args + [
        "--config", args.config_quantized,
        "--checkpoint", args.checkpoint_quantized,
        "--device", dev_quant
    ]
    if args.save_raw:
        orig_args += ["--save-raw-attn", "--save-raw-logits"]
        quant_args += ["--save-raw-attn", "--save-raw-logits"]
    if args.save_full_attn:
        orig_args += ["--save-full-attn"]
        quant_args += ["--save-full-attn"]
    if args.raw_qtokens:
        tok_args = ["--raw-qtokens"] + [str(v) for v in args.raw_qtokens]
        orig_args += tok_args
        quant_args += tok_args
    
    script_dir = Path(__file__).parent
    
    # Start both processes
    p_orig = start_script(str(script_dir / "visualize_layerwise.py"), orig_args, orig_dir)
    p_quant = start_script(str(script_dir / "visualize_layerwise_quantized.py"), quant_args, quant_dir)

    # Wait for completion
    exit_orig = p_orig.wait()
    exit_quant = p_quant.wait()
    
    if exit_orig != 0:
        logging.error(f"Original script failed with exit code {exit_orig}")
    if exit_quant != 0:
        logging.error(f"Quantized script failed with exit code {exit_quant}")
        
    if exit_orig != 0 or exit_quant != 0:
        sys.exit(1)

    # Generate Comparison Report
    gen = ComparisonHTMLGenerator(orig_dir, quant_dir, root_dir)
    gen.generate()


if __name__ == "__main__":
    main()
