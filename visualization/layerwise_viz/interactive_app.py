import gradio as gr
import subprocess
import sys
import os
import io
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np
import re
import base64

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from visualization.interactive_viz.render import normalize_map, to_heatmap
from visualization.interactive_viz.io import infer_token_grid

# Default Paths
DEFAULT_QUANT_CONFIG = "sam2/configs/quantization/main_base_plus.yaml"
DEFAULT_ORIG_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_ORIG_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
# Updated default path based on user history
DEFAULT_QUANT_CHECKPOINT = "sam2_logs/ablations/adaptive_qat_toy_base_plus_20251112_101653/checkpoints/checkpoint.pt"
OUTPUT_ROOT = os.path.abspath("sam2_logs/interactive_viz") # Absolute path for allowed_paths
RUNS_DIR = Path(OUTPUT_ROOT) / "runs"
RUNS_INDEX = RUNS_DIR / "index.json"

def get_annotated_image(image, points, box_points):
    if image is None:
        return None
    
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw points
    for p in points:
        x, y, label = p
        color = "lime" if label == 1 else "red"
        r = 10
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline="white", width=2)
    
    # Draw temporary box points
    for p in box_points:
        x, y = p
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), fill="cyan", outline="white", width=2)
        
    return np.array(img)

def format_prompts(points, box_points):
    lines = []
    if points:
        lines.append("Points (x, y, label):")
        for p in points:
            lbl = "FG" if p[2] == 1 else "BG"
            lines.append(f"  - ({p[0]}, {p[1]}) : {lbl}")
    
    if box_points:
        lines.append("Box Points (Temp):")
        for p in box_points:
            lines.append(f"  - ({p[0]}, {p[1]})")
        if len(box_points) == 2:
            p1, p2 = box_points
            x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])
            lines.append(f"=> Box: [{x0}, {y0}, {x1}, {y1}]")
            
    if not lines:
        return "No prompts selected."
    return "\n".join(lines)

def on_image_select(evt: gr.SelectData, original_image_path, mode, points, box_points):
    # original_image_path comes from the input_image component itself when type="filepath"
    # When select is triggered on an image component, the value is passed automatically?
    # No, we must bind the inputs explicitly.
    
    if original_image_path is None:
        # Sometimes Gradio passes None or weird values if not fully loaded
        return None, points, box_points, "Please upload an image first."
    
    if isinstance(original_image_path, dict):
        # Gradio 4.x sometimes returns dict for image with 'path', 'url' etc if not configured right
        original_image_path = original_image_path.get('path')
        
    x, y = evt.index
    
    # Load original image to draw on fresh
    try:
        image = np.array(Image.open(original_image_path).convert("RGB"))
    except Exception as e:
        return None, points, box_points, f"Error loading image: {e}"
    
    # Important: Create new lists to ensure Gradio state updates
    new_points = list(points)
    new_box_points = list(box_points)
    
    if mode == "Point (Foreground)":
        new_points.append((x, y, 1))
    elif mode == "Point (Background)":
        new_points.append((x, y, 0))
    elif mode == "Box":
        new_box_points.append((x, y))
        if len(new_box_points) > 2:
             new_box_points = [(x, y)]

    annotated = get_annotated_image(image, new_points, new_box_points)
    
    # Draw box rectangle if we have exactly 2 points
    if len(new_box_points) == 2:
        img = Image.fromarray(annotated)
        draw = ImageDraw.Draw(img)
        p1, p2 = new_box_points
        x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])
        draw.rectangle((x0, y0, x1, y1), outline="cyan", width=3)
        annotated = np.array(img)

    prompt_str = format_prompts(new_points, new_box_points)
    return annotated, new_points, new_box_points, prompt_str

def clear_prompts(original_image_path):
    # Reset to original image (path)
    return original_image_path, [], [], "Prompts cleared."

def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {path}: {e}")
        return ""


def _load_report_html(run_dir: Path) -> str:
    report_path = run_dir / "comparison_report.html"
    if not report_path.exists():
        return ""
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            raw_html = f.read()

        def replace_with_compressed_base64(match):
            prefix = match.group(1) # src=' or src="
            rel_path = match.group(2) # original/... or quantized/...
            quote = match.group(3) # ' or "
            abs_path = run_dir / rel_path
            if abs_path.exists():
                try:
                    with open(abs_path, "rb") as img_f:
                        img = Image.open(img_f)
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=70)
                        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        return f"{prefix}data:image/jpeg;base64,{b64}{quote}"
                except Exception:
                    return match.group(0)
            return match.group(0)

        pattern = re.compile(r"(src=['\"])(?!http|data)([^'\"]+)(['\"])")
        embedded_html = pattern.sub(replace_with_compressed_base64, raw_html)
        return f"""
        <div style="width:100%; height:800px; overflow:auto; border:1px solid #ddd;">
            {embedded_html}
        </div>
        """
    except Exception as e:
        return f"<div style='color:red'>Error loading report: {e}</div>"


def _ensure_run_index() -> List[Dict]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNS_INDEX.exists():
        with RUNS_INDEX.open("w", encoding="utf-8") as f:
            json.dump([], f)
    with RUNS_INDEX.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_run_index(entry: Dict) -> None:
    runs = _ensure_run_index()
    runs = [r for r in runs if r.get("run_id") != entry.get("run_id")]
    runs.insert(0, entry)
    with RUNS_INDEX.open("w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)


def list_runs() -> List[Tuple[str, str]]:
    runs = _ensure_run_index()
    items = []
    for r in runs:
        label = f"{r.get('run_id')} | {r.get('image','')}"
        items.append((label, r.get("run_dir")))
    return items


def _select_latest_run() -> Optional[str]:
    runs = _ensure_run_index()
    if not runs:
        return None
    return runs[0].get("run_dir")

def run_visualization(
    original_image_path, points, box_points,
    quant_config, orig_config, orig_ckpt, quant_ckpt, device,
    save_raw, save_full_attn, raw_qtokens, generate_analysis, analysis_layers, analysis_heads
):
    if original_image_path is None:
        return None, None, "Please upload an image.", ""
    
    # Use the original image path directly (no conversion/saving)
    img_path = Path(original_image_path)
    
    if not points and len(box_points) != 2:
        return None, None, "Please add at least one point or a complete box (2 points).", ""
    
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Build Command
    cmd = [
        sys.executable,
        "visualization/layerwise_viz/compare_quantization.py",
        "--config-quantized", quant_config,
        "--config-original", orig_config,
        "--checkpoint-original", orig_ckpt,
        "--checkpoint-quantized", quant_ckpt,
        "--image", str(img_path),
        "--output-dir", str(run_dir),
        "--device", device
    ]
    if isinstance(raw_qtokens, str):
        raw_qtokens = [int(v.strip()) for v in raw_qtokens.split(",") if v.strip().isdigit()]
    if save_raw:
        cmd.append("--save-raw")
    if save_full_attn:
        cmd.append("--save-full-attn")
    if raw_qtokens:
        raw_qtokens = [v for v in raw_qtokens if v is not None]
        if raw_qtokens:
            cmd.append("--raw-qtokens")
            cmd.extend([str(v) for v in raw_qtokens])
    
    # Add points
    if points:
        pt_args = []
        for p in points:
            pt_args.extend([str(p[0]), str(p[1]), str(p[2])])
        cmd.append("--points")
        cmd.extend(pt_args)
        
    # Add box
    if len(box_points) == 2:
        p1, p2 = box_points
        x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])
        cmd.extend(["--box", str(x0), str(y0), str(x1), str(y1)])
        
    # Debug print
    print("="*50)
    print("Executing command:")
    print(" ".join(cmd))
    print("="*50)
    
    # Run
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() # Ensure current dir is in path
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        return None, None, f"Error running script: {e}", ""

    if generate_analysis:
        analysis_cmd = [
            sys.executable,
            "-m",
            "visualization.interactive_viz.generate",
            "--run_dir",
            str(run_dir),
        ]
        if analysis_layers:
            analysis_cmd += ["--layers", analysis_layers]
        if analysis_heads:
            analysis_cmd += ["--heads", analysis_heads]
        try:
            subprocess.check_call(analysis_cmd, env=env)
        except subprocess.CalledProcessError as e:
            return None, None, f"Error running analysis: {e}", ""
        
    # Collect Results
    report_path = run_dir / "comparison_report.html"
    
    # Collect some images for gallery
    gallery_imgs = []
    gallery_imgs = []
    
    orig_pred = run_dir / "original" / "prediction_images" / "mask_0_overlay.png"
    quant_pred = run_dir / "quantized" / "prediction_images" / "mask_0_overlay.png"
    
    if orig_pred.exists():
        gallery_imgs.append((str(orig_pred), "Original Prediction"))
    if quant_pred.exists():
        gallery_imgs.append((str(quant_pred), "Quantized Prediction"))
    
    html_content = _load_report_html(run_dir)
    _append_run_index(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "image": str(img_path),
            "timestamp": time.time(),
        }
    )
    # Update run_latest symlink if possible
    latest_link = Path(OUTPUT_ROOT) / "run_latest"
    try:
        if latest_link.is_symlink() or not latest_link.exists():
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(run_dir)
    except Exception:
        pass
        
    return str(report_path), gallery_imgs, f"Success! Run ID: {run_id}", html_content


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_layer_records(run_dir: Path) -> Dict:
    summary = _load_json(run_dir / "original" / "layerwise_summary.json")
    return summary.get("records", {})


def _get_layer_head_options(run_dir: Path) -> Tuple[List[str], List[str]]:
    records = _load_layer_records(run_dir)
    layers = sorted(records.keys())
    heads = []
    if layers:
        meta = records[layers[0]][0]
        n_heads = int(meta.get("num_heads", 0))
        heads = ["all"] + [str(i) for i in range(n_heads)]
    return layers, heads


def _resolve_raw_paths(run_dir: Path, layer: str, kind: str) -> Optional[Path]:
    summary = _load_json(run_dir / "layerwise_summary.json")
    recs = summary.get("records", {}).get(layer, [])
    if not recs:
        return None
    record = recs[0]
    key = "raw_attn_file" if kind == "attn" else "raw_logits_file"
    path = record.get(key)
    if not path:
        return None
    return Path(path)


def _load_raw_tensor(run_dir: Path, layer: str, kind: str) -> Optional[Dict[str, np.ndarray]]:
    path = _resolve_raw_paths(run_dir, layer, kind)
    if not path or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _pick_qtoken(
    q_indices: np.ndarray,
    q_grid: Optional[Tuple[int, int]],
    q_idx: Optional[int],
    qx: Optional[int],
    qy: Optional[int],
) -> Tuple[int, int]:
    if q_grid and qx is not None and qy is not None:
        q_idx = int(qy) * q_grid[1] + int(qx)
    if q_idx is None:
        q_idx = int(q_indices[0])
    q_idx = int(q_idx)
    # map to index in q_indices
    idx_list = [int(i) for i in q_indices.tolist()]
    if q_idx in idx_list:
        return q_idx, idx_list.index(q_idx)
    # fallback to nearest
    nearest = min(idx_list, key=lambda v: abs(v - q_idx))
    return nearest, idx_list.index(nearest)


def render_attention_view(
    run_dir_str: str,
    layer: str,
    head: str,
    q_idx: Optional[int],
    qx: Optional[int],
    qy: Optional[int],
    mode: str,
    norm_mode: str,
    topk_on: bool,
    topk_k: int,
    view_kind: str,
):
    run_dir = Path(run_dir_str)
    orig = _load_raw_tensor(run_dir / "original", layer, view_kind)
    quant = _load_raw_tensor(run_dir / "quantized", layer, view_kind)
    if not orig or not quant:
        return None, None, "Raw tensors not found. Enable --save-raw when running."
    data_key = "attn" if view_kind == "attn" else "logits"
    attn_fp = orig.get(data_key)
    attn_q = quant.get(data_key)
    if attn_fp is None or attn_q is None:
        return None, None, "Missing tensors in npz."
    q_indices = orig.get("q_indices")
    q_grid = orig.get("q_grid")
    k_grid = orig.get("k_grid")
    if q_indices is None:
        q_indices = np.arange(attn_fp.shape[1], dtype=np.int32)
    q_grid_tuple = tuple(q_grid.tolist()) if q_grid is not None else None
    k_grid_tuple = tuple(k_grid.tolist()) if k_grid is not None else None
    q_sel, q_sel_idx = _pick_qtoken(q_indices, q_grid_tuple, q_idx, qx, qy)

    if head == "all":
        attn_fp_h = attn_fp.mean(axis=0)[q_sel_idx]
        attn_q_h = attn_q.mean(axis=0)[q_sel_idx]
        head_label = "all"
    else:
        h = int(head)
        attn_fp_h = attn_fp[h, q_sel_idx]
        attn_q_h = attn_q[h, q_sel_idx]
        head_label = str(h)

    if topk_on:
        k = max(1, int(topk_k))
        kth = np.partition(attn_fp_h, -k)[-k]
        mask = attn_fp_h >= kth
        attn_fp_h = attn_fp_h * mask
        attn_q_h = attn_q_h * mask

    if mode == "FP":
        data = attn_fp_h
        ref = None
    elif mode == "Quant":
        data = attn_q_h
        ref = None
    elif mode == "Diff":
        data = attn_q_h - attn_fp_h
        ref = attn_fp_h
    elif mode == "AbsDiff":
        data = np.abs(attn_q_h - attn_fp_h)
        ref = attn_fp_h
    else:
        data = attn_q_h / (attn_fp_h + 1e-6)
        ref = attn_fp_h

    if k_grid_tuple:
        grid = data.reshape(k_grid_tuple)
    else:
        grid = data.reshape(1, -1)

    norm = normalize_map(
        grid,
        mode="fp-anchored" if norm_mode == "FP-anchored" else ("per-model" if norm_mode == "Per-model" else "shared"),
        ref=attn_fp_h.reshape(grid.shape),
    )
    display_size = None
    if k_grid_tuple:
        scale = max(4, int(256 / max(k_grid_tuple)))
        max_side = 768
        display_size = (
            min(max_side, k_grid_tuple[1] * scale),
            min(max_side, k_grid_tuple[0] * scale),
        )
    img = to_heatmap(norm, cmap="viridis", size=display_size)

    overlay = None
    summary = _load_json(run_dir / "original" / "layerwise_summary.json")
    image_path = summary.get("input_image")
    if image_path and Path(image_path).exists():
        base = Image.open(image_path).convert("RGB")
        if k_grid_tuple:
            heat = img.resize(base.size, Image.BILINEAR)
            heat = heat.convert("RGB")
            overlay = Image.blend(base, heat, alpha=0.5)
        else:
            overlay = base
        draw = ImageDraw.Draw(overlay)
        if q_grid_tuple:
            w, h = overlay.size
            gx = int((q_sel % q_grid_tuple[1]) * (w / q_grid_tuple[1]))
            gy = int((q_sel // q_grid_tuple[1]) * (h / q_grid_tuple[0]))
            draw.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), fill="cyan", outline="white")

    info = f"Layer={layer} | head={head_label} | qtoken={q_sel}"
    return img, overlay, info


def on_token_pick(evt: gr.SelectData, run_dir_str: str, layer: str):
    run_dir = Path(run_dir_str)
    summary = _load_json(run_dir / "original" / "layerwise_summary.json")
    record = summary.get("records", {}).get(layer, [{}])[0]
    q_grid = record.get("q_grid")
    q_tokens = record.get("q_tokens")
    if q_grid:
        grid = tuple(q_grid)
    else:
        grid = infer_token_grid(int(q_tokens)) if q_tokens else None
    image_path = summary.get("input_image")
    if not grid or not image_path or not Path(image_path).exists():
        return None, None, None
    img = Image.open(image_path)
    w, h = img.size
    x, y = evt.index
    gx = int(x / max(w / grid[1], 1))
    gy = int(y / max(h / grid[0], 1))
    gx = max(0, min(grid[1] - 1, gx))
    gy = max(0, min(grid[0] - 1, gy))
    q_idx = gy * grid[1] + gx
    return gx, gy, q_idx


def load_picker_image(run_dir_str: str):
    summary = _load_json(Path(run_dir_str) / "original" / "layerwise_summary.json")
    image_path = summary.get("input_image")
    if not image_path or not Path(image_path).exists():
        return None
    return np.array(Image.open(image_path).convert("RGB"))


def refresh_runs():
    runs = list_runs()
    choices = [label for label, _ in runs]
    values = {label: path for label, path in runs}
    latest = _select_latest_run()
    latest_label = None
    for label, path in runs:
        if path == latest:
            latest_label = label
            break
    return gr.update(choices=choices, value=latest_label), json.dumps(values)


def load_selected_run(selection: str, mapping_json: str):
    if not selection:
        return None, None, None, "No run selected."
    mapping = json.loads(mapping_json) if mapping_json else {}
    run_dir = mapping.get(selection)
    if not run_dir:
        return None, None, None, "Run path not found."
    run_path = Path(run_dir)
    report_file = run_path / "comparison_report.html"
    html = _load_report_html(run_path)
    # Quick preview
    gallery_imgs = []
    orig_pred = run_path / "original" / "prediction_images" / "mask_0_overlay.png"
    quant_pred = run_path / "quantized" / "prediction_images" / "mask_0_overlay.png"
    if orig_pred.exists():
        gallery_imgs.append((str(orig_pred), "Original Prediction"))
    if quant_pred.exists():
        gallery_imgs.append((str(quant_pred), "Quantized Prediction"))
    return str(report_file), gallery_imgs, html, f"Loaded run: {selection}"


def render_attention(
    run_dir_str,
    layer,
    head,
    q_idx,
    qx,
    qy,
    mode,
    norm_mode,
    topk_on,
    topk_k,
):
    return render_attention_view(
        run_dir_str,
        layer,
        head,
        q_idx,
        qx,
        qy,
        mode,
        norm_mode,
        topk_on,
        topk_k,
        "attn",
    )


def render_logits(
    run_dir_str,
    layer,
    head,
    q_idx,
    mode,
    norm_mode,
):
    return render_attention_view(
        run_dir_str,
        layer,
        head,
        q_idx,
        None,
        None,
        mode,
        norm_mode,
        False,
        0,
        "logits",
    )


def load_layers_heads(run_dir_str: str):
    layers, heads = _get_layer_head_options(Path(run_dir_str))
    layer_value = layers[0] if layers else None
    head_value = heads[0] if heads else None
    return gr.update(choices=layers, value=layer_value), gr.update(choices=heads, value=head_value)


def load_layers_only(run_dir_str: str):
    layers, _ = _get_layer_head_options(Path(run_dir_str))
    layer_value = layers[0] if layers else None
    return gr.update(choices=layers, value=layer_value)


def load_stats(run_dir_str: str, layer: str):
    run_dir = Path(run_dir_str)
    summary = _load_json(run_dir / "analysis" / "summary.json")
    if not summary:
        return {}, "Analysis summary not found."
    layer_data = summary.get("layers", {}).get(layer, {})
    quant_stats = _load_json(run_dir / "quantized" / "quant_stats.json")
    return {"layer_metrics": layer_data, "quant_stats": quant_stats}, "Loaded."


def load_output_impact(run_dir_str: str):
    run_dir = Path(run_dir_str)
    summary = _load_json(run_dir / "analysis" / "summary.json")
    impact = summary.get("output_impact", {})
    if not impact:
        return None, None, None, None, None, "Output impact not found."
    return (
        impact.get("orig_mask"),
        impact.get("quant_mask"),
        impact.get("diff"),
        impact.get("abs_diff"),
        impact.get("edge_weighted"),
        "Loaded.",
    )

with gr.Blocks(title="SAM2 Interactive Visualization") as demo:
    gr.Markdown("# SAM2 Interactive Layerwise Visualization")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="filepath", elem_id="input_image")
            original_image_state = gr.State(None)
            
            # Capture original path on upload
            input_image.upload(lambda x: x, inputs=input_image, outputs=original_image_state)
            # Clear state on clear
            input_image.clear(lambda: None, outputs=original_image_state)
            
            with gr.Row():
                mode = gr.Radio(
                    ["Point (Foreground)", "Point (Background)", "Box"],
                    label="Prompt Mode",
                    value="Point (Foreground)",
                    elem_id="prompt_mode"
                )
                clear_btn = gr.Button("Clear Prompts", elem_id="clear_prompts_btn")
            
            prompt_status = gr.Textbox(label="Current Prompts", value="No prompts selected.", lines=5, elem_id="prompt_status")

            # State for prompts
            points_state = gr.State([])
            box_state = gr.State([])
            
            with gr.Accordion("Configuration", open=True):
                quant_config = gr.Textbox(label="Quantized Config", value=DEFAULT_QUANT_CONFIG, elem_id="quant_config")
                orig_config = gr.Textbox(label="Original Config", value=DEFAULT_ORIG_CONFIG, elem_id="orig_config")
                orig_ckpt = gr.Textbox(label="Original Checkpoint", value=DEFAULT_ORIG_CHECKPOINT, elem_id="orig_ckpt")
                quant_ckpt = gr.Textbox(label="Quantized Checkpoint", value=DEFAULT_QUANT_CHECKPOINT, elem_id="quant_ckpt")
                device = gr.Textbox(label="Device", value="cuda", elem_id="device")
                save_raw = gr.Checkbox(label="Enable Raw Capture (Light)", value=True, elem_id="save_raw")
                save_full_attn = gr.Checkbox(label="Save Full Attention/Logits", value=False, elem_id="save_full_attn")
                raw_qtokens = gr.Textbox(label="Raw QTokens (comma-separated)", value="", elem_id="raw_qtokens")
                generate_analysis = gr.Checkbox(label="Generate Analysis After Run", value=True, elem_id="generate_analysis")
                analysis_layers = gr.Textbox(label="Analysis Layers", value="image_encoder_trunk_blocks_0:23", elem_id="analysis_layers")
                analysis_heads = gr.Textbox(label="Analysis Heads", value="all", elem_id="analysis_heads")
            with gr.Accordion("Run History", open=True):
                run_selector = gr.Dropdown(label="Saved Runs", choices=[], elem_id="run_selector")
                run_map_state = gr.State("{}")
                refresh_runs_btn = gr.Button("Refresh Runs", elem_id="refresh_runs_btn")
                load_run_btn = gr.Button("Load Selected Run", elem_id="load_run_btn")

            run_btn = gr.Button("Run Visualization", variant="primary", elem_id="run_btn")
            
        with gr.Column(scale=2):
            status = gr.Textbox(label="Status", interactive=False, elem_id="status")
            gallery = gr.Gallery(label="Quick Preview", elem_id="gallery")
            
            # Use Tab for better layout
            with gr.Tabs():
                with gr.TabItem("Report View"):
                    # Use Iframe to show HTML content directly
                    report_iframe = gr.HTML(label="Report Preview", elem_id="report_iframe")
                with gr.TabItem("File Download"):
                    report_file = gr.File(label="Download HTML Report", elem_id="report_file")
                with gr.TabItem("Attention"):
                    analysis_run_dir = gr.Textbox(label="Run Dir", value=str(Path(OUTPUT_ROOT) / "run_latest"), elem_id="analysis_run_dir")
                    layer_select = gr.Dropdown(label="Layer", choices=[], elem_id="layer_select")
                    head_select = gr.Dropdown(label="Head", choices=["all"], elem_id="head_select")
                    with gr.Row():
                        q_idx = gr.Number(label="QToken Index", value=0, elem_id="q_idx")
                        qx = gr.Number(label="QToken X", value=None, elem_id="q_x")
                        qy = gr.Number(label="QToken Y", value=None, elem_id="q_y")
                    token_picker = gr.Image(label="Token Picker", elem_id="token_picker")
                    token_load = gr.Button("Load Input Image", elem_id="token_load")
                    mode_select = gr.Radio(
                        ["FP", "Quant", "Diff", "AbsDiff", "Ratio"],
                        label="View Mode",
                        value="Diff",
                        elem_id="view_mode"
                    )
                    norm_select = gr.Radio(
                        ["Shared", "Per-model", "FP-anchored"],
                        label="Normalization",
                        value="Shared",
                        elem_id="norm_mode"
                    )
                    topk_on = gr.Checkbox(label="Top-k Mask", value=False, elem_id="topk_on")
                    topk_k = gr.Slider(label="Top-k", minimum=1, maximum=128, step=1, value=10, elem_id="topk_k")
                    attn_img = gr.Image(label="Attention/Logits Heatmap", elem_id="attn_img")
                    token_overlay = gr.Image(label="QToken Location", elem_id="token_overlay")
                    attn_info = gr.Textbox(label="Info", elem_id="attn_info")
                    load_btn = gr.Button("Load Layers/Heads", elem_id="load_layers_btn")
                    render_btn = gr.Button("Render", elem_id="render_btn")
                with gr.TabItem("Logits"):
                    logits_layer = gr.Dropdown(label="Layer", choices=[], elem_id="logits_layer")
                    logits_head = gr.Dropdown(label="Head", choices=["all"], elem_id="logits_head")
                    logits_q_idx = gr.Number(label="QToken Index", value=0, elem_id="logits_q_idx")
                    logits_mode = gr.Radio(
                        ["FP", "Quant", "Diff", "AbsDiff", "Ratio"],
                        label="View Mode",
                        value="Diff",
                        elem_id="logits_mode"
                    )
                    logits_norm = gr.Radio(
                        ["Shared", "Per-model", "FP-anchored"],
                        label="Normalization",
                        value="Shared",
                        elem_id="logits_norm"
                    )
                    logits_img = gr.Image(label="Logits Heatmap", elem_id="logits_img")
                    logits_overlay = gr.Image(visible=False, elem_id="logits_overlay")
                    logits_info = gr.Textbox(label="Info", elem_id="logits_info")
                    logits_render = gr.Button("Render", elem_id="logits_render")
                with gr.TabItem("Stats"):
                    stats_layer = gr.Dropdown(label="Layer", choices=[], elem_id="stats_layer")
                    stats_json = gr.JSON(label="Stats", elem_id="stats_json")
                    stats_status = gr.Textbox(label="Status", elem_id="stats_status")
                    stats_load = gr.Button("Load Stats", elem_id="stats_load")
                with gr.TabItem("Output Impact"):
                    impact_orig = gr.Image(label="FP Mask", elem_id="impact_orig")
                    impact_quant = gr.Image(label="Quant Mask", elem_id="impact_quant")
                    impact_diff = gr.Image(label="Diff Heatmap", elem_id="impact_diff")
                    impact_abs = gr.Image(label="Abs Diff Heatmap", elem_id="impact_abs")
                    impact_edge = gr.Image(label="Edge-weighted Diff", elem_id="impact_edge")
                    impact_status = gr.Textbox(label="Status", elem_id="impact_status")
                    impact_load = gr.Button("Load Output Impact", elem_id="impact_load")
            
    # Interactions
    input_image.select(
        on_image_select,
        inputs=[original_image_state, mode, points_state, box_state], # Do NOT pass input_image here, it triggers loop?
        # Wait, evt is implicit. The inputs list corresponds to arguments AFTER evt.
        # on_image_select(evt, original_image_path, mode, points, box_points)
        # So inputs should be [original_image_state, mode, points_state, box_state]
        outputs=[input_image, points_state, box_state, prompt_status]
    )
    
    clear_btn.click(
        clear_prompts,
        inputs=[original_image_state], # Return original image to reset view
        outputs=[input_image, points_state, box_state, prompt_status]
    )
    
    run_btn.click(
        run_visualization,
        inputs=[
            original_image_state, points_state, box_state,
            quant_config, orig_config, orig_ckpt, quant_ckpt, device,
            save_raw, save_full_attn, raw_qtokens, generate_analysis, analysis_layers, analysis_heads
        ],
        outputs=[report_file, gallery, status, report_iframe]
    )
    refresh_runs_btn.click(
        refresh_runs,
        inputs=[],
        outputs=[run_selector, run_map_state],
    )
    load_run_btn.click(
        load_selected_run,
        inputs=[run_selector, run_map_state],
        outputs=[report_file, gallery, report_iframe, status],
    )
    demo.load(
        refresh_runs,
        inputs=[],
        outputs=[run_selector, run_map_state],
    )

    load_btn.click(
        load_layers_heads,
        inputs=[analysis_run_dir],
        outputs=[layer_select, head_select],
    )
    load_btn.click(
        load_layers_heads,
        inputs=[analysis_run_dir],
        outputs=[logits_layer, logits_head],
    )
    load_btn.click(
        load_layers_only,
        inputs=[analysis_run_dir],
        outputs=[stats_layer],
    )

    render_btn.click(
        render_attention,
        inputs=[
            analysis_run_dir,
            layer_select,
            head_select,
            q_idx,
            qx,
            qy,
            mode_select,
            norm_select,
            topk_on,
            topk_k,
        ],
        outputs=[attn_img, token_overlay, attn_info],
    )
    token_load.click(
        load_picker_image,
        inputs=[analysis_run_dir],
        outputs=[token_picker],
    )
    token_picker.select(
        on_token_pick,
        inputs=[analysis_run_dir, layer_select],
        outputs=[qx, qy, q_idx],
    )
    logits_render.click(
        render_logits,
        inputs=[
            analysis_run_dir,
            logits_layer,
            logits_head,
            logits_q_idx,
            logits_mode,
            logits_norm,
        ],
        outputs=[logits_img, logits_overlay, logits_info],
    )
    stats_load.click(
        load_stats,
        inputs=[analysis_run_dir, stats_layer],
        outputs=[stats_json, stats_status],
    )
    impact_load.click(
        load_output_impact,
        inputs=[analysis_run_dir],
        outputs=[impact_orig, impact_quant, impact_diff, impact_abs, impact_edge, impact_status],
    )

if __name__ == "__main__":
    # Allow serving files from the output directory
    print(f"Allowed paths: {[OUTPUT_ROOT]}")
    demo.launch(server_name="0.0.0.0", share=True, allowed_paths=[OUTPUT_ROOT])
