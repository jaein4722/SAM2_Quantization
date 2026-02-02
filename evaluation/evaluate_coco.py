"""Evaluate SAM 2 on COCO val2017 using ground-truth boxes as prompts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "datasets/coco2017/val2017"
DEFAULT_ANN_FILE = REPO_ROOT / "datasets/coco2017/annotations/instances_val2017.json"


def _bbox_to_xyxy(bbox: List[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_bool = np.asarray(pred_mask, dtype=bool)
    gt_bool = np.asarray(gt_mask, dtype=bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _format_per_category_stats(
    cat_stats: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    formatted = {}
    for cat_name, stat in cat_stats.items():
        if stat["count"] == 0:
            continue
        formatted[cat_name] = stat["sum"] / stat["count"]
    return dict(sorted(formatted.items(), key=lambda kv: kv[0]))


def evaluate_coco_with_boxes(
    model_cfg: str,
    checkpoint: str,
    image_root: Path,
    ann_file: Path,
    device: str = "cuda",
    max_images: Optional[int] = None,
    multimask_output: bool = True,
    skip_crowd: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    image_root = image_root.expanduser().resolve()
    ann_file = ann_file.expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not ann_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    model = build_sam2(
        config_file=model_cfg,
        ckpt_path=checkpoint,
        device=device,
        apply_postprocessing=True,
    )
    predictor = SAM2ImagePredictor(model)
    cat_id_to_name = {
        cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())
    }

    per_image_rows: List[Dict[str, object]] = []
    cat_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"sum": 0.0, "count": 0.0}
    )
    total_iou = 0.0
    total_objects = 0

    with torch.inference_mode():
        for img_id in tqdm(img_ids, desc="Evaluating COCO"):
            img_info = coco.loadImgs([img_id])[0]
            img_path = image_root / img_info["file_name"]
            if not img_path.is_file():
                print(f"[WARN] Missing image file {img_path}, skipping.")
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                print(f"[WARN] Failed to open {img_path}: {exc}")
                continue

            predictor.set_image(image)
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            obj_ious: List[float] = []
            for ann in anns:
                if skip_crowd and ann.get("iscrowd", 0) == 1:
                    continue
                if not ann.get("segmentation"):
                    continue
                if ann.get("area", 0) <= 0:
                    continue

                bbox_xyxy = _bbox_to_xyxy(ann["bbox"])
                try:
                    masks_np, iou_pred, _ = predictor.predict(
                        box=bbox_xyxy,
                        multimask_output=multimask_output,
                    )
                except RuntimeError as exc:
                    print(f"[WARN] Predictor failed on {img_path}: {exc}")
                    continue

                best_idx = int(np.argmax(iou_pred))
                pred_mask = masks_np[best_idx]
                gt_mask = coco.annToMask(ann)
                iou = _compute_iou(pred_mask, gt_mask)
                obj_ious.append(iou)

                total_iou += iou
                total_objects += 1

                cat_name = cat_id_to_name.get(ann["category_id"], "unknown")
                cat_stats[cat_name]["sum"] += iou
                cat_stats[cat_name]["count"] += 1

            per_image_rows.append(
                {
                    "image_id": img_id,
                    "file_name": img_info["file_name"],
                    "num_objects": len(obj_ious),
                    "mean_iou": float(np.mean(obj_ious)) if obj_ious else 0.0,
                }
            )

    summary = {
        "num_images": len(per_image_rows),
        "num_instances": total_objects,
        "mean_iou": (total_iou / total_objects) if total_objects > 0 else 0.0,
        "per_category": _format_per_category_stats(cat_stats),
    }
    return summary, per_image_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 on COCO val2017 with GT boxes as prompts.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 checkpoint.")
    parser.add_argument(
        "--model-cfg",
        default=DEFAULT_MODEL_CFG,
        help="Hydra config for building SAM2.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=str(DEFAULT_IMAGE_ROOT),
        help="Path to COCO val2017 images.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=str(DEFAULT_ANN_FILE),
        help="Path to COCO instances_val2017.json file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device for SAM2.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on the number of images to evaluate.",
    )
    parser.add_argument(
        "--no-multimask",
        action="store_true",
        help="Disable multimask output during prediction.",
    )
    parser.add_argument(
        "--allow-crowd",
        action="store_true",
        help="Include COCO crowd annotations in evaluation.",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="Optional path to save summary metrics as JSON.",
    )
    parser.add_argument(
        "--per-image-json",
        type=str,
        default=None,
        help="Optional path to save per-image IoU results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, per_image_rows = evaluate_coco_with_boxes(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        image_root=Path(args.image_root),
        ann_file=Path(args.annotation_file),
        device=args.device,
        max_images=args.max_images,
        multimask_output=not args.no_multimask,
        skip_crowd=not args.allow_crowd,
    )

    print("\n===== COCO Bounding Box Prompt Evaluation =====")
    print(f"Images evaluated : {summary['num_images']}")
    print(f"Instances covered: {summary['num_instances']}")
    print(f"Mean IoU         : {summary['mean_iou']:.4f}")
    print("Top categories (by name, IoU):")
    for cat_name, iou in list(summary["per_category"].items())[:10]:
        print(f"  - {cat_name}: {iou:.4f}")

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary metrics to {metrics_path}")

    if args.per_image_json:
        per_image_path = Path(args.per_image_json)
        per_image_path.parent.mkdir(parents=True, exist_ok=True)
        per_image_path.write_text(json.dumps(per_image_rows, indent=2))
        print(f"Wrote per-image metrics to {per_image_path}")


if __name__ == "__main__":
    main()
