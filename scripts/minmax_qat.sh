#!/usr/bin/env bash
set -euo pipefail

exp_dir="sam2_minmax/W6A6_minmax_qat_base_plus_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 python -m projects.minmax_qat.train_minmax_qat \
    --config baselines/minmax_qat/configs/minmax_base_plus.yaml \
    --experiment-dir "$exp_dir"

CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_coco.py \
    --checkpoint "$exp_dir"/checkpoints/checkpoint_sam2.pt \
    --image-root ../datasets/coco2017/val2017 \
    --annotation-file ../datasets/coco2017/annotations/instances_val2017.json \
    --metrics-json "$exp_dir"/COCO_results/coco_val2017_metrics.json \
    --per-image-json "$exp_dir"/COCO_results/coco_val2017_per_image.json

CUDA_VISIBLE_DEVICES=0 python -m projects.sav_finetune.evaluate_sav \
    --checkpoint "$exp_dir"/checkpoints/checkpoint_sam2.pt \
    --video-root ../datasets/sa-v/sav_test/JPEGImages_24fps \
    --annotation-root ../datasets/sa-v/sav_test/Annotations_6fps \
    --video-list ../datasets/sa-v/sav_test/sav_test.txt \
    --output-root "$exp_dir"/sav_test_predictions \
    --metrics-json "$exp_dir"/sav_test_metrics.json

CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_sa1b.py \
    --data_root ../datasets/sa-1b_split/test \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
    --ckpt "$exp_dir"/checkpoints/checkpoint_sam2.pt \
    --eval_mode miou_point \
    --out_dir "$exp_dir"/SA1B_results \
    --viz_percent 5 \
    --viz_metric miou
