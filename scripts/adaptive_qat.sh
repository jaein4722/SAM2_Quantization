#!/usr/bin/env bash
set -euo pipefail

exp_dir=sam2_logs/adaptive_qat_toy_base_plus_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=6 python -m projects.adaptive_qat.train_adaptive_qat \
    --config projects/main/adaptive_qat/configs/toy_base_plus.yaml \
    --experiment-dir $exp_dir

CUDA_VISIBLE_DEVICES=6 python evaluation/evaluate_coco.py \
    --checkpoint $exp_dir/checkpoints/checkpoint.pt \
    --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
    --image-root ../datasets/coco2017/val2017 \
    --annotation-file ../datasets/coco2017/annotations/instances_val2017.json \
    --metrics-json "$exp_dir"/COCO_results/coco_val2017_metrics.json \
    --per-image-json "$exp_dir"/COCO_results/coco_val2017_per_image.json

CUDA_VISIBLE_DEVICES=6 python -m projects.sav_finetune.evaluate_sav \
    --checkpoint $exp_dir/checkpoints/checkpoint.pt \
    --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
    --video-root ../datasets/sa-v/sav_test/JPEGImages_24fps \
    --annotation-root ../datasets/sa-v/sav_test/Annotations_6fps \
    --video-list ../datasets/sa-v/sav_test/sav_test.txt \
    --output-root $exp_dir/sav_test_predictions \
    --metrics-json $exp_dir/sav_test_metrics.json

CUDA_VISIBLE_DEVICES=6 python evaluation/evaluate_sa1b.py \
  --data_root ../datasets/sa-1b_split/test \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --ckpt $exp_dir/checkpoints/checkpoint.pt \
  --eval_mode miou_point \
  --out_dir $exp_dir/SA1B_results\
  --viz_percent 5 \
  --viz_metric miou