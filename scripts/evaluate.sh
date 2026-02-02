CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_sa1b.py \
  --data_root ../datasets/sa-1b_split/test \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --ckpt checkpoints/sam2.1_hiera_base_plus.pt \
  --eval_mode miou_point \
  --out_dir ./eval_results/SA1B/sam2.1_hiera_base_plus \
  --viz_percent 5 \
  --viz_metric miou