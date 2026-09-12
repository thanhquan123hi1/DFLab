#!/bin/bash
# Chạy distributed evaluation trên 2 GPU
torchrun --nproc_per_node=2 training/test.py \
  --ddp \
  --detector_path ./training/config/detector/ln_sspanet_mil.yaml \
  --test_dataset "Celeb-DF-v2" "UADFV" "DFDCP" \
  --weights_path /path/to/ln_sspanet_mil_weights.pth \
  --save_feat
