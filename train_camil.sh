#!/usr/bin/env bash
set -e

python training/train.py \
--detector_path ./training/config/detector/camil.yaml \
--train_dataset "FaceForensics++" \
--test_dataset "Celeb-DF-v2" "$@"
