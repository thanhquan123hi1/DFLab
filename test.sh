python training/test.py \
--detector_path ./training/config/detector/biasln.yaml  \
--test_dataset  "Celeb-DF-v2" "UADFV" "DFDCP" \
--weights_path /path/to/biasln_weights.pth \
--save_feat \
--feat_out_dir /kaggle/tmp/tsne_pkls
