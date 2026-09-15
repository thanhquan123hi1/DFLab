# Kế hoạch và đánh giá LN + SSPANet + CE + MIL

## Quyết định thiết kế

Sửa trực tiếp detector đăng ký tên `biasln`, không tạo detector thay thế. Backbone CLIP ViT-L/14 chỉ tuning LayerNorm gamma/beta; patch token cuối tạo bản đồ Bx1024x16x16. Giữ nguyên ATTN_Block từ mã công khai của tác giả, rồi thêm fusion CLS/patch và patch classifier ở bên ngoài.

`local = GAP(SSPANet(patches))`

`z = normalize(normalize(CLS) + alpha * normalize(local))`

`L = weighted_CE(head(z), y) + lambda_mil * weighted_BCEWithLogits(mean(topk(patch_logits)), y)`

Mặc định alpha khởi tạo 0.1, top-k=16/256, lambda_mil=0.3. CE giữ weight_real=1, weight_fake=2 và label_smoothing=0.1 của baseline để không đồng thời thay đổi quá nhiều yếu tố. MIL dùng cùng class weights, chuẩn hóa theo tổng trọng số batch, không smoothing. Các hệ số là điểm khởi đầu thực nghiệm, không phải hyperparameter tối ưu đã được chứng minh. Với dữ liệu mất cân bằng, cần đánh giá lại trọng số lớp ở một ablation riêng.

Inference dùng head fusion; MIL là auxiliary objective, không trộn tùy tiện xác suất hai head lúc test. `cls_only_prob` là chẩn đoán bỏ nhánh local bằng cùng head đã train với fusion, KHÔNG phải một baseline được train độc lập. MIL top-k là thiết kế chuyển đổi của dự án; không gọi đây là code gốc của một bài contrastive MIL.

## Trình tự thực hiện và tiêu chí nghiệm thu

1. Xác minh nguồn SSPANet: pin commit, giữ file nguyên trạng, ghi khác biệt RMS/standard deviation và số tham số. Nghiệm thu bằng kiểm tra SHA256 và forward/backward trên feature âm.
2. Detector: LN-only, official block, fusion, MIL. Nghiệm thu tensor shapes, gradient đi tới LN/SSPA/head, bias khác trong CLIP không đổi, CE+MIL đúng giá trị, k sai bị từ chối.
3. Dữ liệu và đánh giá: validation split nguồn riêng; sampling theo DeepfakeBench (cập nhật theo yêu cầu ngày 2026-09-13, thay cho sampling trải đều); không thay ảnh lỗi bằng ảnh khác; video ID giữ cả đường dẫn method/dataset. Nghiệm thu trên fixture nhãn một lớp, video trùng tên, giới hạn mẫu.
4. Trainer: scheduler mỗi epoch, số epoch chính xác, lỗi NaN/Inf dừng ngay, log detached không giữ graph, checkpoint strict. Nghiệm thu một chu kỳ train/validation/save/reload/test bằng dữ liệu tổng hợp.
5. Dữ liệu thực: kiểm tra đường dẫn và JSON có `val`, thống kê số video/nhãn/method, bảo đảm source identity/video không giao giữa split. Chạy 1 epoch pilot trước 15 epoch; không đưa target test vào vòng chọn checkpoint.
6. Ablation 3 seed, sau đó 5 seed cho ứng viên cuối. Cùng split, preprocessing, loss weights, optimizer, checkpoint selection. Báo mean/std video AUC và EER, frame AUC, balanced accuracy, TPR ở FPR thấp, VRAM và throughput. Bootstrap CI nên resample VIDEO, không resample frame độc lập.

## Audit repo và thay đổi cần thiết

| Phát hiện | Ảnh hưởng | Xử lý |
|---|---|---|
| Tuning mọi bias | Không đúng LN-only | Bật train theo kiểu module nn.LayerNorm |
| Không có patch head | SSPANet/MIL chưa có đường gradient | Thêm trực tiếp vào BiasLNDetector |
| Scheduler.step nằm sau toàn vòng train | LR gần như cố định | Bước một lần mỗi epoch |
| range(nEpochs + 1) | 15 thành 16 epoch | nEpochs là số epoch |
| Chọn best theo target test, không lưu source FF++ | Lẫn selection với final evaluation | validation_dataset + validation_split + selection_dataset; lưu best nguồn |
| Video metrics dùng tên thư mục cuối | Có thể gộp nhầm method/video | Dùng toàn bộ parent path, kiểm tra nhãn trong video |
| Giảm total_frames trước khi tính step | Lấy frame đầu thay vì trải đều | Giữ hành vi upstream theo yêu cầu ngày 2026-09-13 |
| Ảnh lỗi thay bằng index 0 | Sai nhãn/path/metrics | Fail fast kèm đường dẫn |
| Recorder giữ tensor có graph | Tăng bộ nhớ trong train | detach/item ngay khi ghi |
| strict=False lúc test | Có thể test với nhánh mới ngẫu nhiên | Strict load; kiến trúc lấy từ checkpoint |
| max_samples không cắt batch/path đúng | Lệch độ dài arrays | Cắt chính xác và lưu tên từng mẫu |
| Log trung bình batch AUC | Không phải AUC tập mẫu | Train AUC tính trên toàn cửa sổ log; val/test trên toàn tập |
| DDP chỉ rank0 eval qua wrapper | Có nguy cơ treo BN/buffer collective | Barrier ở ranh giới epoch, eval unwrapped module |

Trainer được thu gọn cho detector duy nhất hiện có. Adam/SGD được hỗ trợ; SAM/SWA bị từ chối rõ ràng vì hai lượt forward và BN statistics cần kiểm chứng riêng. Checkpoint hiện là model + config + epoch, dùng cho evaluation hoặc khởi tạo trọng số, KHÔNG phải exact resume optimizer/scheduler/RNG. Những chức năng chưa được kiểm chứng không được quảng cáo là hoạt động.

## Log dùng để chẩn đoán

| Quan sát | Cách đọc / hành động tiếp theo |
|---|---|
| CE giảm, MIL không giảm | Kiểm tra grad_patch_head, grad_sspanet, class balance, k và lambda |
| MIL tốt, fusion kém | So sánh fusion_alpha, disagreement và cls-only diagnostic; thử fusion khác trong ablation |
| Alpha gần 0 | Nhánh local bị bỏ qua; đối chiếu baseline không SSPANet, không tự kết luận module vô ích |
| Patch entropy gần 1, std nhỏ | Patch score phẳng; có thể weak localization hoặc saturation; xem ảnh cụ thể |
| Patch score gần 0 ở cả hai lớp | Local head thiên real; kiểm tra real_mil_prob/fake_mil_prob và gradient |
| Train tốt, val kém | Overfit, BN domain shift, leakage/shortcut; kiểm tra chất lượng ảnh và method |
| AUC tốt, acc_real kém | Threshold/class-weight/calibration; xem confusion matrix, Brier/ECE |
| Gradient trước clip lớn kéo dài | Kiểm tra RMS gate, learning rate, lambda; không chỉ tăng clip |

Các file: train.jsonl, validation.jsonl, TensorBoard, config.json, trainable_parameters.json; validation predictions_best/last.npz chứa đường dẫn, nhãn, xác suất fusion/CLS/MIL và tối đa 16 bản đồ patch. Standalone test ghi metrics JSON và prediction NPZ, tối đa 32 bản đồ patch; `--save_feat` giữ đầu ra t-SNE.

Patch map là xác suất từ MIL head, không phải bản đồ attention nguyên bản, cũng không phải ground-truth localization. Không có mask thì heatmap chỉ là chẩn đoán. `patch_entropy` đo phân bố khối lượng sigmoid patch score, không phải entropy của attention gate. SSPA relative change đo norm phần residual so với input.

Train DDP log hiện là cửa sổ mẫu của rank0, không phải metric toàn bộ rank. Validation/test chạy toàn bộ tập trên rank0. BN dùng thống kê cục bộ khi train, buffer rank0 được dùng khi eval; chưa kiểm chứng multi-GPU trong môi trường này. Không tự thay SyncBN để giữ nguyên module gốc.

## Ablation bắt buộc

| ID | use_patch | use_sspanet | lambda_mil | Mục đích |
|---|---|---|---|---|
| A | false | false | 0 | LN + CLS + CE |
| B | true | false | 0.3 | Lợi ích của patch/MIL không SSPA |
| C | true | true | 0 | Lợi ích của SSPA không MIL |
| D | true | true | 0.3 | Kiến trúc yêu cầu |

Không đọc chất lượng MIL head ở run C vì head đó không được train. Đối chiếu thêm LN+patch mean pooling+CE nếu cần tách fusion khỏi MIL. Chỉ mở sweep k=8/16/32 và lambda=0.1/0.3 khi D có tín hiệu tích cực. Không chọn k/lambda dựa trên test đích.

## Hướng có thể tốt hơn, chưa mặc định triển khai

1. **Giám sát vùng với pseudo-fake và mask**: Forensics Adapter (CVPR 2025) dùng adapter chuyên biệt học dấu vết biên; LAA-Net (CVPR 2024) dùng heatmap/self-consistency supervision. Đây là hướng trực tiếp hơn top-k MIL để chống patch head tìm shortcut. Đổi lại cần pipeline blend/mask và đánh giá thêm trên fake không có biên ghép.
2. **MIL pooling học được**: tham khảo Attention-based Deep MIL (ICML 2018) và Contrastive Learning for DeepFake Classification and Localization via Multi-Label Ranking (CVPR 2024). Top-k cố định bỏ qua quy mô vùng sửa đổi; attention pooling hoặc multi-k là đối chứng hợp lý, nhưng phải kiểm soát số tham số và tránh suy diễn attention là localization đúng.
3. **Patch tầng trung gian**: CLIP cuối tầng chứa ngữ cảnh toàn ảnh; MIL không còn bảo đảm tính cục bộ. Thử tầng giữa hoặc kết hợp hai tầng, dựa trên động cơ giữ texture/local cues của Multi-Attentional Deepfake Detection (CVPR 2021). Việc áp dụng vào CLIP là giả thuyết mới cần ablation, không phải kết luận của bài đó.
4. **SSPA bottleneck / normalization khác**: 1024 kênh làm block ~7.35M tham số. Giảm kênh 1024->128 hoặc thay BN có thể tiết kiệm và ổn định domain shift, nhưng đây là BIẾN THỂ, không còn nguyên trạng code gốc. Chỉ thử sau đối chứng official.

Không có bằng chứng rằng SSPANet trên MRI sẽ vượt adapter thiết kế cho deepfake. Tính mới không nằm ở ghép tên module/loss; cần chứng minh nhánh RMS/strip khai thác dấu vết có khả năng tổng quát hóa và có lợi vượt baseline patch/MIL đơn giản.

## Nguồn ưu tiên

- SSPANet official: https://github.com/HelloJahid/SSPANet
- Contrastive MIL, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Hong_Contrastive_Learning_for_DeepFake_Classification_and_Localization_via_Multi-Label_Ranking_CVPR_2024_paper.html
- Forensics Adapter, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.html
- LAA-Net, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_LAA-Net_Localized_Artifact_Attention_Network_for_Quality-Agnostic_and_Generalizable_Deepfake_CVPR_2024_paper.pdf
- Attention MIL, ICML 2018: https://proceedings.mlr.press/v80/ilse18a.html
- Multi-Attentional Deepfake Detection, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Multi-Attentional_Deepfake_Detection_CVPR_2021_paper.html

## Chạy thực nghiệm

Sửa rgb_dir và dataset_json_folder trong training/config/train_config.yaml (train/validation) và test_config.yaml (final test). JSON nguồn phải có `train` và `val` đúng chuẩn DeepfakeBench. Không fallback val sang test. Đường dẫn mẫu hiện vẫn là Colab/Kaggle và không tồn tại trong workspace này.

```bash
python training/train.py --detector_path training/config/detector/biasln.yaml --train_dataset FaceForensics++
python training/test.py --weights_path /path/to/run/validation/FaceForensics++/ckpt_best.pth --test_dataset Celeb-DF-v2 DFDCP --output_dir /path/to/evaluation
python -m pytest tests/test_ln_sspanet_mil.py -q
python analysis/make_mil_ablations.py
python analysis/inspect_mil_predictions.py --predictions /path/to/evaluation/Celeb-DF-v2_predictions.npz --rgb_dir /path/to/rgb
```

Không dùng checkpoint BiasLN cũ để test kiến trúc mới: strict load sẽ báo thiếu SSPANet/MIL. `--weights_path` khi train là load model weights cùng kiến trúc, không phải resume đầy đủ. `--test_dataset` trên train được giữ cho tương thích CLI nhưng không điều khiển selection; hãy dùng validation_dataset trong detector YAML.
