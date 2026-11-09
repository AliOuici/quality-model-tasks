# Task 3: YOLOv8 vs MobileNetV2 (same split, same 288-image test set)

Student: Ali Ouici, Group AIb-24 | Seed (student ID): AliOuici2026

## Commands used
```
python make_split_folders.py                        # same data/quality_split as Task 2
python train_yolo_quality.py --model yolov8n-cls.pt --split_dir /content/data/quality_split --out_root /content/results   # Colab, T4 GPU
python train_yolo_quality.py --model yolov8s-cls.pt --split_dir /content/data/quality_split --out_root /content/results   # Colab, T4 GPU
python measure_yolo_latency.py --run results/week7_yolov8n_cls --update_results   # laptop CPU, same machine as Task 2
python measure_yolo_latency.py --run results/week7_yolov8s_cls --update_results
```
Fixed parameters: epochs 50, patience 5, batch 32, seed AliOuici2026. Input: 640x640 for YOLOv8 (course setting), 224x224 for MobileNetV2 (Task 2).
Training device: Google Colab, Tesla T4 GPU. Latency device for all three models: the same laptop CPU used in Task 2.

## Comparison table
| Model | Input | Accuracy | F1 macro | Precision macro | Recall macro | Recall "bad" | Latency CPU (ms/img) | Size (MB) | Params | Epochs run | Train time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 (frozen) | 224 | 0.9931 | 0.9931 | 0.9932 | 0.9931 | 1.000 | 36.75 | 11.59 | 2,422,081 | 29/50 | 994.7 s (CPU) |
| YOLOv8n-cls (full fine-tune) | 640 | 0.9722 | 0.9722 | 0.9737 | 0.9722 | 0.944 | 46.02 | 2.96 | 1,440,850 | 19/50 | 952.6 s (GPU) |
| YOLOv8s-cls (full fine-tune) | 640 | 0.9757 | 0.9757 | 0.9768 | 0.9757 | 0.951 | 278.74 | 10.26 | 5,083,298 | 10/50 | 533.9 s (GPU) |

Latency protocol: 20 warm-up runs, then the mean of 100 timed runs, batch size 1, forward pass only, on the laptop CPU. MobileNetV2 was timed as a compiled TensorFlow graph (tf.function, see Task 2); YOLOv8 as a PyTorch forward pass under torch.inference_mode (same protocol shape, different framework).

## Confusion matrices (test, rows = true, columns = predicted; order: bad, good)
- MobileNetV2: [[144, 0], [2, 142]] - the only 2 errors are good photos rejected as bad (recall of good = 0.986), and both are the same source passport image.
- YOLOv8n-cls: [[136, 8], [0, 144]] - 8 bad photos passed the gate as good; no good photo was rejected.
- YOLOv8s-cls: [[137, 7], [0, 144]] - 7 bad photos passed the gate as good; no good photo was rejected.

## Error analysis (the 15 misclassified bad photos, both YOLO models combined)
| Defect | Count |
|---|---|
| glare | 8 |
| exposure | 5 |
| occlusion | 1 |
| skew | 1 |

All 8 glare errors are on the "inn" document class. This matches the Task 1 acceptance criteria, which named glare as the expected hardest defect (reference F1 = 0.92 in the capstone report). Exposure is the second weakest defect, which fits a known risk of Ultralytics' default training augmentation (hsv_v=0.4 randomly changes brightness during training), making the model less sensitive to the real brightness defect at test time. MobileNetV2 training used no such color/exposure augmentation, which may explain why it made zero bad-class errors.

## Comparison with the Task 2 baseline (MobileNetV2)
Accuracy delta: both YOLOv8 variants score about 2 points lower than MobileNetV2 (97.2% and 97.6% vs 99.3%), and, more importantly for the pipeline, their recall on the "bad" class is 0.944 and 0.951 versus MobileNetV2's perfect 1.000 - so 7 to 8 bad photos out of 288 would reach OCR instead of being rejected, which is the expensive error type defined in Task 1. Latency delta is the most surprising result: despite MobileNetV2 being a classic CNN and YOLOv8n being built for speed, MobileNetV2 is actually the fastest on CPU (36.75 ms) because it runs at 224x224 while YOLOv8 runs at the course-mandated 640x640 (about 8x more pixels), so the much larger input size outweighs YOLOv8n's smaller parameter count (1.44M vs 2.42M). YOLOv8s is both the largest in latency (278.74 ms, clearly over the 100 ms budget) and highly variable (std 128 ms), which makes it unsuitable for a latency-sensitive first gate regardless of its slightly better accuracy than YOLOv8n. Size delta favors YOLOv8n (2.96 MB vs 11.59 MB), which would matter for a future mobile/edge deployment, but on the current CPU-server target speed and bad-class recall matter more than file size. The three models differ mainly in architecture and training regime: MobileNetV2 is frozen (only 164K head parameters are trained) while both YOLOv8 variants are fully fine-tuned (all parameters trained, hence their much faster convergence in fewer epochs: 19 and 10 vs 29), which together with the larger input and the color/exposure augmentation likely explains why YOLOv8 is less conservative and lets more bad photos through.

## Verdict against the Task 1 acceptance criteria
| Criterion | Threshold | MobileNetV2 | YOLOv8n-cls | YOLOv8s-cls |
|---|---|---|---|---|
| Accuracy | >= 0.91 | Met (0.9931) | Met (0.9722) | Met (0.9757) |
| Macro F1 | >= 0.90 | Met | Met | Met |
| Recall "bad" | >= 0.90 | Met (1.000) | Met (0.944) | Met (0.951) |
| Model size | <= 50 MB | Met | Met | Met |
| Latency | <= 100 ms | Met (36.75) | Met (46.02) | Not met (278.74) |

MobileNetV2 remains the strongest candidate for the quality gate after Task 3: it is the only model with zero bad photos reaching OCR, and it is also the fastest on the target CPU. YOLOv8s-cls is rejected outright on latency. YOLOv8n-cls meets every numeric threshold but is weaker on the one metric the pipeline cares about most (bad-class recall), so it is kept as an alternative pending the Task 4 comparison, not as the current recommendation.
