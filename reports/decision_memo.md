# Task 4: Decision Memo — Quality-Gate Model Selection

Student: Ali Ouici, Group AIb-24, IRNTU | Seed (student ID): AliOuici2026 | Date: November 2026

## 1. Decision

**MobileNetV2 (frozen base, 224x224) is selected as the quality gate (pipeline stage 1).**

## 2. Evidence

All four models were trained on the identical data/quality_split (70/15/15 by source image, 1800 total images) and evaluated on the same 288-image test split. Latency was measured on the same laptop CPU for all four models (20 warm-up + 100 timed, batch size 1).

| Model | Input | Accuracy | F1 macro | Recall "bad" | Latency (CPU) | Size |
|---|---|---|---|---|---|---|
| **MobileNetV2** | 224 | **0.9931** | **0.9931** | **1.000** | **36.75 ms** | 11.59 MB |
| EfficientNetB0 | 224 | 0.9861 | 0.9861 | 0.972 | 62.44 ms | 19.01 MB |
| YOLOv8s-cls | 640 | 0.9757 | 0.9757 | 0.951 | 278.74 ms | 10.26 MB |
| YOLOv8n-cls | 640 | 0.9722 | 0.9722 | 0.944 | 46.02 ms | 2.96 MB |

MobileNetV2 wins on the two metrics that matter most for this pipeline stage:
- **Zero bad photos reach OCR** (recall of "bad" = 1.000, confusion matrix [[144,0],[2,142]]). Every other model lets 4 to 8 bad photos through.
- **Fastest on the target CPU** (36.75 ms), well under the 100 ms budget defined in Task 1, and in fact faster than every alternative despite being evaluated at a larger effective parameter count than YOLOv8n.
- Model size (11.59 MB) is far under the 50 MB limit, so size was not a deciding factor among the four candidates.

## 3. Rejected alternatives and why

- **YOLOv8s-cls** — rejected outright: 278.74 ms per image is 2.8x over the 100 ms latency budget, and its latency is also highly variable (std 128 ms in the raw benchmark), making it unsuitable for a user-facing first gate regardless of its accuracy.
- **YOLOv8n-cls** — meets every numeric threshold from Task 1 (accuracy, F1, size, latency) but has the worst recall on "bad" (0.944, 8 missed bad photos) of all four models. Its advantage (smallest file, 2.96 MB) does not matter yet because the deployment target is a CPU server, not an edge device.
- **EfficientNetB0** — the strongest alternative: good accuracy (0.986) and reasonable latency (62.44 ms), but still worse than MobileNetV2 on both latency and bad-class recall (0.972 vs 1.000), and its 19.01 MB checkpoint is the largest of the four. Kept as the backup candidate if MobileNetV2's real-world performance degrades (see Risks).

## 4. Risks and mitigation

- **Synthetic-to-real gap (2.8-5.8%, capstone report, Table 7).** All four test accuracies are from synthetic data; real phone photos are expected to score lower. MobileNetV2's 99.3% has the largest safety margin above the 85% real-world target used in the Task 1 acceptance criteria. Mitigation: re-evaluate on a small set of real phone photos before full deployment, and re-check recall of "bad" specifically, since that is the metric the gap could hurt most.
- **Glare is the dominant failure mode.** Across all 21 test-set errors from all four models, 12 (57%) are glare, concentrated on the "inn" document class; MobileNetV2 itself made zero glare errors on this synthetic test set, but with only 288 test images and 0 observed glare failures its true glare recall is not precisely known. Mitigation: targeted glare augmentation on the training set (Task 5C), and an operator verification step for borderline (near-threshold) predictions rather than a hard auto-accept.
- **Only 2 MobileNetV2 errors exist in total**, both the same source passport image (pale, low-contrast page misread as bad). This is a positive result but also means the error-rate estimate is based on very few failure cases; it should be re-checked once more real-world or harder synthetic data is available.

## 5. Deployment note

CPU server now, no GPU required (36.75 ms per image is well inside the 100 ms budget on a standard CPU). If the product later needs on-device (mobile/edge) classification, MobileNetV2's small size (11.59 MB) and existing TFLite export path from Keras make it a straightforward fit; EfficientNetB0 remains the fallback if a future retrain of MobileNetV2 is required and underperforms this one.
