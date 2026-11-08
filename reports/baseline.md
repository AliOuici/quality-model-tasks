# Task 2: MobileNetV2 Baseline (frozen)

Student: Ali Ouici, Group AIb-24 | Seed (student ID): AliOuici2026

## Setup
- Data: data/quality_split, 1800 images (900 good, 900 bad), split 70/15/15 by source image: train 1230, validation 282, test 288
- Input: RGB, 224 x 224, preprocess_input (mode tf, pixels in [-1, 1]), binary labels (bad = 0, good = 1), threshold 0.5
- Model: MobileNetV2 (ImageNet weights, frozen) -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(1, sigmoid)
- Parameters: 2,422,081 in total, of which 164,097 are trainable (the head)
- Training: Adam lr = 0.001, binary_crossentropy, batch 32, up to 50 epochs, early stopping patience 5 on val_loss with best weights restored, ModelCheckpoint (best only)
- Device: CPU (TensorFlow on native Windows, no GPU)
- Commands:
  - python make_split_folders.py
  - python train_mobilenetv2_baseline.py --split_dir data/quality_split --out_dir results/mobilenetv2_frozen

## Test results (test split only, 288 images)
| Metric | Value |
|---|---|
| Accuracy | 0.9931 |
| Precision (macro) | 0.9932 |
| Recall (macro) | 0.9931 |
| F1 (macro) | 0.9931 |
| Epochs run | 29 / 50 (best epoch 24) |
| Training time | 994.7 s (about 16.6 min) |
| Model size | 11.59 MB |
| Parameters | 2,422,081 |
| Latency (compiled graph, tf.function, batch 1, CPU; 20 warm-up + 100 timed) | 36.75 ms/image |

Latency benchmark (measure_latency.py, same protocol, same CPU, batch size 1):
| Mode | ms per image (mean) | Std |
|---|---|---|
| Compiled graph (tf.function) | 36.75 | 6.63 |
| model.predict | 100.10 | 11.06 |
| Eager call | 173.92 | 13.27 |

The first measurement inside the training script (eager call, taken right after training) gave 417.69 ms. A repeated eager measurement gave 173.92 ms, so the first number was most likely inflated by machine load and eager overhead. The compiled-graph value is used as the reference latency because a deployed service runs a compiled model. Full numbers: results/mobilenetv2_frozen/latency_benchmark.json.

Per class:
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bad | 0.9863 | 1.0000 | 0.9931 | 144 |
| good | 1.0000 | 0.9861 | 0.9930 | 144 |

Confusion matrix (rows = true, columns = predicted):
| | pred bad | pred good |
|---|---|---|
| true bad | 144 | 0 |
| true good | 2 | 142 |

Files: results/mobilenetv2_frozen/confusion_matrix.png, training_curves.png, results.json, config.json, errors/ (2 images)

## Comparison with the Task 1 acceptance criteria
| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Accuracy (synthetic test) | >= 0.91 | 0.9931 | Met |
| Macro F1 | >= 0.90 | 0.9931 | Met |
| Recall of "bad" | >= 0.90 | 1.0000 | Met |
| Glare detection | F1 >= 0.85 | All 144 bad test images were detected, so no glare image was missed (per-defect F1 not computed separately) | Met (recall) |
| Model size | <= 50 MB | 11.59 MB | Met |
| Latency | <= 100 ms | 36.75 ms (compiled graph); predict 100.10 ms; eager 173.92 ms | Met in compiled mode only |

## Interpretation
- Both curves show a fast and healthy fit: validation accuracy is already 0.97 after epoch 1 and stays around 0.99, while validation loss falls from 0.147 to about 0.03.
- Training loss keeps falling (to about 0.01) while validation loss stays flat at about 0.03 after epoch 20. This is a small sign of overfitting. Early stopping ended training at epoch 29 and restored the weights of the best epoch (24).
- Validation loss has a few spikes (epochs 12 and 22), which is normal noise on a 282-image validation set. There is no sign of underfitting.
- The model is not wrong on any bad photo (recall of bad = 1.0), so no bad photo would pass the gate. Its only 2 errors are good photos rejected as bad (recall of good = 0.986): good_passport_131_v2 and good_passport_132_v3. Both are unmodified copies (good_augmentation = none) of the same source image, passport_00010, so they are really one hard source image counted twice (1 of 96 test sources). The page of this passport is pale, with low contrast and faint text, which resembles the synthetic "low contrast" exposure defect, so the model probably confused it with a bad photo (hypothesis, not tested). For the pipeline this is the cheaper error: the user retakes the photo. A possible fix is to add pale but readable passports to the good training examples, or to tune the threshold on the validation split.
- The test score is very high because the defects are generated synthetically and are easy to detect. The known synthetic-to-real gap of 2.8 to 5.8% means real phone photos will probably score lower.
- Latency: the criterion (<= 100 ms per image) is met only when the model is compiled (36.75 ms). With model.predict it is borderline (100.10 ms) and with an eager call it is not met (173.92 ms). For deployment the model must therefore run as a compiled graph, or be exported to TFLite or ONNX (later task), and the other models must be measured with the same protocol so the comparison stays fair.