"""
Task 4 - third model: EfficientNet-B0 (frozen base), same protocol as Task 2 MobileNetV2.

Run (from the project root):
    python train_efficientnet_baseline.py --split_dir data/quality_split --out_dir results/efficientnetb0_frozen

Fixed parameters (course rules): epochs=50, patience=5, batch=32, input 224x224,
Adam lr=0.001, binary_crossentropy, seed = student ID.
Preprocessing: EfficientNet uses mode "torch" (ImageNet mean/std normalization),
NOT mode "tf" like MobileNetV2 - this is the family-correct preprocessing required by Task 4.

Label mapping (alphabetical folder order): bad = 0, good = 1.
"""
import argparse
import json
import os
import random
import shutil
import time
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SEED_ID = "AliOuici2026"
SEED = zlib.crc32(SEED_ID.encode())
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 50
PATIENCE = 5
LR = 0.001
DROPOUT = 0.3
DENSE_UNITS = 128
WEEK = 8
EXPERIMENT = "efficientnetb0_frozen"


def set_seeds():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_split(split_dir, split, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        Path(split_dir) / split, labels="inferred", label_mode="binary",
        class_names=["bad", "good"], color_mode="rgb", image_size=IMG_SIZE,
        batch_size=BATCH, shuffle=shuffle, seed=SEED if shuffle else None,
    )
    file_paths = list(ds.file_paths)
    class_names = list(ds.class_names)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), file_paths, class_names


def build_model():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(base.input, out)
    model.compile(optimizer=Adam(learning_rate=LR), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def measure_latency(model, file_paths):
    """Compiled-graph latency (tf.function), same protocol as Task 2: 20 warm-up + 100 timed, batch 1."""
    arrays = []
    for i in range(120):
        p = file_paths[i % len(file_paths)]
        img = tf.keras.utils.load_img(p, target_size=IMG_SIZE)
        arr = preprocess_input(tf.keras.utils.img_to_array(img))
        arrays.append(np.expand_dims(arr, 0).astype("float32"))

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
    def graph_fn(x):
        return model(x, training=False)

    for a in arrays[:20]:
        graph_fn(tf.constant(a)).numpy()
    times = []
    for a in arrays[20:120]:
        t0 = time.perf_counter()
        graph_fn(tf.constant(a)).numpy()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000.0)


def save_curves(history, path):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["loss"], label="train loss"); ax[0].plot(h["val_loss"], label="val loss")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["accuracy"], label="train acc"); ax[1].plot(h["val_accuracy"], label="val acc")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_confusion(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", type=Path, default=Path("data/quality_split"))
    ap.add_argument("--out_dir", type=Path, default=Path("results") / EXPERIMENT)
    args = ap.parse_args()

    set_seeds()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    train_ds, _, class_names = load_split(args.split_dir, "train", shuffle=True)
    val_ds, _, _ = load_split(args.split_dir, "validation", shuffle=False)
    test_ds, test_paths, _ = load_split(args.split_dir, "test", shuffle=False)
    print("Class order:", class_names, "(0 = bad, 1 = good)")

    model = build_model()
    model.summary()

    ckpt = out / "best_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
    ]
    t0 = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    train_time = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])
    save_curves(history, out / "training_curves.png")

    probs = model.predict(test_ds, verbose=0).ravel()
    y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
    y_pred = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    save_confusion(cm, class_names, out / "confusion_matrix.png")

    err_dir = out / "errors"
    if err_dir.exists():
        shutil.rmtree(err_dir)
    err_dir.mkdir(parents=True)
    for path, t, p in zip(test_paths, y_true, y_pred):
        if t != p:
            name = f"true-{class_names[t]}__pred-{class_names[p]}__{Path(path).name}"
            shutil.copy2(path, err_dir / name)

    latency_ms = measure_latency(model, test_paths)
    size_mb = ckpt.stat().st_size / 1e6
    gpus = tf.config.list_physical_devices("GPU")

    results = {
        "experiment": EXPERIMENT, "week": WEEK, "model": "EfficientNetB0", "regime": "frozen",
        "input_size": list(IMG_SIZE), "seed": SEED_ID, "device": "GPU" if gpus else "CPU",
        "epochs_planned": EPOCHS, "epochs_run": epochs_run, "train_time_s": round(train_time, 2),
        "latency_ms_per_image": round(latency_ms, 3), "model_size_mb": round(size_mb, 3),
        "params": int(model.count_params()), "test_samples": int(len(y_true)),
        "metrics": {
            "accuracy": acc, "f1_macro": float(f_m), "precision_macro": float(p_m), "recall_macro": float(r_m),
            "per_class": {class_names[i]: {"precision": float(p_c[i]), "recall": float(r_c[i]),
                                           "f1": float(f_c[i]), "support": int(s_c[i])} for i in range(2)},
        },
        "confusion_matrix": cm.tolist(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    config = {
        "seed_id": SEED_ID, "seed_int": SEED, "epochs": EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH, "input_size": list(IMG_SIZE), "optimizer": "Adam", "learning_rate": LR,
        "loss": "binary_crossentropy", "head": f"GAP -> Dense({DENSE_UNITS}, relu) -> Dropout({DROPOUT}) -> Dense(1, sigmoid)",
        "base": "EfficientNetB0 (ImageNet, frozen)", "preprocessing": "preprocess_input (mode torch)",
        "threshold": 0.5, "class_order": class_names,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n=== TEST RESULTS ===")
    print(f"samples: {len(y_true)}  accuracy: {acc:.4f}  F1 macro: {f_m:.4f}  "
          f"precision macro: {p_m:.4f}  recall macro: {r_m:.4f}")
    print("confusion matrix [rows = true bad, good; cols = predicted bad, good]:")
    print(cm)
    print(f"epochs run: {epochs_run}/{EPOCHS}  train time: {train_time:.1f}s")
    print(f"latency: {latency_ms:.2f} ms/image  model size: {size_mb:.2f} MB  params: {model.count_params()}")
    print(f"errors copied: {len(list(err_dir.iterdir()))}  ->  {err_dir}")
    print(f"Saved in: {out}")


if __name__ == "__main__":
    main()
