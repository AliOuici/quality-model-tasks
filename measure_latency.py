"""
Re-measure inference latency of the saved MobileNetV2 model on CPU.
Protocol: 20 warm-up predictions, then the average over 100 timed predictions (1 image each).
Three modes are measured so the difference is visible:
    eager     : model(x, training=False)          (what train_mobilenetv2_baseline.py used)
    graph     : the same call wrapped in tf.function (compiled)
    predict   : model.predict(x)

Usage (project root):
    python measure_latency.py
    python measure_latency.py --model results/mobilenetv2_frozen/best_model.keras --test_dir data/quality_split/test
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)


def load_arrays(test_dir, n=120):
    paths = sorted(p for p in Path(test_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"No images in {test_dir}")
    arrays = []
    for i in range(n):
        img = tf.keras.utils.load_img(paths[i % len(paths)], target_size=IMG_SIZE)
        arr = preprocess_input(tf.keras.utils.img_to_array(img)).astype("float32")
        arrays.append(np.expand_dims(arr, 0))
    return arrays


def bench(fn, arrays):
    for a in arrays[:20]:          # warm-up
        fn(a)
    times = []
    for a in arrays[20:120]:       # timed
        t0 = time.perf_counter()
        fn(a)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000), float(np.std(times) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("results/mobilenetv2_frozen/best_model.keras"))
    ap.add_argument("--test_dir", type=Path, default=Path("data/quality_split/test"))
    ap.add_argument("--out", type=Path, default=Path("results/mobilenetv2_frozen/latency_benchmark.json"))
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    arrays = load_arrays(args.test_dir)

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
    def graph_fn(x):
        return model(x, training=False)

    modes = {
        "eager": lambda a: model(a, training=False).numpy(),
        "graph_tf_function": lambda a: graph_fn(tf.constant(a)).numpy(),
        "predict": lambda a: model.predict(a, verbose=0),
    }
    result = {"cpu_count": os.cpu_count(), "protocol": "20 warm-up + 100 timed, batch size 1"}
    for name, fn in modes.items():
        mean_ms, std_ms = bench(fn, arrays)
        result[name] = {"mean_ms_per_image": round(mean_ms, 3), "std_ms": round(std_ms, 3)}
        print(f"{name:<20} {mean_ms:8.2f} ms/image  (std {std_ms:.2f})")

    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
