"""
Measure YOLOv8-cls latency on YOUR laptop CPU (same machine as Task 2) so the comparison is fair.
Forward pass only, batch 1, 640x640: 20 warm-up runs, then the mean of 100 timed runs.

Usage (project root, after downloading the Colab results to results\\):
    python measure_yolo_latency.py --run results/week7_yolov8n_cls
    python measure_yolo_latency.py --run results/week7_yolov8s_cls --update_results
--update_results writes the CPU latency and device into that run's results.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

IMGSZ = 640


def to_tensor(path):
    img = Image.open(path).convert("RGB").resize((IMGSZ, IMGSZ))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="folder that contains weights/best.pt")
    ap.add_argument("--test_dir", type=Path, default=Path("data/quality_split/test"))
    ap.add_argument("--update_results", action="store_true")
    args = ap.parse_args()

    weights = args.run / "weights" / "best.pt"
    nn_model = YOLO(str(weights)).model.float().to("cpu").eval()
    paths = sorted(p for p in args.test_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    tensors = [to_tensor(paths[i % len(paths)]) for i in range(120)]

    times = []
    with torch.inference_mode():
        for t in tensors[:20]:
            nn_model(t)
        for t in tensors[20:120]:
            t0 = time.perf_counter()
            nn_model(t)
            times.append(time.perf_counter() - t0)
    mean_ms, std_ms = float(np.mean(times) * 1000), float(np.std(times) * 1000)
    print(f"{args.run.name}: {mean_ms:.2f} ms/image on CPU (std {std_ms:.2f}), threads={torch.get_num_threads()}, cpus={os.cpu_count()}")

    (args.run / "latency_benchmark_cpu.json").write_text(json.dumps(
        {"mean_ms_per_image": round(mean_ms, 3), "std_ms": round(std_ms, 3), "device": "CPU",
         "protocol": "forward only, batch 1, 640x640, 20 warm-up + 100 timed"}, indent=2), encoding="utf-8")

    if args.update_results:
        rp = args.run / "results.json"
        d = json.loads(rp.read_text(encoding="utf-8"))
        d["latency_ms_per_image"] = round(mean_ms, 3)
        d["device"] = "CPU"
        rp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print("results.json updated.")


if __name__ == "__main__":
    main()
