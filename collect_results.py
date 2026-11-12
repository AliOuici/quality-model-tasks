"""
Task 4: build the full comparison table and charts from results.json files ONLY
(MobileNetV2, YOLOv8n-cls, YOLOv8s-cls, EfficientNetB0 - no numbers typed by hand).

Usage (project root):
    python collect_results.py

Reads:
    results/mobilenetv2_frozen/results.json
    results/week7_yolov8n_cls/results.json
    results/week7_yolov8s_cls/results.json
    results/efficientnetb0_frozen/results.json

Writes:
    reports/comparison_table.md
    reports/comparison_table.csv
    reports/accuracy_vs_latency.png
    reports/confusion_matrices_grid.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    ("MobileNetV2", "results/mobilenetv2_frozen/results.json"),
    ("YOLOv8n-cls", "results/week7_yolov8n_cls/results.json"),
    ("YOLOv8s-cls", "results/week7_yolov8s_cls/results.json"),
    ("EfficientNetB0", "results/efficientnetb0_frozen/results.json"),
]

OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all():
    rows = []
    for label, path in RUNS:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: missing {path} -- skipping {label}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_label"] = label
        rows.append(d)
    return rows


def write_table(rows):
    cols = ["Model", "Input", "Accuracy", "F1 macro", "Precision macro", "Recall macro",
            "Recall bad", "Latency (ms)", "Size (MB)", "Params", "Epochs run", "Train time (s)"]
    md_lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    csv_lines = [",".join(cols)]

    for d in rows:
        m = d["metrics"]
        inp = "x".join(str(x) for x in d["input_size"])
        vals = [
            d["_label"], inp,
            f"{m['accuracy']:.4f}", f"{m['f1_macro']:.4f}", f"{m['precision_macro']:.4f}", f"{m['recall_macro']:.4f}",
            f"{m['per_class']['bad']['recall']:.3f}",
            f"{d['latency_ms_per_image']:.2f}", f"{d['model_size_mb']:.2f}", str(d["params"]),
            f"{d['epochs_run']}/{d['epochs_planned']}", f"{d['train_time_s']:.1f}",
        ]
        md_lines.append("| " + " | ".join(vals) + " |")
        csv_lines.append(",".join(vals))

    (OUT_DIR / "comparison_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    (OUT_DIR / "comparison_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    print("\n".join(md_lines))


def plot_accuracy_vs_latency(rows):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for d in rows:
        ax.scatter(d["latency_ms_per_image"], d["metrics"]["accuracy"] * 100, s=120)
        ax.annotate(d["_label"], (d["latency_ms_per_image"], d["metrics"]["accuracy"] * 100),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.axvline(100, color="red", linestyle="--", linewidth=1, label="100 ms budget (Task 1)")
    ax.set_xlabel("Latency on laptop CPU (ms/image, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    ax.set_title("Accuracy vs latency (all models, same CPU, same 288-image test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_vs_latency.png", dpi=150)
    plt.close(fig)


def plot_confusion_grid(rows):
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    class_names = ["bad", "good"]
    for ax, d in zip(axes, rows):
        cm = np.array(d["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(2)); ax.set_xticklabels(class_names)
        ax.set_yticks(range(2)); ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted"); ax.set_title(d["_label"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_ylabel("true")
    fig.suptitle("Confusion matrices (test split, 288 images each)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices_grid.png", dpi=150)
    plt.close(fig)


def main():
    rows = load_all()
    if not rows:
        raise SystemExit("No results.json files found. Run Tasks 2-4 training first.")
    write_table(rows)
    plot_accuracy_vs_latency(rows)
    plot_confusion_grid(rows)
    print(f"\nSaved: {OUT_DIR}/comparison_table.md, comparison_table.csv, "
          f"accuracy_vs_latency.png, confusion_matrices_grid.png")


if __name__ == "__main__":
    main()
