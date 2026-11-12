"""
Task 4: build the failure gallery (>= 5 misclassified images per model where available)
and a per-model, per-defect error breakdown, read from metadata.jsonl so the dominant
defect is identified automatically (no manual counting).

Usage (project root):
    python build_failure_gallery.py
"""
import json
import re
import shutil
from collections import Counter
from pathlib import Path

RUNS = [
    ("MobileNetV2", "results/mobilenetv2_frozen/errors"),
    ("YOLOv8n-cls", "results/week7_yolov8n_cls/errors"),
    ("YOLOv8s-cls", "results/week7_yolov8s_cls/errors"),
    ("EfficientNetB0", "results/efficientnetb0_frozen/errors"),
]
METADATA = Path("data/demo_quality_dataset/metadata.jsonl")
OUT_DIR = Path("reports/failure_gallery")
MIN_PER_MODEL = 5


def load_defect_lookup():
    """image_id (filename stem without the bad_/good_ class-name prefix match) -> defect_type."""
    lookup = {}
    with METADATA.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lookup[r["image_id"]] = r.get("defect_type", "unknown")
    return lookup


def extract_image_id(error_filename):
    """true-bad__pred-good__bad_inn_glare_122_v2.jpg -> bad_inn_glare_122_v2"""
    m = re.match(r"true-\w+__pred-\w+__(.+)\.\w+$", error_filename)
    return m.group(1) if m else Path(error_filename).stem


def main():
    defect_lookup = load_defect_lookup() if METADATA.exists() else {}
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    summary = {}
    all_defect_counts = Counter()

    for label, err_dir in RUNS:
        err_dir = Path(err_dir)
        if not err_dir.exists():
            print(f"WARNING: {err_dir} not found, skipping {label}")
            continue
        files = sorted(err_dir.iterdir())
        model_out = OUT_DIR / label.replace(" ", "_")
        model_out.mkdir(parents=True)

        defect_counts = Counter()
        rows = []
        for f in files:
            image_id = extract_image_id(f.name)
            defect = defect_lookup.get(image_id, "unknown")
            defect_counts[defect] += 1
            all_defect_counts[defect] += 1
            shutil.copy2(f, model_out / f.name)
            true_label = "bad" if "true-bad" in f.name else "good"
            pred_label = "good" if "pred-good" in f.name else "bad"
            rows.append({"file": f.name, "true": true_label, "pred": pred_label, "defect": defect})

        dominant = defect_counts.most_common(1)[0][0] if defect_counts else "none (no errors)"
        summary[label] = {
            "total_errors": len(files),
            "defect_breakdown": dict(defect_counts),
            "dominant_defect": dominant,
            "meets_min_5": len(files) >= MIN_PER_MODEL,
            "errors": rows,
        }
        flag = "" if len(files) >= MIN_PER_MODEL else f"  <-- fewer than {MIN_PER_MODEL}, note this in the report"
        print(f"{label}: {len(files)} errors, dominant defect = {dominant}{flag}")
        for d, c in defect_counts.most_common():
            print(f"    {d}: {c}")

    overall_dominant = all_defect_counts.most_common(1)[0][0] if all_defect_counts else "none"
    summary["_overall"] = {"defect_counts_all_models": dict(all_defect_counts), "hardest_defect_overall": overall_dominant}

    (OUT_DIR / "failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nOverall hardest defect across all models: {overall_dominant}")
    print(f"All counts: {dict(all_defect_counts)}")
    print(f"\nGallery saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
