"""
Build data/quality_split/ from the Task 04 dataset (metadata.jsonl).

Result:
    data/quality_split/
        train/       good/  bad/
        validation/  good/  bad/
        test/        good/  bad/

The split values come from metadata.jsonl (assigned per SOURCE image in Task 04),
so there is no data leakage. No re-splitting happens here.

Usage (from the project root):
    python make_split_folders.py
    python make_split_folders.py --dataset_dir data/demo_quality_dataset --out_dir data/quality_split
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, default=Path("data/demo_quality_dataset"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/quality_split"))
    args = ap.parse_args()

    meta = args.dataset_dir / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found. Run build_good_bad_dataset.py first.")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    counts = Counter()
    with meta.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            split, label = r["split"], r["quality_label"]
            src = args.dataset_dir / r["image_path"]
            dst_dir = args.out_dir / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)
            counts[(split, label)] += 1

    print(f"Split folders created in: {args.out_dir}\n")
    print(f"{'split':<12}{'good':>8}{'bad':>8}{'total':>8}")
    grand = 0
    for split in ["train", "validation", "test"]:
        g, b = counts[(split, "good")], counts[(split, "bad")]
        grand += g + b
        print(f"{split:<12}{g:>8}{b:>8}{g + b:>8}")
    print(f"{'ALL':<12}{'':>16}{grand:>8}")


if __name__ == "__main__":
    main()
