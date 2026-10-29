"""把外部 YOLO 数据集整理成项目可直接训练的结构。

当前支持的源是 FBID（xxlree/FBID）：
    源：<src>/images/{train,test}/*.jpg  +  <src>/labels/{train,test}/*.txt
    源类别顺序：0 pollution_flashover / 1 broken / 2 insulator
目标：
    <dst>/images/{train,val,test}/
    <dst>/labels/{train,val,test}/
    <dst>/data.yaml
    <dst>/dataset_stats.json

项目类别顺序（与 src/integration/schema.py 一致）：
    0 normal / 1 flashover / 2 broken

用法：
    python scripts/prepare_dataset.py --src ../训练样本/FBID/raw --dst data
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_CLASSES = ["normal", "flashover", "broken"]
FBID_CLASSES = ["pollution_flashover", "broken", "insulator"]
#: 源类别 id -> 项目类别 id
FBID_REMAP = {0: 1, 1: 2, 2: 0}
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """读 YOLO 标签，返回 [(cls, xc, yc, w, h)]；坏行直接跳过。"""
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            print(f"  [warn] {path.name}:{line_no} 字段数 {len(parts)}，跳过")
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            print(f"  [warn] {path.name}:{line_no} 不是数字，跳过")
            continue
        if cls not in FBID_REMAP:
            print(f"  [warn] {path.name}:{line_no} 未知类别 {cls}，跳过")
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            print(f"  [warn] {path.name}:{line_no} 坐标越界，跳过")
            continue
        boxes.append((FBID_REMAP[cls], xc, yc, w, h))
    return boxes


def collect(src: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """按源 split 收集可用样本；缺图或缺标签的样本都丢掉。"""
    splits = []
    for split in ("train", "test"):
        usable = []
        img_dir = src / "images" / split
        lab_dir = src / "labels" / split
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            label = lab_dir / (img.stem + ".txt")
            if not label.exists() or label.stat().st_size == 0:
                continue
            usable.append(img)
        splits.append(usable)
        print(f"[scan] {split}: {len(usable)} 张可用")
    return splits[0], splits[1], splits[2] if len(splits) > 2 else []


def stratify_key(boxes: list[tuple[int, float, float, float, float]]) -> int:
    """按「含不含缺陷」分层，保证 val 里缺陷样本比例接近训练集。"""
    kinds = {b[0] for b in boxes}
    return 2 if kinds == {0} else (1 if kinds else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="FBID raw 目录（含 images/ 与 labels/）")
    ap.add_argument("--dst", default="data", help="项目数据集目录，默认 data")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="从源 train 划出的验证集比例")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--copy", action="store_true", default=True, help="复制图片（默认）；--no-copy 只写标签清单")
    ap.add_argument("--no-copy", dest="copy", action="store_false")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "images" / "train").is_dir():
        print(f"[error] {src} 下没有 images/train，检查 --src")
        return 2

    train_all, test_all, _ = collect(src)
    if not train_all:
        print("[error] 源 train 没有可用样本")
        return 2

    # 分层切 val
    buckets: dict[int, list[Path]] = {}
    boxes_cache: dict[Path, list] = {}
    for img in train_all:
        boxes = parse_label(src / "labels" / "train" / (img.stem + ".txt"))
        boxes_cache[img] = boxes
        buckets.setdefault(stratify_key(boxes), []).append(img)
    rng = random.Random(args.seed)
    val: list[Path] = []
    train: list[Path] = []
    for key in sorted(buckets):
        items = buckets[key][:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * args.val_ratio)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    plan = {"train": train, "val": val, "test": test_all}
    stats: dict = {"source": str(src), "classes": PROJECT_CLASSES, "split": {}, "class_counts": {}}

    for split in SPLITS:
        img_out = dst / "images" / split
        lab_out = dst / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        count = Counter()
        for img in plan[split]:
            src_label = src / "labels" / ("train" if split in ("train", "val") else "test") / (img.stem + ".txt")
            boxes = boxes_cache.get(img) or parse_label(src_label)
            if not boxes:
                continue
            if args.copy:
                shutil.copy2(img, img_out / img.name)
            lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes]
            (lab_out / (img.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
            count.update(b[0] for b in boxes)
        stats["split"][split] = len(list(lab_out.glob("*.txt")))
        stats["class_counts"][split] = {PROJECT_CLASSES[i]: count.get(i, 0) for i in range(len(PROJECT_CLASSES))}
        print(f"[write] {split}: {stats['split'][split]} 张，类别计数 {stats['class_counts'][split]}")

    data_yaml = dst / "data.yaml"
    rel = lambda p: str(p.relative_to(dst)).replace("\\", "/")
    data_yaml.write_text(
        "# 由 scripts/prepare_dataset.py 生成；类别顺序与 src/integration/schema.py 一致\n"
        f"path: {str(dst).replace(chr(92), '/')}\n"
        f"train: {rel(dst / 'images' / 'train')}\n"
        f"val: {rel(dst / 'images' / 'val')}\n"
        f"test: {rel(dst / 'images' / 'test')}\n"
        f"nc: {len(PROJECT_CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(PROJECT_CLASSES)),
        encoding="utf-8",
    )
    stats["data_yaml"] = str(data_yaml)
    (dst / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] data.yaml -> {data_yaml}")
    print(json.dumps(stats["class_counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
