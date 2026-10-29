"""Build an oversampled training list (flashover images repeated) for flashover experiments.

    python scripts/make_flash_oversample.py            # E2: FBID/yolo
    python scripts/make_flash_oversample.py --dataset MERGED   # E3: 合并 296 张新图后
"""
import argparse
import json
import os
from pathlib import Path

import yaml

DEMOS = Path(__file__).resolve().parents[2]  # E:/创新创业/Demos

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="FBID", help="训练样本/ 下的数据集目录名（FBID / MERGED）")
ap.add_argument("--tag", default=None, help="产物后缀，默认用数据集名小写")
ap.add_argument("--repeat-flash", type=int, default=3, help="含 flashover 的图在清单里出现几次（默认 3）")
ap.add_argument("--repeat-broken", type=int, default=1, help="含 broken 的图在清单里出现几次（默认 1，平衡实验用 2）")
args = ap.parse_args()

ROOT = DEMOS / "训练样本" / args.dataset
if (ROOT / "yolo").is_dir():  # FBID 用 yolo 子目录，MERGED 直接在根上
    ROOT = ROOT / "yolo"
TAG = args.tag or args.dataset.lower()
IMG_TRAIN = ROOT / "images/train"
LBL_TRAIN = ROOT / "labels/train"

imgs = sorted(p for p in IMG_TRAIN.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
flash, broken = [], []
for img in imgs:
    lbl = LBL_TRAIN / (img.stem + ".txt")
    classes = set()
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if parts:
                classes.add(int(float(parts[0])))
    if 1 in classes:
        flash.append(img)
    if 2 in classes:
        broken.append(img)


def rel(p: Path) -> str:
    # ultralytics 按 txt 所在目录解析相对路径；不带 ./ 会被当 cwd 相对路径而报 corrupt image
    return "./" + str(p.relative_to(ROOT)).replace("\\", "/")


lines = [rel(p) for p in imgs]
lines += [rel(p) for p in flash for _ in range(args.repeat_flash - 1)]
lines += [rel(p) for p in broken for _ in range(args.repeat_broken - 1)]

suffix = "flash%dx" % args.repeat_flash
if args.repeat_broken > 1:
    suffix += "_broken%dx" % args.repeat_broken
list_path = ROOT / ("train_%s.txt" % suffix)
list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

base = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
cfg = dict(base)
cfg["train"] = list_path.name
out = ROOT / ("data_%s.yaml" % suffix)
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

stats = {
    "dataset": args.dataset,
    "unique_train": len(imgs),
    "flashover_train": len(flash),
    "broken_train": len(broken),
    "repeat_flash": args.repeat_flash,
    "repeat_broken": args.repeat_broken,
    "list_entries": len(lines),
    "list": str(list_path),
    "yaml": str(out),
}
Path(DEMOS / "results/benchmarks").mkdir(parents=True, exist_ok=True)
(DEMOS / ("results/benchmarks/oversample_%s_%s.json" % (TAG, suffix))).write_text(
    json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(stats, ensure_ascii=False, indent=2))
