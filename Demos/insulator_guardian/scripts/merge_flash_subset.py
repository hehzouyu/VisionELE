"""Merge the FLASH-release images that FBID does not already contain into a training-only set.

Background: `zxc060413/Insulator-defect_yolo_format` (release v1.0.0) is the same public
collection as FBID for 82.5 % of its files. Only the remaining stems are new data, so this
script adds exactly those, remaps its classes to the project scheme, and keeps val/test from
FBID untouched (clean comparison against the previous runs).

FLASH class map -> project class ids
    0 insulator -> 0 normal
    1 damaged   -> 2 broken
    2 Flashover -> 1 flashover
    3 hammer    -> dropped (not a project class)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

REMAP = {0: 0, 1: 2, 2: 1, 3: None}
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}
DEMOS = Path(__file__).resolve().parents[2]


def stems_of(labels_dir: Path) -> set[str]:
    return {p.stem for p in labels_dir.glob("*.txt")}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label(src: Path, dst: Path) -> int:
    out_lines = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        new_cls = REMAP.get(cls)
        if new_cls is None:
            continue
        out_lines.append("%d %s" % (new_cls, " ".join(parts[1:5])))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return len(out_lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fbid", default=str(DEMOS / "训练样本/FBID/yolo"))
    ap.add_argument("--flash", default=str(DEMOS / "训练样本/FLASH"))
    ap.add_argument("--dst", default=str(DEMOS / "训练样本/MERGED"))
    args = ap.parse_args()

    fbid, flash, dst = Path(args.fbid), Path(args.flash), Path(args.dst)
    flash_images = flash / "images"
    if not flash_images.exists():
        raise SystemExit("FLASH images not extracted yet: %s" % flash_images)
    # the release zip contains "labels/labels/<split>/*.txt" (two nested labels dirs)
    flash_labels = flash / "labels"
    if (flash_labels / "labels").is_dir():
        flash_labels = flash_labels / "labels"

    fbid_stems = set()
    for split in ("train", "val", "test"):
        fbid_stems |= stems_of(fbid / "labels" / split)

    # copy FBID splits as-is
    for split in ("train", "val", "test"):
        for img in (fbid / "images" / split).iterdir():
            if img.suffix.lower() not in IMG_SUFFIX:
                continue
            link_or_copy(img, dst / "images" / split / img.name)
            lbl = fbid / "labels" / split / (img.stem + ".txt")
            if lbl.exists():
                link_or_copy(lbl, dst / "labels" / split / lbl.name)

    # find FLASH images by stem, add only new ones into train
    index = {}
    for p in flash_images.rglob("*"):
        if p.suffix.lower() in IMG_SUFFIX:
            index.setdefault(p.stem, p)

    added, boxes, skipped_empty = [], 0, 0
    for split in ("train", "val", "test"):
        lbl_dir = flash_labels / split
        for lbl in lbl_dir.glob("*.txt"):
            stem = lbl.stem
            if stem in fbid_stems:
                continue
            img = index.get(stem)
            if img is None:
                continue
            n_boxes = remap_label(lbl, dst / "labels" / "train" / (stem + ".txt"))
            if n_boxes == 0:
                (dst / "labels" / "train" / (stem + ".txt")).unlink(missing_ok=True)
                skipped_empty += 1
                continue
            link_or_copy(img, dst / "images" / "train" / img.name)
            added.append(stem)
            boxes += n_boxes

    base = yaml.safe_load((fbid / "data.yaml").read_text(encoding="utf-8"))
    cfg = dict(base)
    cfg["path"] = str(dst)
    cfg["train"] = "images/train"
    cfg["val"] = "images/val"
    cfg["test"] = "images/test"
    (dst / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    stats = {
        "source": "FBID train/val/test + zxc060413/Insulator-defect_yolo_format v1.0.0 (new stems only)",
        "fbid_train": len(list((fbid / "images/train").glob("*.*"))),
        "flash_new_images": len(added),
        "flash_new_boxes": boxes,
        "flash_skipped_empty": skipped_empty,
        "merged_train": len(list((dst / "images/train").glob("*.*"))),
        "merged_val": len(list((dst / "images/val").glob("*.*"))),
        "merged_test": len(list((dst / "images/test").glob("*.*"))),
        "class_map": {str(k): v for k, v in REMAP.items()},
    }
    (dst / "merge_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
