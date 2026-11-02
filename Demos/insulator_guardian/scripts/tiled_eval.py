"""切片推理对照：整图 640 vs 2×2 切片 640，看缺陷召回是否提升。

只做「召回/精确率」的窗口对照，不替代 mAP 评测：切片推理的收益主要出现在
小目标（broken 断串、远端闪络）上，整图缩放到 640 会把它们压到十几个像素。

用法::

    python scripts/tiled_eval.py --weights ../results/runs/real_degmix_b64_s1/weights/best.pt \\
        --images ../训练样本/FBID/yolo/images/test --labels ../训练样本/FBID/yolo/labels/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.schema import CLASS_NAMES  # noqa: E402
from src.utils.io import ensure_dir, save_json  # noqa: E402

IMGS = (".jpg", ".jpeg", ".png", ".bmp")


def load_gt(label_path: Path, width: int, height: int) -> np.ndarray:
    rows: list[list[float]] = []
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = (float(v) for v in parts[:5])
        rows.append([cls, cx * width, cy * height, w * width, h * height])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    out = np.empty_like(box)
    out[:, 0] = box[:, 0] - box[:, 2] / 2
    out[:, 1] = box[:, 1] - box[:, 3] / 2
    out[:, 2] = box[:, 0] + box[:, 2] / 2
    out[:, 3] = box[:, 1] + box[:, 3] / 2
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ae = xywh_to_xyxy(a[:, 1:5])
    be = xywh_to_xyxy(b[:, 1:5])
    ix1 = np.maximum(ae[:, None, 0], be[None, :, 0])
    iy1 = np.maximum(ae[:, None, 1], be[None, :, 1])
    ix2 = np.minimum(ae[:, None, 2], be[None, :, 2])
    iy2 = np.minimum(ae[:, None, 3], be[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(ae[:, 2] - ae[:, 0], 0, None) * np.clip(ae[:, 3] - ae[:, 1], 0, None)
    area_b = np.clip(be[:, 2] - be[:, 0], 0, None) * np.clip(be[:, 3] - be[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def nms(boxes: np.ndarray, thr: float = 0.55) -> np.ndarray:
    """按类别分组的贪心 NMS；boxes 为 [cls, cx, cy, w, h, conf]。"""
    if boxes.size == 0:
        return boxes
    keep: list[int] = []
    for cls in np.unique(boxes[:, 0]):
        idx = np.where(boxes[:, 0] == cls)[0]
        idx = idx[np.argsort(-boxes[idx, 5])]
        while idx.size:
            i = int(idx[0])
            keep.append(i)
            if idx.size == 1:
                break
            rest = idx[1:]
            ious = iou_matrix(boxes[[i]], boxes[rest])[0]
            idx = rest[ious < thr]
    return boxes[np.asarray(sorted(keep), dtype=int)]


def predict_full(model: Any, image_path: Path, imgsz: int, conf: float, device: str) -> list[list[float]]:
    res = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    names = res.names
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    cf = boxes.conf.cpu().numpy()
    out: list[list[float]] = []
    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, cf):
        name = str(names.get(int(c), "")).lower()
        cid = _class_id(name)
        if cid is None:
            continue
        out.append([cid, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, s])
    return out


def predict_tiled(model: Any, image_path: Path, imgsz: int, conf: float, device: str,
                  tiles: int = 2, overlap: float = 0.2) -> list[list[float]]:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    step_x = w / tiles
    step_y = h / tiles
    pad_x = int(step_x * overlap)
    pad_y = int(step_y * overlap)
    merged: list[list[float]] = []
    for ty in range(tiles):
        for tx in range(tiles):
            x1 = max(0, int(tx * step_x) - pad_x)
            y1 = max(0, int(ty * step_y) - pad_y)
            x2 = min(w, int((tx + 1) * step_x) + pad_x)
            y2 = min(h, int((ty + 1) * step_y) + pad_y)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            res = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            cf = boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), c, s in zip(xyxy, cls, cf):
                name = str(res.names.get(int(c), "")).lower()
                cid = _class_id(name)
                if cid is None:
                    continue
                merged.append([cid, (bx1 + bx2) / 2 + x1, (by1 + by2) / 2 + y1,
                               bx2 - bx1, by2 - by1, s])
    arr = np.asarray(merged, dtype=np.float32).reshape(-1, 6)
    return nms(arr).tolist() if arr.size else []


def _class_id(name: str) -> int | None:
    for cid, cname in CLASS_NAMES.items():
        if cname == name:
            return int(cid)
    return None


def match_ids(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> int:
    """同类别、每个 GT 最多匹配一次，返回命中数。gt: [cls,cx,cy,w,h]，pred: [cls,cx,cy,w,h,conf]。"""
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 5)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
    if gt.size == 0 or pred.size == 0:
        return 0
    hits = 0
    for cid in np.unique(gt[:, 0]):
        gi = np.where(gt[:, 0] == cid)[0]
        pi = np.where(pred[:, 0] == cid)[0]
        if gi.size == 0 or pi.size == 0:
            continue
        ious = iou_matrix(gt[gi], pred[pi])
        used = np.zeros(len(pi), dtype=bool)
        for row in range(len(gi)):
            for j in np.argsort(-ious[row]):
                if used[j] or ious[row, j] < iou_thr:
                    continue
                used[j] = True
                hits += 1
                break
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="切片推理 vs 整图推理的召回对照")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default="../训练样本/FBID/yolo/images/test")
    ap.add_argument("--labels", default="../训练样本/FBID/yolo/labels/test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    img_dir = Path(args.images)
    lab_dir = Path(args.labels)
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMGS)[: args.limit]

    import cv2

    stats = {"full": [0, 0, 0], "tiled": [0, 0, 0]}
    per_class = {name: {"full": [0, 0], "tiled": [0, 0]} for name in CLASS_NAMES.values()}
    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(lab_dir / f"{path.stem}.txt", w, h)
        full = predict_full(model, path, args.imgsz, args.conf, args.device)
        tiled = predict_tiled(model, path, args.imgsz, args.conf, args.device)
        for key, pred in (("full", full), ("tiled", tiled)):
            arr = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
            stats[key][0] += match_ids(gt, arr)
            stats[key][1] += int(gt.shape[0])
            stats[key][2] += int(arr.shape[0])
            for cid, cname in CLASS_NAMES.items():
                gt_c = gt[gt[:, 0] == cid]
                pr_c = arr[arr[:, 0] == cid] if arr.size else np.zeros((0, 6), dtype=np.float32)
                per_class[cname][key][0] += match_ids(gt_c, pr_c)
                per_class[cname][key][1] += int(gt_c.shape[0])
        if i % 20 == 0:
            print(f"[{i}/{len(files)}] full {stats['full']} tiled {stats['tiled']}")

    def recall(key: str) -> float:
        hit, n_gt, _ = stats[key]
        return 100.0 * hit / max(1, n_gt)

    print(f"[recall@0.5] full={recall('full'):.2f}% tiled={recall('tiled'):.2f}%")
    for cname, item in per_class.items():
        r_full = 100.0 * item["full"][0] / max(1, item["full"][1])
        r_tile = 100.0 * item["tiled"][0] / max(1, item["tiled"][1])
        print(f"  {cname:<9} full {r_full:5.1f}%  tiled {r_tile:5.1f}%  (GT {item['full'][1]})")

    out = Path(args.out_dir) if args.out_dir else PROJECT_ROOT.parent / "results" / "benchmarks"
    out = ensure_dir(out)
    payload = {
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "n_images": len(files),
        "full": {"hit": stats["full"][0], "gt": stats["full"][1], "pred": stats["full"][2],
                 "recall": recall("full")},
        "tiled": {"hit": stats["tiled"][0], "gt": stats["tiled"][1], "pred": stats["tiled"][2],
                  "recall": recall("tiled")},
        "per_class": per_class,
    }
    path = save_json(payload, out / "tiled_eval.json")
    print(f"[done] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
