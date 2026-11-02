"""切片推理 vs 整图推理的召回对照（支持 NMS / NMM 两种合并方式）。

集外场景里绝缘子常常占满整幅图、缺陷只占其中几片伞裙：把大图直接缩到 640
会把这些细节抹平，切片推理能放大局部，但同一根绝缘子被切开后需要用合并把
框拼回去。两种合并的差别是：

* ``nms``：重叠就抑制掉弱的那一个 —— 会把被切开的框丢掉一半；
* ``nmm``：重叠/包含的同类框并成外接框（SAHI 的 Non-Maximum Merging）——
  能把切开的两半拼回完整一根，代价是框会稍微「长胖」。

用法::

    python scripts/tiled_eval.py --weights ../results/runs/real_degmix_b64_s1/weights/best.pt \
        --device 0 --limit 120 --tiles 2 --overlap 0.35 --merge both
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .plot_utils import setup_cjk_font
except ImportError:  # 直接 python scripts/xxx.py 时没有包上下文
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.integration.schema import CLASS_NAMES
from src.utils.config import PROJECT_ROOT
from src.utils.io import ensure_dir, save_json

IMGS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_gt(label_path: Path, width: int, height: int) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = (float(v) for v in parts[:5])
        rows.append([cls, cx * width, cy * height, w * width, h * height])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU 矩阵；输入按 cxcywh 解读，允许前面带类别/置信度列（取最后 4 列）。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    ae = xywh_to_xyxy(a[..., 1:5] if a.shape[-1] >= 5 else a)
    be = xywh_to_xyxy(b[..., 1:5] if b.shape[-1] >= 5 else b)
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


def nmm(boxes: np.ndarray, thr: float = 0.2, iom_thr: float = 0.35) -> np.ndarray:
    """SAHI 式非极大值合并；与 NMS 相反，重叠的同类框「并」而不是「抑制」。

    判定可合并：IoU >= thr，或小框有 >= iom_thr 的面积落在另一个框里
    （后者用于处理「大框套小框」的切片伪影）。合并后取外接框，置信度取最大。
    """
    if boxes.size == 0:
        return boxes
    out: list[list[float]] = []
    for cls in np.unique(boxes[:, 0]):
        pool = boxes[boxes[:, 0] == cls].astype(np.float32)
        pool = pool[np.argsort(-pool[:, 5])]
        while len(pool):
            head = pool[0:1]
            if len(pool) == 1:
                out.append(pool[0].tolist())
                break
            rest = pool[1:]
            ious = iou_matrix(head, rest)[0]
            ha = np.clip(head[0, 3], 0, None) * np.clip(head[0, 4], 0, None)
            rb = np.clip(rest[:, 3], 0, None) * np.clip(rest[:, 4], 0, None)
            # 用 IoU 反推的近似交并不可靠，单独算一次交集面积
            he = xywh_to_xyxy(head[0, 1:5])
            re = xywh_to_xyxy(rest[:, 1:5])
            ix1 = np.maximum(he[0], re[:, 0]);
            iy1 = np.maximum(he[1], re[:, 1])
            ix2 = np.minimum(he[2], re[:, 2]);
            iy2 = np.minimum(he[3], re[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            iom = inter / np.maximum(np.minimum(ha, rb), 1e-9)
            link = (ious >= thr) | (iom >= iom_thr)
            if not link.any():
                out.append(head[0].tolist())
                pool = rest
                continue
            group = rest[link]
            everyone = np.vstack([head, group])
            ext = xywh_to_xyxy(everyone[:, 1:5])
            x1, y1 = float(ext[:, 0].min()), float(ext[:, 1].min())
            x2, y2 = float(ext[:, 2].max()), float(ext[:, 3].max())
            out.append([cls,
                        (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
                        float(everyone[:, 5].max())])
            pool = rest[~link]
    return np.asarray(out, dtype=np.float32).reshape(-1, 6)


def _to_rows(res: Any, offset_x: int = 0, offset_y: int = 0) -> list[list[float]]:
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    cf = boxes.conf.cpu().numpy()
    out: list[list[float]] = []
    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, cf):
        cid = _class_id(str(res.names.get(int(c), "")).lower())
        if cid is None:
            continue
        out.append([cid, (x1 + x2) / 2 + offset_x, (y1 + y2) / 2 + offset_y,
                    x2 - x1, y2 - y1, s])
    return out


def predict_full(model: Any, image_path: Path, imgsz: int, conf: float, device: str) -> np.ndarray:
    res = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    return np.asarray(_to_rows(res), dtype=np.float32).reshape(-1, 6)


def predict_tiled(model: Any, image_path: Path, imgsz: int, conf: float, device: str,
                  tiles: int = 2, overlap: float = 0.2,
                  merge: str = "nms", merge_thr: float = 0.2) -> np.ndarray:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((0, 6), dtype=np.float32)
    h, w = img.shape[:2]
    step_x, step_y = w / tiles, h / tiles
    pad_x, pad_y = int(step_x * overlap), int(step_y * overlap)
    rows: list[list[float]] = []
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
            rows.extend(_to_rows(res, x1, y1))
    arr = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    if not arr.size:
        return arr
    if merge == "nmm":
        return nmm(arr, thr=merge_thr)
    return nms(arr, thr=max(merge_thr, 0.55) if merge_thr <= 0.5 else merge_thr)


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
    ap = argparse.ArgumentParser(description="切片推理 vs 整图推理的召回对照（NMS / NMM）")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default="../训练样本/FBID/yolo/images/test")
    ap.add_argument("--labels", default="../训练样本/FBID/yolo/labels/test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--tiles", type=int, default=2, help="每边切几片，2 = 2×2")
    ap.add_argument("--overlap", type=float, default=0.35, help="切片之间的重叠比例")
    ap.add_argument("--merge", choices=["nms", "nmm", "both"], default="both")
    ap.add_argument("--merge-thr", type=float, default=0.2, help="NMM 的 IoU 阈值（NMS 固定 0.55）")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMGS)[: args.limit]
    lab_dir = Path(args.labels)

    modes = ["full", f"tiled_{args.merge}" if args.merge != "both" else "tiled_nms"]
    if args.merge == "both":
        modes = ["full", "tiled_nms", "tiled_nmm"]
    stats = {m: [0, 0, 0] for m in modes}
    per_class = {name: {m: [0, 0] for m in modes} for name in CLASS_NAMES.values()}

    import cv2

    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(lab_dir / f"{path.stem}.txt", w, h)
        preds: dict[str, np.ndarray] = {"full": predict_full(model, path, args.imgsz, args.conf, args.device)}
        if args.merge in ("nms", "both"):
            preds["tiled_nms"] = predict_tiled(model, path, args.imgsz, args.conf, args.device,
                                               args.tiles, args.overlap, "nms", 0.55)
        if args.merge in ("nmm", "both"):
            preds["tiled_nmm"] = predict_tiled(model, path, args.imgsz, args.conf, args.device,
                                               args.tiles, args.overlap, "nmm", args.merge_thr)
        for key, arr in preds.items():
            stats[key][0] += match_ids(gt, arr)
            stats[key][1] += int(gt.shape[0])
            stats[key][2] += int(arr.shape[0])
            for cid, cname in CLASS_NAMES.items():
                gt_c = gt[gt[:, 0] == cid]
                pr_c = arr[arr[:, 0] == cid] if arr.size else np.zeros((0, 6), dtype=np.float32)
                per_class[cname][key][0] += match_ids(gt_c, pr_c)
                per_class[cname][key][1] += int(gt_c.shape[0])
        if i % 20 == 0:
            print(f"[{i}/{len(files)}] " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in stats.items()))

    def recall(key: str) -> float:
        hit, n_gt, _ = stats[key]
        return 100.0 * hit / max(1, n_gt)

    print(f"[recall@0.5] " + "  ".join(f"{k}={recall(k):.2f}%" for k in modes))
    for cname, item in per_class.items():
        line = "  ".join(f"{m} {100.0 * item[m][0] / max(1, item[m][1]):5.1f}%" for m in modes)
        print(f"  {cname:<9} {line}  (GT {item[modes[0]][1]})")

    out = ensure_dir(Path(args.out_dir) if args.out_dir else PROJECT_ROOT.parent / "results" / "benchmarks")
    payload = {
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "tiles": args.tiles,
        "overlap": args.overlap,
        "merge_thr": args.merge_thr,
        "n_images": len(files),
        "modes": {m: {"hit": stats[m][0], "gt": stats[m][1], "pred": stats[m][2],
                      "recall": recall(m)} for m in modes},
        "per_class": per_class,
    }
    path = save_json(payload, out / f"tiled_eval_{args.tiles}x{args.tiles}_ov{int(args.overlap * 100)}.json")
    print(f"[done] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
