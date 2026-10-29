"""Sweep the confidence threshold and report per-class precision / recall / F1.

用于部署侧定阈值（mAP 是阈值无关指标，落地要按 P/R 曲线选 conf）。
默认只盯 broken 类，也可以 --classes 指定多个。

    ..\\.venv\\Scripts\\python.exe scripts/sweep_broken_conf.py --weights ../results/runs/real_ca_sppf_b64_v8n/weights/best.pt --split val --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEMOS = Path(__file__).resolve().parents[2]
PROJECT = DEMOS / "insulator_guardian"
RESULTS = DEMOS / "results"
sys.path.insert(0, str(PROJECT))

from src.utils.config import load_config  # noqa: E402
from src.utils.io import ensure_dir, iso_now  # noqa: E402
from src.integration.schema import CLASS_NAMES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="扫 confidence 阈值，输出逐类 P/R/F1")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data-yaml", type=Path, default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-min", type=float, default=0.05)
    ap.add_argument("--conf-max", type=float, default=0.95)
    ap.add_argument("--conf-step", type=float, default=0.05)
    ap.add_argument("--focus", default="broken", help="重点类，逗号分隔，默认 broken")
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "benchmarks")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    vision_cfg = load_config("vision")
    data_yaml = args.data_yaml
    if data_yaml is None:
        from src.vision.train import build_data_yaml

        data_yaml = build_data_yaml(vision_cfg.get("data"))

    model = YOLO(str(args.weights))
    grid = np.arange(args.conf_min, args.conf_max + 1e-9, args.conf_step)
    rows = []
    for conf in grid:
        metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=int(args.imgsz),
            conf=float(round(conf, 2)),
            iou=0.45,
            device=str(args.device),
            workers=0,
            verbose=False,
        )
        box = metrics.box
        names = dict(getattr(metrics, "names", {}) or CLASS_NAMES)
        entry = {"conf": float(round(conf, 2)), "classes": {}}
        for pos, cls_id in enumerate(getattr(box, "ap_class_index", [])):
            name = str(names.get(int(cls_id), cls_id))
            p = float(box.p[pos])
            r = float(box.r[pos])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            entry["classes"][name] = {"precision": p, "recall": r, "f1": f1}
        rows.append(entry)

    focus = [s.strip() for s in args.focus.split(",") if s.strip()]
    out_dir = ensure_dir(args.out_dir)
    tag = f"{args.weights.parent.parent.name}_{args.split}"
    meta = {"generated_at": iso_now(), "weights": str(args.weights), "split": args.split,
            "imgsz": int(args.imgsz), "focus": focus}
    json_path = out_dir / f"conf_sweep_{tag}.json"
    json_path.write_text(json.dumps({**meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# confidence 阈值扫描（{tag}）", "",
             f"权重 `{args.weights.name}`，split={args.split}，imgsz={args.imgsz}，iou=0.45。", ""]
    for cls in focus:
        lines.append(f"## {cls}")
        lines.append("")
        lines.append("| conf | precision | recall | F1 |")
        lines.append("| ---: | ---: | ---: | ---: |")
        best = None
        for entry in rows:
            info = entry["classes"].get(cls)
            if not info:
                continue
            lines.append(f"| {entry['conf']:.2f} | {info['precision']:.3f} | "
                         f"{info['recall']:.3f} | {info['f1']:.3f} |")
            if best is None or info["f1"] > best[1]["f1"]:
                best = (entry["conf"], info)
        if best:
            lines.append("")
            lines.append(f"F1 最优点：conf={best[0]:.2f}（P={best[1]['precision']:.3f}，"
                         f"R={best[1]['recall']:.3f}，F1={best[1]['f1']:.3f}）")
        lines.append("")
    md_path = out_dir / f"conf_sweep_{tag}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] {json_path}")
    print(f"[done] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
