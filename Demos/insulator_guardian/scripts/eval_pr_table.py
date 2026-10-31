#!/usr/bin/env python
"""对多组权重 × 多种测试条件（干净 / 困难）统一跑一遍，输出逐类 P / R / mAP。

    python scripts/eval_pr_table.py \
        --weights baseline=../results/runs/real_baseline_b64_v8n/weights/best.pt \
        --weights deg=../results/runs/real_deg_v8n/weights/best.pt \
        --weights mix=../results/runs/real_degmix_v8n/weights/best.pt \
        --test clean=../训练样本/DEG/data_s1.yaml \
        --test degraded=../训练样本/DEG/data_s1_testdeg.yaml

产物：results/benchmarks/pr_table_<日期>.json / .md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT.parent / "results"

# 直接 `python scripts/eval_pr_table.py` 时 sys.path[0] 是 scripts/，要手动补项目根
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_pairs(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for it in items or []:
        if "=" in it:
            label, p = it.split("=", 1)
        else:
            label, p = Path(it).stem, it
        out.append((label, Path(p)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="逐类 P/R/mAP 对照表（干净测试 vs 困难测试）")
    ap.add_argument("--weights", action="append", help="名字=权重路径，可重复")
    ap.add_argument("--test", action="append", help="条件名=data.yaml，可重复")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001, help="val 用的 conf（报告 P/R 时保持默认低阈值）")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    weights = _parse_pairs(args.weights)
    tests = _parse_pairs(args.test)
    if not weights or not tests:
        ap.error("至少给一个 --weights 和一个 --test")

    from ultralytics import YOLO

    from src.integration.schema import CLASS_NAMES

    results: dict[str, dict] = {}
    for wlabel, wpath in weights:
        if not wpath.exists():
            print(f"[skip] 权重不存在：{wpath}")
            continue
        model = YOLO(str(wpath))
        for tlabel, tyaml in tests:
            if not tyaml.exists():
                print(f"[skip] data.yaml 不存在：{tyaml}")
                continue
            print(f"[run] {wlabel} × {tlabel}", flush=True)
            r = model.val(
                data=str(tyaml), split=args.split, imgsz=args.imgsz,
                conf=args.conf, iou=0.45, workers=0, device=args.device,
                verbose=False, plots=False,
            )
            idx = list(getattr(r.box, "ap_class_index", []))
            entry = {
                "mAP50": float(r.box.map50),
                "mAP50_95": float(r.box.map),
                "per_class": {},
            }
            for k, cid in enumerate(idx):
                entry["per_class"][CLASS_NAMES.get(int(cid), str(cid))] = {
                    "precision": float(r.box.p[k]),
                    "recall": float(r.box.r[k]),
                    "mAP50": float(r.box.ap50[k]),
                    "mAP50_95": float(r.box.ap[k]),
                }
            results.setdefault(wlabel, {})[tlabel] = entry
            print(f"  mAP50={entry['mAP50']:.4f} mAP50-95={entry['mAP50_95']:.4f}", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").date.today().isoformat()
    (out_dir / f"pr_table_{stamp}.json").write_text(
        json.dumps({"weights": {k: str(v) for k, v in weights},
                    "tests": {k: str(v) for k, v in tests},
                    "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# 逐类 Precision / Recall 对照", "", "| 训练集 | 测试条件 | 类别 | P | R | mAP50 | mAP50-95 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for wlabel, by_test in results.items():
        for tlabel, e in by_test.items():
            for cname, m in e["per_class"].items():
                lines.append(
                    f"| {wlabel} | {tlabel} | {cname} | {m['precision']:.3f} | {m['recall']:.3f} "
                    f"| {m['mAP50']:.3f} | {m['mAP50_95']:.3f} |"
                )
            lines.append(
                f"| **{wlabel}** | **{tlabel}** | **全体** | — | — | **{e['mAP50']:.3f}** | **{e['mAP50_95']:.3f}** |"
            )
    (out_dir / f"pr_table_{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] {out_dir / f'pr_table_{stamp}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
