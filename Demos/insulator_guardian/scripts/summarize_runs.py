"""Collect every ``vision_eval_*.json`` into one comparison table.

    python scripts/summarize_runs.py                 # 打印 + 写 results/benchmarks/training_summary.md
    python scripts/summarize_runs.py --focus broken  # 只看某一类，写 broken_summary.md

文件命名约定：``vision_eval_<run>_<split>.json``，内容里 ``result`` 带
``mAP50 / mAP50_95 / per_class``。脚本只做读与汇总，不改任何评测结果。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEMOS = Path(__file__).resolve().parents[2]
BENCH = DEMOS / "results" / "benchmarks"

CLASSES = ("normal", "flashover", "broken")


def load() -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for p in sorted(BENCH.glob("vision_eval_*.json")):
        stem = p.stem[len("vision_eval_"):]
        # 命名：<run>_<val|test>[_i<imgsz>]，960 分辨率的评测后缀并回 run 名
        m = re.match(r"^(?P<run>.+)_(?P<split>val|test)(?:_i(?P<imgsz>\d+))?$", stem)
        if not m:
            continue
        run, split = m.group("run"), m.group("split")
        if m.group("imgsz"):
            run = f"{run}_i{m.group('imgsz')}"
        data = json.loads(p.read_text(encoding="utf-8"))
        runs.setdefault(run, {})[split] = data.get("result", {})
    return runs


def fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def build_overall(runs: dict[str, dict[str, dict]]) -> str:
    lines = ["# 视觉线训练/评测汇总", "", "每格 `mAP50 / mAP50-95`，整体列是检测总体。", ""]
    for split in ("val", "test"):
        lines.append(f"## {split}")
        lines.append("")
        lines.append("| run | mAP50 | mAP50-95 | " + " | ".join(CLASSES) + " |")
        lines.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in CLASSES) + " |")
        rows = []
        for run, by_split in runs.items():
            r = by_split.get(split)
            if not r:
                continue
            pc = r.get("per_class", {})
            cells = []
            for c in CLASSES:
                info = pc.get(c)
                cells.append(f"{fmt(info['map50'])}/{fmt(info['map50_95'])}" if info else "—")
            rows.append((run, r.get("mAP50"), r.get("mAP50_95"), cells))
        rows.sort(key=lambda t: -(t[2] or 0))
        for run, m50, m5095, cells in rows:
            lines.append(f"| {run} | {fmt(m50)} | {fmt(m5095)} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_focus(runs: dict[str, dict[str, dict]], cls: str, include: list[str] | None) -> str:
    """只盯一个类：val/test 的 mAP50 与 mAP50-95 并排，方便对着一个类做对照。"""
    rows = []
    for run, by_split in runs.items():
        if include and not any(k in run for k in include):
            continue
        cells = []
        for split in ("val", "test"):
            info = (by_split.get(split) or {}).get("per_class", {}).get(cls)
            cells.append(f"{fmt(info['map50'])}/{fmt(info['map50_95'])}" if info else "—")
        if cells == ["—", "—"]:
            continue
        rows.append((run, cells))
    lines = [
        f"# {cls} 单类对照",
        "",
        "每格 `mAP50 / mAP50-95`；同一个 batch 内的 run 才能直接比。",
        "",
        "| run | val | test |",
        "| --- | ---: | ---: |",
    ]
    for run, cells in sorted(rows, key=lambda t: t[1][0], reverse=True):
        lines.append(f"| {run} | {cells[0]} | {cells[1]} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="汇总 results/benchmarks/vision_eval_*.json")
    ap.add_argument("--focus", default=None, choices=list(CLASSES), help="只看某一类，输出 <类名>_summary.md")
    ap.add_argument("--include", default=None, help="逗号分隔的 run 名子串过滤，例如 real_baseline,real_ca")
    args = ap.parse_args(argv)

    runs = load()
    include = [s.strip() for s in args.include.split(",") if s.strip()] if args.include else None
    if args.focus:
        text = build_focus(runs, args.focus, include)
        out = BENCH / f"{args.focus}_summary.md"
    else:
        text = build_overall(runs)
        out = BENCH / "training_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
