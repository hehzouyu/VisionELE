"""评测入口：在验证/测试集上跑 CA-YOLO 的 mAP，并落盘成 JSON + 中文摘要。

真评测必须装 ``ultralytics`` + ``torch``；本机没装，所以缺依赖时打印修复指令
并以退出码 2 结束。``--dry-run`` 不加载模型，只体检数据集与权重是否存在，
因此现在就能确认「命令、路径、数据集」三件事是否接对。

口径提醒：mAP 只反映**检测**能力；报告里要写清这是离线验证集指标，
与 ``demo_pipeline.py`` 那条「仿真检测器」链路不是一回事。

用法::

    python scripts/evaluate_vision.py --help
    python scripts/evaluate_vision.py --dry-run                  # 体检数据集 + 找权重
    python scripts/evaluate_vision.py --weights results/runs/xxx/weights/best.pt
    python scripts/evaluate_vision.py --split test --weights best.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config import cfg_get, load_config
from src.utils.io import date_tag, iso_now, save_json

REQUIRED = {"ultralytics": "ultralytics", "torch": "torch"}
RESULTS_ROOT = _PROJECT_ROOT.parent / "results"


def missing_deps() -> dict[str, str]:
    """返回缺失依赖 ``{导入名: pip 包名}``。"""
    return {mod: pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None}


def print_dependency_error(missing: dict[str, str]) -> None:
    pkgs = " ".join(sorted(set(missing.values())))
    print("[error] 评测视觉模型缺少依赖：", file=sys.stderr)
    for mod, pkg in sorted(missing.items()):
        print(f"  - {mod}（pip 包：{pkg}）", file=sys.stderr)
    print(f"\n修复：pip install {pkgs}", file=sys.stderr)
    print("或先跑 python scripts/evaluate_vision.py --dry-run 只做数据集体检。",
          file=sys.stderr)


def find_latest_weights() -> Path | None:
    """在 ``Demos/results/runs/*/weights/best.pt`` 里找最新的一个。"""
    runs = RESULTS_ROOT / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(
        (p for p in runs.glob("*/weights/best.pt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_weights(arg: str | None, vision_cfg: dict) -> Path | None:
    """优先 --weights，其次最新 run 的 best.pt，最后配置里的 model.weights。"""
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else (_PROJECT_ROOT / p)
    latest = find_latest_weights()
    if latest is not None:
        return latest
    cfg_weights = cfg_get(vision_cfg, "model.weights", None)
    if cfg_weights:
        p = Path(str(cfg_weights))
        return p if p.is_absolute() else (_PROJECT_ROOT / p)
    return None


def dataset_report(vision_cfg: dict) -> dict[str, dict]:
    """不依赖 torch 的数据集体检：图片数 / 标注框数 / 逐类计数。"""
    from src.vision.dataset import load_split

    report: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        try:
            ds = load_split(vision_cfg.get("data"), split)
        except Exception as exc:  # noqa: BLE001
            report[split] = {"error": str(exc)}
            continue
        report[split] = {
            "images": len(ds),
            "boxes": sum(len(s.detections) for s in ds.samples),
            "with_labels": sum(1 for s in ds.samples if s.label_path is not None),
            "defects": ds.defect_count(),
            "class_counts": ds.class_counts(),
        }
    return report


def write_markdown(result: dict[str, Any], meta: dict[str, Any], path: Path) -> None:
    """一页中文摘要：整体 mAP + 逐类 precision/recall。"""
    lines = [
        "# CA-YOLO 验证集评测",
        "",
        f"- 权重：`{result.get('weights')}`",
        f"- 划分：{result.get('split')}",
        f"- 生成时间：{meta['generated_at']}",
        f"- mAP@0.5：**{result.get('mAP50', float('nan')):.4f}**",
        f"- mAP@0.5:0.95：**{result.get('mAP50_95', float('nan')):.4f}**",
        "",
        "| 类别 | mAP@0.5 | mAP@0.5:0.95 | precision | recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, m in (result.get("per_class") or {}).items():
        lines.append(
            f"| {name} | {m.get('map50', 0):.4f} | {m.get('map50_95', 0):.4f} | "
            f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} |"
        )
    lines += [
        "",
        "> 口径说明：本表是离线验证集的**检测**指标（YOLO 标注真值），"
        "不代表整机巡检的缺陷识别精度；集成层 ``sim_detector.py`` 是几何仿真检测器，"
        "两者不可直接比较。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="在验证/测试集上评测 CA-YOLO（需要 ultralytics + torch）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default=None, help="权重路径（默认取最新 run 的 best.pt）")
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="评测划分")
    p.add_argument("--data-yaml", type=Path, default=None, help="外部 data.yaml（默认自动生成）")
    p.add_argument("--device", default="cpu", help="推理设备：cpu / 0 / 0,1（有 GPU 时用 0）")
    p.add_argument("--imgsz", type=int, default=None, help="评测分辨率，覆盖 model.input_size（默认 640）")
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "benchmarks",
                   help="JSON/MD 输出目录")
    p.add_argument("--dry-run", action="store_true", help="只体检数据集与权重路径，不加载模型")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    vision_cfg = load_config("vision")
    weights = resolve_weights(args.weights, vision_cfg)
    report = dataset_report(vision_cfg)

    print("[data] 数据集体检")
    for split, info in report.items():
        if "error" in info:
            print(f"  {split:5s} 读取失败：{info['error']}")
            continue
        print(f"  {split:5s} 图片 {info['images']:5d} | 标注框 {info['boxes']:5d} | "
              f"有标注 {info['with_labels']:5d} | 缺陷框 {info['defects']:5d}")

    if weights is None:
        print("[warn] 没找到任何权重（results/runs/*/weights/best.pt 与配置都为空）。",
              file=sys.stderr)
    elif weights.exists():
        size_mb = weights.stat().st_size / 1024 / 1024
        print(f"[weights] {weights}  ({size_mb:.1f} MB)")
    else:
        print(f"[warn] 权重文件不存在：{weights}", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] 未加载模型；真实评测去装好 ultralytics 后去掉 --dry-run。")
        return 0

    missing = missing_deps()
    if missing:
        print_dependency_error(missing)
        return 2
    if weights is None or not weights.exists():
        print("[error] 没有可用权重，先训练（scripts/train_vision.py）或传 --weights。",
              file=sys.stderr)
        return 2

    from src.vision.train import evaluate

    print(f"[run] 评测 {weights}（split={args.split}, device={args.device}）…")
    result = evaluate(weights, vision_cfg, data_yaml=args.data_yaml, split=args.split,
                      device=args.device, imgsz=args.imgsz)

    tag = f"{weights.parent.parent.name}_{args.split}"
    if args.imgsz and args.imgsz != 640:
        tag += f"_i{args.imgsz}"
    out_dir = args.out_dir if args.out_dir.is_absolute() else (_PROJECT_ROOT / args.out_dir)
    meta = {
        "generated_at": iso_now(),
        "script": "scripts/evaluate_vision.py",
        "date_tag": tag,
        "results_root": str(RESULTS_ROOT),
    }
    json_path = out_dir / f"vision_eval_{tag}.json"
    md_path = out_dir / f"vision_eval_{tag}.md"
    save_json({**meta, "result": result}, json_path)
    write_markdown({**result, "split": result.get("split", args.split)}, meta, md_path)

    print(f"[done] mAP@0.5={result.get('mAP50', float('nan')):.4f} "
          f"mAP@0.5:0.95={result.get('mAP50_95', float('nan')):.4f}")
    print(f"  {json_path}  ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {md_path}  ({md_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
