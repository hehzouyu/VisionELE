"""训练入口：CA-YOLO（ultralytics 后端）。

把 ``src/vision/train.py`` 的库函数包成命令行：读 ``configs/vision.yaml``、
拼 data.yaml、注入 CA、开训，并把 run 目录写回 ``Demos/results/runs/``。

这台机器上**没有装 ultralytics / torch**（只有 numpy、matplotlib、Pillow、PyYAML），
所以本脚本在缺依赖时会打印可执行的修复指令并以退出码 2 结束，
而不是抛一屏 traceback；``--dry-run`` 不碰模型，只检查配置与数据集，
因此现在就能验证链路是否接对。

用法::

    python scripts/train_vision.py --help
    python scripts/train_vision.py --dry-run          # 只体检配置 + 数据集
    python scripts/train_vision.py --epochs 50 --device cpu
    python scripts/train_vision.py --resume --run-name 20261028_yolov12_ca
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config import cfg_get, load_config
from src.utils.io import iso_now

#: 训练所需的三方依赖（导入名 → pip 包名）
REQUIRED = {"ultralytics": "ultralytics", "torch": "torch"}


def missing_deps() -> dict[str, str]:
    """返回缺失依赖 ``{导入名: pip 包名}``。"""
    return {mod: pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None}


def print_dependency_error(missing: dict[str, str]) -> None:
    pkgs = " ".join(sorted(set(missing.values())))
    print("[error] 训练视觉模型缺少依赖：", file=sys.stderr)
    for mod, pkg in sorted(missing.items()):
        print(f"  - {mod}（pip 包：{pkg}）", file=sys.stderr)
    print("\n修复方式（任选其一）：", file=sys.stderr)
    print(f"  pip install {pkgs}", file=sys.stderr)
    print("  conda install pytorch torchvision -c pytorch && pip install ultralytics",
          file=sys.stderr)
    print("\n说明：本项目其余部分（决策线、集成仿真、数据集体检）不需要这两个包；"
          "只有真正开训/推理才需要。加 --dry-run 可只做配置与数据集体检。",
          file=sys.stderr)


def describe_plan(args: argparse.Namespace, vision_cfg: dict) -> dict:
    """整理成 dict，既打印也写进 meta，便于事后复现。"""
    model = vision_cfg.get("model", {}) or {}
    train_cfg = vision_cfg.get("train", {}) or {}
    return {
        "family": model.get("family", "yolov8"),
        "weights": model.get("weights") or f"{model.get('family', 'yolov8')}n.pt",
        "use_ca": bool(model.get("use_ca", True)),
        "ca_layer_index": model.get("ca_layer_index"),
        "ca_inject": args.ca_inject or model.get("ca_inject", "last"),
        "input_size": int(model.get("input_size", 640)),
        "pretrained": bool(model.get("pretrained", True)),
        "epochs": int(args.epochs or train_cfg.get("epochs", 300)),
        "batch": int(train_cfg.get("batch", 32)),
        "device": str(args.device or train_cfg.get("device", "auto")),
        "optimizer": str(train_cfg.get("optimizer", "AdamW")),
        "lr0": float(train_cfg.get("lr0", 1e-3)),
        "seed": int(train_cfg.get("seed", 0) if args.seed is None else args.seed),
        "run_name": args.run_name or train_cfg.get("run_name") or "<date>_<family>_ca",
        "resume": bool(args.resume),
        "data_root": str(cfg_get(vision_cfg, "data.root", "data")),
    }


def dataset_report(vision_cfg: dict) -> dict:
    """不依赖 torch 的数据集体检：各划分的图片数 / 标注框数 / 逐类计数。"""
    from src.vision.dataset import load_split

    report: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        try:
            ds = load_split(vision_cfg.get("data"), split)
        except Exception as exc:  # noqa: BLE001 - 体检阶段不该把整条命令打断
            report[split] = {"error": str(exc)}
            continue
        report[split] = {
            "images": len(ds),
            "boxes": sum(len(s.detections) for s in ds.samples),
            "class_counts": ds.class_counts(),
            "defects": ds.defect_count(),
            "with_labels": sum(1 for s in ds.samples if s.label_path is not None),
        }
    return report


def resolve_device(requested: str) -> str:
    """``auto`` → 有 CUDA 用 0，否则 cpu（缺 torch 时保持原样，交由训练库报错）。"""
    if requested != "auto":
        return requested
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="训练 CA-YOLO（ultralytics 后端），并可先做 --dry-run 体检",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=None, help="覆盖 configs/vision.yaml 的 train.epochs")
    p.add_argument("--batch", type=int, default=None, help="覆盖 train.batch")
    p.add_argument("--patience", type=int, default=None, help="早停耐心轮数，覆盖 train.patience")
    p.add_argument("--cache", default=None, choices=["ram", "disk", "false"], help="数据集缓存方式")
    p.add_argument("--no-ca", action="store_true", help="不做 CA 注入（基线对照）")
    p.add_argument(
        "--use-rfa",
        action="store_true",
        help="在检测头前插入 RFA 循环多尺度聚合（与 --no-ca 组合可单独验证 RFA）",
    )
    p.add_argument(
        "--ca-inject",
        default=None,
        choices=["last", "backbone"],
        help="CA 注入位置：last=仅 P5 入口（默认），backbone=每个 C2f/SPPF 输出",
    )
    p.add_argument("--device", default=None, help="cpu / 0 / auto（默认取配置）")
    p.add_argument("--run-name", default=None, help="run 目录名（默认 <日期>_<family>_ca）")
    p.add_argument("--data-yaml", type=Path, default=None, help="外部 data.yaml（默认自动生成）")
    p.add_argument("--resume", action="store_true", help="从上次 best/last 权重续训")
    p.add_argument("--seed", type=int, default=None, help="随机种子，覆盖 train.seed（多种子对照用）")
    p.add_argument("--imgsz", type=int, default=None, help="输入分辨率，覆盖 model.input_size（默认 640；小目标实验用 960）")
    p.add_argument("--workers", type=int, default=None, help="DataLoader 进程数（沙箱默认 0；0 以外的值可能被环境拒绝）")
    p.add_argument("--dry-run", action="store_true", help="只打印计划与数据集体检，不加载模型")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    vision_cfg = load_config("vision")
    plan = describe_plan(args, vision_cfg)
    plan["device"] = resolve_device(plan["device"])
    plan["generated_at"] = iso_now()

    print("[plan] CA-YOLO 训练计划")
    for key, value in plan.items():
        print(f"  {key:16s} = {value}")

    report = dataset_report(vision_cfg)
    print("[data] 数据集体检")
    for split, info in report.items():
        if "error" in info:
            print(f"  {split:5s} 读取失败：{info['error']}")
            continue
        print(f"  {split:5s} 图片 {info['images']:5d} | 标注框 {info['boxes']:5d} | "
              f"有标注 {info['with_labels']:5d} | 缺陷框 {info['defects']:5d}")
        print(f"        逐类 {info['class_counts']}")

    if args.dry_run:
        from src.vision.train import build_data_yaml

        data_yaml = build_data_yaml(vision_cfg.get("data"))
        print(f"[dry-run] data.yaml 已生成：{data_yaml}")
        print(f"[dry-run] 未加载模型，未写入任何 weights；真实训练请加 --epochs N 运行")
        return 0

    missing = missing_deps()
    if missing:
        print_dependency_error(missing)
        return 2

    from src.vision.train import train

    print("[run] 开始训练（Ctrl-C 可中断；产物写到 Demos/results/runs/）")
    run_dir = train(
        vision_cfg,
        data_yaml=args.data_yaml,
        run_name=args.run_name,
        epochs=args.epochs,
        device=plan["device"],
        resume=args.resume,
        use_ca=not args.no_ca,
        cache=args.cache,
        patience=args.patience,
        batch=args.batch,
        seed=args.seed,
        ca_inject=args.ca_inject,
        imgsz=args.imgsz,
        workers=args.workers,
        use_rfa=bool(args.use_rfa),
    )
    print(f"[done] run 目录：{run_dir}")
    print(f"  best.pt → {Path(run_dir) / 'weights' / 'best.pt'}")
    print("  下一步：python scripts/evaluate_vision.py --weights "
          f"\"{Path(run_dir) / 'weights' / 'best.pt'}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
