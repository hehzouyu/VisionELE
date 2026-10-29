"""推理延迟基准：CA-YOLO 单帧耗时（mean / p50 / p95 / max），可出直方图。

为什么单独一个脚本：论文/报告里「检测一次多久」这句话必须有可复现的来源，
而不是口头估计。``--device`` 允许 cpu/0 切换，报告里会记下算力口径。

真基准必须装 ``ultralytics`` + ``torch``；本机没装，缺依赖时打印修复指令并以
退出码 2 结束。``--dry-run`` 只打印配置与将要跑的参数，不加载模型。

用法::

    python scripts/benchmark_vision.py --help
    python scripts/benchmark_vision.py --dry-run
    python scripts/benchmark_vision.py --n 100 --imgsz 640 --device cpu --plot
    python scripts/benchmark_vision.py --weights results/runs/xxx/weights/best.pt --plot
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config import cfg_get, load_config
from src.utils.io import date_tag, iso_now, percentile, save_json

REQUIRED = {"ultralytics": "ultralytics", "torch": "torch"}
RESULTS_ROOT = _PROJECT_ROOT.parent / "results"


def missing_deps() -> dict[str, str]:
    """返回缺失依赖 ``{导入名: pip 包名}``。"""
    return {mod: pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None}


def print_dependency_error(missing: dict[str, str]) -> None:
    pkgs = " ".join(sorted(set(missing.values())))
    print("[error] 视觉延迟基准缺少依赖：", file=sys.stderr)
    for mod, pkg in sorted(missing.items()):
        print(f"  - {mod}（pip 包：{pkg}）", file=sys.stderr)
    print(f"\n修复：pip install {pkgs}", file=sys.stderr)
    print("或先跑 python scripts/benchmark_vision.py --dry-run 看参数。", file=sys.stderr)


def machine_info() -> dict[str, Any]:
    """机器信息：python/numpy 版本与平台，便于解释跨机器延迟差异。"""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
        )
    except Exception:  # noqa: BLE001
        info["torch"] = None
    return info


def measure_latency(detector, imgsz: int, n: int, warmup: int) -> tuple[list[float], list[float]]:
    """跑 warmup 帧 + n 帧，返回 ``(原始毫秒列表, 排序后列表)``。

    与 ``src.vision.train.benchmark_latency`` 同一口径，但保留原始样本，
    这样能画直方图并算分位线。
    """
    import numpy as np

    frame = np.random.default_rng(0).integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(max(0, warmup)):
        detector.detect(frame)
    raw: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        detector.detect(frame)
        raw.append((time.perf_counter() - t0) * 1000.0)
    return raw, sorted(raw)


def summarise(raw: list[float]) -> dict[str, float]:
    """mean / p50 / p95 / max(ms)。"""
    return {
        "mean_ms": float(mean(raw)),
        "p50_ms": float(percentile(raw, 50)),
        "p95_ms": float(percentile(raw, 95)),
        "max_ms": float(max(raw)),
        "min_ms": float(min(raw)),
    }


def plot_latency(raw: list[float], stats: dict[str, float], out_path: Path) -> Path:
    """延迟直方图 + p95 虚线；中文标签走 plot_utils。"""
    from plot_utils import make_tr, save_fig, setup_cjk_font

    import matplotlib.pyplot as plt

    cjk = setup_cjk_font()
    tr = make_tr(cjk)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(raw, bins=min(30, max(8, len(raw) // 3)), color="#7c5cd6", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(stats["p95_ms"], color="#d64545", linestyle="--", linewidth=1.4,
               label=tr(f"p95 = {stats['p95_ms']:.1f} ms", f"p95 = {stats['p95_ms']:.1f} ms"))
    ax.axvline(stats["mean_ms"], color="#2f7d4f", linestyle="-", linewidth=1.2,
               label=tr(f"mean = {stats['mean_ms']:.1f} ms", f"mean = {stats['mean_ms']:.1f} ms"))
    ax.set_xlabel(tr("单帧推理耗时 (ms)", "Single-frame latency (ms)"))
    ax.set_ylabel(tr("帧数", "Frames"))
    ax.set_title(tr(f"CA-YOLO 推理延迟分布（n={len(raw)}）", f"CA-YOLO latency (n={len(raw)})"))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_path)
    return out_path


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    """中文摘要：延迟表 + 口径说明。"""
    lat = payload["latency"]
    cfg = payload["config"]
    lines = [
        "# CA-YOLO 推理延迟基准",
        "",
        f"- 权重：`{cfg['weights']}`",
        f"- 设备：{cfg['device']}（imgsz={cfg['imgsz']}，n={lat['n']}，warmup={cfg['warmup']}）",
        f"- 生成时间：{payload['meta']['generated_at']}",
        f"- 机器：{payload['machine'].get('platform')} / torch "
        f"{payload['machine'].get('torch') or '未安装'}",
        "",
        "| 指标 | 耗时 (ms) |",
        "| --- | ---: |",
        f"| mean | {lat['mean_ms']:.2f} |",
        f"| p50 | {lat['p50_ms']:.2f} |",
        f"| p95 | {lat['p95_ms']:.2f} |",
        f"| max | {lat['max_ms']:.2f} |",
        "",
        "> 口径说明：这是**单帧纯推理**耗时（含前后处理开销的粗略估计，不含取帧/通信），"
        "CPU 与 GPU 差异极大，跨机器不可直接比较；集成层的 ``latency.cycle_ms`` "
        "是闭环单步耗时（仿真检测器），两者口径不同。",
        "",
    ]
    if payload.get("artifacts", {}).get("plot"):
        lines.insert(5, f"- 延迟直方图：`{payload['artifacts']['plot']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CA-YOLO 单帧推理延迟基准（需要 ultralytics + torch）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default=None, help="权重路径（默认取配置或官方预训练权重）")
    p.add_argument("--imgsz", type=int, default=None, help="推理尺寸（默认取配置 input_size）")
    p.add_argument("--n", type=int, default=50, help="计时帧数")
    p.add_argument("--warmup", type=int, default=5, help="预热帧数（不进统计）")
    p.add_argument("--device", default=None, help="cpu / 0（默认取配置 train.device）")
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "benchmarks",
                   help="JSON/MD 输出目录")
    p.add_argument("--plot", action="store_true", help="额外输出延迟直方图到 results/plots/")
    p.add_argument("--dry-run", action="store_true", help="只打印参数，不加载模型")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    vision_cfg = load_config("vision")
    family = cfg_get(vision_cfg, "model.family", "yolov8")
    weights = args.weights or cfg_get(vision_cfg, "model.weights", None) or f"{family}n.pt"
    imgsz = int(args.imgsz or cfg_get(vision_cfg, "model.input_size", 640))
    device = str(args.device or cfg_get(vision_cfg, "train.device", "cpu"))

    print("[plan] 延迟基准")
    print(f"  weights = {weights}")
    print(f"  imgsz   = {imgsz}")
    print(f"  n       = {args.n}（warmup {args.warmup}）")
    print(f"  device  = {device}")

    if args.dry_run:
        print("[dry-run] 未加载模型；装好 ultralytics 后去掉 --dry-run 即开始计时。")
        return 0

    missing = missing_deps()
    if missing:
        print_dependency_error(missing)
        return 2

    from src.vision.detector import Detector, DetectorConfig

    det_cfg = DetectorConfig.from_dict(vision_cfg)
    det_cfg.weights = str(weights)
    det_cfg.imgsz = imgsz
    det_cfg.device = "cpu" if device == "cpu" else device
    try:
        detector = Detector(det_cfg)
    except Exception as exc:  # noqa: BLE001 - 权重/ultralytics 版本问题统一转成清晰报错
        print(f"[error] 检测器加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"  权重：{det_cfg.weights}", file=sys.stderr)
        print("  若是权重路径问题，可用 --weights 指定或先训练生成 best.pt。", file=sys.stderr)
        return 2

    print("[run] 计时中…")
    raw, _sorted = measure_latency(detector, imgsz, args.n, args.warmup)
    stats = summarise(raw)
    print(f"  mean {stats['mean_ms']:.2f} ms | p50 {stats['p50_ms']:.2f} | "
          f"p95 {stats['p95_ms']:.2f} | max {stats['max_ms']:.2f}")

    tag = date_tag()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (_PROJECT_ROOT / args.out_dir)
    payload: dict[str, Any] = {
        "meta": {
            "generated_at": iso_now(),
            "script": "scripts/benchmark_vision.py",
            "date_tag": tag,
            "results_root": str(RESULTS_ROOT),
        },
        "config": {
            "weights": str(det_cfg.weights),
            "imgsz": imgsz,
            "n": args.n,
            "warmup": args.warmup,
            "device": device,
            "family": family,
        },
        "latency": {"n": len(raw), **stats},
        "machine": machine_info(),
        "artifacts": {},
    }

    if args.plot:
        plot_path = RESULTS_ROOT / "plots" / f"vision_latency_{tag}.png"
        plot_latency(raw, stats, plot_path)
        payload["artifacts"]["plot"] = str(plot_path)

    json_path = out_dir / f"vision_latency_{tag}.json"
    md_path = out_dir / f"vision_latency_{tag}.md"
    save_json(payload, json_path)
    write_markdown(payload, md_path)

    print(f"[done] {json_path}  ({json_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {md_path}  ({md_path.stat().st_size / 1024:.1f} KB)")
    if payload["artifacts"].get("plot"):
        print(f"  {payload['artifacts']['plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
