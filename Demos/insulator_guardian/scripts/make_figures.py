"""把 ``results/benchmarks/decision_*.json`` 画成两张图。

- ``benchmark_success_<日期>.png``：成功率（环图 + 失败原因拆解）
- ``benchmark_steps_<日期>.png``：成功样本到达步数直方图（附失败样本剩余距离）

数据缺失时不静默出空图：打印清楚的报错并以退出码 1 结束，
这样 CI 或报告流水线不会把「没数据」当成「零成功」。

用法::

    python scripts/make_figures.py --help
    python scripts/make_figures.py                      # 取 benchmarks/ 下最新的 decision_*.json
    python scripts/make_figures.py --date 20261028
    python scripts/make_figures.py --json results/benchmarks/decision_20261028.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_ROOT = _PROJECT_ROOT.parent / "results"


def _wrap(text: str, width: int) -> str:
    """按字符数折行（中文每字算 1，够用），避免长标签压到邻图。"""
    import textwrap

    lines = textwrap.wrap(str(text), width=width) or [str(text)]
    return "\n".join(lines)


def latest_benchmark(results_root: Path) -> Path | None:
    """取 ``benchmarks/decision_*.json`` 里最新的一份（按 mtime）。"""
    bench = results_root / "benchmarks"
    if not bench.is_dir():
        return None
    files = sorted(bench.glob("decision_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_benchmark(path: Path) -> dict[str, Any]:
    """读 JSON 并做最小校验，缺字段就报清楚。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到基准结果：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "runs" not in data or "summary" not in data:
        raise ValueError(f"{path} 缺少 runs/summary 字段，不是 benchmark_decision.py 的输出")
    if not data["runs"]:
        raise ValueError(f"{path} 的 runs 为空，没有可画的样本")
    return data


def plot_success(data: dict[str, Any], out_path: Path) -> Path:
    """左：成功/失败环图；右：失败原因柱状。（中文标签走 plot_utils）"""
    from plot_utils import make_tr, save_fig, setup_cjk_font

    import matplotlib.pyplot as plt

    cjk = setup_cjk_font()
    tr = make_tr(cjk)
    runs = data["runs"]
    summary = data["summary"]
    labels = data.get("failure_labels") or {}

    n_ok = sum(1 for r in runs if r.get("success"))
    n_fail = len(runs) - n_ok
    rate = summary.get("success_rate", n_ok / max(1, len(runs)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax0.pie(
        [n_ok, n_fail] if n_fail else [n_ok, 0.0001],
        labels=[tr(f"成功 {n_ok}（{n_ok / len(runs) * 100:.0f}%）",
                   f"success {n_ok} ({n_ok / len(runs) * 100:.0f}%)"),
                tr(f"失败 {n_fail}（{n_fail / len(runs) * 100:.0f}%）",
                   f"fail {n_fail} ({n_fail / len(runs) * 100:.0f}%)")],
        colors=["#2f7d4f", "#d64545"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    ax0.text(0, 0, f"{rate * 100:.1f}%\n{tr('成功率', 'success')}",
             ha="center", va="center", fontsize=13, fontweight="bold")
    ax0.set_title(tr(f"随机栅格图 A*+DWA 到达成功率（n={len(runs)}）",
                     f"A*+DWA success rate (n={len(runs)})"))

    reasons: dict[str, int] = {}
    for r in runs:
        if not r.get("success"):
            reasons[str(r.get("failure_reason") or "unknown")] = (
                reasons.get(str(r.get("failure_reason") or "unknown"), 0) + 1
            )
    if reasons:
        keys = sorted(reasons, key=lambda k: reasons[k], reverse=True)
        # y 轴只放短键，长解释统一放脚注，免得压到左图
        ax1.barh(keys, [reasons[k] for k in keys], height=0.42, color="#d68a3f")
        for i, k in enumerate(keys):
            ax1.text(reasons[k] + 0.2, i, str(reasons[k]), va="center", fontsize=9)
        ax1.set_xlim(0, max(reasons.values()) * 1.15 + 0.5)
        ax1.set_ylim(-0.8, len(keys) - 0.2)
        ax1.invert_yaxis()
        ax1.set_xlabel(tr("样本数", "maps"))
        ax1.set_title(tr("失败原因拆解", "Failure breakdown"))
        ax1.grid(axis="x", alpha=0.25)
        notes = [f"{k} = {labels.get(k, '-')}" for k in keys]
        fig.text(0.5, 0.015, _wrap("；".join(notes), 52), ha="center", va="bottom",
                 fontsize=8.5, color="#555555")
        fig.subplots_adjust(bottom=0.16)
    else:
        ax1.text(0.5, 0.5, tr("无失败样本", "no failures"), ha="center", va="center",
                 transform=ax1.transAxes)
        ax1.set_axis_off()

    fig.tight_layout()
    save_fig(fig, out_path)
    return out_path


def plot_steps(data: dict[str, Any], out_path: Path) -> Path:
    """成功样本步数直方图；右侧小图给出失败样本的最终剩余距离。"""
    from plot_utils import make_tr, save_fig, setup_cjk_font

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    cjk = setup_cjk_font()
    tr = make_tr(cjk)
    runs = data["runs"]
    ok_steps = [int(r["steps"]) for r in runs if r.get("success")]
    if not ok_steps:
        raise ValueError("没有任何成功样本，无法画步数直方图（先修决策线或放宽 --max-steps）")

    import numpy as np

    steps = np.asarray(ok_steps)
    median = float(np.median(steps))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                                   gridspec_kw={"width_ratios": [1.45, 1.0]})
    bins = int(min(20, max(5, round(np.sqrt(len(steps)) * 2))))
    ax0.hist(steps, bins=bins, color="#4f6fb8", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax0.axvline(median, color="#d64545", linestyle="--", linewidth=1.4,
                label=tr(f"中位数 {median:.0f} 步", f"median {median:.0f} steps"))
    ax0.set_xlabel(tr("到达目标所需步数（dt=0.1 s）", "steps to goal (dt=0.1 s)"))
    ax0.set_ylabel(tr("样本数", "maps"))
    ax0.set_title(tr(f"成功样本到达步数（n={len(steps)}）", f"Steps to goal (n={len(steps)})"))
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=9)
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

    fail_dist = [float(r.get("final_distance_m", 0.0)) for r in runs if not r.get("success")]
    if fail_dist:
        ax1.hist(fail_dist, bins=min(12, max(4, len(fail_dist))), color="#d68a3f",
                 alpha=0.85, edgecolor="white", linewidth=0.6)
        ax1.set_xlabel(tr("超时样本最终剩余距离 (m)", "remaining distance at timeout (m)"))
        ax1.set_ylabel(tr("样本数", "maps"))
        ax1.set_title(tr(f"失败样本卡在哪（n={len(fail_dist)}）",
                         f"Where failures stall (n={len(fail_dist)})"))
        ax1.grid(alpha=0.25)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax1.axis("off")

    fig.suptitle(tr("巡检决策线：随机障碍地图上的到达性能",
                    "Navigation performance on random maps"), fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从 decision 基准 JSON 生成成功率与步数两张图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json", type=Path, default=None, help="指定 decision_*.json 路径")
    p.add_argument("--date", default=None, help="日期标签 YYYYMMDD（挑对应文件、也用于输出名）")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT, help="results 根目录")
    p.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认 <results>/plots）")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results_root = args.results_root if args.results_root.is_absolute() else (
        _PROJECT_ROOT / args.results_root)
    if args.json is not None:
        json_path = args.json if args.json.is_absolute() else (_PROJECT_ROOT / args.json)
        tag = args.date or json_path.stem.split("_")[-1]
    elif args.date:
        json_path = results_root / "benchmarks" / f"decision_{args.date}.json"
        tag = args.date
    else:
        json_path = latest_benchmark(results_root)
        if json_path is None:
            print(f"[error] {results_root / 'benchmarks'} 下没有 decision_*.json。"
                  "先跑 python scripts/benchmark_decision.py", file=sys.stderr)
            return 1
        tag = json_path.stem.split("_")[-1]

    try:
        data = load_benchmark(json_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] 无法生成图：{exc}", file=sys.stderr)
        print("  提示：确认路径正确、且文件来自 scripts/benchmark_decision.py。",
              file=sys.stderr)
        return 1

    out_dir = args.out_dir or (results_root / "plots")
    out_dir = out_dir if out_dir.is_absolute() else (_PROJECT_ROOT / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[src] {json_path}")
    try:
        success_png = plot_success(data, out_dir / f"benchmark_success_{tag}.png")
        steps_png = plot_steps(data, out_dir / f"benchmark_steps_{tag}.png")
    except ValueError as exc:
        print(f"[error] 数据不足以下图：{exc}", file=sys.stderr)
        return 1

    summary = data["summary"]
    print(f"[done] 成功率 {summary.get('success_rate', 0) * 100:.1f}% "
          f"（{summary.get('n_success', 0)}/{summary.get('n_maps', len(data['runs']))}）")
    for path in (success_png, steps_png):
        print(f"  {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
