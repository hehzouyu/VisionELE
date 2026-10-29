"""端到端闭环 demo：场景 → 仿真检测 → 调度 → A*+DWA → 台账 / 图 / 摘要。

用法::

    python scripts/demo_pipeline.py                      # 默认 3000 步、seed=7
    python scripts/demo_pipeline.py --max-steps 800 --seed 3 --out-dir ../results

产物（默认 Demos/results/，同日期文件覆盖）::

    plots/nav_trajectory_<YYYYMMDD>.png    栅格底图 + 膨胀障碍 + 轨迹 + 任务/缺陷点
    plots/latency_<YYYYMMDD>.png           检测 / 决策 / 循环延迟分布（标 p95）
    benchmarks/pipeline_<YYYYMMDD>.json    summary + 配置快照 + 机器信息
    benchmarks/pipeline_<YYYYMMDD>.md      一页中文结果摘要
    gifs/nav_demo_<YYYYMMDD>.gif           机器人位置逐帧（≤120 帧）
    logs/defect_ledger_<YYYYMMDD>.jsonl    缺陷台账（本次运行前清空，保证可复现）

口径提醒：检测器是 ``SimulatedDetector``（框由地面真值点反推），
本脚本的检测事件与延迟**不代表真实识别精度**，仅用于验证链路与时延预算。
"""
from __future__ import annotations

import argparse
import math
import platform
import sys
from pathlib import Path
from typing import Any, Callable

# 允许 `python scripts/demo_pipeline.py` 直接运行：把项目根与 scripts/ 放进 sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import plot_utils
from src.decision.map import FREE
from src.integration.coords import CameraModel
from src.integration.pipeline import InspectionPipeline
from src.integration.scene import InspectionScene, build_inspection_field
from src.integration.sim_detector import SimulatedDetector
from src.utils.config import cfg_get, load_config, resolve_path
from src.utils.io import date_tag, ensure_dir, iso_now, percentile, save_json

#: 仿真检测器置信度抖动（固定口径，写进摘要）
CONF_JITTER = 0.02
#: GIF 最大帧数
MAX_GIF_FRAMES = 120


# ---------------------------------------------------------------------- 参数/环境
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绝缘卫士端到端闭环 demo")
    parser.add_argument("--max-steps", type=int, default=3000, help="仿真步数上限（默认 3000）")
    parser.add_argument("--seed", type=int, default=7, help="检测器置信度抖动种子（默认 7）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="产物根目录（默认取 pipeline.yaml 的 paths.results_root）")
    return parser.parse_args()


def machine_info() -> dict[str, str]:
    """记录复现实验所需的机器信息。"""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


# ---------------------------------------------------------------------- 指标
def collect_metrics(scene: InspectionScene, pipe: InspectionPipeline) -> dict[str, Any]:
    """从 StepRecord 里统计碰撞、轨迹长度、最小障碍间距与任务分类。"""
    raw = scene.gridmap
    poses = [rec.pose for rec in pipe.records]
    collisions = 0
    min_clearance = math.inf
    length = 0.0
    for i, (x, y, _yaw) in enumerate(poses):
        row, col = raw.world_to_grid(x, y)
        # 碰撞口径：车身参考点落进原始占据栅格（出界也算）
        if not raw.in_bounds(row, col) or int(raw.grid[row, col]) != FREE:
            collisions += 1
        min_clearance = min(min_clearance, raw.nearest_obstacle_distance(x, y))
        if i:
            px, py = poses[i - 1][0], poses[i - 1][1]
            length += math.hypot(x - px, y - py)
    completed = pipe.completed
    return {
        "collisions": collisions,
        "trajectory_length_m": round(length, 2),
        "min_clearance_m": round(min_clearance, 3) if poses else float("nan"),
        "defect_tasks_completed": sum(t.reason == "defect_revisit" for t in completed),
        "inspection_tasks_completed": sum(t.reason == "inspection" for t in completed),
        "defect_tasks_created": sum(len(rec.created_tasks) for rec in pipe.records),
        "detections_total": sum(len(ev.detections) for ev in pipe.events),
    }


# ---------------------------------------------------------------------- 图：轨迹
def plot_trajectory(
    scene: InspectionScene,
    pipe: InspectionPipeline,
    metrics: dict[str, Any],
    summary: dict[str, Any],
    robot_radius: float,
    out_path: Path,
    tr: Callable[[str, str], str],
) -> Path:
    """栅格地图 + 膨胀障碍轮廓 + 机器人轨迹 + 任务点/缺陷点 + 起终点。"""
    raw = scene.gridmap
    res = raw.resolution
    width_m, height_m = raw.size_m
    x0, y0 = raw.origin
    extent = (x0, x0 + width_m, y0, y0 + height_m)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.imshow(raw.grid, cmap=ListedColormap(["#fafafa", "#3f3f46"]), origin="lower",
              extent=extent, vmin=0, vmax=1, interpolation="nearest", zorder=0)

    # 膨胀障碍（机器人安全半径）用虚线轮廓叠一层
    inflated = raw.inflate(robot_radius)
    xs = x0 + (np.arange(raw.cols) + 0.5) * res
    ys = y0 + (np.arange(raw.rows) + 0.5) * res
    ax.contour(xs, ys, inflated.grid, levels=[0.5], colors="#d95f02",
               linewidths=0.9, linestyles="--", zorder=1)

    poses = [rec.pose for rec in pipe.records]
    if poses:
        ax.plot([p[0] for p in poses], [p[1] for p in poses],
                color="#2563eb", lw=1.6, alpha=0.95, zorder=3,
                label=tr("机器人轨迹", "robot trajectory"))

    normal = [m for m in scene.marks if m.class_id == 0]
    defects = scene.defect_marks
    if normal:
        ax.scatter([m.x for m in normal], [m.y for m in normal], marker="o", s=52,
                   facecolors="none", edgecolors="#0f766e", linewidths=1.3, zorder=4,
                   label=tr("普通绝缘子（真值）", "normal insulator (GT)"))
    if defects:
        ax.scatter([m.x for m in defects], [m.y for m in defects], marker="*", s=170,
                   c="#dc2626", edgecolors="black", linewidths=0.4, zorder=5,
                   label=tr("缺陷点（真值）", "defect point (GT)"))

    done_inspection = [t for t in pipe.completed if t.reason == "inspection"]
    done_defect = [t for t in pipe.completed if t.reason == "defect_revisit"]
    if done_inspection:
        ax.scatter([t.x for t in done_inspection], [t.y for t in done_inspection],
                   marker="s", s=46, c="#0284c7", edgecolors="white", linewidths=0.6,
                   zorder=6, label=tr("已完成巡检点", "completed inspection"))
    if done_defect:
        ax.scatter([t.x for t in done_defect], [t.y for t in done_defect],
                   marker="^", s=64, c="#ea580c", edgecolors="white", linewidths=0.6,
                   zorder=6, label=tr("已完成缺陷复检", "completed defect revisit"))

    sx, sy = scene.start_pose[0], scene.start_pose[1]
    ax.scatter([sx], [sy], marker="o", s=110, c="#16a34a", edgecolors="black",
               linewidths=0.6, zorder=7, label=tr("起点", "start"))
    if poses:
        ex, ey = poses[-1][0], poses[-1][1]
        ax.scatter([ex], [ey], marker="X", s=110, c="#111827", zorder=7,
                   label=tr("终点", "end"))

    ax.set_title(
        f"{tr('绝缘卫士 · 巡检轨迹', 'Insulator Guardian - inspection trajectory')}  "
        f"({summary['steps']} steps / {len(pipe.completed)} tasks / "
        f"{metrics['collisions']} collisions)",
        fontsize=12,
    )
    ax.set_xlabel(tr("x (m)", "x (m)"))
    ax.set_ylabel(tr("y (m)", "y (m)"))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(color="#d4d4d8", lw=0.4, alpha=0.55, zorder=0)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.92)
    fig.tight_layout()
    plot_utils.save_fig(fig, out_path)
    return out_path


# ---------------------------------------------------------------------- 图：延迟
def plot_latency(pipe: InspectionPipeline, out_path: Path,
                 tr: Callable[[str, str], str]) -> Path:
    """检测 / 决策 / 循环三类延迟的分布，标出 p95。"""
    items = [
        ("detect_ms", tr("检测延迟", "Detection"), "#2563eb"),
        ("decision_ms", tr("决策延迟（A*+DWA）", "Decision (A*+DWA)"), "#d95f02"),
        ("cycle_ms", tr("单循环总耗时", "Cycle"), "#0f766e"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for ax, (key, title, color) in zip(axes, items):
        vals = [float(v) for v in pipe.latencies.get(key, [])]
        if not vals:
            ax.set_title(title)
            ax.text(0.5, 0.5, tr("无数据", "no data"), ha="center", va="center",
                    transform=ax.transAxes)
            continue
        med = percentile(vals, 50)
        p95 = percentile(vals, 95)
        # 长尾明显时用对数分箱，否则线性
        if max(vals) > 15.0 * max(med, 1e-9):
            bins = np.logspace(np.log10(max(min(vals), 1e-3)), np.log10(max(vals)), 40)
            ax.set_xscale("log")
        else:
            bins = 40
        ax.hist(vals, bins=bins, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(p95, color="#b91c1c", ls="--", lw=1.3,
                   label=f"p95 = {p95:.2f} ms")
        ax.set_title(f"{title}\nmean {sum(vals) / len(vals):.2f} ms · "
                     f"p50 {med:.2f} ms", fontsize=10)
        ax.set_xlabel("ms")
        ax.set_ylabel(tr("次数", "count"))
        ax.legend(fontsize=8)
        ax.grid(color="#e4e4e7", lw=0.4, alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(tr("单步延迟分布（仿真，含 Python 开销）",
                    "Per-step latency (simulation, incl. Python overhead)"),
                 fontsize=12)
    fig.tight_layout()
    plot_utils.save_fig(fig, out_path)
    return out_path


# ---------------------------------------------------------------------- 动图
def save_gif(scene: InspectionScene, pipe: InspectionPipeline, out_path: Path,
             tr: Callable[[str, str], str]) -> Path | None:
    """逐帧机器人位置（scatter 更新，底图只画一次）；帧数 ≤ 120。"""
    poses = [rec.pose for rec in pipe.records]
    if not poses:
        return None
    raw = scene.gridmap
    x0, y0 = raw.origin
    extent = (x0, x0 + raw.size_m[0], y0, y0 + raw.size_m[1])
    idx = np.unique(np.linspace(0, len(poses) - 1,
                                min(MAX_GIF_FRAMES, len(poses))).astype(int))

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=80)
    ax.imshow(raw.grid, cmap=ListedColormap(["#fafafa", "#3f3f46"]), origin="lower",
              extent=extent, vmin=0, vmax=1, interpolation="nearest", zorder=0)
    normal = [m for m in scene.marks if m.class_id == 0]
    defects = scene.defect_marks
    ax.scatter([m.x for m in normal], [m.y for m in normal], marker="o", s=24,
               facecolors="none", edgecolors="#0f766e", linewidths=1.0, zorder=3)
    ax.scatter([m.x for m in defects], [m.y for m in defects], marker="*", s=80,
               c="#dc2626", edgecolors="black", linewidths=0.3, zorder=4)
    for t in pipe.completed:
        ax.scatter([t.x], [t.y], marker="s" if t.reason == "inspection" else "^",
                   s=26, c="#0284c7" if t.reason == "inspection" else "#ea580c",
                   edgecolors="white", linewidths=0.4, zorder=5)

    xs_all = [p[0] for p in poses]
    ys_all = [p[1] for p in poses]
    trail, = ax.plot([], [], color="#2563eb", lw=1.2, alpha=0.85, zorder=6)
    dot = ax.scatter([], [], s=64, c="#dc2626", edgecolors="black", linewidths=0.5,
                     zorder=7)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(tr("x (m)", "x (m)"))
    ax.set_ylabel(tr("y (m)", "y (m)"))

    def update(k: int):
        i = int(idx[k])
        trail.set_data(xs_all[: i + 1], ys_all[: i + 1])
        dot.set_offsets([[xs_all[i], ys_all[i]]])
        ax.set_title(f"step {i + 1} / {len(poses)}   "
                     f"({tr('轨迹', 'trail')})", fontsize=10)
        return trail, dot

    try:
        anim = FuncAnimation(fig, update, frames=len(idx), blit=False, interval=100)
        anim.save(out_path, writer=PillowWriter(fps=10))
    except Exception as exc:  # noqa: BLE001 - GIF 只是附加产物，失败不影响主结果
        plt.close(fig)
        print(f"[warn] GIF 生成失败（跳过）: {exc}")
        return None
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------- 摘要
def write_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    scene: InspectionScene,
    args: argparse.Namespace,
    ledger: Path,
    artifacts: dict[str, Path | None],
) -> Path:
    """一页中文结果摘要（含口径说明，避免把仿真数字当真实指标）。"""
    lat = summary["latency"]
    sched = summary["scheduler_stats"]
    tasks = summary["steps"]
    lines: list[str] = [
        "# 绝缘卫士 · 端到端闭环结果摘要",
        "",
        f"- 生成时间：{iso_now()}",
        f"- 场景：{scene.description}",
        f"- 检测器：SimulatedDetector（conf_jitter={CONF_JITTER}, seed={args.seed}, "
        f"max_range={args.max_range_m:.1f} m）；**非真实识别精度**",
        f"- 仿真步长：{args.dt:.2f} s；步数上限：{args.max_steps}；实际步数：{tasks}",
        f"- 台账：`{ledger}`",
        "",
        "## 1. 关键指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 仿真步数 | {tasks} |",
        f"| 完成任务数 | {summary['tasks_completed']}（巡检 {metrics['inspection_tasks_completed']}"
        f" + 缺陷复检 {metrics['defect_tasks_completed']}） |",
        f"| 缺陷复检任务（新建） | {metrics['defect_tasks_created']} |",
        f"| 检测事件帧数 | {summary['events']} |",
        f"| 含缺陷的帧数 | {summary['defect_events']} |",
        f"| 缺陷检测次数 | {sched['defect_detections']}（重复跳过 {sched['duplicates_skipped']}） |",
        f"| 碰撞次数 | {metrics['collisions']} |",
        f"| 轨迹长度 | {metrics['trajectory_length_m']:.2f} m |",
        f"| 最小障碍间距 | {metrics['min_clearance_m']:.2f} m |",
        f"| 重规划次数 | {summary['replans']} |",
        f"| 剩余待执行任务 | {summary['tasks_pending']} |",
        f"| 总耗时（墙钟） | {summary['duration_s']:.2f} s |",
        "",
        "## 2. 延迟（ms，单次 `perf_counter` 实测）",
        "",
        "| 环节 | mean | p50 | p95 | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, name in (("detect_ms", "检测"), ("decision_ms", "决策"), ("cycle_ms", "单循环")):
        st = lat[key]
        lines.append(
            f"| {name} | {st['mean']:.3f} | {st['p50']:.3f} | "
            f"{st['p95']:.3f} | {st['max']:.3f} |"
        )
    lines += [
        "",
        "## 3. 已完成任务",
        "",
        "| # | 任务点 (x, y) | 类型 |",
        "| --- | --- | --- |",
    ]
    for i, task in enumerate(summary["completed_tasks"], start=1):
        kind = "缺陷复检" if task["reason"] == "defect_revisit" else "常规巡检"
        lines.append(f"| {i} | ({task['x']:.2f}, {task['y']:.2f}) | {kind} |")
    lines += [
        "",
        "## 4. 口径说明（重要）",
        "",
        "1. **检测不是真实模型**：`SimulatedDetector` 按相机内参/视场/距离裁剪，"
        "从地面真值点反推像素框，置信度只加了 ±"
        f"{CONF_JITTER} 抖动；因此 mAP、漏检率、真实延迟都不适用，"
        "真实精度需训练 CA-YOLO 后用 `scripts/benchmark_vision.py` 另测。",
        "2. **单目地面投影**：世界坐标由 `coords.py` 假设缺陷落点在地面 z=0 反算，"
        "是仿真阶段近似，实机需标定或换测距/双目。",
        "3. **延迟口径**：单步墙钟时间，含 Python 开销与调度器入队；"
        "p95 为线性插值分位数；max 长尾来自首次调用/内存分配，不代表稳态。",
        "4. **碰撞口径**：车体参考点落入原始占据栅格即计一次；本场景全程为 0，"
        "但膨胀图上仍留 0.3 m 安全半径。",
        "",
        "## 5. 产物",
        "",
    ]
    for name, p in artifacts.items():
        lines.append(f"- {name}：`{p}`" if p is not None else f"- {name}：未生成")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- 主流程
def main() -> int:
    args = parse_args()
    cjk = plot_utils.setup_cjk_font()
    tr = plot_utils.make_tr(cjk)
    if not cjk:
        print("[warn] 未找到中文字体，图上标签退回英文")

    decision_cfg = load_config("decision")
    pipeline_cfg = load_config("pipeline")
    args.max_range_m = float(cfg_get(pipeline_cfg, "detector.max_range_m", 6.0))
    args.dt = float(cfg_get(pipeline_cfg, "pipeline.dt", 0.1))
    robot_radius = float(cfg_get(decision_cfg, "map.robot_radius", 0.3))

    root = (Path(args.out_dir).resolve() if args.out_dir
            else resolve_path(cfg_get(pipeline_cfg, "paths.results_root", "../results")))
    plots_dir = ensure_dir(root / "plots")
    bench_dir = ensure_dir(root / "benchmarks")
    gifs_dir = ensure_dir(root / "gifs")
    logs_dir = ensure_dir(root / "logs")
    tag = date_tag()

    ledger = logs_dir / f"defect_ledger_{tag}.jsonl"
    if ledger.exists():
        ledger.unlink()  # 同日期重跑：覆盖，保证台账只对应本次运行

    scene = build_inspection_field()
    camera = CameraModel.from_dict(pipeline_cfg.get("camera"))
    detector = SimulatedDetector(
        scene.marks, camera,
        max_range_m=args.max_range_m,
        conf_jitter=CONF_JITTER,
        seed=args.seed,
    )
    pipe = InspectionPipeline.build(
        decision_cfg, pipeline_cfg, scene.gridmap, detector,
        start_pose=scene.start_pose,
        inspect_points=scene.inspect_points,
        ledger_path=ledger,
    )

    print(f"[run] 场景 {scene.gridmap.size_m[0]:.0f}×{scene.gridmap.size_m[1]:.0f} m，"
          f"max_steps={args.max_steps}, seed={args.seed}")
    summary = pipe.run(max_steps=args.max_steps)
    metrics = collect_metrics(scene, pipe)
    summary.update(metrics)
    summary["completed_tasks"] = [t.to_dict() for t in pipe.completed]
    summary["pending_tasks"] = [t.to_dict() for t in pipe.scheduler.pending()]
    print(f"[run] steps={summary['steps']} tasks={summary['tasks_completed']} "
          f"collisions={metrics['collisions']} "
          f"cycle_p95={summary['latency']['cycle_ms']['p95']:.2f} ms")

    traj_png = plots_dir / f"nav_trajectory_{tag}.png"
    lat_png = plots_dir / f"latency_{tag}.png"
    gif_path = gifs_dir / f"nav_demo_{tag}.gif"
    json_path = bench_dir / f"pipeline_{tag}.json"
    md_path = bench_dir / f"pipeline_{tag}.md"

    plot_trajectory(scene, pipe, metrics, summary, robot_radius, traj_png, tr)
    plot_latency(pipe, lat_png, tr)
    gif_out = save_gif(scene, pipe, gif_path, tr)

    payload = {
        "meta": {
            "generated_at": iso_now(),
            "script": "scripts/demo_pipeline.py",
            "date_tag": tag,
            "seed": args.seed,
            "conf_jitter": CONF_JITTER,
            "max_steps_requested": args.max_steps,
            "max_steps_used": summary["steps"],
            "scene": {
                "description": scene.description,
                "size_m": list(scene.gridmap.size_m),
                "resolution": scene.gridmap.resolution,
                "start_pose": list(scene.start_pose),
                "inspect_points": [list(p) for p in scene.inspect_points],
                "marks": [
                    {"x": m.x, "y": m.y, "class_id": m.class_id,
                     "class_name": {0: "normal", 1: "flashover", 2: "broken"}[m.class_id],
                     "confidence": m.confidence}
                    for m in scene.marks
                ],
            },
            "config": {"decision": decision_cfg, "pipeline": pipeline_cfg},
            "machine": machine_info(),
        },
        "summary": summary,
        "ledger": str(ledger),
        "artifacts": {
            "trajectory_png": str(traj_png),
            "latency_png": str(lat_png),
            "gif": None if gif_out is None else str(gif_out),
            "summary_json": str(json_path),
            "summary_md": str(md_path),
        },
        "caveat": ("SimulatedDetector: boxes are back-projected from ground-truth "
                   "points; detection metrics are NOT real recognition accuracy."),
    }
    save_json(payload, json_path)
    write_markdown(
        md_path,
        summary=summary,
        metrics=metrics,
        scene=scene,
        args=args,
        ledger=ledger,
        artifacts={
            "轨迹图": traj_png,
            "延迟图": lat_png,
            "动图": gif_out,
            "JSON": json_path,
            "台账": ledger,
        },
    )

    print("[done] 产物：")
    for p in (traj_png, lat_png, gif_out, json_path, md_path, ledger):
        if p is not None and p.exists():
            print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
