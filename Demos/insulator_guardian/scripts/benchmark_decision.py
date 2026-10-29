"""决策线随机地图基准：A*（全局）+ DWA（局部闭环）的可达性、效率与安全间距。

为什么单独做这个基准
--------------------
``demo_pipeline.py`` 只用一张手工场景，看不出规划器在窄通道/绕行/长距离
任务上的鲁棒性。本脚本生成固定种子的随机栅格图，在每张图上选一对
「可达且相距足够远」的起终点，用 :class:`Planner` 跑完整闭环，记录：

- 是否在步数预算内到达（失败原因如实分类，不剔除样本）；
- 到达步数（仿真步，×dt 得秒）、实际行驶路程与全局路径长度；
- 最小障碍间距（原始地图 ``nearest_obstacle_distance``，分别采样全局
  路径点与实跑位姿）；
- 重规划次数、碰撞次数、停顿步数。

用法::

    python scripts/benchmark_decision.py                 # 50 张图，seed 固定
    python scripts/benchmark_decision.py --maps 20 --max-steps 1500
    python scripts/benchmark_decision.py --out-dir ../results

产物：``results/benchmarks/decision_<YYYYMMDD>.json`` / ``.md``。
"""
from __future__ import annotations

import argparse
import math
import platform
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from src.decision.astar import AStarConfig, astar, densify, path_length, to_world_path
from src.decision.dwa import DWAConfig
from src.decision.map import FREE, GridMap
from src.decision.planner import Planner, PlannerConfig
from src.integration.pipeline import SimRobot
from src.integration.schema import TaskPoint
from src.utils.config import cfg_get, load_config, resolve_path
from src.utils.io import date_tag, ensure_dir, iso_now, percentile, save_json

#: 失败原因分类（写进 md，便于对照）
FAILURE_LABELS: dict[str, str] = {
    "no_valid_pair": "地图碎片化，找不到「可达且足够远」的起终点",
    "planning_failed": "A* 在膨胀地图上无解（通道窄于车体）",
    "timeout": "步数耗尽仍未到达（窄缝/死角里反复倒车-转向）",
    "collision": "车体参考点进入原始占据栅格",
}


# ---------------------------------------------------------------------- 参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绝缘卫士决策线随机地图基准")
    p.add_argument("--maps", type=int, default=50, help="随机地图数量（默认 50）")
    p.add_argument("--rows", type=int, default=80, help="地图行数（默认 80，0.1 m/格 → 8 m）")
    p.add_argument("--cols", type=int, default=80, help="地图列数（默认 80）")
    p.add_argument("--obstacle-ratio", type=float, default=0.12, help="障碍占比（默认 0.12）")
    p.add_argument("--min-start-goal-m", type=float, default=3.0,
                   help="起终点最小直线距离（米，默认 3.0）")
    p.add_argument("--max-steps", type=int, default=1500, help="单图步数预算（默认 1500）")
    p.add_argument("--seed", type=int, default=2026, help="地图种子起点（默认 2026）")
    p.add_argument("--out-dir", type=Path, default=None, help="产物根目录（默认 results/）")
    return p.parse_args()


def machine_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------------- 连通域
def largest_component(gridmap: GridMap) -> list[tuple[int, int]]:
    """返回最大可通行连通域（4 邻域）的格子列表；无可行格返回空表。

    单次遍历 + 全局 visited，避免 ``reachable_from`` 为每个孤立格重算一遍整图。
    """
    visited = np.zeros(gridmap.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for start in np.argwhere(gridmap.grid == FREE):
        sr, sc = int(start[0]), int(start[1])
        if visited[sr, sc]:
            continue
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        visited[sr, sc] = True
        component: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if (0 <= nr < gridmap.rows and 0 <= nc < gridmap.cols
                        and not visited[nr, nc] and gridmap.grid[nr, nc] == FREE):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        if len(component) > len(best):
            best = component
    return best


def pick_start_goal(
    raw: GridMap,
    component: Sequence[tuple[int, int]],
    min_dist_m: float,
    rng: np.random.Generator,
    max_tries: int = 60,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """在连通域里挑一对相距足够远的起终点（世界坐标，格子中心）。"""
    if len(component) < 2:
        return None
    cells = np.asarray(component, dtype=int)
    for _ in range(max_tries):
        i, j = rng.integers(0, len(cells), size=2)
        sr, sc = int(cells[i][0]), int(cells[i][1])
        gr, gc = int(cells[j][0]), int(cells[j][1])
        sx, sy = raw.grid_to_world(sr, sc)
        gx, gy = raw.grid_to_world(gr, gc)
        if math.hypot(gx - sx, gy - sy) >= min_dist_m:
            return (sx, sy), (gx, gy)
    return None


# ---------------------------------------------------------------------- 单图实验
def run_one(
    raw: GridMap,
    planner: Planner,
    start: tuple[float, float],
    goal: tuple[float, float],
    max_steps: int,
    dt: float,
    goal_tolerance: float,
) -> dict[str, Any]:
    """在一张图上跑一次 A*+DWA 闭环，返回逐图记录。"""
    sx, sy = start
    gx, gy = goal
    yaw0 = math.atan2(gy - sy, gx - sx)
    robot = SimRobot(sx, sy, yaw0)
    planner.reset()
    planner.set_goal(TaskPoint(gx, gy, reason="inspection"))
    initial_path = planner.plan(robot.pose)   # 首次全局规划（同时计数第一次 replan）
    if initial_path is None:
        return {
            "status": "failure", "failure_reason": "planning_failed", "success": False,
            "steps": 0, "replans": planner.replans, "collisions": 0,
            "traveled_length_m": 0.0, "global_path_length_m": None,
            "min_clearance_path_m": None, "min_clearance_traveled_m": None,
            "final_distance_m": math.hypot(gx - robot.x, gy - robot.y),
            "dwell_steps": 0, "stuck_trace": [], "wall_time_s": 0.0,
        }

    global_len = path_length(initial_path)
    # 平滑后的折线点很稀，按 0.1 m 重采样再查距离场，避免漏掉贴墙段
    dense = densify(initial_path, step=0.1)
    min_clearance_path = min(
        (raw.nearest_obstacle_distance(px, py) for px, py in dense), default=None
    )

    t0 = time.perf_counter()
    steps = 0
    traveled = 0.0
    collisions = 0
    dwell = 0
    min_clearance_traveled = math.inf
    prev = robot.pose
    trace: list[list[float]] = []
    arrived = False
    while steps < max_steps:
        steps += 1
        cmd = planner.step(robot.pose, dt)
        robot.apply(cmd, dt)
        x, y, _ = robot.pose
        traveled += math.hypot(x - prev[0], y - prev[1])
        prev = robot.pose
        row, col = raw.world_to_grid(x, y)
        if not raw.in_bounds(row, col) or int(raw.grid[row, col]) != FREE:
            collisions += 1
        min_clearance_traveled = min(min_clearance_traveled, raw.nearest_obstacle_distance(x, y))
        if abs(cmd.linear) < 0.01:
            dwell += 1
        trace.append([round(cmd.linear, 3), round(cmd.angular, 3)])
        if len(trace) > 12:
            trace.pop(0)
        dist = math.hypot(gx - x, gy - y)
        if planner.arrived or dist <= goal_tolerance:
            arrived = True
            break
        if collisions >= 3:            # 已经"穿墙"多次，再跑没有意义
            break
    wall = time.perf_counter() - t0

    final_dist = math.hypot(gx - robot.x, gy - robot.y)
    if collisions >= 3:
        status, reason = "failure", "collision"
    elif arrived:
        status, reason = "success", None
    else:
        status, reason = "failure", "timeout"
    return {
        "status": status,
        "failure_reason": reason,
        "success": arrived,
        "steps": steps,
        "replans": planner.replans,
        "collisions": collisions,
        "traveled_length_m": round(traveled, 3),
        "global_path_length_m": round(global_len, 3),
        "min_clearance_path_m": None if min_clearance_path is None else round(min_clearance_path, 3),
        "min_clearance_traveled_m": (round(min_clearance_traveled, 3)
                                     if math.isfinite(min_clearance_traveled) else None),
        "final_distance_m": round(final_dist, 3),
        "dwell_steps": dwell,
        "stuck_trace": trace if not arrived else [],
        "wall_time_s": round(wall, 3),
    }


# ---------------------------------------------------------------------- 汇总
def _stat(values: Iterable[float]) -> dict[str, float] | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return {
        "mean": round(statistics.fmean(vals), 3),
        "std": round(statistics.pstdev(vals), 3),
        "min": round(min(vals), 3),
        "median": round(statistics.median(vals), 3),
        "max": round(max(vals), 3),
        "p95": round(percentile(vals, 95), 3),
        "n": len(vals),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in runs if r["success"]]
    failures: dict[str, int] = {}
    for r in runs:
        if not r["success"]:
            key = r["failure_reason"] or "unknown"
            failures[key] = failures.get(key, 0) + 1
    attempted = [r for r in runs if r["failure_reason"] != "no_valid_pair"]
    return {
        "n_maps": len(runs),
        "n_success": len(ok),
        "n_failure": len(runs) - len(ok),
        "success_rate": round(len(ok) / len(runs), 4) if runs else 0.0,
        "n_attempted": len(attempted),
        "success_rate_attempted": (round(len(ok) / len(attempted), 4) if attempted else 0.0),
        "failure_breakdown": failures,
        "collisions_total": sum(r["collisions"] for r in runs),
        "steps": _stat(r["steps"] for r in ok),
        "traveled_length_m": _stat(r["traveled_length_m"] for r in ok),
        "replans": _stat(r["replans"] for r in ok),
        "min_clearance_m": _stat(r["min_clearance_traveled_m"] for r in ok),
        "wall_time_s_total": round(sum(r["wall_time_s"] for r in runs), 2),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> Path:
    s = payload["summary"]
    meta = payload["meta"]

    def fmt(stat: dict[str, float] | None) -> str:
        if not stat:
            return "—"
        return f"{stat['mean']:.2f} ± {stat['std']:.2f}"

    lines: list[str] = [
        "# 决策线随机地图基准（A\\* + DWA 闭环）",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 地图：{meta['rows']}×{meta['cols']} 格 @ {meta['resolution']:.2f} m "
        f"（{meta['rows'] * meta['resolution']:.1f}×{meta['cols'] * meta['resolution']:.1f} m），"
        f"障碍占比 {meta['obstacle_ratio']:.2f}，种子 {meta['seed']} 起连续 {s['n_maps']} 张",
        f"- 闭环：A\\*（8 邻域 + 转弯代价 + DP/Chaikin 平滑）→ DWA（v_max={meta['dwa']['v_max']} m/s），"
        f"单图步数预算 {meta['max_steps']}，dt={meta['dt']} s",
        f"- 起终点：最大连通域内随机采样，直线距离 ≥ {meta['min_start_goal_m']:.1f} m",
        "",
        "## 1. 汇总",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 样本数 | {s['n_maps']} |",
        f"| 成功 / 失败 | {s['n_success']} / {s['n_failure']} |",
        f"| **成功率（全样本）** | **{s['success_rate'] * 100:.1f}%** |",
        f"| 成功率（可连通样本，剔除 no_valid_pair） | {s['success_rate_attempted'] * 100:.1f}%"
        f"（{s['n_attempted']} 张） |",
        f"| 碰撞次数合计 | {s['collisions_total']} |",
        f"| 到达步数 | {fmt(s['steps'])}（n={s['steps']['n'] if s['steps'] else 0}） |",
        f"| 行驶路程 (m) | {fmt(s['traveled_length_m'])} |",
        f"| 重规划次数 | {fmt(s['replans'])} |",
        f"| 最小障碍间距 (m) | {fmt(s['min_clearance_m'])} |",
        f"| 全部实验墙钟耗时 | {s['wall_time_s_total']:.1f} s |",
        "",
        "## 2. 失败原因分解",
        "",
        "| 原因 | 张数 | 含义 |",
        "| --- | --- | --- |",
    ]
    for key, count in sorted(s["failure_breakdown"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{key}` | {count} | {FAILURE_LABELS.get(key, '未分类')} |")
    if not s["failure_breakdown"]:
        lines.append("| — | 0 | 无失败样本 |")

    lines += [
        "",
        "## 3. 逐图记录",
        "",
        "| # | 结果 | 步数 | 行驶 (m) | 全局路径 (m) | 最小间距 (m) | 重规划 | 碰撞 | 残留距离 (m) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in payload["runs"]:
        glyph = "OK" if r["success"] else f"FAIL({r.get('failure_reason') or '-'})"
        global_len = r["global_path_length_m"]
        clearance = r["min_clearance_traveled_m"]
        rest = r["final_distance_m"]
        lines.append(
            f"| {r['index']} | {glyph} | {r['steps']} | "
            f"{r['traveled_length_m']:.2f} | "
            f"{'-' if global_len is None else f'{global_len:.2f}'} | "
            f"{'-' if clearance is None else f'{clearance:.2f}'} | "
            f"{r['replans']} | {r['collisions']} | "
            f"{'-' if rest is None else f'{rest:.2f}'} |"
        )

    stuck = [r for r in payload["runs"] if not r["success"] and r["stuck_trace"]]
    lines += [
        "",
        "## 4. 失败样本的末端指令（诊断用）",
        "",
        "连续 12 拍的 `(v, w)` 指令。典型失败形态是 `v=0` 且 `w=±0.4` **逐拍变号**："
        "机器人卡在 `Planner._pursuit` 纯跟踪兜底里——`line_clear_world` 判不可清时 `v=0`，"
        "而误差角已在 ±0.05 rad 内时用「固定 ±0.4、符号跟误差」的方式强行转向，"
        "转一拍误差就过零，下一拍符号翻转，转量互相抵消，位置永远不动；"
        "同时 `_update_stall` 每 10 拍把 `_recovery_steps` 重置成 40，兜底被无限重触发。"
        "这是决策线的已知缺陷（`src/decision/planner.py` 第 205-206、240-241 行），"
        "不是本基准脚本的统计错误，也未被本脚本「洗白」成成功。",
        "",
    ]
    for r in stuck[:8]:
        trace = " ".join(f"({v:+.2f},{w:+.2f})" for v, w in r["stuck_trace"])
        rest = r["final_distance_m"]
        lines.append(f"- #{r['index']}（{r['failure_reason']}，{r['steps']} 步，"
                     f"残留 {'—' if rest is None else f'{rest:.2f}'} m）：{trace}")
    if not stuck:
        lines.append("- 无失败样本。")

    lines += [
        "",
        "## 5. 口径与局限",
        "",
        "1. **随机地图不代表真实变电站布局**：`GridMap.random_map` 摆随机矩形块，"
        "通道宽度分布比真实巡检通道更极端，成功率的绝对值只用于比较算法版本，"
        "不能当作现场指标。",
        "2. **失败样本全部计入分母**，包括连通性不足的碎片化地图；"
        "表里同时给出剔除 `no_valid_pair` 后的成功率，两种口径都不隐藏。",
        "3. **最小间距**取原始（未膨胀）地图的 `nearest_obstacle_distance`，"
        "分别采样全局路径点与实跑位姿；膨胀半径 "
        f"{meta['robot_radius']:.2f} m 之内的值即视为擦边。",
        "4. **到达判据**：`planner.arrived` 或终点距离 ≤ goal_tolerance"
        f"（{meta['goal_tolerance']:.2f} m）；后者是必要的，因为 `arrived` 在下一步才置位。",
        "5. 步数是仿真步（×dt=秒），墙钟耗时不等于真实算力下的耗时。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- 主流程
def main() -> int:
    args = parse_args()
    if args.maps < 1:
        print("[error] --maps 必须 >= 1", file=sys.stderr)
        return 2
    decision_cfg = load_config("decision")
    pipeline_cfg = load_config("pipeline")
    dt = float(cfg_get(pipeline_cfg, "pipeline.dt", 0.1))
    robot_radius = float(cfg_get(decision_cfg, "map.robot_radius", 0.3))

    astar_cfg = AStarConfig.from_dict(decision_cfg.get("astar"))
    dwa_cfg = DWAConfig.from_dict(decision_cfg.get("dwa"), robot_radius=robot_radius)
    planner_cfg = PlannerConfig.from_dict(decision_cfg.get("planner"))
    goal_tolerance = planner_cfg.goal_tolerance

    root = (Path(args.out_dir).resolve() if args.out_dir
            else resolve_path(cfg_get(pipeline_cfg, "paths.results_root", "../results")))
    bench_dir = ensure_dir(root / "benchmarks")
    tag = date_tag()

    rng = np.random.default_rng(args.seed)
    runs: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    resolution = 0.1

    for i in range(args.maps):
        map_seed = args.seed + i
        raw = GridMap.random_map(rows=args.rows, cols=args.cols,
                                 obstacle_ratio=args.obstacle_ratio,
                                 seed=map_seed, border=True)
        resolution = raw.resolution
        planner = Planner(raw, astar_cfg=astar_cfg, dwa_cfg=dwa_cfg, planner_cfg=planner_cfg)
        component = largest_component(planner.planning_map)
        pair = pick_start_goal(raw, component, args.min_start_goal_m, rng)
        if pair is None:
            rec = {
                "status": "failure", "failure_reason": "no_valid_pair", "success": False,
                "steps": 0, "replans": 0, "collisions": 0,
                "traveled_length_m": 0.0, "global_path_length_m": None,
                "min_clearance_path_m": None, "min_clearance_traveled_m": None,
                "final_distance_m": None, "dwell_steps": 0, "stuck_trace": [],
                "wall_time_s": 0.0,
                "component_cells": len(component),
            }
        else:
            start, goal = pair
            rec = run_one(raw, planner, start, goal, args.max_steps, dt, goal_tolerance)
            rec["start"] = [round(start[0], 3), round(start[1], 3)]
            rec["goal"] = [round(goal[0], 3), round(goal[1], 3)]
            rec["start_goal_distance_m"] = round(
                math.hypot(goal[0] - start[0], goal[1] - start[1]), 3)
            rec["component_cells"] = len(component)
        rec.update({"index": i + 1, "map_seed": map_seed})
        runs.append(rec)
        print(f"[{i + 1:3d}/{args.maps}] seed={map_seed} status={rec['status']}"
              f"{'' if rec['success'] else ' (' + str(rec['failure_reason']) + ')'}"
              f" steps={rec['steps']} replans={rec['replans']}", flush=True)

    summary = summarize(runs)
    payload = {
        "meta": {
            "generated_at": iso_now(),
            "script": "scripts/benchmark_decision.py",
            "date_tag": tag,
            "seed": args.seed,
            "rows": args.rows,
            "cols": args.cols,
            "resolution": resolution,
            "obstacle_ratio": args.obstacle_ratio,
            "min_start_goal_m": args.min_start_goal_m,
            "max_steps": args.max_steps,
            "dt": dt,
            "robot_radius": robot_radius,
            "goal_tolerance": goal_tolerance,
            "astar": astar_cfg.__dict__,
            "dwa": {k: getattr(dwa_cfg, k) for k in
                    ("v_max", "v_acc", "w_max", "w_acc", "horizon", "lookahead", "stop_distance")},
            "planner": planner_cfg.__dict__,
            "machine": machine_info(),
            "elapsed_s": round(time.perf_counter() - t_start, 2),
        },
        "summary": summary,
        "runs": runs,
        "failure_labels": FAILURE_LABELS,
        "caveat": ("Random rectangle maps are a stress test, not a substation layout; "
                   "absolute success rate is only comparable across algorithm versions."),
    }
    json_path = bench_dir / f"decision_{tag}.json"
    md_path = bench_dir / f"decision_{tag}.md"
    save_json(payload, json_path)
    write_markdown(md_path, payload)

    print(f"[done] 成功率 {summary['success_rate'] * 100:.1f}% "
          f"({summary['n_success']}/{summary['n_maps']})，"
          f"失败原因 {summary['failure_breakdown']}")
    for p in (json_path, md_path):
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
