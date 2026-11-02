#!/usr/bin/env python
"""视觉-决策误差传播仿真：把检测器的漏检/误检按实测比例注入闭环，看任务侧怎么变。

动机：端到端 demo 用的是「完美」仿真检测器（只负责几何投影，不回答准不准）。
真实权重在困难测试集上的逐类 P/R 大约是 broken P 0.85 / R 0.86、normal P 0.99 / R 0.99，
所以这里按可调的漏检率 p_miss 与误检率 p_fp 建模，观察：
    完成任务数、缺陷复检任务数、平均步数、碰撞、误检引入的额外任务。

用法：
    python scripts/sim_noise.py                 # 默认跑 clean / measured / severe 三档
    python scripts/sim_noise.py --p-miss 0.3 --p-fp 0.1 --out-dir ../results/noise
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.pipeline import InspectionPipeline
from src.integration.scene import build_inspection_field
from src.integration.schema import Detection, DEFECT_CLASS_IDS
from src.integration.coords import CameraModel
from src.integration.sim_detector import SimulatedDetector
from src.utils.config import cfg_get, load_config, resolve_path
from src.utils.io import date_tag, ensure_dir, save_json
from src.decision.map import FREE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_JITTER = 0.05


class NoisyDetector:
    """包一层检测噪声：按类别丢弃检测（漏检），并以 p_fp 注入假缺陷（误检）。"""

    def __init__(self, base, gridmap, p_miss: float, p_fp: float, seed: int = 0):
        self.base = base
        self.gridmap = gridmap
        self.p_miss = float(p_miss)
        self.p_fp = float(p_fp)
        self.rng = random.Random(seed)
        self.stats = {"calls": 0, "dropped": 0, "injected": 0}

    def _random_free_point(self) -> tuple[float, float]:
        gm = self.gridmap
        for _ in range(64):
            x = self.rng.uniform(0.5, gm.size_m[0] - 0.5)
            y = self.rng.uniform(0.5, gm.size_m[1] - 0.5)
            if gm.is_free_world(x, y):
                return x, y
        return gm.size_m[0] / 2.0, gm.size_m[1] / 2.0

    def __call__(self, frame):
        dets = list(self.base(frame))
        self.stats["calls"] += 1
        kept = []
        for det in dets:
            # 缺陷类漏检率用 p_miss，普通类按实测 0.99 取十分之一
            p = self.p_miss if det.class_id in DEFECT_CLASS_IDS else self.p_miss / 10.0
            if self.rng.random() < p:
                self.stats["dropped"] += 1
                continue
            kept.append(det)
        if self.p_fp > 0 and self.rng.random() < self.p_fp:
            x, y = self._random_free_point()
            kept.append(Detection(
                class_id=2,
                confidence=0.62,
                bbox=(320.0, 240.0, 380.0, 300.0),
                world_xy=(x, y),
            ))
            self.stats["injected"] += 1
        return kept


def collect(scene, pipe) -> dict[str, Any]:
    raw = scene.gridmap
    poses = [rec.pose for rec in pipe.records]
    collisions = 0
    length = 0.0
    min_clear = math.inf
    for i, (x, y, _yaw) in enumerate(poses):
        row, col = raw.world_to_grid(x, y)
        if not raw.in_bounds(row, col) or int(raw.grid[row, col]) != FREE:
            collisions += 1
        min_clear = min(min_clear, raw.nearest_obstacle_distance(x, y))
        if i:
            length += math.hypot(x - poses[i - 1][0], y - poses[i - 1][1])
    completed = pipe.completed
    created = sum(len(rec.created_tasks) for rec in pipe.records)
    return {
        "steps": len(pipe.records),
        "tasks_completed": len(completed),
        "inspection_completed": sum(t.reason == "inspection" for t in completed),
        "defect_completed": sum(t.reason == "defect_revisit" for t in completed),
        "tasks_created": created,
        # 场景真值缺陷是 3 个，多出来的都是误检造成的重复/幽灵任务
        "spurious_tasks": max(0, created - 3),
        "collisions": collisions,
        "trajectory_length_m": round(length, 2),
        "min_clearance_m": round(min_clear, 3) if poses else float("nan"),
    }


def run_case(decision_cfg, pipeline_cfg, p_miss: float, p_fp: float, seed: int,
             max_steps: int, ledger: Path) -> dict[str, Any]:
    scene = build_inspection_field()
    camera = CameraModel.from_dict(pipeline_cfg.get("camera"))
    base = SimulatedDetector(
        scene.marks, camera,
        max_range_m=float(cfg_get(pipeline_cfg, "detector.max_range_m", 6.0)),
        conf_jitter=CONF_JITTER,
        seed=seed,
    )
    det = NoisyDetector(base, scene.gridmap, p_miss, p_fp, seed=seed)
    pipe = InspectionPipeline.build(
        decision_cfg, pipeline_cfg, scene.gridmap, det,
        start_pose=scene.start_pose,
        inspect_points=scene.inspect_points,
        ledger_path=ledger,
    )
    pipe.run(max_steps=max_steps)
    out = collect(scene, pipe)
    out.update({"p_miss": p_miss, "p_fp": p_fp, "detector_stats": det.stats})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="视觉误检/漏检对巡检闭环的影响")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p-miss", type=float, nargs="*", default=None,
                    help="漏检率列表（省略则跑 clean / 仅漏检 / 仅误检 / 实测 / 加倍 五档）")
    ap.add_argument("--p-fp", type=float, nargs="*", default=None,
                    help="误检率列表（省略则与下面默认档位对应）")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    decision_cfg = load_config("decision")
    pipeline_cfg = load_config("pipeline")
    root = (Path(args.out_dir).resolve() if args.out_dir
            else resolve_path(cfg_get(pipeline_cfg, "paths.results_root", "../results")) / "noise")
    root = ensure_dir(root)
    tag = date_tag()

    misses = args.p_miss if args.p_miss is not None else [0.0, 0.15, 0.0, 0.15, 0.30]
    fps = args.p_fp if args.p_fp is not None else [0.0, 0.0, 0.15, 0.15, 0.30]

    rows = []
    for p_miss, p_fp in zip(misses, fps):
        ledger = root / f"ledger_{tag}_{p_miss:.2f}.jsonl"
        if ledger.exists():
            ledger.unlink()
        row = run_case(decision_cfg, pipeline_cfg, p_miss, p_fp, args.seed, args.max_steps, ledger)
        rows.append(row)
        print(f"[case] p_miss={p_miss:.2f} p_fp={p_fp:.2f} -> steps={row['steps']} "
              f"done={row['tasks_completed']} (defect {row['defect_completed']}) "
              f"created={row['tasks_created']} collisions={row['collisions']}")

    json_path = root / f"noise_{tag}.json"
    save_json({"cases": rows, "seed": args.seed, "max_steps": args.max_steps}, json_path)

    md = ["# 视觉误差传播：漏检/误检对巡检闭环的影响", "",
          "仿真检测器按测得 P/R 注入噪声（broken P/R≈0.85，normal≈0.99）。", "",
          "| p_miss | p_fp | 步数 | 完成 | 缺陷复检 | 创建任务 | 误检多建 | 碰撞 | 轨迹 m |",
          "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        md.append(f"| {r['p_miss']:.2f} | {r['p_fp']:.2f} | {r['steps']} | {r['tasks_completed']} | "
                  f"{r['defect_completed']} | {r['tasks_created']} | {r['spurious_tasks']} | "
                  f"{r['collisions']} | {r['trajectory_length_m']} |")
    md_path = root / f"noise_{tag}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] {json_path}")
    print(f"[done] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
