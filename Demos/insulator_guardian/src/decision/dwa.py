"""动态窗口法 DWA（D3）：速度/角速度采样 + 轨迹预测 + 碰撞检测 + 评价。

实现要点
--------
- **向量化**：一次把所有 (v, ω) 组合的预测轨迹算成 ``(M, steps)`` 数组，
  再一次性查占据栅格与距离场；60×60 地图上单次决策约 1–3 ms；
- **动态窗口**：按加速度在 ``window_dt`` 内可达的速度范围内采样，采样含 0；
- **评价函数**：朝向 / 到预瞄点距离 / 速度 / 余量 四项归一化后加权，
  权重全部来自 ``configs/decision.yaml``；
- **急停**：最近障碍距离 < ``stop_distance``，或无一条可行轨迹，返回 stop=True。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..integration.schema import VelocityCommand
from .map import GridMap


@dataclass(eq=False)
class DWAConfig:
    v_max: float = 1.0
    v_min: float = 0.0
    w_max: float = 1.5
    w_min: float = -1.5
    v_acc: float = 0.5
    w_acc: float = 1.0
    window_dt: float = 0.5
    v_res: float = 0.05
    w_res: float = 0.1
    dt: float = 0.1
    horizon: float = 2.0
    lookahead: float = 0.6
    stop_distance: float = 0.25
    approach_radius: float = 0.8
    clearance_cap: float = 0.6
    robot_radius: float = 0.3
    weights: dict[str, float] = field(
        default_factory=lambda: {"heading": 0.8, "dist": 0.6, "vel": 0.4, "clearance": 0.2}
    )

    @classmethod
    def from_dict(cls, cfg: dict | None, robot_radius: float = 0.3) -> "DWAConfig":
        cfg = cfg or {}
        defaults = cls()
        weights = dict(defaults.weights)
        weights.update({k: float(v) for k, v in (cfg.get("weights") or {}).items()})
        values = {k: cfg.get(k, getattr(defaults, k)) for k in (
            "v_max", "v_min", "w_max", "w_min", "v_acc", "w_acc", "window_dt",
            "v_res", "w_res", "dt", "horizon", "lookahead", "stop_distance", "clearance_cap",
            "approach_radius",
        )}
        return cls(robot_radius=float(robot_radius), weights=weights,
                   **{k: float(v) for k, v in values.items()})

    @property
    def steps(self) -> int:
        return max(1, int(round(self.horizon / self.dt)))


@dataclass(eq=False)
class DWAResult:
    v: float
    w: float
    trajectory: np.ndarray          # (n+1, 2)
    target: tuple[float, float]
    score: float = -math.inf
    clearance: float = math.inf
    stop: bool = False
    reason: str = "ok"

    @property
    def command(self) -> VelocityCommand:
        return VelocityCommand(linear=self.v, angular=self.w, stop=self.stop)


def _wrap_angle(a: np.ndarray | float):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _sample_range(lo: float, hi: float, res: float) -> np.ndarray:
    if hi - lo <= 1e-9:
        return np.array([lo], dtype=float)
    n = max(1, int(round((hi - lo) / max(res, 1e-9))))
    return np.linspace(lo, hi, n + 1)


class DWAPlanner:
    """在给定（已膨胀的）栅格地图上做局部规划。

    ``raw_map`` 可选：碰撞可行性用膨胀图判定，而「离障碍有多近」的
    急停判断用原始图（否则会离真实墙还有 0.3 m 就自己吓停）。
    """

    def __init__(self, gridmap: GridMap, config: DWAConfig | None = None):
        self.map = gridmap
        self.cfg = config or DWAConfig()
        self.raw_map: GridMap | None = None

    # ------------------------------------------------------------------ 查询
    def obstacle_distance(self, pos: Sequence[float]) -> float:
        ref = self.raw_map if self.raw_map is not None else self.map
        return ref.nearest_obstacle_distance(float(pos[0]), float(pos[1]))

    def lookahead_point(
        self,
        path: Sequence[Sequence[float]] | np.ndarray,
        pos: Sequence[float],
    ) -> tuple[float, float]:
        """从路径最近点向前看，绝不回头（nav_demo 里踩过的坑）。"""
        if len(path) == 0:
            return float(pos[0]), float(pos[1])
        arr = np.asarray(path, dtype=float).reshape(-1, 2)
        d = np.hypot(arr[:, 0] - float(pos[0]), arr[:, 1] - float(pos[1]))
        k = int(np.argmin(d))
        i = k
        while i < len(arr) - 1 and math.hypot(arr[i, 0] - pos[0], arr[i, 1] - pos[1]) < self.cfg.lookahead:
            i += 1
        return float(arr[i, 0]), float(arr[i, 1])

    # ------------------------------------------------------------------ 采样与预测
    def sample_window(self, v_cur: float, w_cur: float) -> tuple[np.ndarray, np.ndarray]:
        dv = self.cfg.v_acc * self.cfg.window_dt
        dw = self.cfg.w_acc * self.cfg.window_dt
        v_lo = max(self.cfg.v_min, v_cur - dv)
        v_hi = min(self.cfg.v_max, v_cur + dv)
        w_lo = max(self.cfg.w_min, w_cur - dw)
        w_hi = min(self.cfg.w_max, w_cur + dw)
        return _sample_range(v_lo, v_hi, self.cfg.v_res), _sample_range(w_lo, w_hi, self.cfg.w_res)

    def predict(
        self,
        pose: Sequence[float],
        v: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        """向量化预测轨迹，返回 ``(M, steps, 2)``（不含起点）。"""
        x0, y0, heading = float(pose[0]), float(pose[1]), float(pose[2])
        n = self.cfg.steps
        dt = self.cfg.dt
        k = np.arange(1, n + 1, dtype=float) * dt
        theta = heading + np.outer(w, k)
        xs = x0 + np.cumsum(v[:, None] * dt * np.cos(theta), axis=1)
        ys = y0 + np.cumsum(v[:, None] * dt * np.sin(theta), axis=1)
        return np.stack([xs, ys], axis=-1)

    def evaluate(
        self,
        pose: Sequence[float],
        v_samples: np.ndarray,
        w_samples: np.ndarray,
        target: tuple[float, float],
    ):
        """批量评价，返回 (scores, valid_mask, clearance, vv, ww, trajectories)。"""
        vv, ww = np.meshgrid(v_samples, w_samples, indexing="ij")
        vv = vv.ravel()
        ww = ww.ravel()
        traj = self.predict(pose, vv, ww)                       # (M, n, 2)

        rows = np.floor((traj[..., 1] - self.map.origin[1]) / self.map.resolution).astype(np.int64)
        cols = np.floor((traj[..., 0] - self.map.origin[0]) / self.map.resolution).astype(np.int64)
        in_bounds = ((rows >= 0) & (rows < self.map.rows) & (cols >= 0) & (cols < self.map.cols))
        rc = np.clip(rows, 0, self.map.rows - 1)
        cc = np.clip(cols, 0, self.map.cols - 1)
        free = (self.map.grid[rc, cc] == 0) & in_bounds
        valid = free.all(axis=1)

        clearance = np.where(in_bounds, self.map.obstacle_distance_field()[rc, cc], 0.0).min(axis=1)

        end = traj[:, -1, :]
        dist = np.hypot(target[0] - end[:, 0], target[1] - end[:, 1])
        end_heading = float(pose[2]) + ww * self.cfg.steps * self.cfg.dt
        desired = np.arctan2(target[1] - end[:, 1], target[0] - end[:, 0])
        dth = np.abs(_wrap_angle(desired - end_heading))

        wts = self.cfg.weights
        scores = (
            wts.get("heading", 0.8) * (1.0 - dth / math.pi)
            + wts.get("dist", 0.6) * np.exp(-dist / max(self.cfg.lookahead, 1e-6))
            + wts.get("vel", 0.4) * (vv / max(self.cfg.v_max, 1e-9))
            + wts.get("clearance", 0.2) * np.clip(clearance / max(self.cfg.clearance_cap, 1e-9), 0.0, 1.0)
        )
        return scores, valid, clearance, vv, ww, traj

    # ------------------------------------------------------------------ 主入口
    def compute(
        self,
        pose: Sequence[float],
        path: Sequence[Sequence[float]] | np.ndarray,
        v_cur: float = 0.0,
        w_cur: float = 0.0,
    ) -> DWAResult:
        pos = (float(pose[0]), float(pose[1]))
        target = self.lookahead_point(path, pos)
        start = np.array([[pos[0], pos[1]]], dtype=float)

        obstacle_dist = self.obstacle_distance(pos)
        if obstacle_dist < self.cfg.stop_distance:
            return DWAResult(0.0, 0.0, start, target, clearance=obstacle_dist,
                             stop=True, reason="obstacle_too_close")

        v_samples, w_samples = self.sample_window(float(v_cur), float(w_cur))
        # 接近终点时限速：否则 1.0 m/s × 0.1 s 的单步位移会让机器人在目标点附近
        # 反复超越、绕圈（arrival orbiting）。限速按「到路径终点的距离」线性收敛。
        goal = np.asarray(path[-1], dtype=float) if len(path) else np.asarray(target, dtype=float)
        goal_dist = float(np.hypot(goal[0] - pos[0], goal[1] - pos[1]))
        if goal_dist < self.cfg.approach_radius:
            ratio = max(goal_dist, 1e-3) / max(self.cfg.approach_radius, 1e-6)
            v_cap = max(self.cfg.v_min + 0.05, self.cfg.v_max * ratio)
            v_samples = v_samples[v_samples <= v_cap]
            if v_samples.size == 0:
                v_samples = np.array([min(self.cfg.v_min + 0.05, self.cfg.v_max)])
        scores, valid, clearance, vv, ww, traj = self.evaluate(pose, v_samples, w_samples, target)
        if not np.any(valid):
            return DWAResult(0.0, 0.0, start, target, clearance=obstacle_dist,
                             stop=True, reason="no_feasible_trajectory")

        idx = int(np.argmax(np.where(valid, scores, -np.inf)))
        v_best, w_best = float(vv[idx]), float(ww[idx])
        best_traj = np.vstack([start, traj[idx]])
        stop = abs(v_best) < 1e-6 and abs(w_best) < 1e-6
        return DWAResult(v_best, w_best, best_traj, target, score=float(scores[idx]),
                         clearance=float(clearance[idx]), stop=stop,
                         reason="ok" if not stop else "zero_velocity")
