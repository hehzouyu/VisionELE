"""全局/局部协同（D2+D3 的汇合点）：跟踪路径、触发重规划、输出速度指令。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..integration.schema import TaskPoint, VelocityCommand
from .astar import AStarConfig, astar, smooth_path, to_world_path
from .dwa import DWAConfig, DWAPlanner, DWAResult
from .map import GridMap


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PlannerConfig:
    goal_tolerance: float = 0.15
    deviation_threshold: float = 0.5
    replan_cooldown: float = 0.2
    max_replans: int = 2000
    safety_margin: float = 0.1        # 膨胀半径 = robot_radius + safety_margin

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "PlannerConfig":
        cfg = cfg or {}
        defaults = cls()
        return cls(**{k: type(getattr(defaults, k))(cfg.get(k, getattr(defaults, k)))
                      for k in ("goal_tolerance", "deviation_threshold",
                                "replan_cooldown", "max_replans", "safety_margin")})


class Planner:
    """A* 全局路径 + DWA 局部避障的协同规划器。

    参数
    ----
    gridmap
        原始（未膨胀）地图。内部默认按 ``robot_radius`` 膨胀后再规划，
        因此 A* 的路径本身就满足车体安全半径。
    inflate
        是否膨胀。测试里想直接控制安全距离时可关掉。
    """

    def __init__(
        self,
        gridmap: GridMap,
        astar_cfg: AStarConfig | None = None,
        dwa_cfg: DWAConfig | None = None,
        planner_cfg: PlannerConfig | None = None,
        inflate: bool = True,
    ):
        self.raw_map = gridmap
        self.astar_cfg = astar_cfg or AStarConfig()
        self.dwa_cfg = dwa_cfg or DWAConfig()
        self.cfg = planner_cfg or PlannerConfig()
        self.planning_map = (
            gridmap.inflate(self.dwa_cfg.robot_radius + self.cfg.safety_margin)
            if inflate else gridmap
        )
        self.dwa = DWAPlanner(self.planning_map, self.dwa_cfg)
        self.dwa.raw_map = gridmap       # 急停判断看真实墙距，不看膨胀边界

        self.goal: TaskPoint | None = None
        self.path: list[tuple[float, float]] | None = None
        self.path_cells: list[tuple[int, int]] | None = None
        self.arrived: bool = False
        self.blocked: bool = False
        self.last_result: DWAResult | None = None
        self.replans: int = 0
        self.v_prev: float = 0.0
        self.w_prev: float = 0.0
        self._steps_since_plan: int = 0
        # 停滞检测：位置几乎不变时切到纯跟踪兜底，避免 DWA 在窄角原地打转
        self._prev_xy: tuple[float, float] | None = None
        self._stall_steps: int = 0
        self._recovery_steps: int = 0

    # ------------------------------------------------------------------ 目标
    def set_goal(self, goal: TaskPoint | Sequence[float], yaw: float | None = None) -> bool:
        """设置目标；返回是否真的发生了变化。"""
        if isinstance(goal, TaskPoint):
            task = goal
        else:
            task = TaskPoint(float(goal[0]), float(goal[1]), yaw or 0.0, "inspection")
        if self.goal is not None and math.hypot(self.goal.x - task.x, self.goal.y - task.y) < 1e-9:
            return False
        self.goal = task
        self.path = None
        self.path_cells = None
        self.arrived = False
        self.blocked = False
        self._prev_xy = None
        self._stall_steps = 0
        self._recovery_steps = 0
        return True

    def clear_goal(self) -> None:
        self.goal = None
        self.path = None
        self.path_cells = None
        self.arrived = False
        self.blocked = False
        self._prev_xy = None
        self._stall_steps = 0
        self._recovery_steps = 0

    @property
    def goal_xy(self) -> tuple[float, float] | None:
        return None if self.goal is None else self.goal.xy

    # ------------------------------------------------------------------ 全局规划
    def plan(self, pose: Sequence[float]) -> list[tuple[float, float]] | None:
        if self.goal is None:
            return None
        start_xy = self._snap(pose)
        goal_xy = self._snap(self.goal.xy)
        start_cell = self.planning_map.world_to_grid(*start_xy)
        goal_cell = self.planning_map.world_to_grid(*goal_xy)
        cells = astar(self.planning_map, start_cell, goal_cell, self.astar_cfg)
        if cells is None:
            self.path = None
            self.path_cells = None
            return None
        raw_world = to_world_path(self.planning_map, cells)
        self.path_cells = cells
        self.path = smooth_path(self.planning_map, raw_world, self.astar_cfg)
        self.replans += 1
        self._steps_since_plan = 0
        return self.path

    def _snap(self, xy: Sequence[float]) -> tuple[float, float]:
        """把落在障碍里/地图外的点吸到最近可通行格。"""
        x, y = float(xy[0]), float(xy[1])
        row, col = self.planning_map.world_to_grid(x, y)
        if self.planning_map.is_free_cell(row, col):
            return (x, y)
        snapped = self.planning_map.nearest_free_world(x, y, max_radius_m=3.0)
        return snapped if snapped is not None else (x, y)

    def path_deviation(self, pose: Sequence[float]) -> float:
        """当前位姿到路径折线的最短距离（米）。"""
        if not self.path:
            return math.inf
        p = np.array([float(pose[0]), float(pose[1])])
        pts = np.asarray(self.path, dtype=float)
        best = math.inf
        for a, b in zip(pts, pts[1:]):
            ab = b - a
            denom = float(ab @ ab)
            t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
            q = a + t * ab
            best = min(best, float(np.hypot(*(p - q))))
        if len(pts) == 1:
            best = float(np.hypot(*(p - pts[0])))
        return best

    # ------------------------------------------------------------------ 单步
    def _needs_replan(self, pose: Sequence[float]) -> bool:
        if self.path is None or len(self.path) < 2:
            return True
        if self.replans >= self.cfg.max_replans:
            return False
        cooldown_steps = max(1, int(round(self.cfg.replan_cooldown / self.dwa_cfg.dt)))
        if self._steps_since_plan < cooldown_steps:
            return False
        return self.path_deviation(pose) > self.cfg.deviation_threshold

    def step(self, pose: Sequence[float], dt: float | None = None) -> VelocityCommand:
        """执行一个控制周期，返回速度指令。"""
        dt = self.dwa_cfg.dt if dt is None else float(dt)
        self._steps_since_plan += 1
        if self.goal is None:
            return VelocityCommand.zero(stop=True)

        dx = self.goal.x - float(pose[0])
        dy = self.goal.y - float(pose[1])
        if math.hypot(dx, dy) <= self.cfg.goal_tolerance:
            self.arrived = True
            self.v_prev = self.w_prev = 0.0
            return VelocityCommand.zero(stop=True)

        if self._needs_replan(pose) and self.plan(pose) is None:
            self.blocked = True
            self.v_prev = self.w_prev = 0.0
            return VelocityCommand.zero(stop=True)

        assert self.path is not None
        self._update_stall(pose)
        if self._recovery_steps > 0:
            return self._pursuit(pose)
        # 近终点且朝向偏得厉害时先原地转向：DWA 在目标点附近容易「绕着目标画圈」，
        # 因为最小转弯半径 = v_max / w_max ≈ 0.67 m，目标落在圈内就永远切不进去。
        goal_dist = math.hypot(dx, dy)
        goal_err = _wrap_angle(math.atan2(dy, dx) - float(pose[2]))
        if goal_dist < 0.6 and abs(goal_err) > 0.6:
            w = max(-self.dwa_cfg.w_max, min(self.dwa_cfg.w_max, 2.5 * goal_err))
            self.v_prev, self.w_prev = 0.0, w
            return VelocityCommand(0.0, w, stop=False)
        result = self.dwa.compute(pose, self.path, self.v_prev, self.w_prev)
        self.last_result = result
        if result.stop and not self.arrived:
            # 恢复行为：
            #  - 贴墙太近 / 局部窗口无可行轨迹（通常是被膨胀层卡住）→ 先确认车后
            #    0.4 m 在膨胀图上可通行，再倒车退出；否则原地转向。
            clearance = self.dwa.obstacle_distance(pose)
            if (result.reason in ("obstacle_too_close", "no_feasible_trajectory")
                    and clearance >= self.dwa_cfg.stop_distance
                    and self._can_reverse(pose)):
                self.v_prev, self.w_prev = 0.0, 0.0
                self.blocked = True
                return VelocityCommand(-0.1, 0.0, stop=False)
            err = _wrap_angle(math.atan2(self.goal.y - float(pose[1]),
                                         self.goal.x - float(pose[0])) - float(pose[2]))
            w = max(-self.dwa_cfg.w_max, min(self.dwa_cfg.w_max, 2.0 * err))
            if abs(w) < 0.15:
                w = 0.4 if err >= 0.0 else -0.4
            self.v_prev, self.w_prev = 0.0, w
            self.blocked = True
            return VelocityCommand(0.0, w, stop=False)
        self.v_prev, self.w_prev = result.v, result.w
        self.blocked = result.stop
        return VelocityCommand(result.v, result.w, stop=result.stop)

    # ------------------------------------------------------------------ 停滞兜底
    def _can_reverse(self, pose: Sequence[float], dist: float = 0.4) -> bool:
        """车后 ``dist`` 米内（膨胀图）是否整段可通行：倒车前先看有没有退路。"""
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        back = (x - math.cos(yaw) * dist, y - math.sin(yaw) * dist)
        row, col = self.planning_map.world_to_grid(back[0], back[1])
        if not self.planning_map.in_bounds(row, col):
            return False
        if not self.planning_map.is_free_world(back[0], back[1]):
            return False
        return bool(self.planning_map.line_clear_world(pose, back))
    def _update_stall(self, pose: Sequence[float]) -> None:
        """统计「几乎没挪窝」的连续步数，超过阈值就切纯跟踪。"""
        xy = (float(pose[0]), float(pose[1]))
        if self._prev_xy is not None:
            moved = math.hypot(xy[0] - self._prev_xy[0], xy[1] - self._prev_xy[1])
            self._stall_steps = self._stall_steps + 1 if moved < 0.01 else 0
        self._prev_xy = xy
        if self._stall_steps >= 10:
            self._recovery_steps = 40
            self._stall_steps = 0

    def _pursuit(self, pose: Sequence[float]) -> VelocityCommand:
        """纯跟踪兜底：沿已验证的全局路径小步前进，DWA 无法脱困时用。

        取「能直线看通的最远路点」作为跟踪目标：机器人可能因膨胀层被判定
        在障碍里，此时 ``line_clear`` 对远点恒为假；退到相邻路点（A* 已验证
        该段可通行）仍能给出有效方向，不至于原地打转。
        """
        self._recovery_steps -= 1
        assert self.path is not None
        pts = np.asarray(self.path, dtype=float)
        d = np.hypot(pts[:, 0] - float(pose[0]), pts[:, 1] - float(pose[1]))
        k = int(np.argmin(d))
        j = min(k + 1, len(pts) - 1)
        while j + 1 < len(pts):
            cand = (float(pts[j + 1, 0]), float(pts[j + 1, 1]))
            if math.hypot(cand[0] - float(pose[0]), cand[1] - float(pose[1])) > self.dwa_cfg.lookahead:
                break
            if not self.planning_map.line_clear_world(pose, cand):
                break
            j += 1
        target = (float(pts[j, 0]), float(pts[j, 1]))
        err = _wrap_angle(math.atan2(target[1] - float(pose[1]),
                                     target[0] - float(pose[0])) - float(pose[2]))
        w = max(-self.dwa_cfg.w_max, min(self.dwa_cfg.w_max, 2.5 * err))
        if abs(err) > 1.2:
            v = 0.0
        else:
            v = max(0.12, 0.55 * math.cos(err) ** 2)
        v = min(v, self.dwa_cfg.v_max)
        # 与 DWA 一致的终点限速，避免兜底阶段冲过头
        goal_dist = self.remaining_distance(pose)
        if goal_dist < self.dwa_cfg.approach_radius:
            v = min(v, max(0.08, self.dwa_cfg.v_max * goal_dist / self.dwa_cfg.approach_radius))
        if abs(w) < 0.15 and abs(err) > 0.15:
            w = 0.4 if err >= 0.0 else -0.4
        self.v_prev, self.w_prev = v, w
        self.blocked = self._recovery_steps <= 0
        return VelocityCommand(v, w, stop=False)

    def _path_point_ahead(self, pose: Sequence[float]) -> tuple[float, float]:
        """路径上离当前位姿最近点往前 lookahead 处的一点。"""
        assert self.path is not None
        pts = np.asarray(self.path, dtype=float)
        d = np.hypot(pts[:, 0] - float(pose[0]), pts[:, 1] - float(pose[1]))
        k = int(np.argmin(d))
        acc = 0.0
        for i in range(k, len(pts) - 1):
            seg = math.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1])
            acc += seg
            if acc >= self.dwa_cfg.lookahead:
                return float(pts[i + 1, 0]), float(pts[i + 1, 1])
        return float(pts[-1, 0]), float(pts[-1, 1])

    # ------------------------------------------------------------------ 工具
    def remaining_distance(self, pose: Sequence[float]) -> float:
        if self.goal is None:
            return math.inf
        return math.hypot(self.goal.x - float(pose[0]), self.goal.y - float(pose[1]))

    def reset(self) -> None:
        self.clear_goal()
        self.replans = 0
        self.v_prev = self.w_prev = 0.0
        self._steps_since_plan = 0
        self.last_result = None
        self._prev_xy = None
        self._stall_steps = 0
        self._recovery_steps = 0
