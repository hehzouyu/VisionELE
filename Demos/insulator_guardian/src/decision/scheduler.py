"""巡检调度（D4）：任务队列 + 缺陷事件触发 + 复检机位生成。

调度口径
--------
- 队列分两段：**缺陷复检优先**（``defect_revisit``），其次常规巡检点（``inspection``）；
- 收到含 ``flashover`` / ``broken`` 且置信度 ≥ 阈值的 ``DetectionEvent`` 时，
  在缺陷点周围 ``revisit_range`` 的一圈机位里挑一个「可通行、能看见缺陷、
  离当前位姿最近」的位置，生成任务点；
- 同一处缺陷在 ``dedupe_radius`` 内已有待执行任务、或已经复检过时，
  不再重复入队（按**缺陷点位置**判重，而不是按机位判重）。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..integration.schema import DetectionEvent, TaskPoint
from .map import GridMap


@dataclass
class SchedulerConfig:
    defect_confidence: float = 0.5
    revisit_range: float = 3.0
    view_angles: int = 24
    dedupe_radius: float = 1.0
    max_queue: int = 64
    # 连续确认帧数：同一个缺陷点要在 K 个不同帧里被看到才建任务。
    # 1 = 关闭确认（旧行为）；2~3 可显著抑制误检引起的幽灵任务。
    confirm_frames: int = 1

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "SchedulerConfig":
        cfg = cfg or {}
        defaults = cls()
        return cls(defect_confidence=float(cfg.get("defect_confidence", defaults.defect_confidence)),
                   revisit_range=float(cfg.get("revisit_range", defaults.revisit_range)),
                   view_angles=int(cfg.get("view_angles", defaults.view_angles)),
                   dedupe_radius=float(cfg.get("dedupe_radius", defaults.dedupe_radius)),
                   max_queue=int(cfg.get("max_queue", defaults.max_queue)),
                   confirm_frames=int(cfg.get("confirm_frames", defaults.confirm_frames)))


class InspectionScheduler:
    """巡检任务队列。线程模型：单线程调用（与 pipeline 一致）。"""

    def __init__(self, config: SchedulerConfig | None = None, gridmap: GridMap | None = None):
        self.cfg = config or SchedulerConfig()
        self.map = gridmap
        self._defect_tasks: deque[TaskPoint] = deque()
        self._inspection_tasks: deque[TaskPoint] = deque()
        # id(任务点) -> 缺陷点世界坐标：判重看缺陷本身，不看机位
        self._claims: dict[int, tuple[float, float]] = {}
        self._visited: list[tuple[float, float]] = []
        # 候选缺陷点：{键: {"xy": (x, y), "frames": set(frame_id)}}，键用格点量化
        self._candidates: dict[tuple[int, int], dict] = {}
        self.stats: dict[str, int] = {
            "events": 0,
            "defect_detections": 0,
            "tasks_created": 0,
            "duplicates_skipped": 0,
            "no_feasible_pose": 0,
            "pending_confirm": 0,
        }

    # ------------------------------------------------------------------ 队列
    def set_map(self, gridmap: GridMap | None) -> None:
        self.map = gridmap

    @property
    def queue_size(self) -> int:
        return len(self._defect_tasks) + len(self._inspection_tasks)

    def pending(self) -> list[TaskPoint]:
        return list(self._defect_tasks) + list(self._inspection_tasks)

    def next_task(self) -> TaskPoint | None:
        if self._defect_tasks:
            return self._defect_tasks[0]
        if self._inspection_tasks:
            return self._inspection_tasks[0]
        return None

    def pop_task(self) -> TaskPoint | None:
        if self._defect_tasks:
            return self._defect_tasks.popleft()
        if self._inspection_tasks:
            return self._inspection_tasks.popleft()
        return None

    def complete(self, task: TaskPoint | None) -> bool:
        """把任务标记为完成（从队列里移除）。"""
        if task is None:
            return False
        claim = self._claims.pop(id(task), None)
        if claim is not None and claim not in self._visited:
            self._visited.append(claim)
        for queue in (self._defect_tasks, self._inspection_tasks):
            for i, item in enumerate(queue):
                if _same_task(item, task):
                    del queue[i]
                    return True
        return False

    def add_inspection_point(self, point: TaskPoint | Sequence[float]) -> TaskPoint:
        task = point if isinstance(point, TaskPoint) else TaskPoint(float(point[0]), float(point[1]), 0.0, "inspection")
        if len(self._inspection_tasks) < self.cfg.max_queue:
            self._inspection_tasks.append(task)
        return task

    def add_inspection_points(self, points: Iterable[Sequence[float]]) -> list[TaskPoint]:
        return [self.add_inspection_point(p) for p in points]

    def clear(self) -> None:
        self._defect_tasks.clear()
        self._inspection_tasks.clear()
        self._claims.clear()
        self._candidates.clear()
        # _visited 保留：已复检过的缺陷不该因为清队列又被重新排队

    def reset(self) -> None:
        """整段实验复位：队列、claim、已复检记录、统计全部归零。

        与 ``clear()`` 的区别：``clear()`` 保留 ``_visited``，避免同一个缺陷
        在一次巡检中被反复派单；``reset()`` 用于同一个 pipeline 对象反复
        ``run()`` 复现实验。
        """
        self._defect_tasks.clear()
        self._inspection_tasks.clear()
        self._claims.clear()
        self._candidates.clear()
        self._visited.clear()
        for key in self.stats:
            self.stats[key] = 0

    # ------------------------------------------------------------------ 缺陷事件
    def submit(
        self,
        event: DetectionEvent,
        current_pose: Sequence[float] | None = None,
    ) -> list[TaskPoint]:
        """处理一帧事件，返回本次新建的复检任务点。"""
        self.stats["events"] += 1
        created: list[TaskPoint] = []
        for det in event.defect_detections(self.cfg.defect_confidence):
            self.stats["defect_detections"] += 1
            if det.world_xy is None:
                continue
            if self._is_duplicate(det.world_xy):
                self.stats["duplicates_skipped"] += 1
                continue
            if not self._confirm(det.world_xy, event.frame_id):
                self.stats["pending_confirm"] += 1
                continue
            pose = self.plan_revisit_pose(det.world_xy, current_pose)
            if pose is None:
                self.stats["no_feasible_pose"] += 1
                continue
            task = TaskPoint(pose[0], pose[1], pose[2], "defect_revisit")
            if len(self._defect_tasks) < self.cfg.max_queue:
                self._defect_tasks.append(task)
                self._claims[id(task)] = (float(det.world_xy[0]), float(det.world_xy[1]))
                created.append(task)
                self.stats["tasks_created"] += 1
        return created

    def _confirm(self, xy: Sequence[float], frame_id: int) -> bool:
        """K 帧连续确认：同一缺陷点要在 K 个不同帧里被看到才建任务。

        ``confirm_frames <= 1`` 时直接放行（保持旧行为）。候选点按
        ``dedupe_radius`` 就近归并，帧号用 set 去重，避免同一帧多个框重复计数。
        """
        k = max(1, int(self.cfg.confirm_frames))
        if k <= 1:
            return True
        x, y = float(xy[0]), float(xy[1])
        radius = max(float(self.cfg.dedupe_radius), 0.05)
        best_key, best_dist = None, math.inf
        for key, item in self._candidates.items():
            cx, cy = item["xy"]
            d = math.hypot(cx - x, cy - y)
            if d <= radius and d < best_dist:
                best_key, best_dist = key, d
        if best_key is None:
            key = (int(x / radius), int(y / radius))
            self._candidates[key] = {"xy": (x, y), "frames": {int(frame_id)}}
            return False
        item = self._candidates[best_key]
        item["frames"].add(int(frame_id))
        if len(item["frames"]) >= k:
            self._candidates.pop(best_key, None)
            return True
        return False

    def _is_duplicate(self, xy: Sequence[float]) -> bool:
        r = self.cfg.dedupe_radius
        x, y = float(xy[0]), float(xy[1])
        for cx, cy in self._claims.values():
            if math.hypot(cx - x, cy - y) <= r:
                return True
        for cx, cy in self._visited:
            if math.hypot(cx - x, cy - y) <= r:
                return True
        return False

    # ------------------------------------------------------------------ 机位生成
    def plan_revisit_pose(
        self,
        defect_xy: Sequence[float],
        current_pose: Sequence[float] | None = None,
    ) -> tuple[float, float, float] | None:
        """围着缺陷点找可观测机位，返回 ``(x, y, yaw)``；找不到返回 None。"""
        dx, dy = float(defect_xy[0]), float(defect_xy[1])
        current = (0.0, 0.0) if current_pose is None else (float(current_pose[0]), float(current_pose[1]))
        ranges = [self.cfg.revisit_range,
                  self.cfg.revisit_range * 0.75,
                  self.cfg.revisit_range * 0.5,
                  self.cfg.revisit_range * 1.25]
        best: tuple[float, float, float, float] | None = None
        for r in ranges:
            for i in range(max(4, self.cfg.view_angles)):
                ang = 2.0 * math.pi * i / max(4, self.cfg.view_angles)
                cx = dx + r * math.cos(ang)
                cy = dy + r * math.sin(ang)
                if not self._pose_usable(cx, cy, dx, dy):
                    continue
                score = math.hypot(cx - current[0], cy - current[1])
                if best is None or score < best[0]:
                    best = (score, cx, cy, ang)
            if best is not None:
                break
        if best is None:
            return None
        _, x, y, _ = best
        yaw = math.atan2(dy - y, dx - x)
        return (x, y, yaw)

    def _pose_usable(self, x: float, y: float, dx: float, dy: float) -> bool:
        if self.map is None:
            return True
        if not self.map.is_free_world(x, y):
            return False
        # 机位到缺陷点之间不能有实体障碍挡住视线
        mid_x, mid_y = (x + dx) / 2.0, (y + dy) / 2.0
        if not self.map.is_free_world(mid_x, mid_y):
            return False
        return self.map.line_clear_world((x, y), (dx, dy), step=self.map.resolution * 1.0)


def _same_task(a: TaskPoint, b: TaskPoint, tol: float = 1e-6) -> bool:
    return (a.reason == b.reason
            and abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol)
