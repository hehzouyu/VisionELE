"""端到端联动（D5）：图像帧 → 检测事件 → 任务点 → 导航 → 台账。

一次循环（``tick``）
--------------------
1. 取当前任务点（缺陷复检优先），必要时重置规划目标；
2. 执行一个控制周期：A*+DWA → ``VelocityCommand`` → 仿真车体积分；
3. 用当前位置「拍」一帧 → 检测器 → 像素框 → 地面投影填 ``world_xy``；
4. ``DetectionEvent`` 交调度器（可能生成新的复检任务）；
5. 到点则完成任务，检测事件与任务流转写 ``defect_ledger.jsonl``。

时延预算：检测 < 200 ms + 决策 < 30 ms → 单循环 < 500 ms（``latency()`` 会给出实测）。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..decision.dwa import DWAConfig
from ..decision.astar import AStarConfig
from ..decision.map import GridMap
from ..decision.planner import Planner, PlannerConfig
from ..decision.scheduler import InspectionScheduler, SchedulerConfig
from ..utils.config import cfg_get, resolve_path
from ..utils.io import append_jsonl, date_tag, ensure_dir, iso_now, new_id, percentile, save_json
from ..utils.logger import get_logger
from .coords import CameraModel, fill_world_xy
from .schema import Detection, DetectionEvent, TaskPoint, VelocityCommand


@dataclass(eq=False)
class Frame:
    """一帧相机数据（仿真里 image 可以是 None）。"""

    frame_id: int = 0
    image: Any = None
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp: str = field(default_factory=iso_now)


@dataclass(eq=False)
class SimRobot:
    """单差速运动学积分（仿真阶段代替实机）。"""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    def apply(self, command: VelocityCommand, dt: float) -> tuple[float, float, float]:
        if command.stop:
            return self.pose
        self.yaw = _wrap(self.yaw + command.angular * dt)
        self.x += command.linear * dt * math.cos(self.yaw)
        self.y += command.linear * dt * math.sin(self.yaw)
        return self.pose


@dataclass(eq=False)
class StepRecord:
    step: int
    pose: tuple[float, float, float]
    command: VelocityCommand
    task: TaskPoint | None
    path: list[tuple[float, float]]
    event: DetectionEvent | None
    created_tasks: list[TaskPoint] = field(default_factory=list)
    completed_task: TaskPoint | None = None
    detect_ms: float = 0.0
    decision_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "pose": list(self.pose),
            "command": self.command.to_dict(),
            "task": None if self.task is None else self.task.to_dict(),
            "path": [list(p) for p in self.path],
            "event": None if self.event is None else self.event.to_dict(),
            "created_tasks": [t.to_dict() for t in self.created_tasks],
            "completed_task": None if self.completed_task is None else self.completed_task.to_dict(),
            "detect_ms": self.detect_ms,
            "decision_ms": self.decision_ms,
        }


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class InspectionPipeline:
    """把视觉事件与决策导航缝在一起的最小可运行闭环。"""

    def __init__(
        self,
        gridmap: GridMap,
        planner: Planner | None = None,
        scheduler: InspectionScheduler | None = None,
        detector: Callable[[Frame], list[Detection]] | None = None,
        camera: CameraModel | None = None,
        robot: SimRobot | None = None,
        dt: float = 0.1,
        max_steps: int = 3000,
        goal_tolerance: float = 0.15,
        ledger_path: str | Path | None = None,
        logger=None,
        inspect_points: Iterable[Sequence[float]] | None = None,
    ):
        self.map = gridmap
        self.planner = planner or Planner(gridmap)
        self.scheduler = scheduler or InspectionScheduler(gridmap=gridmap)
        # 预设巡检点：reset() 时重新入队，保证复位后还能从头复现整条链路
        self.inspect_points: list[tuple[float, float]] = [
            (float(p[0]), float(p[1])) for p in (inspect_points or [])
        ]
        if self.inspect_points:
            self.scheduler.add_inspection_points(self.inspect_points)
        self.detector = detector
        self.camera = camera or CameraModel()
        self.robot = robot or SimRobot()
        self.start_pose: tuple[float, float, float] = tuple(self.robot.pose)  # type: ignore[assignment]
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.goal_tolerance = float(goal_tolerance)
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.log = logger or get_logger("pipeline")

        self.step_index = 0
        self.events: list[DetectionEvent] = []
        self.records: list[StepRecord] = []
        self.completed: list[TaskPoint] = []
        self.latencies: dict[str, list[float]] = {"detect_ms": [], "decision_ms": [], "cycle_ms": []}

    # ------------------------------------------------------------------ 构建
    @classmethod
    def build(
        cls,
        decision_cfg: dict,
        pipeline_cfg: dict,
        gridmap: GridMap,
        detector: Callable[[Frame], list[Detection]] | None = None,
        start_pose: Sequence[float] = (0.0, 0.0, 0.0),
        inspect_points: Iterable[Sequence[float]] | None = None,
        ledger_path: str | Path | None = None,
        logger=None,
    ) -> "InspectionPipeline":
        map_cfg = decision_cfg.get("map", {}) or {}
        robot_radius = float(map_cfg.get("robot_radius", 0.3))
        planner = Planner(
            gridmap,
            astar_cfg=AStarConfig.from_dict(decision_cfg.get("astar")),
            dwa_cfg=DWAConfig.from_dict(decision_cfg.get("dwa"), robot_radius=robot_radius),
            planner_cfg=PlannerConfig.from_dict(decision_cfg.get("planner")),
        )
        scheduler = InspectionScheduler(SchedulerConfig.from_dict(decision_cfg.get("scheduler")), gridmap)
        # 复检机位必须按「机器人可达」判定，所以用膨胀后的地图
        scheduler.set_map(gridmap.inflate(robot_radius))

        if ledger_path is None:
            results_root = resolve_path(cfg_get(pipeline_cfg, "paths.results_root", "../results"))
            ledger_path = ensure_dir(results_root / "logs") / f"defect_ledger_{date_tag()}.jsonl"

        camera = CameraModel.from_dict(pipeline_cfg.get("camera"))
        return cls(
            gridmap=gridmap,
            planner=planner,
            scheduler=scheduler,
            detector=detector,
            camera=camera,
            robot=SimRobot(float(start_pose[0]), float(start_pose[1]), float(start_pose[2])),
            dt=float(cfg_get(pipeline_cfg, "pipeline.dt", 0.1)),
            max_steps=int(cfg_get(pipeline_cfg, "pipeline.max_steps", 3000)),
            goal_tolerance=planner.cfg.goal_tolerance,
            ledger_path=ledger_path,
            logger=logger,
            inspect_points=inspect_points,
        )

    # ------------------------------------------------------------------ 单步
    def make_frame(self, frame_id: int | None = None) -> Frame:
        return Frame(
            frame_id=self.step_index if frame_id is None else int(frame_id),
            image=None,
            pose=self.robot.pose,
            timestamp=iso_now(),
        )

    def detect(self, frame: Frame) -> list[Detection]:
        if self.detector is None:
            return []
        return list(self.detector(frame))

    def process(self, frame: Frame) -> tuple[DetectionEvent, list[TaskPoint]]:
        detections = self.detect(frame)
        event = DetectionEvent(frame_id=frame.frame_id, timestamp=frame.timestamp, detections=detections)
        if detections:
            fill_world_xy(event, self.camera, frame.pose)
        created = self.scheduler.submit(event, current_pose=frame.pose)
        if detections:
            self.events.append(event)
            self._ledger({
                "type": "detection_event",
                "event": event.to_dict(),
                "pose": list(frame.pose),
            })
            for task in created:
                self._ledger({"type": "task_created", "event_id": event.event_id, "task": task.to_dict()})
        return event, created

    def tick(self, frame: Frame | None = None) -> StepRecord:
        self.step_index += 1
        cycle_t0 = time.perf_counter()

        task = self.scheduler.next_task()
        if task is not None:
            self.planner.set_goal(task)

        t0 = time.perf_counter()
        command = self.planner.step(self.robot.pose, self.dt)
        decision_ms = (time.perf_counter() - t0) * 1000.0
        self.robot.apply(command, self.dt)

        frame = frame or self.make_frame()
        t1 = time.perf_counter()
        event, created = self.process(frame)
        detect_ms = (time.perf_counter() - t1) * 1000.0

        completed: TaskPoint | None = None
        if task is not None:
            dist = math.hypot(task.x - self.robot.x, task.y - self.robot.y)
            if self.planner.arrived or dist <= self.goal_tolerance:
                completed = task
                self.scheduler.complete(task)
                self.completed.append(task)
                self.planner.clear_goal()
                self._ledger({"type": "task_completed", "task": task.to_dict()})

        record = StepRecord(
            step=self.step_index,
            pose=self.robot.pose,
            command=command,
            task=task,
            path=list(self.planner.path or []),
            event=event,
            created_tasks=created,
            completed_task=completed,
            detect_ms=detect_ms,
            decision_ms=decision_ms,
        )
        self.records.append(record)
        self.latencies["detect_ms"].append(detect_ms)
        self.latencies["decision_ms"].append(decision_ms)
        self.latencies["cycle_ms"].append((time.perf_counter() - cycle_t0) * 1000.0)
        return record

    # ------------------------------------------------------------------ 整段
    def run(
        self,
        max_steps: int | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> dict[str, Any]:
        limit = int(max_steps or self.max_steps)
        t_start = time.perf_counter()
        idle_ticks = 0
        while self.step_index < limit:
            record = self.tick()
            if on_step is not None:
                on_step(record)
            done = self.scheduler.next_task() is None and self.planner.goal is None
            idle_ticks = idle_ticks + 1 if done else 0
            if idle_ticks >= 3:      # 连续几拍都没有任务，收工
                break
        return self.summary(duration_s=time.perf_counter() - t_start)

    # ------------------------------------------------------------------ 统计
    def latency(self) -> dict[str, dict[str, float]]:
        return {
            key: {
                "mean": float(sum(vals) / len(vals)) if vals else float("nan"),
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
                "max": max(vals) if vals else float("nan"),
                "n": len(vals),
            }
            for key, vals in self.latencies.items()
        }

    def summary(self, duration_s: float | None = None) -> dict[str, Any]:
        clearance: list[float] = []
        for rec in self.records:
            try:
                clearance.append(float(self.map.nearest_obstacle_distance(rec.pose[0], rec.pose[1])))
            except Exception:  # pragma: no cover - 地图接口不支持时跳过
                break
        radius = float(getattr(getattr(getattr(self.planner, "dwa", None), "cfg", None), "robot_radius", 0.3))
        return {
            "steps": self.step_index,
            "events": len(self.events),
            "defect_events": sum(1 for e in self.events if e.has_defect),
            "tasks_completed": len(self.completed),
            "completed_tasks": [t.to_dict() for t in self.completed],
            "tasks_pending": self.scheduler.queue_size,
            "replans": self.planner.replans,
            "scheduler_stats": dict(self.scheduler.stats),
            "latency": self.latency(),
            "duration_s": duration_s,
            "final_pose": list(self.robot.pose),
            "min_clearance_m": min(clearance) if clearance else None,
            "collisions": sum(1 for d in clearance if d < radius),
            "ledger": None if self.ledger_path is None else str(self.ledger_path),
        }

    def save_summary(self, path: str | Path) -> Path:
        return save_json(self.summary(), path)

    # ------------------------------------------------------------------ 内部
    def _ledger(self, record: dict[str, Any]) -> None:
        if self.ledger_path is None:
            return
        record = {"timestamp": iso_now(), **record}
        append_jsonl(record, self.ledger_path)

    def reset(self) -> None:
        self.step_index = 0
        self.events.clear()
        self.records.clear()
        self.completed.clear()
        for key in self.latencies:
            self.latencies[key].clear()
        # 位姿也回到起点：同一个 pipeline 对象可以反复 run() 复现同一段实验
        self.robot.x, self.robot.y, self.robot.yaw = self.start_pose
        self.planner.reset()
        self.scheduler.reset()
        reset_detector = getattr(self.detector, "reset", None)
        if callable(reset_detector):
            reset_detector()
        if self.inspect_points:
            self.scheduler.add_inspection_points(self.inspect_points)
