"""全项目共用的数据结构（口径固定，改动需同步本文档与三个 config）。

口径
----
- ``class_id``：0=normal 1=flashover 2=broken（全文一致，不再加别的类别）
- ``bbox``：像素坐标 ``(x1, y1, x2, y2)``，左上原点
- ``world_xy``：地图世界坐标（米），由 ``coords.py`` 填写
- ``VelocityCommand``：标准「线速度 + 角速度」指令，ROS2 / 串口都能接

所有结构都能 ``to_dict()`` / ``from_dict()`` / ``to_json()`` / ``from_json()``，
往返不丢字段（测试里断言过）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..utils.io import iso_now, new_id

CLASS_NAMES: dict[int, str] = {0: "normal", 1: "flashover", 2: "broken"}
DEFECT_CLASS_IDS: frozenset[int] = frozenset({1, 2})

BBox = tuple[float, float, float, float]


def _as_bbox(value: Sequence[float]) -> BBox:
    if len(value) != 4:
        raise ValueError(f"bbox 必须是 4 个数，收到 {len(value)} 个")
    x1, y1, x2, y2 = (float(v) for v in value)
    return (x1, y1, x2, y2)


@dataclass(eq=True)
class Detection:
    """单个检测框。"""

    class_id: int
    confidence: float
    bbox: BBox
    world_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.class_id = int(self.class_id)
        if self.class_id not in CLASS_NAMES:
            raise ValueError(f"非法 class_id={self.class_id}，只允许 {sorted(CLASS_NAMES)}")
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence 必须在 [0, 1]，收到 {self.confidence}")
        self.bbox = _as_bbox(self.bbox)
        if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
            raise ValueError(f"bbox 需要 x2>=x1 且 y2>=y1，收到 {self.bbox}")
        if self.world_xy is not None:
            self.world_xy = (float(self.world_xy[0]), float(self.world_xy[1]))

    # -- 便捷属性 ---------------------------------------------------------
    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def is_defect(self) -> bool:
        return self.class_id in DEFECT_CLASS_IDS

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "world_xy": list(self.world_xy) if self.world_xy is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        return cls(
            class_id=data["class_id"],
            confidence=data["confidence"],
            bbox=tuple(data["bbox"]),
            world_xy=tuple(data["world_xy"]) if data.get("world_xy") is not None else None,
        )


@dataclass(eq=True)
class DetectionEvent:
    """一帧的检测结果，台账里一行一条。"""

    event_id: str = field(default_factory=lambda: new_id())
    frame_id: int = 0
    timestamp: str = field(default_factory=iso_now)
    detections: list[Detection] = field(default_factory=list)

    def add(self, detection: Detection) -> "DetectionEvent":
        self.detections.append(detection)
        return self

    def defect_detections(self, min_confidence: float = 0.0) -> list[Detection]:
        return [d for d in self.detections if d.is_defect and d.confidence >= min_confidence]

    @property
    def has_defect(self) -> bool:
        return any(d.is_defect for d in self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionEvent":
        return cls(
            event_id=data["event_id"],
            frame_id=int(data["frame_id"]),
            timestamp=data["timestamp"],
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
        )

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "DetectionEvent":
        return cls.from_dict(json.loads(text))


@dataclass(eq=True)
class TaskPoint:
    """巡检任务点（调度器输出，规划器输入）。"""

    x: float
    y: float
    yaw: float = 0.0
    reason: str = "inspection"   # inspection / defect_revisit

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.yaw = float(self.yaw)
        if self.reason not in ("inspection", "defect_revisit"):
            raise ValueError(f"reason 只允许 inspection / defect_revisit，收到 {self.reason!r}")

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, xy: Sequence[float]) -> float:
        return math.hypot(self.x - float(xy[0]), self.y - float(xy[1]))

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskPoint":
        return cls(x=data["x"], y=data["y"], yaw=data.get("yaw", 0.0),
                   reason=data.get("reason", "inspection"))


@dataclass(eq=True)
class VelocityCommand:
    """决策线输出：标准速度指令。"""

    linear: float = 0.0     # m/s
    angular: float = 0.0    # rad/s
    stop: bool = False

    def __post_init__(self) -> None:
        self.linear = float(self.linear)
        self.angular = float(self.angular)
        self.stop = bool(self.stop)

    @classmethod
    def zero(cls, stop: bool = True) -> "VelocityCommand":
        return cls(0.0, 0.0, stop)

    @property
    def is_moving(self) -> bool:
        return not self.stop and (abs(self.linear) > 1e-9 or abs(self.angular) > 1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {"linear": self.linear, "angular": self.angular, "stop": self.stop}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VelocityCommand":
        return cls(linear=data.get("linear", 0.0), angular=data.get("angular", 0.0),
                   stop=data.get("stop", False))
