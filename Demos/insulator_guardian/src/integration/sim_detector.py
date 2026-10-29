"""仿真检测器：把「地面真值缺陷点」按相机口径投成像素框。

用途只有一个——在接入真 CA-YOLO 之前，让「检测 → 任务点 → 导航 → 台账」
这条链路能跑通、能计时、能可视化。它**不**代表识别精度：框是由真值点
反推的，置信度也是合成值。真实 mAP / 延迟请用 ultralytics 训练后测
（见 ``scripts/benchmark_vision.py``）。

检测器的接口与 ``vision.detector.Detector`` 完全一致：
``detector(frame) -> list[Detection]``，方便同一个 pipeline 换模型。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..integration.coords import CameraModel, ground_to_pixel
from ..integration.schema import Detection


@dataclass(frozen=True)
class InsulatorMark:
    """地面真值：缺陷点在地面的投影位置（仿真用）。"""

    x: float
    y: float
    class_id: int
    size_m: float = 0.25       # 绝缘子片在画面里的近似尺度（米）
    confidence: float = 0.9    # 合成置信度（缺陷通常更高）


@dataclass
class SimulatedDetector:
    """按相机视场/距离裁剪的合成检测器。"""

    marks: Sequence[InsulatorMark]
    camera: CameraModel
    max_range_m: float = 6.0
    fov_deg: float | None = None
    conf_jitter: float = 0.05
    seed: int = 0

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.marks = list(self.marks)
        self._rng = np.random.default_rng(int(self.seed))
        if self.fov_deg is None:
            self.fov_deg = self.camera.horizontal_fov_deg

    def reset(self, seed: int | None = None) -> None:
        """复位随机数发生器：同一 pipeline 反复 run() 时保证噪声序列一致。"""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(int(self.seed))

    def __call__(self, frame) -> list[Detection]:
        pose = frame.pose
        x0, y0 = float(pose[0]), float(pose[1])
        half_fov = math.radians(float(self.fov_deg)) / 2.0
        detections: list[Detection] = []
        for mark in self.marks:
            dx, dy = mark.x - x0, mark.y - y0
            dist = math.hypot(dx, dy)
            if dist > self.max_range_m or dist < 1e-6:
                continue
            bearing = math.atan2(dy, dx) - float(pose[2])
            bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
            if abs(bearing) > half_fov:
                continue
            if not self._visible(mark, pose):
                continue
            u, v, inside = ground_to_pixel(self.camera, pose, mark.x, mark.y)
            if not inside:
                continue
            half = 0.5 * mark.size_m * self.camera.fx / max(dist, 1e-6)
            bbox = (u - half, v - half, u + half, v + half)
            conf = mark.confidence + float(self._rng.normal(0.0, self.conf_jitter))
            conf = float(min(0.999, max(0.05, conf)))
            detections.append(Detection(class_id=mark.class_id, confidence=conf, bbox=bbox))
        return detections

    def _visible(self, mark: InsulatorMark, pose) -> bool:
        """车体前方 180° 才可能被拍到（相机朝前下）。"""
        dx, dy = mark.x - float(pose[0]), mark.y - float(pose[1])
        if math.hypot(dx, dy) < 1e-6:
            return False
        rel = math.atan2(dy, dx) - float(pose[2])
        rel = (rel + math.pi) % (2.0 * math.pi) - math.pi
        return abs(rel) < math.pi / 2.0
