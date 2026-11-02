"""像素 ↔ 世界坐标（单目地面投影口径）。

假设：缺陷落点在地面平面 ``z = 0``，相机固定车体、已知内参与俯仰角。
这是仿真阶段的近似口径（实机需要标定，或换测距/双目），README 里写明了限制。

车体系：x 前、y 左、z 上；相机系：x 右、y 下、z 前（与 OpenCV 一致）。
两系之间：``(x_r, y_r, z_r) = (z_c, -x_c, -y_c)``，再绕 y 轴下俯 ``pitch``。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schema import Detection, DetectionEvent


@dataclass
class CameraModel:
    width: int = 640
    height: int = 480
    fx: float = 480.0
    fy: float = 480.0
    cx: float = 320.0
    cy: float = 240.0
    height_m: float = 1.5
    pitch_deg: float = 25.0
    roll_deg: float = 0.0
    mount_x: float = 0.0
    mount_y: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "CameraModel":
        cfg = cfg or {}
        defaults = cls()
        return cls(**{k: type(getattr(defaults, k))(cfg.get(k, getattr(defaults, k)))
                      for k in ("width", "height", "fx", "fy", "cx", "cy",
                                "height_m", "pitch_deg", "roll_deg", "mount_x", "mount_y")})

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    def intrinsics(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=float)


def _rot_x(angle: float) -> np.ndarray:
    """绕相机光轴滚转：正角度 = 画面顺时针。"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def _rot_y(angle: float) -> np.ndarray:
    """绕车体 y 轴（左）旋转：正角度 = 俯视向下。"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def camera_position(camera: CameraModel, pose: Sequence[float]) -> np.ndarray:
    """相机光心的世界坐标。"""
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    offset = _rot_z(yaw) @ np.array([camera.mount_x, camera.mount_y, 0.0])
    return np.array([x + offset[0], y + offset[1], camera.height_m], dtype=float)


def pixel_to_ground(
    camera: CameraModel,
    pose: Sequence[float],
    u: float,
    v: float,
) -> tuple[float, float] | None:
    """像素 → 地面世界坐标；射线朝上（打不到地面）时返回 None。"""
    ray_cam = np.array([(float(u) - camera.cx) / camera.fx,
                        (float(v) - camera.cy) / camera.fy,
                        1.0], dtype=float)
    ray_roll = _rot_x(math.radians(camera.roll_deg)) @ ray_cam
    ray_robot = np.array([ray_roll[2], -ray_roll[0], -ray_roll[1]], dtype=float)
    ray_tilt = _rot_y(camera.pitch_rad) @ ray_robot
    ray_world = _rot_z(float(pose[2])) @ ray_tilt
    if ray_world[2] >= -1e-9:
        return None
    origin = camera_position(camera, pose)
    t = -origin[2] / ray_world[2]
    if t <= 0:
        return None
    point = origin + t * ray_world
    return (float(point[0]), float(point[1]))


def ground_to_pixel(
    camera: CameraModel,
    pose: Sequence[float],
    x: float,
    y: float,
) -> tuple[float, float, bool]:
    """地面世界坐标 → 像素 ``(u, v, visible)``。"""
    px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    rel = np.array([float(x) - px, float(y) - py, 0.0])
    local = _rot_z(-yaw) @ rel
    local = np.array([local[0] - camera.mount_x, local[1] - camera.mount_y, -camera.height_m])
    ray_robot = _rot_y(-camera.pitch_rad) @ local
    x_c, y_c, z_c = -ray_robot[1], -ray_robot[2], ray_robot[0]
    if z_c <= 1e-6:
        return (0.0, 0.0, False)
    cam = _rot_x(-math.radians(camera.roll_deg)) @ np.array([x_c, y_c, z_c], dtype=float)
    x_c, y_c, z_c = float(cam[0]), float(cam[1]), float(cam[2])
    u = camera.fx * x_c / z_c + camera.cx
    v = camera.fy * y_c / z_c + camera.cy
    visible = (0.0 <= u < camera.width) and (0.0 <= v < camera.height)
    return (float(u), float(v), bool(visible))


def bearing_range(camera: CameraModel, pose: Sequence[float], x: float, y: float) -> tuple[float, float]:
    """目标相对车体的方位角（rad）与距离（m）。"""
    dx = float(x) - float(pose[0])
    dy = float(y) - float(pose[1])
    return (math.atan2(dy, dx), math.hypot(dx, dy))


def fill_world_xy(
    event: DetectionEvent,
    camera: CameraModel,
    pose: Sequence[float],
) -> DetectionEvent:
    """把事件里所有检测框中心的像素坐标填成世界坐标（原地修改并返回）。"""
    for det in event.detections:
        if det.world_xy is not None:
            continue
        u, v = det.center
        point = pixel_to_ground(camera, pose, u, v)
        if point is not None:
            det.world_xy = point
    return event


def detection_world_xy(
    detection: Detection,
    camera: CameraModel,
    pose: Sequence[float],
) -> tuple[float, float] | None:
    u, v = detection.center
    return pixel_to_ground(camera, pose, u, v)
