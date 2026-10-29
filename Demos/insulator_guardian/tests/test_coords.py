"""像素 ↔ 地面坐标单测。

口径：相机模型是「视觉量测 → 决策世界坐标」的唯一换算口，这里验三件事——
往返一致（正对相机的地面点误差 < 5 cm）、抬高相机 / 改变俯角后可见性随之
变化、以及 ``fill_world_xy`` 只补空缺、不覆盖已有世界坐标。
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.integration.coords import (
    CameraModel,
    fill_world_xy,
    ground_to_pixel,
    pixel_to_ground,
)
from src.integration.schema import Detection, DetectionEvent

POSE = (0.0, 0.0, 0.0)


def test_center_pixel_maps_to_expected_ground_distance(camera):
    ground = pixel_to_ground(camera, POSE, camera.cx, camera.cy)
    assert ground is not None
    # 正对相机中心的地面点距离 = 相机高度 / tan(俯角)
    expected = camera.height_m / math.tan(camera.pitch_rad)
    assert ground[0] == pytest.approx(expected, rel=1e-6)
    assert ground[1] == pytest.approx(0.0, abs=1e-9)


def test_pixel_ground_roundtrip_error_under_5cm(camera):
    for u, v in [(320, 240), (200, 260), (450, 300), (100, 400), (500, 200), (300, 360)]:
        ground = pixel_to_ground(camera, POSE, u, v)
        assert ground is not None, (u, v)
        u2, v2, visible = ground_to_pixel(camera, POSE, *ground)
        assert visible is True
        back = pixel_to_ground(camera, POSE, u2, v2)
        assert back is not None
        assert math.hypot(back[0] - ground[0], back[1] - ground[1]) < 0.05


def test_horizon_pixel_has_no_ground_point(camera):
    # 画面上边缘的射线朝上，打不到地面
    assert pixel_to_ground(camera, POSE, camera.cx, 0) is None


def test_pitch_and_height_change_visibility(camera):
    assert ground_to_pixel(camera, POSE, 3.2, 0.0)[2] is True

    # 俯角 90°：镜头垂直向下，3.2 m 外的地面点落到画面外
    straight_down = replace(camera, pitch_deg=90.0)
    assert ground_to_pixel(straight_down, POSE, 3.2, 0.0)[2] is False

    # 相机抬到 6 m：3.2 m 的点跑到画面下方之外，8 m 的点重新可见
    high = replace(camera, height_m=6.0)
    assert ground_to_pixel(high, POSE, 3.2, 0.0)[2] is False
    assert ground_to_pixel(high, POSE, 8.0, 0.0)[2] is True


def test_ground_to_pixel_moves_with_yaw(camera):
    u0, _, visible0 = ground_to_pixel(camera, (0.0, 0.0, 0.0), 3.2, 0.0)
    # 车体左转 0.3 rad 后，正前方的点应偏到画面右侧
    u1, _, visible1 = ground_to_pixel(camera, (0.0, 0.0, 0.3), 3.2, 0.0)
    assert visible0 is True and visible1 is True
    assert u1 > u0


def test_fill_world_xy_fills_only_missing(camera):
    event = DetectionEvent()
    filled = Detection(0, 0.9, (300.0, 230.0, 340.0, 250.0))
    kept = Detection(1, 0.8, (300.0, 230.0, 340.0, 250.0), world_xy=(1.0, 2.0))
    event.add(filled)
    event.add(kept)

    out = fill_world_xy(event, camera, POSE)
    assert out is event
    assert filled.world_xy is not None
    assert filled.world_xy[0] == pytest.approx(3.2167603807643377, rel=1e-9)
    assert filled.world_xy[1] == pytest.approx(0.0, abs=1e-9)
    assert kept.world_xy == (1.0, 2.0)  # 已有的世界坐标不被覆盖
    assert out.detections[1] is kept
