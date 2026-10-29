"""模拟检测器单测。

口径：SimulatedDetector 只用 frame.pose 做几何可见性判断，因此这里钉住
四条视线规则——正前方可见、超 max_range 不可见、车体背后不可见、水平视场
外不可见；再验输出框的像素合法性。
"""
from __future__ import annotations

import pytest

from src.integration.coords import CameraModel
from src.integration.pipeline import Frame
from src.integration.sim_detector import InsulatorMark, SimulatedDetector


def _detector(marks, **kwargs) -> SimulatedDetector:
    kwargs.setdefault("seed", 0)
    return SimulatedDetector(marks=marks, camera=CameraModel(), **kwargs)


def test_visible_mark_gives_valid_bbox():
    detections = _detector([InsulatorMark(3.2, 0.0, 0)])(Frame(pose=(0.0, 0.0, 0.0)))
    assert len(detections) == 1
    det = detections[0]
    x1, y1, x2, y2 = det.bbox
    assert x2 > x1 and y2 > y1
    assert 0.0 <= x1 < x2 <= 640.0
    assert 0.0 <= y1 < y2 <= 480.0
    assert det.class_name == "normal"
    assert 0.05 <= det.confidence <= 0.999
    assert det.world_xy is None  # 世界坐标由 coords 层填充
    assert (x1 + x2) / 2 == pytest.approx(320.0, abs=1.0)


def test_out_of_range_is_invisible():
    detector = _detector([InsulatorMark(7.0, 0.0, 0)], max_range_m=6.0)
    assert detector(Frame(pose=(0.0, 0.0, 0.0))) == []


def test_mark_behind_robot_is_invisible():
    detector = _detector([InsulatorMark(-3.2, 0.0, 0)])
    assert detector(Frame(pose=(0.0, 0.0, 0.0))) == []


def test_mark_outside_fov_is_invisible():
    mark = InsulatorMark(3.2, 1.2, 1)  # 方位角约 20.6°
    narrow = _detector([mark], fov_deg=10.0)
    assert narrow(Frame(pose=(0.0, 0.0, 0.0))) == []
    wide = _detector([mark], fov_deg=70.0)
    assert len(wide(Frame(pose=(0.0, 0.0, 0.0)))) == 1


def test_pose_moves_the_visible_region():
    mark = InsulatorMark(4.2, 0.0, 0)
    detector = _detector([mark])
    assert len(detector(Frame(pose=(1.0, 0.0, 0.0)))) == 1
    assert detector(Frame(pose=(1.0, 0.0, 3.141592653589793))) == []


def test_only_marks_in_front_are_returned():
    marks = [InsulatorMark(3.2, 0.0, 0), InsulatorMark(3.4, 0.2, 2),
             InsulatorMark(20.0, 0.0, 1)]
    detections = _detector(marks, max_range_m=6.0)(Frame(pose=(0.0, 0.0, 0.0)))
    assert {det.class_id for det in detections} == {0, 2}


def test_confidence_is_deterministic_for_seed():
    mark = InsulatorMark(3.2, 0.0, 0, confidence=0.9)
    a = _detector([mark])(Frame(pose=(0.0, 0.0, 0.0)))[0].confidence
    b = _detector([mark])(Frame(pose=(0.0, 0.0, 0.0)))[0].confidence
    assert a == pytest.approx(b)
    assert 0.05 <= a <= 0.999
