"""数据契约单测：检测框 / 检测事件 / 任务点 / 速度指令。

口径：schema 是视觉线与决策线之间唯一的「合同」，这里只验合同本身——
序列化能往返、类别与缺陷判定正确、非法输入会被拒绝。
"""
from __future__ import annotations

import json

import pytest

from src.integration.schema import (
    CLASS_NAMES,
    DEFECT_CLASS_IDS,
    Detection,
    DetectionEvent,
    TaskPoint,
    VelocityCommand,
)


def _det(class_id: int = 0, conf: float = 0.9) -> Detection:
    return Detection(class_id=class_id, confidence=conf, bbox=(10.0, 20.0, 30.0, 40.0))


# ---------------------------------------------------------------- Detection
def test_detection_roundtrip_preserves_fields():
    det = Detection(class_id=1, confidence=0.87, bbox=(10.0, 20.0, 30.0, 40.0),
                    world_xy=(1.5, 2.5))
    again = Detection.from_dict(det.to_dict())
    assert again == det
    assert again.world_xy == (1.5, 2.5)


def test_detection_class_name_and_defect_flag():
    assert _det(0).class_name == "normal"
    assert _det(1).class_name == "flashover"
    assert _det(2).class_name == "broken"
    assert _det(0).is_defect is False
    for cid in DEFECT_CLASS_IDS:
        assert _det(cid).is_defect is True
    assert set(DEFECT_CLASS_IDS) == {1, 2}


def test_detection_geometry_properties():
    det = _det()
    assert det.center == (20.0, 30.0)
    assert det.area == pytest.approx(20.0 * 20.0)


def test_detection_rejects_bad_input():
    with pytest.raises(ValueError):
        Detection(class_id=0, confidence=1.5, bbox=(0, 0, 1, 1))
    with pytest.raises(ValueError):
        Detection(class_id=0, confidence=0.5, bbox=(0, 0, 1))
    with pytest.raises(ValueError):
        Detection(class_id=0, confidence=0.5, bbox=(10, 0, 5, 1))  # x2 < x1
    with pytest.raises(ValueError):
        Detection(class_id=3, confidence=0.5, bbox=(0, 0, 1, 1))  # 未知类别


# ---------------------------------------------------------------- DetectionEvent
def test_event_defect_filtering_and_has_defect():
    event = DetectionEvent(frame_id=3)
    event.add(_det(0, 0.95))
    event.add(_det(1, 0.40))
    event.add(_det(2, 0.80))

    assert event.has_defect is True
    assert len(event.defect_detections()) == 2
    # 置信度阈值只留下 broken
    high = event.defect_detections(min_confidence=0.5)
    assert len(high) == 1 and high[0].class_name == "broken"

    empty = DetectionEvent()
    assert empty.has_defect is False
    assert empty.defect_detections() == []


def test_event_json_roundtrip_includes_world_xy():
    event = DetectionEvent(frame_id=7)
    event.add(Detection(1, 0.8, (1.0, 2.0, 3.0, 4.0), world_xy=(5.0, 6.0)))

    text = event.to_json()
    assert json.loads(text) == event.to_dict()
    assert DetectionEvent.from_json(text).to_dict() == event.to_dict()
    assert DetectionEvent.from_dict(event.to_dict()) == event
    assert event.to_dict()["event_id"] == event.event_id


# ---------------------------------------------------------------- TaskPoint
def test_taskpoint_roundtrip_and_reason_validation():
    point = TaskPoint(x=1.5, y=2.5, yaw=0.5, reason="defect_revisit")
    assert point.xy == (1.5, 2.5)
    assert point.distance_to((4.5, 6.5)) == pytest.approx(5.0)
    assert TaskPoint.from_dict(point.to_dict()) == point

    assert TaskPoint(1.0, 2.0).reason == "inspection"
    with pytest.raises(ValueError):
        TaskPoint(1.0, 2.0, reason="patrol")


# ---------------------------------------------------------------- VelocityCommand
def test_velocity_command_stop_and_moving():
    zero = VelocityCommand.zero()
    assert zero.stop is True
    assert zero.is_moving is False
    assert VelocityCommand(linear=0.4, angular=0.0).is_moving is True
    assert VelocityCommand.zero(stop=False).is_moving is False


def test_velocity_command_roundtrip():
    cmd = VelocityCommand(linear=0.3, angular=-0.2, stop=False)
    assert VelocityCommand.from_dict(cmd.to_dict()) == cmd
    assert CLASS_NAMES[0] == "normal"
