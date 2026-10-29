"""巡检调度单测。

口径：调度器把「缺陷事件」翻译成「复检机位任务」。这里在 8×8 m 空地上验
任务顺序（缺陷优先）、缺陷去重（1 m 内只建一个任务）、完成后再检测不再入队
（_visited 生效）、以及生成的机位必须真的可通行。
"""
from __future__ import annotations

import pytest

from src.decision.scheduler import InspectionScheduler, SchedulerConfig
from src.integration.schema import Detection, DetectionEvent, TaskPoint


def _defect_event(x: float, y: float, *, cls: int = 1, conf: float = 0.9,
                  frame_id: int = 0) -> DetectionEvent:
    event = DetectionEvent(frame_id=frame_id)
    event.add(Detection(cls, conf, (100.0, 100.0, 200.0, 200.0), world_xy=(x, y)))
    return event


@pytest.fixture()
def scheduler(open_map) -> InspectionScheduler:
    return InspectionScheduler(SchedulerConfig(), open_map)


def test_inspection_points_keep_insertion_order(scheduler):
    points = scheduler.add_inspection_points([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    assert [p.xy for p in points] == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    assert [p.xy for p in scheduler.pending()] == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    assert scheduler.queue_size == 3

    scheduler.add_inspection_point((4.0, 4.0))
    assert scheduler.pending()[-1].xy == (4.0, 4.0)


def test_defect_task_has_priority(scheduler):
    scheduler.add_inspection_points([(1.0, 1.0), (2.0, 2.0)])
    created = scheduler.submit(_defect_event(6.0, 6.0), current_pose=(1.0, 1.0))
    assert len(created) == 1
    assert created[0].reason == "defect_revisit"
    assert scheduler.next_task() is created[0]  # 取队首但不弹出
    assert scheduler.complete(created[0]) is True
    # 缺陷任务完成后，剩下的仍是巡检点
    assert [t.xy for t in scheduler.pending()] == [(1.0, 1.0), (2.0, 2.0)]


def test_same_defect_within_dedupe_radius_creates_one_task(scheduler):
    first = scheduler.submit(_defect_event(6.0, 6.0, frame_id=0), current_pose=(1.0, 1.0))
    assert len(first) == 1
    second = scheduler.submit(_defect_event(6.5, 6.0, frame_id=1), current_pose=(1.0, 1.0))
    assert second == []
    assert scheduler.stats["tasks_created"] == 1
    assert scheduler.stats["duplicates_skipped"] >= 1


def test_completed_defect_is_not_queued_again(scheduler):
    created = scheduler.submit(_defect_event(6.0, 6.0))
    task = scheduler.next_task()
    assert task is created[0]
    assert scheduler.complete(task) is True

    again = scheduler.submit(_defect_event(6.2, 6.1))
    assert again == []
    assert scheduler.stats["tasks_created"] == 1
    assert scheduler.stats["duplicates_skipped"] >= 1


def test_revisit_pose_is_traversable(open_map, scheduler):
    pose = scheduler.plan_revisit_pose((6.0, 6.0), current_pose=(1.0, 1.0))
    assert pose is not None
    x, y, yaw = pose
    assert bool(open_map.is_free_world(x, y)) is True
    distance = ((x - 6.0) ** 2 + (y - 6.0) ** 2) ** 0.5
    assert 2.9 < distance < 3.1  # 默认复检距离 3 m
    assert open_map.line_clear_world((x, y), (6.0, 6.0)) is True
    assert -3.1416 < yaw <= 3.1416


def test_low_confidence_defect_is_ignored(scheduler):
    event = _defect_event(6.0, 6.0, conf=0.2)
    assert scheduler.submit(event) == []
    assert scheduler.stats["tasks_created"] == 0
