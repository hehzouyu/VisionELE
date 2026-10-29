"""Planner（A* + DWA）单测。

口径：Planner 是决策线的总入口。这里在 8×8 m 空地上做闭环仿真：从
(0.5, 0.5) 出发走到 (5, 5)，每步 dt=0.1、最多 2000 步，要求到达且每一步
都落在原始地图的可通行格里；另外单独验重规划触发与停滞兜底（允许调用
私有方法，但断言的是行为：不会卡死、最终会 blocked）。
"""
from __future__ import annotations

import math

import pytest

from src.decision.planner import Planner
from src.integration.schema import TaskPoint


def _integrate(pose, command, dt):
    """与 SimRobot 相同的单差速积分：先转后走，stop 时不移动。"""
    x, y, yaw = pose
    if command.stop:
        return (x, y, yaw)
    yaw = math.atan2(math.sin(yaw + command.angular * dt), math.cos(yaw + command.angular * dt))
    x += command.linear * dt * math.cos(yaw)
    y += command.linear * dt * math.sin(yaw)
    return (x, y, yaw)


@pytest.fixture()
def planner(open_map) -> Planner:
    return Planner(open_map)


def test_set_and_clear_goal(planner):
    assert planner.set_goal((5.0, 5.0)) is True
    assert planner.goal_xy == (5.0, 5.0)
    # 同一个目标点不重复规划
    assert planner.set_goal((5.0, 5.0)) is False
    assert planner.set_goal(TaskPoint(5.0, 5.0)) is False

    planner.clear_goal()
    assert planner.goal_xy is None

    assert planner.set_goal((1.0, 1.0), yaw=1.2) is True
    assert planner.goal_xy == (1.0, 1.0)
    assert planner.goal is not None and planner.goal.yaw == pytest.approx(1.2)


def test_closed_loop_reaches_goal_without_collision(open_map, planner):
    goal = (5.0, 5.0)
    assert planner.set_goal(goal) is True
    pose = (0.5, 0.5, 0.0)
    steps = 0
    for steps in range(2000):
        command = planner.step(pose, dt=0.1)
        pose = _integrate(pose, command, 0.1)
        assert open_map.is_free_world(*pose[:2]), f"第 {steps} 步走出地图/撞墙: {pose}"
        if math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= 0.15:
            break
    assert math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= 0.16, pose
    assert steps < 1999
    assert planner.replans >= 1


def test_path_deviation_triggers_replan(planner):
    assert planner.set_goal((5.0, 5.0)) is True
    assert planner.plan((0.5, 0.5)) is not None
    first = planner.replans
    assert first == 1

    pose = (3.0, 0.5, 0.0)  # 离计划路径 1.7 m 以上
    assert planner.path_deviation(pose) > planner.cfg.deviation_threshold
    for _ in range(3):
        pose = _integrate(pose, planner.step(pose, dt=0.1), 0.1)
    assert planner.replans > first


def test_stall_recovery_is_reachable_and_eventually_blocks(planner):
    assert planner.set_goal((5.0, 5.0)) is True
    pose = (0.5, 0.5, 0.0)
    for _ in range(15):
        command = planner.step(pose, dt=0.1)  # 位置始终不变 -> 触发停滞兜底
    assert getattr(planner, "_recovery_steps", 0) > 0
    assert command.stop is False

    for _ in range(80):
        command = planner._pursuit(pose)  # 每次调用消耗恢复步数
    assert planner.blocked is True
    assert command.stop is False
    assert 0.0 <= command.linear <= planner.dwa_cfg.v_max
