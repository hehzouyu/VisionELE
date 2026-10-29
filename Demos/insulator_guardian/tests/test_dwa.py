"""DWA 局部规划单测。

口径：DWA 只对「未来 horizon 秒」负责，因此这里用 8 m 空图隔离环境，
验四件事——正前方无遮挡时给出前进速度；前方 0.4 m 有墙时预测轨迹不穿墙
且明显减速；贴墙到急停距离时走 obstacle_too_close 分支；预测轨迹形状
为 (M, steps, 2)。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.decision.dwa import DWAConfig, DWAPlanner
from src.decision.map import OCCUPIED


@pytest.fixture()
def open_map(make_open_map):
    return make_open_map(8.0, 0.1)


@pytest.fixture()
def dwa(open_map) -> DWAPlanner:
    return DWAPlanner(open_map, DWAConfig())


def test_clear_forward_selects_positive_speed(dwa):
    pose = (3.0, 3.0, 0.0)
    result = dwa.compute(pose, [(3.0, 3.0), (6.0, 3.0)])
    assert result.stop is False
    assert result.reason == "ok"
    assert result.v > 0.1
    assert result.w == pytest.approx(0.0, abs=1e-9)
    assert result.target[0] > 3.4  # 前视点朝目标方向


def test_prediction_shape_and_no_motion_behind_start(dwa):
    pose = (3.0, 3.0, 0.0)
    traj = dwa.predict(pose, np.array([0.2]), np.array([0.0]))
    assert traj.shape == (1, dwa.cfg.steps, 2)
    assert np.all(traj[..., 0] > pose[0])
    # evaluate 的批量形状 (M, steps, 2)
    v_samples, w_samples = dwa.sample_window(0.2, 0.0)
    scores, valid, clearance, vv, ww, batch = dwa.evaluate(
        pose, v_samples, w_samples, target=(6.0, 3.0)
    )
    assert batch.shape == (len(scores), dwa.cfg.steps, 2)
    assert scores.shape == valid.shape == clearance.shape == vv.shape == ww.shape
    assert dwa.cfg.steps == 20


def test_near_wall_reduces_speed_and_keeps_trajectory_free(open_map, dwa):
    # 机器人位于 (5,5) 朝 +x，0.4 m 处立一道墙
    open_map.grid[1:79, 54:57] = OCCUPIED
    pose = (5.0, 5.0, 0.0)
    result = dwa.compute(pose, [(5.0, 5.0), (7.0, 5.0)], v_cur=0.0)
    assert result.stop is False
    assert result.v <= 0.1  # 明显减速
    for x, y in result.trajectory:
        assert open_map.is_free_world(x, y)


def test_obstacle_too_close_stops(open_map, dwa):
    open_map.grid[1:79, 52:57] = OCCUPIED  # 0.2 m 处贴脸
    pose = (5.0, 5.0, 0.0)
    result = dwa.compute(pose, [(5.0, 5.0), (7.0, 5.0)])
    assert result.stop is True
    assert result.reason == "obstacle_too_close"
    assert result.v == 0.0
    assert result.clearance == pytest.approx(0.2, abs=1e-6)


def test_result_trajectory_includes_start(dwa):
    result = dwa.compute((3.0, 3.0, 0.0), [(3.0, 3.0), (6.0, 3.0)])
    assert result.trajectory.shape == (dwa.cfg.steps + 1, 2)
    assert result.trajectory[0] == pytest.approx((3.0, 3.0))
    assert result.command.linear == pytest.approx(result.v)
    assert result.command.stop is False
