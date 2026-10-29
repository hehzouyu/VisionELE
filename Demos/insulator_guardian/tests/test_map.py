"""栅格地图单测。

口径：地图是决策线所有几何判断的地基。这里固定一张 20×20、分辨率 0.5 m
的图（唯一障碍在格 (10,10)），逐项验坐标互逆、可通行判定、膨胀语义、
连线可行性、最近自由格与距离场；所有数值都按「格距 0.5 m」手算，不依赖
随机地图。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.decision.map import FREE, OCCUPIED, GridMap

RES = 0.5
ROWS = COLS = 20


@pytest.fixture()
def one_obstacle_map() -> GridMap:
    grid = np.full((ROWS, COLS), FREE, dtype=np.uint8)
    grid[10, 10] = OCCUPIED
    return GridMap(grid=grid, resolution=RES)


def test_world_grid_roundtrip(one_obstacle_map):
    for row, col in [(0, 0), (3, 7), (10, 10), (19, 19)]:
        x, y = one_obstacle_map.grid_to_world(row, col)
        assert one_obstacle_map.world_to_grid(x, y) == (row, col)
    # 格子内的任意点都落回同一个格子
    assert one_obstacle_map.world_to_grid(0.01, 0.01) == (0, 0)
    assert one_obstacle_map.world_to_grid(5.49, 5.49) == (10, 10)


def test_is_free_world_and_bounds(one_obstacle_map):
    x, y = one_obstacle_map.grid_to_world(10, 10)
    assert not one_obstacle_map.is_free_world(x, y)
    free_x, free_y = one_obstacle_map.grid_to_world(2, 3)
    assert one_obstacle_map.is_free_world(free_x, free_y)
    assert not one_obstacle_map.is_free_world(-1.0, 1.0)
    assert not one_obstacle_map.is_free_world(10.1, 1.0)


def test_inflate_radius_semantics(one_obstacle_map):
    inflated = one_obstacle_map.inflate(1.0)
    # 原图保持不变
    assert one_obstacle_map.count(OCCUPIED) == 1
    assert inflated.count(OCCUPIED) > 1

    def is_occ(row: int, col: int) -> bool:
        return inflated.grid[row, col] != FREE

    assert is_occ(10, 10)  # 障碍自身
    assert is_occ(10, 9)    # 距障碍 0.5 m
    assert is_occ(10, 8)    # 距障碍 1.0 m（等于半径，也算障碍）
    assert not is_occ(10, 7)  # 距障碍 1.5 m，仍是自由格


def test_line_clear_world_detects_walls_and_bounds(one_obstacle_map):
    assert one_obstacle_map.line_clear_world((3.0, 5.25), (7.0, 5.25)) is False  # 穿过障碍
    assert one_obstacle_map.line_clear_world((3.0, 7.25), (7.0, 7.25)) is True
    assert one_obstacle_map.line_clear_world((1.0, 1.0), (15.0, 1.0)) is False  # 出界
    assert one_obstacle_map.line_clear_world((1.0, 1.0), (1.0, 1.0)) is True


def test_nearest_free_world(one_obstacle_map):
    x, y = one_obstacle_map.grid_to_world(10, 10)
    nearest = one_obstacle_map.nearest_free_world(x, y)
    assert nearest == pytest.approx(one_obstacle_map.grid_to_world(9, 10))  # 最近的是下方格
    # 本身是自由格时返回自己的格中心
    free_x, free_y = one_obstacle_map.grid_to_world(2, 2)
    assert one_obstacle_map.nearest_free_world(free_x, free_y) == pytest.approx(
        one_obstacle_map.grid_to_world(2, 2)
    )


def test_nearest_free_world_returns_none_when_blocked():
    grid = np.full((20, 20), OCCUPIED, dtype=np.uint8)
    grid[9:12, 9:12] = OCCUPIED
    grid[10, 10] = OCCUPIED
    grid[8, 12] = FREE  # 只有远处一个自由格
    blocked = GridMap(grid=grid, resolution=RES)
    x, y = blocked.grid_to_world(10, 10)
    assert blocked.nearest_free_world(x, y, max_radius_m=0.5) is None


def test_obstacle_distance_field_monotonic(one_obstacle_map):
    field = one_obstacle_map.obstacle_distance_field()
    assert field.shape == (ROWS, COLS)
    assert field[10, 10] == pytest.approx(0.0)
    row = field[10, 8:14]
    assert row == pytest.approx([1.0, 0.5, 0.0, 0.5, 1.0, 1.5])
    assert np.all(np.diff(field[10, 10:]) >= -1e-12)
    assert np.all(np.diff(field[10:, 10]) >= -1e-12)
    # 查询接口与距离场一致；越界返回 0
    x, y = one_obstacle_map.grid_to_world(10, 12)
    assert one_obstacle_map.nearest_obstacle_distance(x, y) == pytest.approx(1.0)
    assert one_obstacle_map.nearest_obstacle_distance(-5.0, -5.0) == 0.0


def test_reachability_and_sealing(one_obstacle_map):
    assert one_obstacle_map.is_connected((1, 1), (18, 18)) is True
    sealed = one_obstacle_map.copy()
    sealed.grid[:, 5] = OCCUPIED  # 整列封死 -> 不连通
    assert not sealed.is_connected((10, 1), (10, 9))
    sealed.grid[10, 5] = FREE  # 只开一个缺口 -> 重新连通
    assert sealed.is_connected((10, 1), (10, 9))
