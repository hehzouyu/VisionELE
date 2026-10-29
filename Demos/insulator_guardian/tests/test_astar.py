"""A* 全局规划单测。

口径：栅格 A* 只负责「找一条可通行的折线」，所以这里验四件事——
空图最短、封死返回 None、绕障路径每段都真的可通行、平滑后仍然可通行；
再加 50 张固定种子随机图，验证「只要起终点连通，规划就必成且无碰撞」。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.decision.astar import (
    AStarConfig,
    astar,
    path_length,
    segments_clear,
    smooth_path,
    to_world_path,
)
from src.decision.map import FREE, OCCUPIED, GridMap


@pytest.fixture()
def empty_map() -> GridMap:
    grid = np.full((80, 80), FREE, dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCCUPIED
    return GridMap(grid=grid, resolution=0.1)


def test_straight_line_is_shortest(empty_map):
    cells = astar(empty_map, (5, 5), (5, 25))
    assert cells is not None
    world = to_world_path(empty_map, cells)
    assert path_length(world) == pytest.approx(2.0)
    assert segments_clear(empty_map, world)


def test_diagonal_length_is_euclidean(empty_map):
    cells = astar(empty_map, (5, 5), (25, 25))
    assert cells is not None
    length = path_length(to_world_path(empty_map, cells))
    assert length == pytest.approx(math.sqrt(8.0), rel=1e-6)


def test_same_cell_returns_single_point(empty_map):
    cells = astar(empty_map, (5, 5), (5, 5))
    assert cells == [(5, 5)]


def test_sealed_map_returns_none(empty_map):
    sealed = empty_map.copy()
    sealed.grid[1:79, 30] = OCCUPIED  # 竖墙把地图一分为二
    assert astar(sealed, (5, 5), (5, 35)) is None
    # 起终点不可通行也应直接失败
    assert astar(sealed, (30, 30), (5, 35)) is None


def test_detour_is_collision_free(empty_map):
    walled = empty_map.copy()
    walled.grid[1:60, 40] = OCCUPIED
    cells = astar(walled, (5, 5), (5, 70))
    assert cells is not None
    world = to_world_path(walled, cells)
    assert segments_clear(walled, world)
    # 绕路一定比直线长
    assert path_length(world) > math.hypot(0.0, (70 - 5) * 0.1) * 0.99
    smoothed = smooth_path(walled, world, AStarConfig())
    assert len(smoothed) >= 2
    assert segments_clear(walled, smoothed)


def test_random_maps_reachable_pairs_always_planned():
    config = AStarConfig()
    runs = 50
    for seed in range(runs):
        grid = GridMap.random_map(rows=40, cols=40, obstacle_ratio=0.18,
                                  seed=seed).inflate(0.2)
        free_cells = np.argwhere(grid.grid == FREE)
        rng = np.random.default_rng(1000 + seed)
        pair = None
        for _ in range(400):
            a = free_cells[rng.integers(0, len(free_cells))]
            b = free_cells[rng.integers(0, len(free_cells))]
            ca = (int(a[0]), int(a[1]))
            cb = (int(b[0]), int(b[1]))
            if abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]) < 10:
                continue
            if not grid.is_connected(ca, cb):
                continue
            pair = (ca, cb)
            break
        assert pair is not None, f"seed={seed} 找不到连通点对"
        cells = astar(grid, pair[0], pair[1], config)
        assert cells is not None, f"seed={seed} 连通却规划失败"
        world = to_world_path(grid, cells)
        assert all(grid.is_free_world(*p) for p in world), f"seed={seed} 路径经过障碍"
        assert segments_clear(grid, world), f"seed={seed} 存在穿墙线段"
        smoothed = smooth_path(grid, world, config)
        assert segments_clear(grid, smoothed), f"seed={seed} 平滑后穿墙"
        assert len(smoothed) >= 2
