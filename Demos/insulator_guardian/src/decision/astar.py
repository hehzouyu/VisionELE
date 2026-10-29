"""改进 A*（D2）：8 邻域 + 转弯代价 + 路径平滑。

设计要点
--------
1. **转弯代价**：节点状态带上「进入方向」，转向 d×45° 追加
   ``turn_penalty * d``，让规划器偏好少拐弯的路线（绝缘子巡检线束更直、更省电）；
2. **启发式**：8 邻域用 octile（``max + (√2-1)·min``），4 邻域用曼哈顿，
   在非负转向代价下仍可采纳；
3. **平滑**：Douglas-Peucker 简化 → Chaikin（二次 B 样条等价写法）光滑；
   每一段都用 :meth:`GridMap.line_clear_world` 验证，不通过就自动降级，
   所以**平滑结果一定无碰撞**（测试断言）。

坐标：入参/出参都是栅格 ``(row, col)``，``to_world_path`` 转世界坐标。
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .map import GridMap

# 方向按 45° 递增：0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
DIRS8: tuple[tuple[int, int, float], ...] = (
    (0, 1, 1.0),
    (1, 1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, -1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
)
DIRS4: tuple[tuple[int, int, float], ...] = (
    (0, 1, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (-1, 0, 1.0),
)
_NO_DIR = 8


@dataclass(frozen=True)
class AStarConfig:
    connectivity: int = 8
    turn_penalty: float = 0.5
    heuristic: str = "octile"
    smooth_method: str = "douglas_peucker"
    smooth_epsilon: float = 0.15
    bspline_iterations: int = 2

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "AStarConfig":
        cfg = cfg or {}
        smooth = cfg.get("smooth", {}) or {}
        return cls(
            connectivity=int(cfg.get("connectivity", 8)),
            turn_penalty=float(cfg.get("turn_penalty", 0.5)),
            heuristic=str(cfg.get("heuristic", "octile")),
            smooth_method=str(smooth.get("method", "douglas_peucker")),
            smooth_epsilon=float(smooth.get("epsilon", 0.15)),
            bspline_iterations=int(smooth.get("bspline_iterations", 2)),
        )

    @property
    def directions(self) -> tuple[tuple[int, int, float], ...]:
        return DIRS8 if self.connectivity == 8 else DIRS4

    @property
    def diag_cost(self) -> float:
        return math.sqrt(2.0) if self.connectivity == 8 else 1.0


def heuristic_cost(a: tuple[int, int], b: tuple[int, int], connectivity: int = 8) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if connectivity != 8:
        return float(dr + dc)
    return float(max(dr, dc) + (math.sqrt(2.0) - 1.0) * min(dr, dc))


def astar(
    gridmap: GridMap,
    start: tuple[int, int],
    goal: tuple[int, int],
    config: AStarConfig | None = None,
) -> list[tuple[int, int]] | None:
    """栅格 A*，返回 ``[(row, col), ...]``；不可达返回 None。"""
    cfg = config or AStarConfig()
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])
    if not gridmap.is_free_cell(sr, sc) or not gridmap.is_free_cell(gr, gc):
        return None
    if (sr, sc) == (gr, gc):
        return [(sr, sc)]

    dirs = cfg.directions
    goal_cell = (gr, gc)
    start_key = (sr, sc, _NO_DIR)
    came: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_key: None}
    g_score: dict[tuple[int, int, int], float] = {start_key: 0.0}
    counter = 0
    open_heap: list[tuple[float, float, int, int, int, int]] = [
        (heuristic_cost((sr, sc), goal_cell, cfg.connectivity), 0.0, counter, sr, sc, _NO_DIR)
    ]

    found: tuple[int, int, int] | None = None
    while open_heap:
        _, g_cur, _, r, c, d = heapq.heappop(open_heap)
        key = (r, c, d)
        if g_cur > g_score.get(key, math.inf) + 1e-12:
            continue
        if (r, c) == goal_cell:
            found = key
            break
        for nd, (dr, dc, step_cost) in enumerate(dirs):
            nr, nc = r + dr, c + dc
            if not gridmap.is_free_cell(nr, nc):
                continue
            turn = 0.0
            if d != _NO_DIR and nd != d:
                delta = min((d - nd) % 8, (nd - d) % 8)
                turn = cfg.turn_penalty * delta
            ng = g_cur + step_cost + turn
            nkey = (nr, nc, nd)
            if ng < g_score.get(nkey, math.inf) - 1e-12:
                g_score[nkey] = ng
                came[nkey] = key
                counter += 1
                heapq.heappush(
                    open_heap,
                    (ng + heuristic_cost((nr, nc), goal_cell, cfg.connectivity), ng, counter, nr, nc, nd),
                )

    if found is None:
        return None
    path: list[tuple[int, int]] = []
    node: tuple[int, int, int] | None = found
    while node is not None:
        path.append((node[0], node[1]))
        node = came.get(node)
    path.reverse()
    return path


def to_world_path(gridmap: GridMap, path: Sequence[tuple[int, int]]) -> list[tuple[float, float]]:
    return [gridmap.grid_to_world(int(r), int(c)) for r, c in path]


def path_length(points: Sequence[Sequence[float]]) -> float:
    return float(sum(math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
                     for i in range(len(points) - 1)))


# ---------------------------------------------------------------- 平滑工具
def densify(points: Sequence[Sequence[float]], step: float = 0.05) -> list[tuple[float, float]]:
    """按 step 等距插值，防止跟踪器切角。"""
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        for k in range(1, n + 1):
            out.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    return out


def douglas_peucker(points: Sequence[Sequence[float]], epsilon: float) -> list[tuple[float, float]]:
    """Douglas-Peucker 折线简化（递归实现，路径点数量级很小）。"""
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) <= 2 or epsilon <= 0:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True

    def _rec(i: int, j: int) -> None:
        if j <= i + 1:
            return
        (x1, y1), (x2, y2) = pts[i], pts[j]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        best_d, best_k = -1.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            if norm < 1e-12:
                d = math.hypot(px - x1, py - y1)
            else:
                d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
            if d > best_d:
                best_d, best_k = d, k
        if best_d > epsilon and best_k > 0:
            keep[best_k] = True
            _rec(i, best_k)
            _rec(best_k, j)

    _rec(0, len(pts) - 1)
    return [p for p, k in zip(pts, keep) if k]


def chaikin(points: Sequence[Sequence[float]], iterations: int, keep_ends: bool = True) -> list[tuple[float, float]]:
    """Chaikin 角切割：等价于二次 B 样条的一次细分，收敛到光滑曲线。"""
    pts = [tuple(map(float, p)) for p in points]
    for _ in range(max(0, iterations)):
        if len(pts) < 3:
            break
        new_pts: list[tuple[float, float]] = []
        if keep_ends:
            new_pts.append(pts[0])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            new_pts.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            new_pts.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        if keep_ends:
            new_pts.append(pts[-1])
        pts = new_pts
    return pts


def segments_clear(gridmap: GridMap, points: Sequence[Sequence[float]]) -> bool:
    return all(gridmap.line_clear_world(points[i], points[i + 1])
               for i in range(len(points) - 1))


def smooth_path(
    gridmap: GridMap,
    points: Sequence[Sequence[float]],
    config: AStarConfig | None = None,
) -> list[tuple[float, float]]:
    """DP 简化 + Chaikin 平滑；每层都做碰撞校验，失败自动降级。

    返回的折线保证 ``segments_clear(gridmap, out) is True``。
    """
    cfg = config or AStarConfig()
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) <= 2:
        return pts
    simplified = douglas_peucker(pts, cfg.smooth_epsilon)
    if not segments_clear(gridmap, simplified):
        # DP 切角穿墙：退回原始栅格路径，只做点加密
        if segments_clear(gridmap, pts):
            return pts
        return densify(pts, step=gridmap.resolution * 0.5)
    for iters in range(cfg.bspline_iterations, -1, -1):
        candidate = chaikin(simplified, iters) if iters > 0 else simplified
        if segments_clear(gridmap, candidate):
            return candidate
    return simplified
