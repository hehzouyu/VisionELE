"""栅格占据地图（D1）。

坐标口径（全项目统一）
----------------------
- 栅格数组 ``grid[row, col]``，``row`` 沿 y 轴、``col`` 沿 x 轴；
- 取值 ``FREE=0`` / ``OCCUPIED=1`` / ``UNKNOWN=2``；
- ``origin`` 是栅格 (0, 0) **左下角**的世界坐标，故格子中心为
  ``origin + (col + 0.5, row + 0.5) * resolution``。

ROS map 格式（.yaml + .pgm）直接可读，见 :meth:`GridMap.load_ros_map`。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

FREE = 0
OCCUPIED = 1
UNKNOWN = 2

_CELL_OFFSETS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(eq=False)
class GridMap:
    """二维占据栅格地图。"""

    grid: np.ndarray
    resolution: float = 0.1
    origin: tuple[float, float] = (0.0, 0.0)
    _distance_field: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.grid)
        if arr.ndim != 2:
            raise ValueError(f"grid 必须是二维数组，收到 shape={arr.shape}")
        if arr.size == 0:
            raise ValueError("grid 不能为空")
        self.grid = arr.astype(np.uint8, copy=False)
        self.resolution = float(self.resolution)
        if self.resolution <= 0:
            raise ValueError("resolution 必须为正")
        self.origin = (float(self.origin[0]), float(self.origin[1]))

    # ------------------------------------------------------------------ 基本信息
    @property
    def rows(self) -> int:
        return int(self.grid.shape[0])

    @property
    def cols(self) -> int:
        return int(self.grid.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def size_m(self) -> tuple[float, float]:
        """地图物理尺寸 (宽, 高)，单位米。"""
        return (self.cols * self.resolution, self.rows * self.resolution)

    def copy(self) -> "GridMap":
        return GridMap(self.grid.copy(), self.resolution, self.origin)

    # ------------------------------------------------------------------ 栅格访问
    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_free_cell(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == FREE

    def is_free_world(self, x: float, y: float) -> bool:
        row, col = self.world_to_grid(x, y)
        return self.is_free_cell(row, col)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """世界坐标 → 栅格索引 (row, col)，向下取整，越界不裁剪。"""
        col = int(math.floor((float(x) - self.origin[0]) / self.resolution))
        row = int(math.floor((float(y) - self.origin[1]) / self.resolution))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """栅格索引 → 格子中心的世界坐标。"""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    def free_mask(self) -> np.ndarray:
        return self.grid == FREE

    def obstacle_cells(self) -> np.ndarray:
        """所有非可通行格（障碍 + 未知），返回 (N, 2) 的 (row, col)。"""
        return np.argwhere(self.grid != FREE)

    def count(self, value: int) -> int:
        return int(np.count_nonzero(self.grid == value))

    # ------------------------------------------------------------------ 路径可行性
    def line_clear_world(
        self,
        start: Sequence[float],
        end: Sequence[float],
        step: float | None = None,
    ) -> bool:
        """世界坐标两点连线是否全程可通行（按 step 采样查格）。"""
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        if step is None:
            step = self.resolution * 0.5
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(dist / step)))
        for k in range(n + 1):
            t = k / n
            if not self.is_free_world(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
                return False
        return True

    def nearest_free_world(self, x: float, y: float, max_radius_m: float = 2.0) -> tuple[float, float] | None:
        """找离 (x, y) 最近的可通行格中心；超出 max_radius_m 返回 None。"""
        row, col = self.world_to_grid(x, y)
        if self.is_free_cell(row, col):
            return self.grid_to_world(row, col)
        max_cells = int(math.ceil(max_radius_m / self.resolution))
        best: tuple[float, int, int] | None = None
        for dr in range(-max_cells, max_cells + 1):
            for dc in range(-max_cells, max_cells + 1):
                r, c = row + dr, col + dc
                if not self.is_free_cell(r, c):
                    continue
                d = math.hypot(dr, dc)
                if best is None or d < best[0]:
                    best = (d, r, c)
        if best is None:
            return None
        return self.grid_to_world(best[1], best[2])

    # ------------------------------------------------------------------ 障碍膨胀
    def inflate(self, radius_m: float, treat_unknown_as_obstacle: bool = True) -> "GridMap":
        """按半径膨胀障碍（机器人安全半径）。返回新地图，不修改自身。"""
        if radius_m <= 0:
            return self.copy()
        source = (self.grid != FREE) if treat_unknown_as_obstacle else (self.grid == OCCUPIED)
        r_cells = radius_m / self.resolution
        R = int(math.ceil(r_cells))
        out = np.zeros_like(self.grid)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dy * dy + dx * dx > r_cells * r_cells + 1e-9:
                    continue
                r0, r1 = max(0, dy), self.rows + min(0, dy)
                c0, c1 = max(0, dx), self.cols + min(0, dx)
                if r0 >= r1 or c0 >= c1:
                    continue
                out[r0:r1, c0:c1] |= source[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
        return GridMap(out, self.resolution, self.origin)

    # ------------------------------------------------------------------ 距离场
    def obstacle_distance_field(self) -> np.ndarray:
        """到最近障碍的距离场（米），BFS 近似（四邻域，斜向略偏保守）。"""
        if self._distance_field is not None:
            return self._distance_field
        big = float(np.hypot(*self.size_m)) + self.resolution
        dist = np.full(self.grid.shape, big, dtype=np.float64)
        q: deque[tuple[int, int]] = deque()
        for r, c in np.argwhere(self.grid != FREE):
            dist[r, c] = 0.0
            q.append((int(r), int(c)))
        while q:
            r, c = q.popleft()
            nd = dist[r, c] + self.resolution
            for dr, dc in _CELL_OFFSETS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and nd < dist[nr, nc] - 1e-12:
                    dist[nr, nc] = nd
                    q.append((nr, nc))
        self._distance_field = dist
        return dist

    def nearest_obstacle_distance(self, x: float, y: float) -> float:
        """走距离场，O(1)。"""
        row, col = self.world_to_grid(x, y)
        if not self.in_bounds(row, col):
            return 0.0
        return float(self.obstacle_distance_field()[row, col])

    # ------------------------------------------------------------------ 连通性
    def reachable_from(self, cell: tuple[int, int]) -> np.ndarray:
        """从 cell 出发做 4 邻域 BFS，返回 bool 掩码。"""
        row, col = cell
        mask = np.zeros(self.grid.shape, dtype=bool)
        if not self.is_free_cell(row, col):
            return mask
        q: deque[tuple[int, int]] = deque([(row, col)])
        mask[row, col] = True
        while q:
            r, c = q.popleft()
            for dr, dc in _CELL_OFFSETS_4:
                nr, nc = r + dr, c + dc
                if self.is_free_cell(nr, nc) and not mask[nr, nc]:
                    mask[nr, nc] = True
                    q.append((nr, nc))
        return mask

    def is_connected(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        if not (self.is_free_cell(*a) and self.is_free_cell(*b)):
            return False
        mask = self.reachable_from(a)
        return bool(mask[b])

    # ------------------------------------------------------------------ 构造/序列化
    @classmethod
    def random_map(
        cls,
        rows: int = 60,
        cols: int = 60,
        obstacle_ratio: float = 0.18,
        seed: int = 0,
        border: bool = True,
        resolution: float = 0.1,
    ) -> "GridMap":
        """随机矩形障碍地图（测试与仿真用）。"""
        rng = np.random.default_rng(seed)
        grid = np.zeros((rows, cols), dtype=np.uint8)
        if border:
            grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCCUPIED
        target = int(rows * cols * obstacle_ratio)
        placed = 0
        guard = 0
        while placed < target and guard < 10000:
            guard += 1
            h = int(rng.integers(2, max(3, rows // 6)))
            w = int(rng.integers(2, max(3, cols // 6)))
            r = int(rng.integers(1, max(2, rows - h - 1)))
            c = int(rng.integers(1, max(2, cols - w - 1)))
            grid[r:r + h, c:c + w] = OCCUPIED
            placed += h * w
        return cls(grid, resolution=resolution)

    @classmethod
    def from_ascii(cls, text: str, resolution: float = 0.1,
                   origin: tuple[float, float] = (0.0, 0.0)) -> "GridMap":
        """``#``/``1`` 障碍，``.``/``0`` 可通行；第一行是**地图上方**（y 最大）。"""
        lines = [ln.rstrip("\n") for ln in text.strip("\n").split("\n")]
        width = max(len(ln) for ln in lines)
        grid = np.zeros((len(lines), width), dtype=np.uint8)
        for row, line in enumerate(lines):
            for col, ch in enumerate(line.ljust(width, ".")):
                if ch in "#1X":
                    grid[len(lines) - 1 - row, col] = OCCUPIED  # 翻转成左下原点
        return cls(grid, resolution=resolution, origin=origin)

    def to_ascii(self) -> str:
        chars = {FREE: ".", OCCUPIED: "#", UNKNOWN: "?"}
        lines = []
        for row in range(self.rows - 1, -1, -1):
            lines.append("".join(chars.get(int(v), "?") for v in self.grid[row]))
        return "\n".join(lines)

    # ------------------------------------------------------------------ ROS map
    @classmethod
    def load_ros_map(cls, yaml_path: str | Path) -> "GridMap":
        """读 ROS map（``map.yaml`` + ``.pgm``），按 occupied/free 阈值二值化。"""
        import yaml

        yaml_path = Path(yaml_path)
        meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        image = yaml_path.parent / str(meta["image"])
        pixels = read_pgm(image)
        negate = int(meta.get("negate", 0))
        occ_th = float(meta.get("occupied_thresh", 0.65))
        free_th = float(meta.get("free_thresh", 0.196))
        # ROS 口径：occ = (255 - value)/255；negate=1 时反转
        occ = (255.0 - pixels.astype(np.float64)) / 255.0
        if negate:
            occ = 1.0 - occ
        grid = np.full(pixels.shape, UNKNOWN, dtype=np.uint8)
        grid[occ <= free_th] = FREE
        grid[occ >= occ_th] = OCCUPIED
        grid = np.flipud(grid).copy()          # PGM 第一行在上方
        resolution = float(meta.get("resolution", 0.05))
        origin = tuple(float(v) for v in meta.get("origin", [0.0, 0.0, 0.0])[:2])
        return cls(grid, resolution=resolution, origin=origin)

    # ------------------------------------------------------------------ 便捷
    def cell_center_path(self, cells: Iterable[tuple[int, int]]) -> list[tuple[float, float]]:
        return [self.grid_to_world(r, c) for r, c in cells]

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        w, h = self.size_m
        return f"GridMap({self.rows}x{self.cols}, {w:.1f}m x {h:.1f}m, res={self.resolution})"


def read_pgm(path: str | Path) -> np.ndarray:
    """极简 PGM 读取（P2 ASCII / P5 二进制），返回 uint8 数组。"""
    data = Path(path).read_bytes()
    pos = 0
    tokens: list[str] = []

    def _next_token() -> str:
        nonlocal pos
        while pos < len(data):
            if data[pos:pos + 1] == b"#":
                while pos < len(data) and data[pos:pos + 1] not in (b"\n", b"\r"):
                    pos += 1
                continue
            if data[pos:pos + 1].isspace():
                pos += 1
                continue
            break
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        return data[start:pos].decode("ascii")

    magic = _next_token()
    if magic not in ("P2", "P5"):
        raise ValueError(f"不支持的 PGM 魔数: {magic}")
    width = int(_next_token())
    height = int(_next_token())
    maxval = int(_next_token())
    if magic == "P2":
        values = np.array([int(t) for t in data[pos:].split()][: width * height], dtype=np.float64)
        if values.size != width * height:
            raise ValueError("P2 数据量不足")
        img = values.reshape(height, width)
    else:
        pos += 1  # 跳过唯一分隔空白
        if maxval > 255:
            img = np.frombuffer(data[pos:pos + width * height * 2], dtype=">u2")
        else:
            img = np.frombuffer(data[pos:pos + width * height], dtype=np.uint8)
        img = img.reshape(height, width).astype(np.float64)
    return (img * (255.0 / maxval)).round().astype(np.uint8)
