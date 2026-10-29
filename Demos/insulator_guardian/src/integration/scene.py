"""巡检场景：一条可复现的「变电站走廊」地图 + 绝缘子真值点。

由 ``tests`` 与 ``scripts/demo_pipeline.py`` 共用，保证图和测试看的是同一个场。
地图 12 m × 8 m，四根立柱围出回字形走廊；真值点沿走廊布置，
3 个缺陷点（2 处闪络 + 1 处破损）用来触发「检测 → 复检任务」闭环。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..decision.map import FREE, OCCUPIED, GridMap
from .sim_detector import InsulatorMark


@dataclass
class InspectionScene:
    gridmap: GridMap
    marks: list[InsulatorMark]
    start_pose: tuple[float, float, float]
    inspect_points: list[tuple[float, float]]
    description: str = ""

    @property
    def defect_marks(self) -> list[InsulatorMark]:
        return [m for m in self.marks if m.class_id != 0]


def build_inspection_field(resolution: float = 0.1) -> InspectionScene:
    """构建 12×8 m 的巡检走廊。"""
    res = float(resolution)
    rows, cols = int(round(8.0 / res)), int(round(12.0 / res))
    grid = np.full((rows, cols), FREE, dtype=np.uint8)
    grid[0, :] = grid[-1, :] = OCCUPIED
    grid[:, 0] = grid[:, -1] = OCCUPIED

    def block(r0: int, r1: int, c0: int, c1: int) -> None:
        grid[r0:r1, c0:c1] = OCCUPIED

    # 四根立柱（各 1.6m × 1.6m），留出 ≥1.8m 的走廊
    block(20, 36, 20, 36)
    block(20, 36, 84, 100)
    block(44, 60, 20, 36)
    block(44, 60, 84, 100)
    # 走廊中段再放一根柱子，制造「拐弯 + 局部避障」
    block(32, 44, 56, 68)

    gridmap = GridMap(grid=grid, resolution=res, origin=(0.0, 0.0))

    marks = [
        InsulatorMark(1.0, 1.0, class_id=0, confidence=0.92),
        InsulatorMark(4.0, 1.0, class_id=0, confidence=0.90),
        InsulatorMark(8.0, 1.2, class_id=1, confidence=0.66),   # 闪络
        InsulatorMark(10.8, 3.2, class_id=0, confidence=0.91),
        InsulatorMark(10.8, 6.6, class_id=2, confidence=0.72),  # 破损
        InsulatorMark(5.0, 7.0, class_id=0, confidence=0.93),
        InsulatorMark(1.2, 4.0, class_id=1, confidence=0.61),   # 闪络（低置信）
    ]
    inspect_points = [(m.x, m.y) for m in marks if m.class_id == 0]
    return InspectionScene(
        gridmap=gridmap,
        marks=marks,
        start_pose=(1.2, 1.2, 0.0),
        inspect_points=inspect_points,
        description="12×8m 变电站走廊，四柱 + 中段一柱，7 个绝缘子真值点（含 3 缺陷：2 闪络 + 1 破损）",
    )
