"""测试公共夹具。

口径说明
--------
- ``scene``：项目自带的 12×8 m 巡检走廊（``build_inspection_field``），
  与 ``scripts/demo_pipeline.py`` 看的是同一张图；
- ``open_map`` / ``make_open_map``：只有四周障碍的空地，用来隔离导航与调度
  逻辑。这里刻意不用 ``random_map``：碎片化随机图会让「可达」「最近点」
  一类断言随种子漂移，没法稳定复现；
- ``camera``：默认相机参数（640×480、fx=480、俯角 25°）；
- ``decision_cfg`` / ``pipeline_cfg``：直接读 configs/ 下的 YAML，避免测试
  里再抄一份参数。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.decision.map import FREE, OCCUPIED, GridMap
from src.integration.coords import CameraModel
from src.integration.scene import build_inspection_field
from src.utils.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 测试环境兼容：Python 3.13 在 Windows 上会按 mode 给新目录写 DACL，
# ``os.mkdir(path, 0o700)`` 会生成一张「仅创建者」的权限表；在受限令牌
# （沙箱 / AppContainer）下连创建进程自己都写不进去。而 pytest 的 tmp_path
# 正是用 ``tempfile.mkdtemp``（0700）建目录，于是出现
# ``PermissionError: [Errno 13]``。这里只在测试进程内把 0700 放宽成 0777，
# 不影响任何生产代码，也不改变目录以外的东西。
if os.name == "nt":
    _real_mkdir = os.mkdir

    def _mkdir_mode_compat(path, mode=0o777, *, dir_fd=None):
        if mode == 0o700:
            mode = 0o777
        return _real_mkdir(path, mode, dir_fd=dir_fd)

    os.mkdir = _mkdir_mode_compat


def pytest_configure(config):
    """把 pytest 的临时根目录放到工作区内。

    系统 TEMP 在部分受限环境下不可写（且可能残留旧版本建出来的 0700 目录），
    放到项目内的 ``.pytest_tmp`` 既稳定，也满足「不写工作区外文件」的约束。
    """
    if getattr(config.option, "basetemp", None) is None:
        base = os.path.join(ROOT, ".pytest_tmp")
        os.makedirs(base, exist_ok=True)
        config.option.basetemp = base


@pytest.fixture(scope="session")
def decision_cfg() -> dict:
    """decision.yaml：地图 / A* / DWA / planner / scheduler 参数。"""
    return load_config("decision")


@pytest.fixture(scope="session")
def pipeline_cfg() -> dict:
    """pipeline.yaml：相机、闭环参数、检测器（sim 模式）与产物路径。"""
    return load_config("pipeline")


@pytest.fixture()
def scene():
    """12×8 m 巡检走廊场景（含 7 个绝缘子真值点）。"""
    return build_inspection_field()


@pytest.fixture()
def camera() -> CameraModel:
    """默认相机模型。"""
    return CameraModel()


@pytest.fixture()
def make_open_map():
    """构造只带四周墙的空地地图：``make_open_map(8.0, 0.1)``。"""

    def _make(size_m: float = 8.0, resolution: float = 0.1, border: bool = True) -> GridMap:
        res = float(resolution)
        n = max(4, int(round(float(size_m) / res)))
        grid = np.full((n, n), FREE, dtype=np.uint8)
        if border:
            grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCCUPIED
        return GridMap(grid=grid, resolution=res)

    return _make


@pytest.fixture()
def open_map(make_open_map) -> GridMap:
    """8 m × 8 m、四周一圈障碍的空地（80×80 格，分辨率 0.1 m）。"""
    return make_open_map(8.0, 0.1)
