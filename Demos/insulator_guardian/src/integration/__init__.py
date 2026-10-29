"""集成层：跨模块共享的数据结构与坐标转换。

注意：本包 ``__init__`` 只导出无副作用的内容（schema / coords），
``pipeline`` 会 import decision 层，请用 ``from src.integration.pipeline import ...``
显式导入，避免 ``vision → integration → decision`` 的循环导入。
"""
from .coords import CameraModel, fill_world_xy, ground_to_pixel, pixel_to_ground  # noqa: F401
from .schema import (  # noqa: F401
    CLASS_NAMES,
    DEFECT_CLASS_IDS,
    Detection,
    DetectionEvent,
    TaskPoint,
    VelocityCommand,
)
