"""检测可视化单测。

口径：画框与掩码叠加都是「返回新图、不改输入」的纯函数；这里在纯白图上
验证像素确实被改写、原图保持干净、掩码形状与取值范围合法，以及调色板
键与 schema 的类别集合一致。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.integration.schema import CLASS_NAMES, Detection
from src.vision.viz import CLASS_COLORS, draw_detections, overlay_masks


def _white(width: int = 200, height: int = 150) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8))


def test_draw_detections_changes_pixels_but_not_input():
    image = _white()
    before = np.asarray(image).copy()
    det = Detection(1, 0.9, (20.0, 20.0, 100.0, 80.0))

    drawn = draw_detections(image, [det])
    assert isinstance(drawn, Image.Image)
    assert drawn.size == image.size
    out = np.asarray(drawn)
    assert np.any(out != before)  # 框画上去了
    assert np.array_equal(np.asarray(image), before)  # 原图未被修改


def test_draw_detections_on_empty_list_is_identity():
    image = _white(60, 40)
    drawn = draw_detections(image, [])
    assert np.array_equal(np.asarray(drawn), np.asarray(image))


def test_overlay_masks_shape_and_values():
    image = _white(100, 80)
    masks = np.zeros((2, 80, 100), dtype=bool)
    masks[1, 20:40, 30:50] = True

    blended = overlay_masks(image, masks)
    assert isinstance(blended, Image.Image)
    out = np.asarray(blended)
    assert out.shape == (80, 100, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255

    inside = out[30, 40]
    outside = out[60, 60]
    assert not np.array_equal(inside, outside)
    assert np.all(outside == 255)  # 掩码外保持白色
    assert np.array_equal(np.asarray(image), np.full((80, 100, 3), 255, np.uint8))


def test_overlay_masks_rejects_wrong_shape():
    image = _white(100, 80)
    with pytest.raises(ValueError):
        overlay_masks(image, np.zeros((100, 80), dtype=bool))  # (w, h) 而不是 (h, w)


def test_class_colors_match_schema_classes():
    assert set(CLASS_COLORS) == set(CLASS_NAMES)
    for color in CLASS_COLORS.values():
        assert len(color) == 3
        assert all(0 <= v <= 255 for v in color)
