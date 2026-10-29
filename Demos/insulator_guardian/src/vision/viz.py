"""可视化：检测框、分割掩码叠加、预览图（matplotlib 只在绘图函数里按需导入）。"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..integration.schema import CLASS_NAMES, Detection

# 类别配色（RGB）：正常=蓝灰，闪络=琥珀，破损=红
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (96, 165, 250),    # normal
    1: (251, 191, 36),    # flashover
    2: (239, 68, 68),     # broken
}


def _font(size: int = 14) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: np.ndarray | Image.Image,
    detections: Sequence[Detection],
    class_names: dict[int, str] | None = None,
    show_confidence: bool = True,
    line_width: int = 2,
) -> Image.Image:
    """在图上画检测框，返回新的 PIL.Image（不改原图）。"""
    names = class_names or CLASS_NAMES
    canvas = _to_pil(image).copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    for det in detections:
        color = CLASS_COLORS.get(det.class_id, (148, 163, 184))
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = names.get(det.class_id, str(det.class_id))
        if show_confidence:
            label = f"{label} {det.confidence:.2f}"
        text_w = draw.textlength(label, font=font)
        ty = max(0.0, y1 - 18)
        draw.rectangle([x1, ty, x1 + text_w + 6, ty + 18], fill=color)
        draw.text((x1 + 3, ty + 1), label, fill=(17, 24, 39), font=font)
    return canvas


def overlay_masks(
    image: np.ndarray | Image.Image,
    masks: Sequence[np.ndarray],
    alpha: float = 0.35,
    color: tuple[int, int, int] = (239, 68, 68),
    outline: bool = True,
) -> Image.Image:
    """把布尔掩码半透明地叠到图上（``masks`` 为 (H, W) bool 列表）。"""
    base = np.asarray(_to_pil(image)).astype(np.float32)
    h, w = base.shape[:2]
    out = base.copy()
    for mask in masks:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (h, w):
            raise ValueError(f"掩码尺寸 {m.shape} 与图像 {(h, w)} 不一致")
        tint = np.array(color, dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * tint
    canvas = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if outline:
        draw = ImageDraw.Draw(canvas)
        for mask in masks:
            _draw_mask_outline(draw, np.asarray(mask, dtype=bool), color)
    return canvas


def _draw_mask_outline(draw: ImageDraw.ImageDraw, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    """沿掩码边界画一圈细线（4 邻域差分）。"""
    m = mask.astype(bool)
    edge = m ^ _shrink(m)
    ys, xs = np.nonzero(edge)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((x, y), fill=color)


def _shrink(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def plot_preview(
    image: np.ndarray | Image.Image,
    detections: Sequence[Detection] | None = None,
    masks: Sequence[np.ndarray] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
):
    """返回 matplotlib Figure；给了 ``save_path`` 就顺便存盘。"""
    import matplotlib.pyplot as plt

    canvas = _to_pil(image)
    if masks:
        canvas = overlay_masks(canvas, masks)
    if detections:
        canvas = draw_detections(canvas, detections)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.asarray(canvas))
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
