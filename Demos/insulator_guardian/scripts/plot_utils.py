"""绘图公共设置：无头后端 + 中文字体探测。

matplotlib 在本机默认后端是 qtagg；脚本一律切到 Agg，保证无显示环境也能出图。
中文字体若一个都找不到，调用方应把图上标签退回英文（见 :func:`make_tr`），
避免出现豆腐块。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

#: 按优先级排列的中文字体候选（Windows 常见 + 开源备选）
CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
)


def setup_cjk_font() -> bool:
    """启用一个可用中文字体；返回 False 表示只能用英文标签。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


def make_tr(cjk: bool):
    """返回 ``tr(zh, en)``：有中文字体用中文，否则用英文。"""

    def tr(zh: str, en: str) -> str:
        return zh if cjk else en

    return tr


def save_fig(fig, path, dpi: int = 160) -> None:
    """统一保存并关闭图（项目里不依赖交互窗口）。"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
