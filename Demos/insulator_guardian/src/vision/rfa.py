"""RFA（Recurrent Feature Aggregation）循环特征聚合模块。

小目标缺陷常在多尺度特征之间来回「跳」，单一尺度容易漏。RFA 用一条
可控的循环回路把高层语义（P5）逐级回灌到低层（P4 → P3），再用门控
决定每一层保留多少旧特征：

    h_3 = f_3 + g_3 ⊙ up(h_4)         # 自顶向下
    out_i = gate_i(h_i)               # 每层再过一次 1x1 残差

与 FPN 的区别：不是简单相加，而是 gated + recurrent，参数共享、
轻量。参考 "Recurrent Feature Aggregation" 系列工作（用于小目标检测
的循环特征融合思想）。

**torch 延迟导入**：没装 torch 时 import 本模块不报错，构造时才提示。
"""
from __future__ import annotations

from typing import Any

try:  # 视觉线依赖；缺库时整模块降级
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 无 torch 环境
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

__all__ = ["RFABlock", "RFA", "RFAFusion", "TORCH_AVAILABLE"]


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("RFA 模块需要 torch：pip install torch torchvision")


if TORCH_AVAILABLE:
    class RFABlock(nn.Module):
        """一次自顶向下的门控循环聚合。"""

        def __init__(self, channels: list[int]):
            super().__init__()
            if len(channels) < 2:
                raise ValueError("RFA 至少需要两层特征")
            self.channels = list(channels)
            self.lateral = nn.ModuleList(
                nn.Conv2d(c, c, kernel_size=1, bias=False) for c in channels
            )
            # 门控要把「上一层上采样后的特征」压到当前层的通道数再逐元素相乘，
            # 所以第 i 个门控是 (channels[i+1] → channels[i])，不是一个方阵卷积。
            self.gates = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
                for i in range(len(channels) - 1)
            )
            self.refine = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.SiLU(inplace=True),
                )
                for c in channels
            )
            self.out_channels = channels[0]

        def forward(self, features: list["torch.Tensor"]) -> list["torch.Tensor"]:
            if len(features) != len(self.channels):
                raise ValueError(f"期望 {len(self.channels)} 层特征，收到 {len(features)}")
            outs: list[Any] = [None] * len(features)
            top = features[-1]
            outs[-1] = self.refine[-1](self.lateral[-1](top))
            for i in range(len(features) - 2, -1, -1):
                up = F.interpolate(outs[i + 1], size=features[i].shape[-2:], mode="nearest")
                # 门控同时完成两件事：通道对齐（ch_{i+1} → ch_i）与权重分配
                fused = features[i] + self.gates[i](up)
                outs[i] = self.refine[i](self.lateral[i](fused))
            return outs

    class RFA(nn.Module):
        """把任意数量的骨干特征 → 循环聚合后输出同尺度列表（方便接检测头）。"""

        def __init__(self, channels: list[int], ratio: float = 0.5):
            super().__init__()
            self.block = RFABlock(channels)

        def forward(self, features: list["torch.Tensor"]) -> list["torch.Tensor"]:
            # 各层通道数不变，只做特征层面的循环聚合，直接返回即可
            return self.block(features)

    class RFAFusion(nn.Module):
        """接在检测头之前的「薄壳」：nn.Sequential 里的一层，输入输出都是 list。

        定义在模块顶层是为了能被打包进 checkpoint——闭包/hook 都不可 pickle，
        训练完的 ``best.pt`` 会被 torch.save 整个模型对象，闭包会直接让存盘失败。
        """

        def __init__(self, channels: list[int], ratio: float = 0.5):
            super().__init__()
            self.rfa = RFA(channels, ratio=ratio)
            self.channels = list(channels)
            # 供 Ultralytics 的 _forward_once 使用（与普通层同构）
            self.f: list[int] = [-1]
            self.i: int = 0
            self.type: str = "rfa"
            self.np: int = sum(p.numel() for p in self.parameters())
            self.inplace: bool = True

        def forward(self, features: list["torch.Tensor"]) -> list["torch.Tensor"]:
            if isinstance(features, (list, tuple)):
                return list(self.rfa(list(features)))
            return self.rfa([features])

else:  # pragma: no cover - 无 torch 环境
    class RFABlock:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()

    class RFA:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()

    class RFAFusion:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()
