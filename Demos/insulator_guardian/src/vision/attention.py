"""CA（坐标注意力，Coordinate Attention）模块。

YOLO 的卷积只在小邻域内聚合信息，绝缘子缺陷（小目标、长条形、与背景
纹理相近）容易在深层特征里被「糊」掉。CA 把特征沿 x / y 两个方向做
全局池化，得到带方向信息的通道注意图，再分别加权回原特征——

    z_c(h) = 1/W Σ_w x_c(h, w)          # 高度方向池化
    z_c(w) = 1/H Σ_h x_c(h, w)          # 宽度方向池化
    a_c(h) = σ(Conv1x1([z_h, z_w]))     # 方向敏感注意力

好处是：计算量约 O(C/(H+W))，几乎不增加参数；对长条形的破损、
闪络痕迹（沿绝缘子串方向）尤其友好。参考 Hou et al., CVPR 2021。

本文件只依赖 torch，不依赖 ultralytics，方便被任意骨干网直接调用。
**torch 延迟导入**：没装 torch 时 import 本模块不报错，构造/注入时才提示。
"""
from __future__ import annotations

from typing import Any

try:  # 视觉线依赖；缺库时整模块降级，决策线与测试不受影响
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 无 torch 环境
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

__all__ = ["CoordAtt", "CoordAttWrapper", "inject_coordatt", "param_count", "TORCH_AVAILABLE"]


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("CA 注意力需要 torch：pip install torch torchvision")


if TORCH_AVAILABLE:
    class CoordAtt(nn.Module):
        """坐标注意力块（即插即用，输入输出同形状）。"""

        def __init__(self, channels: int, reduction: int = 32,
                     act: type[nn.Module] = nn.Hardswish):
            super().__init__()
            mid = max(8, channels // max(1, reduction))
            self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid)
            self.act = act(inplace=True) if act is nn.Hardswish else act()
            self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
            self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, _, h, w = x.shape
            # 两个方向各自池化成 (N, C, H, 1) 与 (N, C, 1, W)
            x_h = x.mean(dim=3, keepdim=True)
            x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

            y = torch.cat([x_h, x_w], dim=2)
            y = self.act(self.bn1(self.conv1(y)))
            y_h, y_w = torch.split(y, [h, w], dim=2)
            y_w = y_w.permute(0, 1, 3, 2)

            a_h = torch.sigmoid(self.conv_h(y_h))
            a_w = torch.sigmoid(self.conv_w(y_w))
            return x * a_h * a_w

        def extra_repr(self) -> str:
            return f"channels={self.conv1.in_channels}, mid={self.conv1.out_channels}"

    class CoordAttWrapper(nn.Module):
        """给已有层加上 CA 的透明包装器。

        Ultralytics 在 ``_predict_once`` 里按层读取 ``m.f`` / ``m.i`` / ``m.type`` /
        ``m.np``，所以包装器要把这几个属性透传出去，否则前向会报
        ``'CoordAttWrapper' object has no attribute 'f'``。
        """

        def __init__(self, layer: nn.Module, channels: int):
            super().__init__()
            self.layer = layer
            self.ca = CoordAtt(channels)
            self.f = getattr(layer, "f", -1)
            self.i = getattr(layer, "i", -1)
            self.type = getattr(layer, "type", "")
            self.np = sum(p.numel() for p in layer.parameters()) + sum(
                p.numel() for p in self.ca.parameters()
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.ca(self.layer(x))

    def _out_channels(layer: nn.Module) -> int | None:
        """尽量推断一层的输出通道数。

        Ultralytics 的算子有多种写法（原生 ``nn.Conv2d``、自定义 ``Conv``、
        ``C2f`` 里 ``cv2`` 是 ``Conv`` 而非 ``Sequential``），这里按
        「直接属性 → 子模块 ``.conv`` → 顺序模块首元素」逐层探测。
        """
        if isinstance(layer, nn.Conv2d):
            return int(layer.out_channels)
        out = getattr(layer, "out_channels", None)
        if isinstance(out, int):
            return int(out)
        for name in ("conv", "cv2", "cv3", "c2"):
            sub = getattr(layer, name, None)
            if isinstance(sub, nn.Conv2d):
                return int(sub.out_channels)
            inner = getattr(sub, "conv", None)
            if isinstance(inner, nn.Conv2d):
                return int(inner.out_channels)
            if isinstance(sub, (list, tuple, nn.Sequential)) and len(sub) > 0:
                first = sub[0]
                if isinstance(first, nn.Conv2d):
                    return int(first.out_channels)
                conv = getattr(first, "conv", None)
                if isinstance(conv, nn.Conv2d):
                    return int(conv.out_channels)
        if isinstance(layer, nn.BatchNorm2d):
            return int(layer.num_features)
        return None

    def _inject_one(layers, layer_index: int) -> int:
        target = layers[layer_index]
        channels = _out_channels(target)
        if channels is None:
            raise ValueError(f"第 {layer_index} 层无法推断输出通道")
        layers[layer_index] = CoordAttWrapper(target, channels)
        return layer_index

    def inject_coordatt(
        model: nn.Module,
        layer_index: int | None = None,
        mode: str = "last",
    ) -> list[int]:
        """把 CoordAtt 挂到模型的输出上（原地修改）。

        Ultralytics 的 ``YOLO.model.model`` 是 ``nn.Sequential``，第 i 个元素通常
        是 C2f / Conv 等算子。这里用一个轻量包装器替换该层：

            layer(x)  →  coordatt(layer(x))

        ``mode="last"``（默认）：挂在单层上。``layer_index=None`` 时优先选
        SPPF / SPP / SPPCSPC（Backbone 末端，P5 特征入口）；模型里没有这类层时
        退回「通道数 ≥ 256 的最后一层」。

        ``mode="backbone"``：挂在 Backbone 里每个 C2f / C3 / C3k2（含 SPPF）
        的输出上，做「多点注入」消融；``layer_index`` 此时被忽略。
        返回被注入的层号列表。
        """
        base = getattr(model, "model", model)
        if isinstance(base, (list, tuple, nn.Sequential)):
            layers = base
        else:
            raise TypeError("注入失败：模型没有可迭代的 .model 层列表")

        if mode == "backbone":
            hits = []
            for i, layer in enumerate(layers):
                name = type(layer).__name__
                if name in {"C2f", "C3", "C3k2", "C2fCIB", "SPPF", "SPP", "SPPCSPC"}:
                    if _out_channels(layer) is None:
                        continue
                    hits.append(_inject_one(layers, i))
            if not hits:
                raise ValueError("backbone 模式没有找到可注入的 C2f/SPPF 层")
            return hits

        if layer_index is None:
            layer_index = len(layers) - 1
            for i, layer in enumerate(layers):
                if type(layer).__name__ in {"SPPF", "SPP", "SPPCSPC"}:
                    layer_index = i
                    break
            else:
                for i, layer in enumerate(layers):
                    out_ch = _out_channels(layer)
                    if out_ch is not None and out_ch >= 256:
                        layer_index = i

        return [_inject_one(layers, layer_index)]

    def param_count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

else:  # pragma: no cover - 无 torch 环境
    class CoordAtt:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()

    class CoordAttWrapper:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch()

    def inject_coordatt(*args: Any, **kwargs: Any) -> list[int]:  # type: ignore[misc]
        _require_torch()
        return []

    def param_count(*args: Any, **kwargs: Any) -> int:  # type: ignore[misc]
        _require_torch()
        return 0
