"""cDCGAN 小样本生成（构想 1.3.4）的 PyTorch 骨架。

- 5 层转置卷积生成器 + 条件输入（类别标签）+ 谱归一化判别器 + hinge 损失；
- 生成样本的筛选口径：FID + 特征中心距（``screen_generated``）。

本模块**延迟导入 torch**：环境没装 torch 时导入本文件不会报错，
调用训练/构建函数才抛 ``RuntimeError`` 并给出安装提示——这样决策线、
测试套件都能在没有 torch 的机器上跑通。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

try:  # 视觉线依赖，缺库时整模块降级
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 无 torch 环境
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

__all__ = ["CdcganConfig", "build_generator", "build_discriminator",
           "hinge_losses", "screen_generated", "TORCH_AVAILABLE"]


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("cDCGAN 需要 torch：pip install torch torchvision")


@dataclass
class CdcganConfig:
    latent_dim: int = 100
    num_classes: int = 3
    image_size: int = 64
    base_channels: int = 256
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    hinge_margin: float = 1.0


if TORCH_AVAILABLE:
    class ConditionalGenerator(nn.Module):
        """5 层转置卷积生成器：z ⊕ 类别嵌入 → (nc, S, S)。"""

        def __init__(self, cfg: CdcganConfig):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.num_classes, cfg.latent_dim)
            self.fc = nn.Linear(cfg.latent_dim * 2, cfg.base_channels * 4 * 4)
            c = cfg.base_channels
            self.net = nn.Sequential(
                nn.BatchNorm2d(c), nn.ReLU(True),
                nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.BatchNorm2d(c // 2), nn.ReLU(True),
                nn.ConvTranspose2d(c // 2, c // 4, 4, 2, 1), nn.BatchNorm2d(c // 4), nn.ReLU(True),
                nn.ConvTranspose2d(c // 4, c // 8, 4, 2, 1), nn.BatchNorm2d(c // 8), nn.ReLU(True),
                nn.ConvTranspose2d(c // 8, 3, 4, 2, 1), nn.Tanh(),
            )

        def forward(self, z: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
            cond = self.embed(labels)
            h = self.fc(torch.cat([z, cond], dim=1)).view(-1, self.cfg.base_channels, 4, 4)
            return self.net(h)

    class ConditionalDiscriminator(nn.Module):
        """谱归一化判别器：图像 + 类别嵌入 → 真伪分数。"""

        def __init__(self, cfg: CdcganConfig):
            super().__init__()
            self.embed = nn.Embedding(cfg.num_classes, 16 * 16)
            c = cfg.base_channels // 8 or 32
            self.net = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(3 + 1, c, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(c, c * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                nn.utils.parametrizations.spectral_norm(nn.Conv2d(c * 2, c * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                nn.Flatten(), nn.Linear(c * 4 * (cfg.image_size // 8) ** 2, 1),
            )

        def forward(self, images: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
            cond = self.embed(labels).view(-1, 1, 16, 16)
            cond = torch.nn.functional.interpolate(cond, size=images.shape[-2:], mode="nearest")
            return self.net(torch.cat([images, cond], dim=1)).squeeze(1)

    def build_generator(cfg: CdcganConfig) -> "ConditionalGenerator":
        _require_torch()
        return ConditionalGenerator(cfg)

    def build_discriminator(cfg: CdcganConfig) -> "ConditionalDiscriminator":
        _require_torch()
        return ConditionalDiscriminator(cfg)

    def hinge_losses(real_scores: "torch.Tensor", fake_scores: "torch.Tensor",
                     margin: float = 1.0) -> tuple["torch.Tensor", "torch.Tensor"]:
        """hinge GAN：判别器 ``relu(m - real) + relu(m + fake)``；生成器 ``-fake.mean()``。"""
        d_loss = torch.relu(margin - real_scores).mean() + torch.relu(margin + fake_scores).mean()
        g_loss = -fake_scores.mean()
        return d_loss, g_loss
else:  # pragma: no cover - 无 torch 环境
    def build_generator(cfg: CdcganConfig) -> Any:
        _require_torch()

    def build_discriminator(cfg: CdcganConfig) -> Any:
        _require_torch()

    def hinge_losses(*args: Any, **kwargs: Any) -> Any:
        _require_torch()


def screen_generated(
    generated_features: Sequence[Sequence[float]],
    real_features: Sequence[Sequence[float]],
    keep_ratio: float = 0.5,
) -> list[int]:
    """生成样本筛选：按到真实特征中心的距离保留最近的 ``keep_ratio``。

    这是构想里「FID + 特征中心距」中**中心距**那一半的可执行版本；
    FID 需要 Inception 特征，留到接入真实模型后再算（README 里写明口径）。
    """
    gen = [list(map(float, row)) for row in generated_features]
    real = [list(map(float, row)) for row in real_features]
    if not gen or not real:
        return []
    dim = min(len(gen[0]), len(real[0]))
    center = [0.0] * dim
    for row in real:
        for i in range(dim):
            center[i] += row[i]
    center = [v / len(real) for v in center]
    scored = []
    for idx, row in enumerate(gen):
        d = math.sqrt(sum((row[i] - center[i]) ** 2 for i in range(dim)))
        scored.append((d, idx))
    scored.sort()
    keep = max(1, int(round(len(scored) * float(keep_ratio))))
    return sorted(idx for _, idx in scored[:keep])
