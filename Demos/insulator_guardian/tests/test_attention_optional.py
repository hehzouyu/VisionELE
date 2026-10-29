"""CoordAttention 可选依赖单测。

口径：CoordAtt 依赖 torch，环境没有 torch 时必须整体跳过而不是报错；
有 torch 时验「注意力层不改变特征图形状」「inject_coordatt 能原地替换
指定层且模型仍可前向」。
"""
from __future__ import annotations

import pytest

from src.vision import attention

pytestmark = pytest.mark.skipif(
    not attention.TORCH_AVAILABLE,
    reason="环境未安装 torch，跳过 CoordAttention 测试",
)


def test_coordatt_forward_keeps_shape():
    torch = pytest.importorskip("torch")
    layer = attention.CoordAtt(32)
    x = torch.randn(2, 32, 16, 16)
    assert layer(x).shape == x.shape


def test_coordatt_wrapper_is_callable_and_keeps_shape():
    torch = pytest.importorskip("torch")
    wrapper = attention.CoordAttWrapper(torch.nn.Identity(), 16)
    x = torch.randn(1, 16, 8, 8)
    assert wrapper(x).shape == x.shape


def test_inject_coordatt_into_sequential():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, 3, padding=1),
    )
    injected = attention.inject_coordatt(model)
    assert isinstance(injected, list) and injected
    x = torch.randn(1, 3, 8, 8)
    assert model(x).shape == (1, 16, 8, 8)

    model2 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.Conv2d(8, 16, 3, padding=1),
    )
    assert attention.inject_coordatt(model2, layer_index=1) == [1]
    assert model2(torch.randn(1, 3, 8, 8)).shape == (1, 16, 8, 8)


def test_param_count_positive():
    assert attention.param_count(attention.CoordAtt(32)) > 0
