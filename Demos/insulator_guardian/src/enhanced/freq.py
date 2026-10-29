"""频域增强分支（构想 1.3.2）：CWT（Morlet）/ VMD 频带分解 + SE 融合。

口径
----
``F_enhance = F_spatial + γ1 · SE(F_cwt) + γ2 · SE(F_vmd)``

- **CWT 分支**：Morlet 小波在尺度 {1,2,4,8} 上做可分离卷积，取复响应幅度
  并在尺度维取均值，得到与输入同形状的特征响应（雾霾/逆光下空间域退化时，
  频域能量更稳）；
- **VMD 分支**：真正的变分模态分解是迭代优化，这里用**频域带通分解**做近似
  （``K`` 个径向频带重建），口径与 FastVMD 的「有限模态滤波组」思想一致，
  但实现是 FFT 掩码，速度快、可单测；
- **SE 通道注意力**：对频域分支做通道重标定后再叠加，避免直接相加把
  空间域的判别性盖掉。

全部实现只依赖 numpy，输入是单通道或 ``(C, H, W)`` 的特征图，输出形状不变。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "morlet_wavelet",
    "cwt_morlet",
    "cwt_response",
    "vmd_bands",
    "vmd_response",
    "se_attention",
    "fuse_features",
]


def morlet_wavelet(t: np.ndarray, scale: float, center_freq: float = 0.8) -> np.ndarray:
    """离散 Morlet 小波（复数）：``exp(i·ω·t)·exp(-t²/2)``，按尺度归一化。"""
    t = np.asarray(t, dtype=float)
    if scale <= 0:
        raise ValueError(f"scale 必须 > 0，收到 {scale}")
    w = center_freq / float(scale)
    gauss = np.exp(-0.5 * (t / float(scale)) ** 2)
    wavelet = (1.0 / np.sqrt(float(scale))) * np.exp(1j * w * t) * gauss
    return wavelet.astype(np.complex128)


def cwt_morlet(signal: Sequence[float], scales: Iterable[float] = (1, 2, 4, 8),
               center_freq: float = 0.8) -> np.ndarray:
    """1D 连续小波变换，返回 ``(len(scales), len(signal))`` 复响应。"""
    x = np.asarray(signal, dtype=float).ravel()
    out = np.empty((0, x.size), dtype=np.complex128)
    for scale in scales:
        half = int(math.ceil(3.0 * float(scale)))
        t = np.arange(-half, half + 1, dtype=float)
        kernel = morlet_wavelet(t, float(scale), center_freq)
        full = np.convolve(x, kernel, mode="full")
        start = (len(full) - x.size) // 2
        conv = full[start:start + x.size]
        out = np.vstack([out, conv[np.newaxis, :]])
    return out


def cwt_response(x: np.ndarray, scales: Iterable[float] = (1, 2, 4, 8)) -> np.ndarray:
    """2D/3D 特征图的 CWT 响应（可分离：先横后纵），返回同形状的幅度特征。

    ``x`` 支持 ``(H, W)`` 或 ``(C, H, W)``；尺度维取均值后与空间维同形。
    """
    arr = np.asarray(x, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"cwt_response 需要 (H,W) 或 (C,H,W)，收到 {arr.shape}")
    scales = list(scales)
    out = np.zeros_like(arr)
    for c in range(arr.shape[0]):
        acc = np.zeros(arr.shape[1:], dtype=float)
        for scale in scales:
            rows = cwt_morlet(arr[c].mean(axis=1), scales=[scale])
            cols = cwt_morlet(arr[c].mean(axis=0), scales=[scale])
            # 行/列方向的 1D 响应按广播近似成 2D（可分离卷积的秩一近似）
            acc += np.abs(rows[0])[:, None] * np.abs(cols[0])[None, :]
        out[c] = acc / max(len(scales), 1)
    return out[0] if single else out


def vmd_bands(signal: Sequence[float], modes: int = 3) -> np.ndarray:
    """1D 频带分解近似 VMD：按径向频带切 FFT 并重建，返回 ``(modes, len)``。"""
    x = np.asarray(signal, dtype=float).ravel()
    if modes < 1:
        raise ValueError(f"modes 必须 >= 1，收到 {modes}")
    spectrum = np.fft.rfft(x)
    n = spectrum.size
    bounds = np.linspace(0, n, modes + 1).astype(int)
    out = np.zeros((modes, x.size), dtype=float)
    for k in range(modes):
        lo, hi = bounds[k], max(bounds[k + 1], bounds[k] + 1 if k < modes - 1 else bounds[k + 1])
        hi = min(hi, n)
        part = np.zeros_like(spectrum)
        if hi > lo:
            part[lo:hi] = spectrum[lo:hi]
        out[k] = np.fft.irfft(part, n=x.size)
    return out


def vmd_response(x: np.ndarray, modes: int = 3) -> np.ndarray:
    """2D/3D 特征图的 VMD 近似响应：按行、按列做频带分解后外积叠加。"""
    arr = np.asarray(x, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"vmd_response 需要 (H,W) 或 (C,H,W)，收到 {arr.shape}")
    out = np.zeros_like(arr)
    for c in range(arr.shape[0]):
        acc = np.zeros(arr.shape[1:], dtype=float)
        for band in vmd_bands(arr[c].mean(axis=1), modes):
            acc += np.abs(np.fft.fft(band)).mean() * np.outer(np.abs(band), np.ones(arr.shape[2]))
        for band in vmd_bands(arr[c].mean(axis=0), modes):
            acc += np.abs(np.fft.fft(band)).mean() * np.outer(np.ones(arr.shape[1]), np.abs(band))
        out[c] = acc / (2.0 * max(modes, 1))
    return out[0] if single else out


def se_attention(x: np.ndarray, reduction: int = 16, eps: float = 1e-6) -> np.ndarray:
    """SE 通道注意力：全局平均池化 → 两层全连接（sigmoid 门控）→ 重标定。

    ``x`` 为 ``(C, H, W)``；单通道输入原样返回。权重用固定随机种子初始化，
    不引入可训练状态（训练时把本函数换成 torch 版 SqueezeExcite 即可）。
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2:
        return arr.copy()
    if arr.ndim != 3:
        raise ValueError(f"se_attention 需要 (C,H,W)，收到 {arr.shape}")
    c = arr.shape[0]
    if c <= 1:
        return arr.copy()
    hidden = max(1, c // max(int(reduction), 1))
    rng = np.random.default_rng(0)
    w1 = rng.normal(0.0, 1.0 / math.sqrt(c), size=(c, hidden))
    w2 = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, c))
    pooled = arr.mean(axis=(1, 2))
    gate = 1.0 / (1.0 + np.exp(-(pooled @ w1) @ w2))
    return arr * gate[:, None, None]


def fuse_features(
    spatial: np.ndarray,
    gamma_cwt: float = 0.3,
    gamma_vmd: float = 0.2,
    scales: Iterable[float] = (1, 2, 4, 8),
    modes: int = 3,
    se_reduction: int = 16,
    use_vmd: bool = True,
) -> np.ndarray:
    """按 ``F + γ1·SE(CWT) + γ2·SE(VMD)`` 融合，形状与 ``spatial`` 一致。"""
    base = np.asarray(spatial, dtype=float)
    if base.ndim not in (2, 3):
        raise ValueError(f"fuse_features 需要 (H,W) 或 (C,H,W)，收到 {base.shape}")
    out = base.copy()
    if gamma_cwt:
        out = out + float(gamma_cwt) * se_attention(cwt_response(base, scales), se_reduction)
    if use_vmd and gamma_vmd:
        out = out + float(gamma_vmd) * se_attention(vmd_response(base, modes), se_reduction)
    return out
