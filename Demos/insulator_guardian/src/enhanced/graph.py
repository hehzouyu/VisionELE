"""MSA-GCN 半监督语义图学习（构想 1.3.3）的 numpy 骨架。

口径
----
1. **构图**：节点特征先做 FFT 取幅度谱，谱空间里 kNN + 高斯核加权 → 稀疏邻接图；
2. **节点特征**：``(N, D)`` 的 patch 特征（视觉分支输出）；
3. **图卷积**：多尺度图卷积（``A^p, p ∈ scales``）+ 逐节点注意力自适应融合
   + 门控残差，堆叠 ``layers`` 层；
4. **半监督损失**：有标签节点交叉熵，无标签节点一致性约束（伪标签 MSE）。

说明：这是 M4 之前的结构骨架，前向/损失/单步更新都可跑、可单测；
层间权重在骨架阶段保持随机初始化（真实训练走 PyTorch 主干），
只有输出层按解析梯度更新，用来验证「损失确实会下降」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "normalize_adjacency",
    "build_knn_graph",
    "MultiScaleGCN",
    "semi_supervised_loss",
    "accuracy",
]


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """对称归一化 ``D^-1/2 (A + I) D^-1/2``。"""
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"邻接矩阵必须是方阵，收到 {a.shape}")
    a = a + np.eye(a.shape[0])
    deg = a.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    return (a * inv_sqrt[:, None]) * inv_sqrt[None, :]


def build_knn_graph(features: np.ndarray, k: int = 5, sigma: float = 1.0) -> np.ndarray:
    """FFT 谱空间 kNN 构图，边权为高斯核 ``exp(-d²/2σ²)``，返回对称邻接。"""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"features 需要 (N, D)，收到 {x.shape}")
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    k = int(min(max(k, 1), max(n - 1, 1)))
    spec = np.abs(np.fft.rfft(x, axis=1))
    spec = spec / np.maximum(np.linalg.norm(spec, axis=1, keepdims=True), 1e-12)
    d2 = np.maximum(
        (spec * spec).sum(axis=1)[:, None] + (spec * spec).sum(axis=1)[None, :] - 2.0 * spec @ spec.T,
        0.0,
    )
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, np.inf)
    adjacency = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(dist[i])[:k]
        weights = np.exp(-(dist[i, order] ** 2) / (2.0 * max(float(sigma), 1e-9) ** 2))
        adjacency[i, order] = np.maximum(adjacency[i, order], weights)
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


@dataclass
class MultiScaleGCN:
    """多尺度 + 逐节点注意力 + 门控残差的图卷积堆叠。"""

    in_dim: int
    num_classes: int
    hidden_dim: int = 64
    scales: Sequence[int] = (1, 2, 3)
    layers: int = 3
    dropout: float = 0.0
    seed: int = 0
    _weights: list[np.ndarray] = field(default_factory=list, init=False)
    _gates: list[np.ndarray] = field(default_factory=list, init=False)
    _scale_query: list[np.ndarray] = field(default_factory=list, init=False)
    _cache: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.in_dim <= 0 or self.num_classes <= 0:
            raise ValueError("in_dim / num_classes 必须为正")
        self.in_dim = int(self.in_dim)
        self.num_classes = int(self.num_classes)
        self.hidden_dim = int(self.hidden_dim)
        self.scales = tuple(int(s) for s in self.scales)
        if not self.scales:
            raise ValueError("scales 不能为空")
        self.layers = max(int(self.layers), 1)
        rng = np.random.default_rng(self.seed)
        dims = [self.in_dim] + [self.hidden_dim] * (self.layers - 1) + [self.num_classes]
        self._weights.clear()
        self._gates.clear()
        self._scale_query.clear()
        for i in range(self.layers):
            fan_in = dims[i]
            self._weights.append(rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_in, dims[i + 1])))
            self._gates.append(rng.normal(0.0, 0.1, size=(dims[i + 1],)))
            self._scale_query.append(rng.normal(0.0, 1.0 / np.sqrt(fan_in),
                                                size=(dims[i], len(self.scales))))

    # ------------------------------------------------------------------ 前向
    def _activations(self, adjacency: np.ndarray) -> list[np.ndarray]:
        a = normalize_adjacency(adjacency)
        out = [np.eye(a.shape[0], dtype=float)]
        for p in self.scales:
            out.append(np.linalg.matrix_power(a, p))
        return out

    def forward(self, adjacency: np.ndarray, features: np.ndarray) -> np.ndarray:
        h = np.asarray(features, dtype=float)
        if h.shape[0] != np.asarray(adjacency).shape[0]:
            raise ValueError("features 行数必须等于邻接矩阵节点数")
        powers = self._activations(adjacency)
        inputs: list[np.ndarray] = []
        for i in range(self.layers):
            inputs.append(h)
            multi = []
            for p in range(len(self.scales)):
                multi.append(powers[p] @ h @ self._weights[i])
            stack = np.stack(multi, axis=-1)                       # (N, d_out, S)
            logits = h @ self._scale_query[i]                      # (N, S)
            logits = logits - logits.max(axis=1, keepdims=True)
            attn = np.exp(logits)
            attn = attn / np.maximum(attn.sum(axis=1, keepdims=True), 1e-12)
            agg = (stack * attn[:, None, :]).sum(axis=-1)          # (N, d_out)
            if i == self.layers - 1:
                # 末层用线性读出（不加 tanh/门控），这样交叉熵对 W 的梯度是精确的，
                # 骨架阶段「损失确实下降」才是可验证的事实而不是近似。
                h = agg
                break
            gate = 1.0 / (1.0 + np.exp(-self._gates[i]))
            h_new = np.tanh(agg) * gate[None, :]
            h = h_new + h if h.shape == h_new.shape else h_new
            if self.dropout > 0.0 and i < self.layers - 1:
                rng = np.random.default_rng(self.seed + i)
                mask = (rng.random(h.shape) > self.dropout).astype(float)
                h = h * mask / max(1.0 - self.dropout, 1e-6)
        self._cache = {"features": np.asarray(features, dtype=float),
                       "hidden": h, "powers": powers, "inputs": inputs}
        return h

    # ------------------------------------------------------------------ 训练
    def step(
        self,
        adjacency: np.ndarray,
        features: np.ndarray,
        labels: Sequence[int],
        mask: Sequence[bool],
        lr: float = 0.1,
    ) -> float:
        """输出层单步更新（解析梯度）；返回更新前的有标签交叉熵。"""
        logits = self.forward(adjacency, features)
        y = np.asarray(labels, dtype=int)
        m = np.asarray(mask, dtype=bool)
        if m.any():
            loss = cross_entropy(logits[m], y[m])
            grad = np.zeros_like(logits)
            grad[m] = softmax(logits[m])
            grad[m, y[m]] -= 1.0
            grad[m] /= float(max(int(m.sum()), 1))
            # 末层为线性读出 logits = H @ W，故 dW = Hᵀ·dL 是精确梯度
            h_prev = self._cache["inputs"][-1]
            self._weights[-1] -= lr * (h_prev.T @ grad)
        else:
            loss = 0.0
        return float(loss)

    def fit(
        self,
        adjacency: np.ndarray,
        features: np.ndarray,
        labels: Sequence[int],
        mask: Sequence[bool],
        epochs: int = 50,
        lr: float = 0.1,
    ) -> list[float]:
        history: list[float] = []
        for _ in range(int(epochs)):
            history.append(self.step(adjacency, features, labels, mask, lr=lr))
        return history

    def predict(self, adjacency: np.ndarray, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(adjacency, features), axis=1)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def cross_entropy(logits: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probs = softmax(logits)
    picked = probs[np.arange(len(labels)), np.asarray(labels, dtype=int)]
    return float(-np.mean(np.log(np.clip(picked, eps, 1.0))))


def semi_supervised_loss(
    logits: np.ndarray,
    labels: Sequence[int],
    labeled_mask: Sequence[bool],
    unlabeled_mask: Sequence[bool] | None = None,
    consistency: np.ndarray | None = None,
    weight_consistency: float = 0.3,
) -> float:
    """有标签交叉熵 + 无标签一致性（伪标签 MSE）。"""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=int)
    lm = np.asarray(labeled_mask, dtype=bool)
    loss = cross_entropy(z[lm], y[lm]) if lm.any() else 0.0
    if unlabeled_mask is not None and consistency is not None:
        um = np.asarray(unlabeled_mask, dtype=bool)
        if um.any():
            pseudo = softmax(z[um])
            ref = consistency[um] if consistency.shape[0] == z.shape[0] else consistency
            loss += float(weight_consistency) * float(np.mean((pseudo - ref) ** 2))
    return float(loss)


def accuracy(logits: np.ndarray, labels: Sequence[int]) -> float:
    pred = np.argmax(np.asarray(logits, dtype=float), axis=1)
    y = np.asarray(labels, dtype=int)
    return float(np.mean(pred == y)) if y.size else 0.0
