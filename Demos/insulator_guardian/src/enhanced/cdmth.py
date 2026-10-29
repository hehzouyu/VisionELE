"""CDMTH 元迁移检测头（构想 1.3.5）的原型网络骨架。

口径
----
- 每类一个**原型向量**（支持集特征的 L2 归一化均值）；
- 查询样本与原型做余弦相似度，softmax(·/T) 出后验；
- ``K ∈ {5, 10}`` 支持集微调，冻结主干、只更新原型；
- 触发条件：检测置信度 < 0.5 **且** 电气异常 > 0.7 才走元迁移分支（进阶双模态）。

纯 numpy 实现，可单测；真接 YOLO 时把特征换成 neck 输出即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["ProtoNetHead", "should_trigger"]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


@dataclass
class ProtoNetHead:
    """支持集建原型的元迁移分类头。"""

    num_classes: int
    temperature: float = 0.1
    prototypes: np.ndarray | None = field(default=None, init=False)
    support_counts: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.num_classes = int(self.num_classes)
        if self.num_classes <= 0:
            raise ValueError("num_classes 必须为正")
        if self.temperature <= 0:
            raise ValueError("temperature 必须 > 0")

    # ------------------------------------------------------------------ 拟合
    def fit(self, features: np.ndarray, labels: Sequence[int]) -> "ProtoNetHead":
        """用支持集重建每类原型；返回 self 便于链式调用。"""
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features 需要 (N, D) 且与 labels 等长")
        if len(x) == 0:
            raise ValueError("支持集为空")
        unit = _l2_normalize(x)
        protos = np.zeros((self.num_classes, x.shape[1]), dtype=float)
        self.support_counts.clear()
        for c in range(self.num_classes):
            picked = unit[y == c]
            self.support_counts[c] = int(len(picked))
            if len(picked):
                protos[c] = _l2_normalize(picked.mean(axis=0))
        self.prototypes = protos
        return self

    # ------------------------------------------------------------------ 推理
    def logits(self, features: np.ndarray) -> np.ndarray:
        if self.prototypes is None:
            raise RuntimeError("请先调用 fit() 建立原型")
        query = _l2_normalize(np.asarray(features, dtype=float))
        return query @ self.prototypes.T / float(self.temperature)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        z = self.logits(features)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.probabilities(features), axis=1)

    def evaluate(self, features: np.ndarray, labels: Sequence[int]) -> float:
        y = np.asarray(labels, dtype=int)
        if y.size == 0:
            return 0.0
        return float(np.mean(self.predict(features) == y))


def should_trigger(confidence: float, electrical_abnormal: float,
                   conf_threshold: float = 0.5, electrical_threshold: float = 0.7) -> bool:
    """元迁移分支触发条件：视觉低置信 **且** 电气侧明显异常。"""
    return float(confidence) < float(conf_threshold) and float(electrical_abnormal) > float(electrical_threshold)
