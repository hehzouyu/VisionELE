"""进阶语义增强（构想 1.3 节的 Demos 落地骨架）。

模块划分
--------
- ``freq``   CWT / VMD 频域增强分支 + SE 通道注意力融合（纯 numpy，可单测）；
- ``graph``  MSA-GCN 半监督语义图学习（纯 numpy，可单测）；
- ``cdmth``  CDMTH 元迁移检测头：原型网络 + 触发条件（纯 numpy，可单测）；
- ``cdcgan`` 条件 DCGAN 小样本生成（依赖 torch，缺库时导入即降级）。

这些模块是「第二阶段」的骨架：数据结构、算子口径、损失形式都按构想 1.3
实现，但还没接进 YOLO 训练主干（M4 才接）。单测只验证数学口径与形状，
不宣称任何精度指标。
"""
from __future__ import annotations

__all__ = ["freq", "graph", "cdmth", "cdcgan"]
