"""CA-YOLO 检测器封装（推理侧）。

- 依赖 ``ultralytics``（训练/推理），本模块按需导入，没装时报清晰错误；
- 支持从 ``runs/.../best.pt`` 直接加载（CA 权重已烤进 .pt），
  也支持「从官方权重出发 + 现场注入 CA」（训练脚本用）；
- 输出统一成 ``schema.Detection``，交给集成层填世界坐标。

推理口径：``conf=0.25, iou=0.45``（configs/vision.yaml），逐帧 < 200 ms（M4 验证）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..integration.schema import CLASS_NAMES, Detection

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DetectorConfig:
    weights: str | None = None
    family: str = "yolov8"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 100
    device: str = "cpu"
    ca: bool = False                  # 只对「新训练」有意义：加载后注入 CA
    ca_layer_index: int | None = None
    half: bool = False

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "DetectorConfig":
        cfg = cfg or {}
        model = cfg.get("model", {}) or {}
        infer = cfg.get("infer", {}) or {}
        weights = model.get("weights") or None
        return cls(
            weights=None if weights is None else str(_resolve(weights)),
            family=str(model.get("family", "yolov8")),
            imgsz=int(model.get("input_size", 640)),
            conf=float(infer.get("conf", 0.25)),
            iou=float(infer.get("iou", 0.45)),
            max_det=int(infer.get("max_det", 100)),
            device=str((cfg.get("train", {}) or {}).get("device", "cpu")),
            ca=bool(model.get("use_ca", True)),
            ca_layer_index=model.get("ca_layer_index"),
            half=bool(infer.get("half", False)),
        )


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


class Detector:
    """``Detector.detect(image) -> list[Detection]``；也可直接 ``detector(image)``。"""

    def __init__(self, cfg: DetectorConfig | None = None, model: Any | None = None):
        self.cfg = cfg or DetectorConfig()
        self.model = model
        self._names: dict[int, str] = dict(CLASS_NAMES)
        self._class_map: dict[int, int] | None = None
        self._ca_layers: list[int] = []
        if self.model is None:
            self.load()

    # ---------------------------------------------------------------- 加载
    def load(self) -> "Detector":
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "检测器需要 ultralytics：pip install ultralytics（或见 requirements.txt）"
            ) from exc
        if not self.cfg.weights:
            raise RuntimeError(
                "weights 为空：先在 configs/vision.yaml 填 runs/.../best.pt，"
                "或训练得到权重后指向它"
            )
        self.model = YOLO(str(self.cfg.weights))
        if self.cfg.ca:
            from .attention import inject_coordatt

            self._ca_layers = inject_coordatt(self.model.model, self.cfg.ca_layer_index)
        names = getattr(self.model, "names", None) or {}
        self._names = {int(k): str(v) for k, v in dict(names).items()} or dict(CLASS_NAMES)
        return self

    # ---------------------------------------------------------------- 推理
    def detect(self, image: np.ndarray | str | Path) -> list[Detection]:
        if self.model is None:
            raise RuntimeError("模型未加载")
        results = self.model.predict(
            source=self._as_array(image),
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )
        return detections_from_result(results[0], class_names=self._names)

    __call__ = detect

    def warmup(self, iterations: int = 1) -> None:
        """跑几帧空推理，让 CUDA/线程池完成首次分配（延迟测量前调用）。"""
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)

    @staticmethod
    def _as_array(image: np.ndarray | str | Path) -> Any:
        if isinstance(image, (str, Path)):
            return str(image)
        return np.asarray(image)


def detections_from_result(
    result: Any,
    class_names: dict[int, str] | None = None,
) -> list[Detection]:
    """把 ultralytics 的 ``Results`` 转成 ``schema.Detection``。

    类别 id 按「名字 ↔ schema」对齐；模型不认识的类别直接丢弃（避免越界）。
    """
    names = class_names or dict(CLASS_NAMES)
    name_to_id = {v: k for k, v in CLASS_NAMES.items()}
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

    out: list[Detection] = []
    for box, score, raw_id in zip(xyxy, conf, cls):
        model_name = names.get(int(raw_id), str(int(raw_id)))
        schema_id = name_to_id.get(str(model_name).lower())
        if schema_id is None:
            continue
        out.append(
            Detection(
                class_id=schema_id,
                confidence=float(score),
                bbox=tuple(float(v) for v in box),
            )
        )
    return out


def load_detector(vision_cfg: dict | None = None) -> Detector:
    """按 vision.yaml 构造检测器（weights 为 null 时给出清晰报错）。"""
    return Detector(DetectorConfig.from_dict(vision_cfg))
