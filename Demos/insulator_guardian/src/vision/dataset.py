"""绝缘子数据集加载（YOLO 格式，纯 numpy/PIL，不依赖 ultralytics）。

目录约定（``data/`` 下）::

    data/
      images/train/*.jpg        # 或 data/raw/*.jpg（平铺）
      labels/train/*.txt        # 与图片同名的 YOLO txt
      splits/train.txt          # 可选：每行一个图片路径，覆盖目录扫描

YOLO txt 每行：``class cx cy w h``（全归一化到 [0,1]，cx/cy 为框中心）。
类别口径固定为 ``0=normal 1=flashover 2=broken``（与 ``schema.py`` 一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from ..integration.schema import CLASS_NAMES, Detection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image(path: str | Path) -> np.ndarray:
    """读成 RGB uint8 的 ``(H, W, 3)`` 数组。"""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def load_yolo_labels(label_path: str | Path, width: int, height: int) -> list[Detection]:
    """读 YOLO txt → ``list[Detection]``（像素坐标，置信度记 1.0）。"""
    detections: list[Detection] = []
    path = Path(label_path)
    if not path.exists():
        return detections
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} 需要 5 列（cls cx cy w h），收到 {len(parts)}")
        cls, cx, cy, bw, bh = (float(v) for v in parts)
        cls_id = int(cls)
        if cls_id not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_no} 非法类别 {cls_id}，只允许 {sorted(CLASS_NAMES)}")
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        detections.append(
            Detection(class_id=cls_id, confidence=1.0, bbox=(x1, y1, x2, y2))
        )
    return detections


def save_yolo_labels(detections: Sequence[Detection], width: int, height: int) -> str:
    """``list[Detection]`` → YOLO txt 文本（训练/评测脚本导出标注用）。"""
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


@dataclass(eq=False)
class InsulatorSample:
    image_path: Path
    label_path: Path | None = None
    width: int = 0
    height: int = 0
    detections: list[Detection] = field(default_factory=list)
    _image: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = load_image(self.image_path)
            self.height, self.width = self._image.shape[:2]
        return self._image

    @property
    def num_defects(self) -> int:
        return sum(1 for d in self.detections if d.is_defect)


class InsulatorDataset:
    """最小数据集接口：``len()``、``[i] -> (image, detections)``。"""

    def __init__(
        self,
        samples: Sequence[InsulatorSample],
        preload_labels: bool = True,
    ):
        self.samples = list(samples)
        if preload_labels:
            self._load_labels()

    def _load_labels(self) -> None:
        for sample in self.samples:
            if sample.width and sample.height:
                continue
            with Image.open(sample.image_path) as img:
                w, h = img.size
            sample.width, sample.height = int(w), int(h)
            if sample.label_path is not None:
                sample.detections = load_yolo_labels(sample.label_path, w, h)

    # ---------------------------------------------------------------- 构造器
    @classmethod
    def from_yolo(
        cls,
        root: str | Path = "data",
        split: str | None = None,
        label_dir: str | Path | None = None,
        load_images: bool = False,
    ) -> "InsulatorDataset":
        root = Path(root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        split = split or "train"

        images = _collect_images(root, split)
        labels = _resolve_label_dir(root, split, label_dir)

        samples = []
        for image_path in images:
            label_path = None
            if labels is not None:
                candidate = labels / (image_path.stem + ".txt")
                label_path = candidate if candidate.exists() else None
            sample = InsulatorSample(image_path=image_path, label_path=label_path)
            if load_images:
                _ = sample.image
            samples.append(sample)
        return cls(samples, preload_labels=True)

    # ---------------------------------------------------------------- 访问
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[Detection]]:
        return self.sample(index)

    def sample(self, index: int) -> tuple[np.ndarray, list[Detection]]:
        s = self.samples[index]
        return s.image, s.detections

    def __iter__(self) -> Iterable[tuple[np.ndarray, list[Detection]]]:
        for i in range(len(self)):
            yield self.sample(i)

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES.values()}
        for s in self.samples:
            for det in s.detections:
                counts[det.class_name] += 1
        return counts

    def defect_count(self) -> int:
        return sum(s.num_defects for s in self.samples)

    def write_split_file(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(str(s.image_path) for s in self.samples), encoding="utf-8")
        return path


def _collect_images(root: Path, split: str) -> list[Path]:
    split_file = root / "splits" / f"{split}.txt"
    if split_file.exists():
        images = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            images.append(p if p.is_absolute() else PROJECT_ROOT / p)
        return [p for p in images if p.exists()]
    for candidate in (root / "images" / split, root / split, root / "raw", root):
        if candidate.is_dir():
            files = sorted(p for p in candidate.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if files:
                return files
    return []


def _resolve_label_dir(root: Path, split: str, label_dir: str | Path | None) -> Path | None:
    candidates = []
    if label_dir is not None:
        p = Path(label_dir)
        candidates.append(p if p.is_absolute() else PROJECT_ROOT / p)
    candidates += [root / "labels" / split, root / "labels"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def load_split(data_cfg: dict | None, split: str = "train") -> InsulatorDataset:
    """按 vision.yaml 的 ``data:`` 段加载一个划分。"""
    cfg = data_cfg or {}
    root = cfg.get("root", "data")
    label_dir = cfg.get("label_dir")
    return InsulatorDataset.from_yolo(root=root, split=split, label_dir=label_dir)
