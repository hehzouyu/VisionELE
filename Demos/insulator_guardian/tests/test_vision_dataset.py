"""YOLO 数据集读取单测。

口径：在 tmp_path 造一个极小的 YOLO 目录（1 张图 + 2 行标注），验
归一化坐标 → 像素框的换算、Dataset 取样本的返回结构、类别统计，以及
splits/train.txt 优先级。所有文件都写在 tmp_path 内。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.integration.schema import Detection
from src.vision.dataset import (
    InsulatorDataset,
    load_split,
    load_yolo_labels,
    save_yolo_labels,
)

LABELS = "1 0.5 0.5 0.2 0.1\n2 0.25 0.75 0.1 0.2\n"


def _make_yolo_root(tmp_path: Path) -> Path:
    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    image = np.full((240, 320, 3), 180, dtype=np.uint8)
    Image.fromarray(image).save(root / "images" / "train" / "a.jpg")
    (root / "labels" / "train" / "a.txt").write_text(LABELS, encoding="utf-8")
    return root


def test_load_yolo_labels_normalizes_to_pixels(tmp_path):
    root = _make_yolo_root(tmp_path)
    detections = load_yolo_labels(root / "labels" / "train" / "a.txt", 320, 240)
    assert len(detections) == 2
    assert detections[0].bbox == pytest.approx((128.0, 108.0, 192.0, 132.0))
    assert detections[0].class_id == 1 and detections[0].confidence == 1.0
    assert detections[1].bbox == pytest.approx((64.0, 156.0, 96.0, 204.0))
    assert detections[1].class_name == "broken"


def test_missing_label_file_returns_empty(tmp_path):
    assert load_yolo_labels(tmp_path / "nope.txt", 320, 240) == []


def test_invalid_label_file_raises(tmp_path):
    bad_columns = tmp_path / "bad_cols.txt"
    bad_columns.write_text("0 0.5 0.5 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yolo_labels(bad_columns, 320, 240)

    bad_class = tmp_path / "bad_class.txt"
    bad_class.write_text("9 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yolo_labels(bad_class, 320, 240)


def test_save_labels_roundtrip(tmp_path):
    detections = [Detection(1, 0.9, (80.0, 45.0, 120.0, 55.0))]
    text = save_yolo_labels(detections, 320, 240)
    assert len(text.splitlines()) == 1
    path = tmp_path / "roundtrip.txt"
    path.write_text(text, encoding="utf-8")
    back = load_yolo_labels(path, 320, 240)
    assert len(back) == 1
    assert back[0].bbox == pytest.approx(detections[0].bbox, rel=1e-4)


def test_dataset_getitem_and_class_counts(tmp_path):
    root = _make_yolo_root(tmp_path)
    dataset = InsulatorDataset.from_yolo(root=root, split="train")
    assert len(dataset) == 1

    image, labels = dataset[0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (240, 320, 3) and image.dtype == np.uint8
    assert len(labels) == 2

    assert dataset.class_counts() == {"normal": 0, "flashover": 1, "broken": 1}
    assert dataset.defect_count() == 2  # 缺陷框总数（两行都是缺陷类）


def test_split_file_takes_priority(tmp_path):
    root = _make_yolo_root(tmp_path)
    (root / "splits").mkdir()
    image_path = root / "images" / "train" / "a.jpg"
    (root / "splits" / "train.txt").write_text(f"{image_path}\n", encoding="utf-8")
    dataset = InsulatorDataset.from_yolo(root=root, split="train")
    assert len(dataset) == 1
    assert dataset.samples[0].image_path == image_path


def test_load_split_uses_data_cfg_root(tmp_path):
    root = _make_yolo_root(tmp_path)
    dataset = load_split({"root": str(root)}, split="train")
    assert isinstance(dataset, InsulatorDataset)
    assert len(dataset) == 1
