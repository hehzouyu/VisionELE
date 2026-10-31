"""CA-YOLO 训练 / 导出 / 评测库函数（后端：ultralytics）。

约定
----
- 训练超参一律来自 ``configs/vision.yaml`` 的 ``model:`` / ``train:`` / ``augment:``；
- 数据集用 ``dataset.build_data_yaml`` 生成的 data.yaml（Ultralytics 原生格式）；
- CA 只在「从官方权重新建训练」时注入；训练产物 ``best.pt`` 已含 CA，
  推理侧 ``vision.detector.Detector`` 直接加载即可。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..integration.schema import CLASS_NAMES
from ..utils.config import resolve_path
from ..utils.io import ensure_dir, save_json, time_tag
from ..utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT.parent / "results"   # Demos/results

# 训练期 CA 注入的「请求上下文」：由 train() 填写，_ca_trainer() 生成的 trainer
# 在 get_model() 里读取。用模块级字典是因为 Ultralytics 的 trainer 只接受
# overrides 字典，塞自定义键会被 check_dict_alignment 警告。
_CA_SPEC: dict[str, Any] = {}
# RFA（循环特征聚合）注入请求：与 CA 互不冲突，可同时打开。
_RFA_SPEC: dict[str, Any] = {}


def attach_rfa(model, log=None) -> dict:
    """把 ``RFAFusion`` 插到检测头之前（``model.model[-1]`` 是 Detect）。

    做法：在 nn.Sequential 末尾插一层自定义模块，把 Detect 的输入索引指到它，
    这样 ``_forward_once`` 正常执行，Detect 收到的仍然是「P3/P4/P5 列表」。
    必须是模块实例而不是 forward hook——hook 里的闭包无法被 torch.save 序列化。
    """
    from ..vision.rfa import RFAFusion

    layers = list(model.model)
    detect = layers[-1]
    ch = getattr(detect, "ch", None)
    if not ch:
        # Ultralytics 的 Detect 不保存输入通道，从首层卷积反推（cv2[i] = Conv(...)）
        ch = []
        for i in range(int(getattr(detect, "nl", 3))):
            m = detect.cv2[i][0]
            conv = getattr(m, "conv", None)
            if conv is None:
                raise RuntimeError("RFA 注入失败：无法推断检测头输入通道")
            ch.append(int(conv.in_channels))
    if not ch:
        raise RuntimeError("RFA 注入失败：检测头通道为空")
    idx = len(layers) - 1                     # RFA 将来占据的位置
    rfa = RFAFusion([int(c) for c in ch])
    rfa.i = idx
    # YOLOv8 的 Detect 用 f=[16,19,22] 从 y 里取三层特征，而不是吃上一层的单张张量；
    # 所以 RFA 必须继承同一组索引，才能收到 P3/P4/P5 列表。
    src_idx = list(detect.f) if isinstance(detect.f, (list, tuple)) else [int(detect.f)]
    rfa.f = [int(i) for i in src_idx]
    detect.f = idx                            # Detect 改吃 RFA 的输出（int，才能拿到整个 list）
    layers.insert(-1, rfa)
    model.model = type(model.model)(*layers)  # 重建 nn.Sequential
    # 让 y[] 里保留 RFA 的输出，供 Detec t读取
    if hasattr(model, "save"):
        model.save = list(getattr(model, "save", []) or []) + [idx]
    info = {"index": idx, "channels": list(rfa.channels), "params": rfa.np}
    _RFA_SPEC.update(info)
    if log is not None:
        log.info("RFA 注入检测头之前：通道 %s，参数 %d", info["channels"], info["params"])
    return info


def _custom_trainer(log):
    """返回一个在 get_model() 里按需注入 CA / RFA 的 DetectionTrainer 子类。

    Ultralytics 的 ``Model.train()`` 会重建 DetectionModel（按 yaml）再载入
    预训练权重，所以在 ``YOLO(...)`` 实例上提前注入的包装层会被丢掉；
    只有 trainer 自己建出来的模型才是真正被训练、被保存的那个。
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    from .attention import inject_coordatt

    class _CustomTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore[override]
            model = super().get_model(cfg, weights, verbose)
            if _CA_SPEC:
                injected = inject_coordatt(
                    model,
                    _CA_SPEC.get("layer_index"),
                    mode=_CA_SPEC.get("mode", "last"),
                )
                _CA_SPEC["injected"] = injected
                log.info("CA 注入 trainer.model 层号：%s（mode=%s）", injected, _CA_SPEC.get("mode", "last"))
            if _RFA_SPEC.get("enabled"):
                attach_rfa(model, log)
            return model

    return _CustomTrainer




def build_data_yaml(data_cfg: dict | None, out_path: str | Path | None = None) -> Path:
    """把 ``data:`` 段写成 Ultralytics 的 data.yaml。"""
    cfg = data_cfg or {}
    root = resolve_path(cfg.get("root", "data"), base=PROJECT_ROOT)
    out = Path(out_path) if out_path else root / "data.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    names = ", ".join(f"{i}: {n}" for i, n in sorted(CLASS_NAMES.items()))
    out.write_text(
        f"# 自动生成，勿手改（源：configs/vision.yaml data 段）\n"
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names: {{ {names} }}\n",
        encoding="utf-8",
    )
    return out


def _import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "训练需要 ultralytics：pip install ultralytics；另需 torch>=2.0"
        ) from exc
    return YOLO


def _run_name(vision_cfg: dict, override: str | None) -> str:
    if override:
        return override
    train_cfg = vision_cfg.get("train", {}) or {}
    name = train_cfg.get("run_name")
    if name:
        return str(name)
    family = str((vision_cfg.get("model", {}) or {}).get("family", "yolov8"))
    return f"{time_tag()}_{family}_ca"


def train(
    vision_cfg: dict,
    data_yaml: str | Path | None = None,
    run_name: str | None = None,
    epochs: int | None = None,
    device: str | None = None,
    resume: bool = False,
    use_ca: bool | None = None,
    cache: str | bool | None = None,
    patience: int | None = None,
    batch: int | None = None,
    seed: int | None = None,
    ca_inject: str | None = None,
    imgsz: int | None = None,
    workers: int | None = None,
    use_rfa: bool | None = None,
) -> Path:
    """训练 CA-YOLO，返回 run 目录。

    ``use_ca=None`` 时读配置；``cache`` 传 ``"ram"`` 可把数据集缓存进内存
    （FBID 约 1.3 GB，本机 32 GB 内存足够），能明显加快小数据集上的迭代。
    """
    YOLO = _import_yolo()
    model_cfg = vision_cfg.get("model", {}) or {}
    train_cfg = vision_cfg.get("train", {}) or {}
    aug_cfg = vision_cfg.get("augment", {}) or {}
    log = get_logger("train")

    data_path = Path(data_yaml) if data_yaml else build_data_yaml(vision_cfg.get("data"))
    data_path = data_path if data_path.is_absolute() else PROJECT_ROOT / data_path

    family = str(model_cfg.get("family", "yolov8"))
    weights = model_cfg.get("weights")
    if resume:                      # 断点续训：从上次的 last.pt 继续
        weights = str(Path(weights)) if weights else f"{family}n.pt"
    if not weights:
        weights = f"{family}n.pt" if model_cfg.get("pretrained", True) else f"{family}n.yaml"

    log.info("加载 %s，注入 CA=%s", weights, use_ca if use_ca is not None else model_cfg.get("use_ca", True))
    model = YOLO(str(weights))
    ca_on = bool(model_cfg.get("use_ca", True)) if use_ca is None else bool(use_ca)
    _CA_SPEC.clear()
    if ca_on:
        # 注意：不能在 YOLO(...) 上直接注入——Ultralytics 的 trainer 会按 yaml 重建
        # DetectionModel 后再载入预训练权重，之前的包装层会被丢掉。必须在
        # trainer.get_model() 里就地注入，才会进入真正被训练的那个模型。
        _CA_SPEC["layer_index"] = model_cfg.get("ca_layer_index")
        # "last" = 只挂 P5 入口（原实验）；"backbone" = 挂到 Backbone 每个 C2f/SPPF 输出
        _CA_SPEC["mode"] = str(ca_inject or model_cfg.get("ca_inject", "last"))

    _RFA_SPEC.clear()
    rfa_on = bool(model_cfg.get("use_rfa", False)) if use_rfa is None else bool(use_rfa)
    if rfa_on:
        _RFA_SPEC["enabled"] = True

    name = _run_name(vision_cfg, run_name)
    project_dir = ensure_dir(RESULTS_ROOT / "runs")

    results = model.train(
        **({"trainer": _custom_trainer(log)} if (_CA_SPEC or _RFA_SPEC) else {}),
        data=str(data_path),
        epochs=int(epochs or train_cfg.get("epochs", 300)),
        batch=int(batch or train_cfg.get("batch", 32)),
        patience=int(patience or train_cfg.get("patience", 100)),
        cache=False if cache is None else cache,
        imgsz=int(imgsz or model_cfg.get("input_size", 640)),
        lr0=float(train_cfg.get("lr0", 1e-3)),
        lrf=float(train_cfg.get("lrf", 0.01)),
        optimizer=str(train_cfg.get("optimizer", "AdamW")),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        warmup_epochs=float(train_cfg.get("warmup_epochs", 3)),
        workers=int(train_cfg.get("workers", 4)) if workers is None else int(workers),
        device=str(device or train_cfg.get("device", "cpu")),
        seed=int(train_cfg.get("seed", 0) if seed is None else seed),
        project=str(project_dir),
        name=name,
        exist_ok=True,
        hsv_h=float(aug_cfg.get("hsv_h", 0.015)),
        hsv_s=float(aug_cfg.get("hsv_s", 0.5)),
        hsv_v=float(aug_cfg.get("hsv_v", 0.4)),
        fliplr=float(aug_cfg.get("fliplr", 0.5)),
        mosaic=float(aug_cfg.get("mosaic", 1.0)),
        resume=bool(resume),
    )
    run_dir = Path(results.save_dir)
    save_json({"run_dir": str(run_dir), "data": str(data_path), "weights": str(weights)},
              run_dir / "train_meta.json")
    return run_dir


def evaluate(
    weights: str | Path,
    vision_cfg: dict,
    data_yaml: str | Path | None = None,
    split: str = "val",
    device: str = "cpu",
    imgsz: int | None = None,
) -> dict[str, Any]:
    """跑验证集，返回 mAP@0.5 / mAP@0.5:0.95 与逐类指标。"""
    YOLO = _import_yolo()
    model_cfg = vision_cfg.get("model", {}) or {}
    data_path = Path(data_yaml) if data_yaml else build_data_yaml(vision_cfg.get("data"))
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_path),
        split=split,
        imgsz=int(imgsz or model_cfg.get("input_size", 640)),
        device=str(device),
        workers=0,  # DSH 沙箱里多进程 DataLoader 的命名管道会被拒（WinError 5）
        verbose=False,
    )
    result: dict[str, Any] = {
        "weights": str(weights),
        "split": split,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": {},
    }
    names = dict(getattr(metrics, "names", {}) or CLASS_NAMES)
    box = metrics.box
    # ultralytics 的 per-class 数组只统计「本 split 出现过」的类，索引与 ap_class_index 对齐
    idx = list(getattr(box, "ap_class_index", []))
    for pos, cls_id in enumerate(idx):
        try:
            result["per_class"][str(names.get(int(cls_id), cls_id))] = {
                "map50": float(box.ap50[pos]),
                "map50_95": float(box.ap[pos]),
                "precision": float(box.p[pos]),
                "recall": float(box.r[pos]),
            }
        except Exception:  # pragma: no cover - ultralytics 版本差异
            continue
    return result


@dataclass
class LatencyReport:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "mean_ms": self.mean_ms, "p50_ms": self.p50_ms,
                "p95_ms": self.p95_ms, "max_ms": self.max_ms}


def benchmark_latency(
    detector,
    imgsz: int | None = None,
    n: int = 50,
    warmup: int = 5,
) -> LatencyReport:
    """单帧推理延迟（不含前后处理中的像素搬运计数误差可忽略）。

    注意：必须 ``detector`` 已加载（CPU 上可用 yolov8n 做冒烟）。
    """
    import time

    import numpy as np

    size = int(imgsz or getattr(detector.cfg, "imgsz", 640))
    dummy = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    for _ in range(warmup):
        detector.detect(dummy)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        detector.detect(dummy)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    import math

    def pct(q: float) -> float:
        idx = min(len(times) - 1, max(0, int(round(q / 100.0 * (len(times) - 1)))))
        return float(times[idx])

    return LatencyReport(
        n=len(times),
        mean_ms=float(sum(times) / len(times)),
        p50_ms=pct(50),
        p95_ms=pct(95),
        max_ms=float(times[-1]),
    )
