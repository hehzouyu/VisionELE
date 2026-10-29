"""IO 与通用小工具：JSON / JSONL、时间戳、目录创建。"""
from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    """dataclass / numpy 标量 / Path 的兜底序列化。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    # numpy 标量/数组（延迟导入，numpy 是本项目硬依赖）
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"无法序列化的对象: {type(obj)!r}")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=_json_default)
        fh.write("\n")
    return p


def append_jsonl(record: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_json_default))
        fh.write("\n")
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iso_now() -> str:
    """本地时区 ISO8601（带偏移），秒级精度。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def date_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def time_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    return p


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数（避免依赖 numpy 的接口差异）。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)
