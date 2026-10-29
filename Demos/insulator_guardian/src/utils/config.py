"""配置加载：``configs/*.yaml`` 是唯一事实源。

约定
----
- 相对路径一律相对 ``insulator_guardian/`` 解析（不是相对 CWD），
  这样从任何目录调用 ``python scripts/xxx.py`` 都能找到配置。
- ``cfg_get(cfg, "dwa.weights.heading", 0.8)`` 支持点号取嵌套值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件，返回 dict。相对路径按项目根目录解析。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {p}")
    return data


def deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并，``extra`` 覆盖 ``base``，不改动入参。"""
    out: dict[str, Any] = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def load_config(name: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """按名字加载配置，如 ``load_config("decision")``。"""
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    cfg = load_yaml(CONFIG_DIR / filename)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """按 ``"a.b.c"`` 取嵌套值，缺失返回 default。"""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    """把配置里的相对路径解析成绝对路径（相对项目根目录）。"""
    p = Path(value)
    if p.is_absolute():
        return p
    root = Path(base) if base is not None else PROJECT_ROOT
    return (root / p).resolve()
