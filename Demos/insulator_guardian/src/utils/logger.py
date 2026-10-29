"""日志：统一格式、可配等级、可选写文件。重复调用不会叠加 handler。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(
    name: str = "insulator_guardian",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    fmt: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """取一个配置好的 logger。

    Parameters
    ----------
    log_file
        给定时追加写入该文件（目录自动创建）；``None`` 只输出到 stderr。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(fmt)

    if not any(isinstance(h, logging.StreamHandler) and h.get_name() == "ig-stderr"
               for h in logger.handlers):
        stream = logging.StreamHandler(sys.stderr)
        stream.set_name("ig-stderr")
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        abs_path = str(path.resolve())
        exists = any(isinstance(h, logging.FileHandler) and h.baseFilename == abs_path
                     for h in logger.handlers)
        if not exists:
            fh = logging.FileHandler(abs_path, encoding="utf-8")
            fh.set_name(f"ig-file-{abs_path}")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
