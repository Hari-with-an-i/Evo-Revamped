"""
Central logging configuration for Evo-Revamped.

Usage in any module:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")

Log level is controlled by the LOG_LEVEL env var (default: INFO).
Log format is controlled by LOG_FORMAT env var: "json" (default) or "text".

JSON format is structured and machine-readable — suitable for log aggregators.
Text format is human-readable — useful during local development.

Set LOG_FILE env var to also write logs to a rotating file (max 10 MB, 5 backups).
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from typing import Any


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via `extra={"key": val}`
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Setup (called once at import time)
# ---------------------------------------------------------------------------

def _setup() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    log_file = os.getenv("LOG_FILE", "")

    formatter: logging.Formatter
    if log_format == "text":
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = _JsonFormatter()

    root = logging.getLogger("evo")
    root.setLevel(level)
    root.propagate = False

    # Console handler
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        # Optional file handler
        if log_file:
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            fh.setFormatter(formatter)
            root.addHandler(fh)


_setup()


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'evo' root logger.

    Args:
        name: typically __name__ of the calling module.
    """
    # Strip the 'src.' prefix so logger names read as e.g. 'evo.orchestrator'
    short = name.removeprefix("src.").replace(".", "/")
    return logging.getLogger(f"evo.{short}")


# ---------------------------------------------------------------------------
# Convenience context manager for timing LLM calls
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self, log: logging.Logger, label: str, extra: dict | None = None):
        self._log = log
        self._label = label
        self._extra = extra or {}

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = round(time.perf_counter() - self._start, 3)
        if exc_type:
            self._log.error(
                f"{self._label} failed after {elapsed}s",
                extra={**self._extra, "elapsed_s": elapsed, "error": str(exc_val)},
            )
        else:
            self._log.info(
                f"{self._label} completed in {elapsed}s",
                extra={**self._extra, "elapsed_s": elapsed},
            )
        return False  # don't suppress exceptions


def timer(log: logging.Logger, label: str, **extra) -> _Timer:
    """Use as: `with timer(log, 'llm_call', node='orchestrator'):`"""
    return _Timer(log, label, extra)
