"""
app/utilities/logger.py
-----------------------
Structured logger for the Gaming Video Agent.

Every record carries: timestamp, level, module, job_id, message, extras.
Outputs: coloured console  +  JSON-lines file (logs/<date>.jsonl).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

_LEVEL_COLOURS: dict[str, str] = {
    "DEBUG":    Fore.CYAN,
    "INFO":     Fore.GREEN,
    "WARNING":  Fore.YELLOW,
    "ERROR":    Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}

_SKIP_FIELDS = frozenset({
    "args","asctime","created","exc_info","exc_text","filename","funcName",
    "id","levelname","levelno","lineno","message","module","msecs","msg",
    "name","pathname","process","processName","relativeCreated","stack_info",
    "taskName","thread","threadName",
})


class _JsonLineHandler(logging.FileHandler):
    """Writes one JSON object per line."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry: dict[str, Any] = {
                "ts":      datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level":   record.levelname,
                "module":  record.name,
                "message": record.getMessage(),
            }
            for key, value in record.__dict__.items():
                if key not in _SKIP_FIELDS:
                    entry[key] = value
            if record.exc_info:
                entry["exc_info"] = "".join(traceback.format_exception(*record.exc_info))
            self.stream.write(json.dumps(entry, default=str) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


class _ConsoleHandler(logging.StreamHandler):
    """Writes coloured lines to stdout safely on all platforms."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            colour = _LEVEL_COLOURS.get(record.levelname, "")
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            job_id: str = getattr(record, "job_id", "")
            job_part = f" [{job_id}]" if job_id else ""
            line = (
                f"{Fore.WHITE}{ts}{Style.RESET_ALL} "
                f"{colour}{record.levelname:<8}{Style.RESET_ALL} "
                f"{Fore.BLUE}{record.name}{Style.RESET_ALL}"
                f"{Fore.YELLOW}{job_part}{Style.RESET_ALL}"
                f"  {record.getMessage()}"
            )
            if record.exc_info:
                line += "\n" + "".join(traceback.format_exception(*record.exc_info))

            # Encode safely for console
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout.buffer.write((line + "\n").encode(sys.stdout.encoding or "utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            else:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Return a configured logger for *name*."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    console = _ConsoleHandler()
    console.setLevel(level)
    logger.addHandler(console)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_handler = _JsonLineHandler(
        str(log_path / f"agent_{date_str}.jsonl"), encoding="utf-8"
    )
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
