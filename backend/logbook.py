"""Server log, kept for the studio's Logs page (there is no console window to read it in).

Every record that reaches the root logger (the studio, uvicorn, warnings,
uncaught exceptions) is kept in a ring of recent lines with increasing ids,
so the page can ask for "everything after id N", and written to a rotating
file under ``data/logs`` for support.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

CAPACITY = 3000
FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass(frozen=True, slots=True)
class LogLine:
    id: int
    time: float
    level: str
    logger: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class Logbook(logging.Handler):
    """A logging handler that remembers the most recent lines."""

    def __init__(self, capacity: int = CAPACITY) -> None:
        super().__init__(level=logging.DEBUG)
        self._lines: deque[LogLine] = deque(maxlen=capacity)
        self._next_id = 1
        self._lock_lines = threading.Lock()
        self._changed = threading.Condition(self._lock_lines)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + logging.Formatter().formatException(record.exc_info)
        except Exception:  # never let logging break the studio
            message = str(record.msg)
        with self._changed:
            self._lines.append(LogLine(self._next_id, record.created, record.levelname, record.name, message))
            self._next_id += 1
            self._changed.notify_all()

    @property
    def last_id(self) -> int:
        return self._next_id - 1

    def since(self, after: int = 0, minimum_level: str | None = None, limit: int = 1000) -> list[LogLine]:
        threshold = logging.getLevelName(minimum_level.upper()) if minimum_level else logging.DEBUG
        if not isinstance(threshold, int):
            threshold = logging.DEBUG
        with self._lock_lines:
            lines = [line for line in self._lines if line.id > after
                     and logging.getLevelName(line.level) >= threshold]
        return lines[-limit:]

    def wait(self, after: int, timeout: float) -> bool:
        """Block until a line newer than ``after`` exists (or ``timeout``); True if there is one."""
        deadline = time.monotonic() + timeout
        with self._changed:
            while self.last_id <= after:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._changed.wait(remaining)
            return True


logbook = Logbook()


def configure_logging(level: str, log_dir: Path | None) -> None:
    """Root logging: the logbook, a rotating file and (when there is one) the console."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(FORMAT)
    if logbook not in root.handlers:
        root.addHandler(logbook)
    if sys.stderr is not None and not any(type(h) is logging.StreamHandler for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    if log_dir is not None and not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file = RotatingFileHandler(log_dir / "server.log", maxBytes=2 * 1024 * 1024, backupCount=3,
                                       encoding="utf-8")
            file.setFormatter(formatter)
            root.addHandler(file)
        except OSError:
            logging.getLogger(__name__).warning("Could not write the log file in %s", log_dir)
    # uvicorn configures its own loggers; route them through the root handlers.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        named = logging.getLogger(name)
        named.handlers.clear()
        named.propagate = True
    logging.captureWarnings(True)

    def log_uncaught(kind, value, traceback) -> None:
        logging.getLogger("uncaught").error("Unhandled exception", exc_info=(kind, value, traceback))

    sys.excepthook = log_uncaught
