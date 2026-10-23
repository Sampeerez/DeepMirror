"""Build the web UI on startup when its compiled copy is missing or stale.

Release archives ship ``frontend/dist`` prebuilt, so Node.js is only needed
when running from a source checkout that has changed since the last build.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_SOURCES = ("src", "public", "index.html", "package.json", "package-lock.json", "vite.config.js")


def _newest_mtime(paths: list[Path]) -> float:
    newest = 0.0
    for path in paths:
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
        elif path.is_dir():
            newest = max([newest, *(p.stat().st_mtime for p in path.rglob("*") if p.is_file())])
    return newest


def frontend_is_stale(frontend: Path, dist: Path) -> bool:
    index = dist / "index.html"
    if not index.is_file():
        return True
    return _newest_mtime([frontend / name for name in _SOURCES]) > index.stat().st_mtime


def ensure_frontend(frontend: Path, dist: Path) -> None:
    """Rebuild ``dist`` with npm if needed and possible; never fails startup."""
    if not (frontend / "package.json").is_file() or not frontend_is_stale(frontend, dist):
        return
    npm = shutil.which("npm")
    if npm is None:
        if not (dist / "index.html").is_file():
            logger.warning("The web UI is not built and Node.js is not installed. Install Node.js 20+ "
                           "or use a release archive, which includes the built UI.")
        return
    logger.info("Building the web UI (first run or sources changed)...")
    try:
        if not (frontend / "node_modules").is_dir():
            subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True)
        subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.error("Web UI build failed (%s); serving the previous build if there is one.", exc)
