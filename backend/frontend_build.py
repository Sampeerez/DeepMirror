"""Build the web UI on startup when its compiled copy is missing or stale.

Release archives ship ``frontend/dist`` prebuilt. From a source checkout the
UI is built with the Node.js runtime bundled as a Python wheel
(``nodejs-wheel-binaries``), so no system Node.js is needed and the version
is always one Vite supports.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_SOURCES = ("src", "public", "index.html", "package.json", "package-lock.json", "vite.config.js")

NpmRunner = Callable[[list[str], Path], int]


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


def _npm_runner() -> NpmRunner | None:
    """The bundled npm when available, else a system npm."""
    try:
        import nodejs_wheel

        return lambda args, cwd: int(nodejs_wheel.npm(args, cwd=cwd))  # type: ignore[arg-type]
    except ImportError:
        pass
    npm = shutil.which("npm")
    if npm is None:
        return None
    return lambda args, cwd: subprocess.run([npm, *args], cwd=cwd, check=False).returncode


def ensure_frontend(frontend: Path, dist: Path, npm: NpmRunner | None = None) -> bool:
    """Rebuild ``dist`` if needed; returns whether a usable build exists. Never raises."""
    if not (frontend / "package.json").is_file() or not frontend_is_stale(frontend, dist):
        return (dist / "index.html").is_file()
    npm = npm or _npm_runner()
    if npm is None:
        logger.warning("The web UI needs building but no Node.js runtime is available. "
                       "Run `uv sync` (it installs one) or use a release archive.")
        return (dist / "index.html").is_file()

    logger.info("Building the web UI (first run or sources changed)...")
    install = ["ci", "--no-audit", "--no-fund"]
    try:
        if not (frontend / "node_modules").is_dir() and npm(install, frontend) != 0:
            raise RuntimeError("npm ci failed")
        if npm(["run", "build"], frontend) != 0:
            # node_modules from another platform (a copied folder) lacks the
            # right native bindings: reinstall once and retry.
            logger.info("Reinstalling frontend dependencies and retrying the build")
            shutil.rmtree(frontend / "node_modules", ignore_errors=True)
            if npm(install, frontend) != 0 or npm(["run", "build"], frontend) != 0:
                raise RuntimeError("npm run build failed")
    except (OSError, RuntimeError) as exc:
        logger.error("Web UI build failed (%s); see the output above.", exc)
    return (dist / "index.html").is_file()


MISSING_UI_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenVoiceChanger</title>
<style>body{font:15px/1.6 system-ui,sans-serif;background:#0a0a0c;color:#e4e4e7;max-width:40rem;
margin:4rem auto;padding:0 1rem}code{background:#18181b;padding:.1rem .35rem;border-radius:4px}</style></head>
<body><h1>The web UI is not built yet</h1>
<p>The server is running, but <code>frontend/dist</code> is missing and could not be built.
Check the server log for the build error, then restart with the start script
(<code>start.bat</code> or <code>./start.sh</code>), which builds it automatically.</p>
<p>Alternatively download a release archive, which includes the built UI.</p></body></html>"""
