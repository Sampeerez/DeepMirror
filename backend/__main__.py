"""Command-line entry point: ``uv run openvoicechanger`` (or ``python -m backend``)."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import urllib.request
import webbrowser

import uvicorn

from backend.app import create_app
from backend.config import ROOT, Settings
from backend.frontend_build import ensure_frontend
from backend.lifecycle import Lifecycle
from backend.logbook import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="openvoicechanger", description="Realtime RVC voice changer studio")
    parser.add_argument("--host", help="interface to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port to listen on (default 8000)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "xpu", "cpu"], help="inference device")
    parser.add_argument("--no-browser", action="store_true", help="do not open the studio in a browser")
    parser.add_argument("--skip-ui-build", action="store_true", help="never rebuild the web UI on startup")
    parser.add_argument("--keep-alive", action="store_true",
                        help="keep running when the studio page is closed (default: exit with the page)")
    return parser.parse_args(argv)


FORCE_EXIT_SECONDS = 8.0


def force_exit_later(seconds: float) -> None:
    time.sleep(seconds)
    logging.getLogger(__name__).warning("Shutdown did not finish in %.0f s; exiting now", seconds)
    logging.shutdown()
    os._exit(0)


def studio_running(url: str) -> bool:
    """Whether an OpenVoiceChanger server already answers at ``url``."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1.5) as response:
            return response.status == 200 and b'"ok"' in response.read()
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {key: value for key, value in (("HOST", args.host), ("PORT", args.port), ("DEVICE", args.device))
                 if value is not None}
    settings = Settings(**overrides)
    configure_logging(settings.LOG_LEVEL, settings.DATA_DIR / "logs")

    url = f"http://{'127.0.0.1' if settings.HOST in {'0.0.0.0', '::'} else settings.HOST}:{settings.PORT}"
    if studio_running(url):
        # Started again (shortcut, start script) while it is open: show it instead of failing quietly.
        logging.getLogger(__name__).info("OpenVoiceChanger is already running; opening %s", url)
        if not args.no_browser:
            webbrowser.open(url)
        return

    if not args.skip_ui_build:
        ensure_frontend(ROOT / "frontend", settings.FRONTEND_DIST)

    # Opened from the app's startup hook, i.e. only once the server answers.
    on_ready = None if args.no_browser else (lambda: threading.Timer(0.5, webbrowser.open, args=(url,)).start())
    server: uvicorn.Server | None = None

    def stop() -> None:
        if server is not None:
            server.should_exit = True
        # A clean shutdown takes well under a second. On Windows, uvicorn can wait
        # forever on sockets the browser already dropped; never leave an invisible
        # process behind (the studio has no window to close it from).
        threading.Thread(target=force_exit_later, args=(FORCE_EXIT_SECONDS,), daemon=True).start()

    # Opened in a browser by us: closing that page ends the studio (it has no console window).
    lifecycle = Lifecycle(enabled=not args.no_browser and not args.keep_alive, on_exit=stop)
    app = create_app(settings, on_ready=on_ready, lifecycle=lifecycle)
    config = uvicorn.Config(app, host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL.lower(),
                            log_config=None)
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
