"""Command-line entry point: ``uv run deepmirror`` (or ``python -m backend``)."""

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
from backend.config import APP_NAME, ROOT, Settings
from backend.frontend_build import ensure_frontend
from backend.lifecycle import Lifecycle
from backend.logbook import configure_logging
from backend.process_guard import contain_children


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="deepmirror", description="Realtime voice and face cloning studio")
    parser.add_argument("--host", help="interface to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port to listen on (default 8000)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "xpu", "cpu"], help="inference device")
    parser.add_argument("--no-browser", action="store_true", help="do not open the studio in a browser")
    parser.add_argument("--skip-ui-build", action="store_true", help="never rebuild the web UI on startup")
    parser.add_argument("--close-with-page", action="store_true",
                        help="stop when the studio page is closed (default: keep running until Ctrl+C)")
    parser.add_argument("--keep-alive", action="store_true", help=argparse.SUPPRESS)   # the default now
    return parser.parse_args(argv)


FORCE_EXIT_SECONDS = 8.0


def force_exit_later(seconds: float) -> None:
    time.sleep(seconds)
    logging.getLogger(__name__).warning("Shutdown did not finish in %.0f s; exiting now", seconds)
    logging.shutdown()
    os._exit(0)


def studio_running(url: str) -> bool:
    """Whether the studio already answers at ``url``."""
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
        logging.getLogger(__name__).info("%s is already running; opening %s", APP_NAME, url)
        if not args.no_browser:
            webbrowser.open(url)
        return

    if not args.skip_ui_build:
        ensure_frontend(ROOT / "frontend", settings.FRONTEND_DIST)

    # Every process the studio starts (face, head) ends with it, however it ends.
    contain_children()

    def ready() -> None:
        print(f"\n  {APP_NAME} is running at {url}\n  Press Ctrl+C here to stop it.\n", flush=True)
        if not args.no_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    # Called from the app's startup hook, i.e. only once the server answers.
    on_ready = ready
    server: uvicorn.Server | None = None

    def stop() -> None:
        if server is not None:
            server.should_exit = True
        # A clean shutdown takes well under a second. On Windows, uvicorn can wait
        # forever on sockets the browser already dropped; never leave an invisible
        # process behind (the studio has no window to close it from).
        threading.Thread(target=force_exit_later, args=(FORCE_EXIT_SECONDS,), daemon=True).start()

    # It runs until stopped with Ctrl+C in its console; closing the page stops it only on request.
    close_with_page = args.close_with_page or os.environ.get("OVC_CLOSE_WITH_PAGE", "").strip() in {"1", "true", "yes"}
    lifecycle = Lifecycle(enabled=close_with_page and not args.no_browser, on_exit=stop)
    app = create_app(settings, on_ready=on_ready, lifecycle=lifecycle)
    config = uvicorn.Config(app, host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL.lower(),
                            log_config=None)
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
