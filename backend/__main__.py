"""Command-line entry point: ``uv run openvoicechanger`` (or ``python -m backend``)."""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from backend.app import create_app
from backend.config import Settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="openvoicechanger", description="Realtime RVC voice changer studio")
    parser.add_argument("--host", help="interface to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port to listen on (default 8000)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "xpu", "cpu"], help="inference device")
    parser.add_argument("--no-browser", action="store_true", help="do not open the studio in a browser")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {key: value for key, value in (("HOST", args.host), ("PORT", args.port), ("DEVICE", args.device))
                 if value is not None}
    settings = Settings(**overrides)
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    url = f"http://{'127.0.0.1' if settings.HOST in {'0.0.0.0', '::'} else settings.HOST}:{settings.PORT}"
    # Opened from the app's startup hook, i.e. only once the server answers.
    on_ready = None if args.no_browser else (lambda: threading.Timer(0.5, webbrowser.open, args=(url,)).start())
    uvicorn.run(create_app(settings, on_ready=on_ready), host=settings.HOST, port=settings.PORT,
                log_level=settings.LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
