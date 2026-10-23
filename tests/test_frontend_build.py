"""Startup UI build check."""

from __future__ import annotations

import os

from backend.frontend_build import ensure_frontend, frontend_is_stale


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_missing_or_older_build_is_stale(tmp_path):
    frontend, dist = tmp_path / "frontend", tmp_path / "frontend" / "dist"
    _touch(frontend / "src" / "App.jsx", 200)
    assert frontend_is_stale(frontend, dist)
    _touch(dist / "index.html", 100)
    assert frontend_is_stale(frontend, dist)
    _touch(dist / "index.html", 300)
    assert not frontend_is_stale(frontend, dist)


def test_without_npm_startup_continues(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    _touch(frontend / "package.json", 100)
    monkeypatch.setattr("backend.frontend_build.shutil.which", lambda _name: None)
    ensure_frontend(frontend, frontend / "dist")  # logs a hint, never raises
