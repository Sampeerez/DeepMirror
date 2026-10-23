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


def test_without_node_startup_continues(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    _touch(frontend / "package.json", 100)
    monkeypatch.setattr("backend.frontend_build._npm_runner", lambda: None)
    assert ensure_frontend(frontend, frontend / "dist") is False  # logs a hint, never raises


def test_foreign_node_modules_are_reinstalled_once(tmp_path):
    frontend = tmp_path / "frontend"
    _touch(frontend / "package.json", 100)
    (frontend / "node_modules").mkdir()
    calls = []

    def npm(args, cwd):
        calls.append(args[0] if args[0] == "ci" else "build")
        if calls == ["build"]:
            return 1  # native bindings from another OS
        if args[0] == "run":
            _touch(frontend / "dist" / "index.html", 500)
        return 0

    assert ensure_frontend(frontend, frontend / "dist", npm) is True
    assert calls == ["build", "ci", "build"]


def test_missing_ui_serves_instructions(client):
    response = client.get("/")
    assert response.status_code == 503 and "not built yet" in response.text
