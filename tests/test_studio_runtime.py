"""Logs page, closing with the page, and the virtual microphone."""

from __future__ import annotations

import logging
import time

from backend.lifecycle import Lifecycle
from backend.logbook import Logbook
from backend.virtual_mic import VirtualMic

ORIGIN = {"origin": "http://127.0.0.1:5173"}


class TestLogbook:
    def test_keeps_recent_lines_with_increasing_ids(self):
        book = Logbook(capacity=3)
        logger = logging.getLogger("test.logbook")
        logger.addHandler(book)
        logger.setLevel(logging.DEBUG)
        try:
            for number in range(5):
                logger.info("line %d", number)
            logger.warning("careful")
        finally:
            logger.removeHandler(book)
        lines = book.since(0)
        assert [line.message for line in lines] == ["line 3", "line 4", "careful"]
        assert book.since(lines[-2].id)[0].message == "careful"
        assert [line.level for line in book.since(0, "WARNING")] == ["WARNING"]

    def test_wait_returns_when_a_new_line_arrives(self):
        book = Logbook()
        assert book.wait(book.last_id, 0.05) is False
        book.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None))
        assert book.wait(0, 0.05) is True


class TestLogsApi:
    def test_lines_are_served_after_an_id(self, client):
        logging.getLogger("backend.test").warning("visible in the page")
        body = client.get("/api/logs?level=WARNING").json()
        assert any(line["message"] == "visible in the page" for line in body["lines"])
        newer = client.get(f"/api/logs?after={body['last_id']}").json()["lines"]
        assert all(line["id"] > body["last_id"] for line in newer)

    def test_unknown_training_log_is_404(self, client):
        assert client.get("/api/training/jobs/abcdef123456/log").status_code == 404


class TestLifecycle:
    def test_exits_after_the_last_page_closes_but_not_on_a_reload(self):
        exits = []
        life = Lifecycle(True, lambda: exits.append(1), grace=0.2, first_contact=60)
        life.connect()
        assert not life.should_exit()
        life.disconnect()
        life.connect()          # a reload reconnects within the grace period
        time.sleep(0.25)
        assert not life.should_exit()
        life.disconnect()
        time.sleep(0.25)
        assert life.should_exit()

    def test_exits_if_no_page_ever_connects(self):
        life = Lifecycle(True, lambda: None, grace=60, first_contact=0.1)
        time.sleep(0.15)
        assert life.should_exit()

    def test_disabled_without_a_browser(self):
        assert not Lifecycle(False, lambda: None, grace=0, first_contact=0).should_exit()

    def test_presence_socket_counts_pages(self, client):
        lifecycle = client.app.state.lifecycle
        with client.websocket_connect("/ws/presence", headers=ORIGIN):
            assert lifecycle.pages == 1
        assert lifecycle.pages == 0


class TestVirtualMic:
    def test_status_names_the_devices_to_pick(self, tmp_path, monkeypatch):
        mic = VirtualMic(tmp_path)
        monkeypatch.setattr(mic, "_detect", lambda: False)
        status = mic.status()
        assert {"supported", "installed", "studio_output", "call_input", "state"} <= status.keys()
        if status["platform"] == "Windows":
            assert status["call_input"].startswith("CABLE Output")

    def test_status_endpoint(self, client, monkeypatch):
        monkeypatch.setattr(client.app.state.virtual_mic, "_detect", lambda: True)
        body = client.get("/api/virtual-mic").json()
        assert body["installed"] is True or body["supported"] is False


def test_closing_with_the_page_lets_a_training_run_finish(settings):
    from backend.training.jobs import TrainingJobs

    jobs = TrainingJobs(settings)
    (jobs.root / "abcdef123456").mkdir(parents=True)

    class Running:
        def poll(self):
            return None

    jobs._processes["abcdef123456"] = Running()  # type: ignore[assignment]
    jobs.close(stop_workers=False)
    assert not (jobs.root / "abcdef123456" / "stop").exists()
    jobs.close()
    assert (jobs.root / "abcdef123456" / "stop").exists()


class TestVirtualCamera:
    def _zip(self, path, modules):
        import zipfile

        with zipfile.ZipFile(path, "w") as archive:
            for name, data in modules.items():
                archive.writestr(name, data)
            archive.writestr("bin/64bit/obs64.exe", b"not needed")
        return zipfile.ZipFile(path)

    def test_takes_only_the_camera_modules_checked_by_sha256(self, tmp_path):
        import hashlib

        from backend.virtual_camera import extract_modules

        modules = {"core/win-dshow/data/obs-virtualcam-module64.dll": b"sixty-four",
                   "core/win-dshow/data/obs-virtualcam-module32.dll": b"thirty-two"}
        known = {name: hashlib.sha256(data).hexdigest() for name, data in modules.items()}
        with self._zip(tmp_path / "obs.zip", modules) as archive:
            paths = extract_modules(archive, tmp_path / "out", known)
        assert sorted(path.name for path in paths) == ["obs-virtualcam-module32.dll", "obs-virtualcam-module64.dll"]
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(path.name for path in paths)

    def test_refuses_a_module_that_does_not_match(self, tmp_path):
        import pytest

        from backend.virtual_camera import extract_modules

        modules = {"core/win-dshow/data/obs-virtualcam-module64.dll": b"tampered"}
        with self._zip(tmp_path / "obs.zip", modules) as archive, pytest.raises(RuntimeError, match="checksum"):
            extract_modules(archive, tmp_path / "out", {name: "0" * 64 for name in modules})
        assert not (tmp_path / "out" / "obs-virtualcam-module64.dll").exists()

    def test_registers_both_modules_with_one_prompt(self, tmp_path, monkeypatch):
        from backend import virtual_camera
        from backend.virtual_camera import MODULES, VirtualCamera

        camera = VirtualCamera(tmp_path)
        camera.folder.mkdir(parents=True)
        for name in MODULES:
            (camera.folder / name.rsplit("/", 1)[1]).write_bytes(b"dll")
        commands = []

        class Done:
            returncode = 0

        monkeypatch.setattr(virtual_camera, "_run", lambda command, timeout=20.0: commands.append(command) or Done())
        camera._install_windows()
        assert len(commands) == 1 and "-Verb RunAs" in commands[0][-1]
        script = (camera.folder / "register.cmd").read_text()
        assert script.count("regsvr32.exe /s") == 2

    def test_calls_endpoint_reports_microphone_and_camera(self, client, monkeypatch):
        monkeypatch.setattr(client.app.state.virtual_mic, "_detect", lambda: True)
        monkeypatch.setattr(client.app.state.virtual_camera, "_detect", lambda: False)
        body = client.get("/api/calls").json()
        assert set(body) == {"microphone", "camera"}
        assert body["camera"]["installed"] is False
        assert body["camera"]["device"] == "OBS Virtual Camera"
