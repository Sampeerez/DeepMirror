"""The 3D head: build status, the live session around the drive process, and the setting."""

from __future__ import annotations

import numpy as np

from backend.video.avatar import HEAD_FILE, HeadBuilder, LamInstall

from .conftest import ROOT


def _drive():
    """head3d/drive.py (it runs in LAM's environment; its image helpers need only NumPy and OpenCV)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("drive", ROOT / "backend/video/head3d/drive.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_3d_layer_goes_over_the_frame_where_it_sits_and_nowhere_else():
    from backend.video import hair

    frame = np.full((120, 160, 3), 10, np.uint8)
    layer = np.zeros((40, 50, 4), np.uint8)
    layer[..., :3] = 255
    layer[5:35, 5:45, 3] = 255             # opaque inside, transparent at its border
    out = hair.over(frame, layer, 100, 90)  # partly past the frame's corner: clipped, not wrapped
    assert (out[95:120, 105:145] == 255).all()
    assert (out[:90] == 10).all() and (out[:, :100] == 10).all()
    assert (frame == 10).all()               # a copy unless asked otherwise


def test_the_face_hole_covers_the_face_and_not_the_hair():
    drive = _drive()
    mesh = np.zeros((478, 2), np.float32)
    # A face 200 px wide centred at (320, 260): an oval, brows, eyes and chin where MediaPipe puts them.
    for i, k in enumerate(drive.FACE_OVAL):
        angle = 2 * np.pi * i / len(drive.FACE_OVAL) - np.pi / 2
        mesh[k] = (320 + 100 * np.cos(angle), 260 + 130 * np.sin(angle))
    mesh[drive.BROWS] = np.column_stack([np.linspace(270, 370, len(drive.BROWS)), np.full(len(drive.BROWS), 232)])
    mesh[33], mesh[263], mesh[152], mesh[4] = (280, 240), (360, 240), (320, 390), (320, 300)
    hole = drive.face_hole((520, 640), mesh)
    assert hole[300, 320] > 0.95          # cheeks, nose, mouth: the face swap's
    assert hole[140, 320] < 0.05          # the top of the head: the 3D head's hair


def test_edges_are_crisp_and_take_their_colour_from_inside():
    import torch

    drive = _drive()
    image = torch.zeros(64, 64, 4)
    image[..., :3] = torch.tensor([40, 60, 90]) / 255   # the head's colour
    image[16:48, 16:48, 3] = 1.0
    image[14:16, 16:48, 3] = 0.25          # a faint fringe...
    image[14:16, 16:48, :3] = 0.98         # ...carrying the photo's white background
    out = drive.clean_edges(image)
    assert out[14, 30, 3] < 0.12           # the faint fringe goes
    assert out[15, 30, :3].max() < 0.6     # and loses its white
    assert out[30, 30, 3] == 1.0


def test_build_status_without_the_install(tmp_path):
    builder = HeadBuilder(tmp_path, lambda: tmp_path / "parser.onnx", lambda: False,
                          install=LamInstall(tmp_path / "lam"))
    assert builder.available is False
    assert builder.status("0123456789ab") == {"state": "unavailable"}
    (tmp_path / "0123456789ab").mkdir()
    (tmp_path / "0123456789ab" / HEAD_FILE).write_bytes(b"x")
    assert builder.status("0123456789ab")["state"] == "ready"   # built earlier: still usable


def test_3d_is_a_live_setting_and_status_says_if_it_is_installed(client):
    status = client.get("/api/video/").json()
    assert status["head3d"] is False
    assert client.patch("/api/video/settings", json={"mode": "3d"}).json()["settings"]["mode"] == "3d"
    assert client.post("/api/video/faces/0123456789ab/head3d").status_code == 404
    assert client.get("/api/video/faces/0123456789ab/head3d.jpg").status_code == 404
    assert status["head3d_install"]["state"] == "unavailable"   # OVC_HEAD3D=off in tests


def test_environment_names_are_merged_as_windows_compares_them():
    from backend.video.avatar import merge_environment

    env = {"PATH": "ours", "Other": "1"}
    merge_environment(env, [("Path", "vs;ours")])   # cmd's spelling: one variable, never two
    assert env == {"Path": "vs;ours", "Other": "1"}


def test_the_installer_is_off_when_asked(monkeypatch, tmp_path):
    from backend.video.head3d_install import Head3DInstaller

    monkeypatch.setenv("OVC_HEAD3D", "off")
    installer = Head3DInstaller(tmp_path, install=LamInstall(tmp_path / "lam"))
    installer.start_if_needed()
    assert installer._process is None
    assert installer.status() == {"state": "unavailable", "error": "Turned off (OVC_HEAD3D)"}


def test_the_pose_is_always_in_front_of_the_camera():
    import cv2

    drive = _drive()
    # A head model (y up, as FLAME) seen by a camera looking along +z with y down.
    rng = np.random.default_rng(0)
    model = rng.uniform(-0.08, 0.08, (51, 3))
    camera = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    true_r, true_t = np.array([[np.pi], [0.2], [0.0]]), np.array([[0.02], [-0.03], [0.6]])
    points = cv2.projectPoints(model, true_r, true_t, camera, None)[0][:, 0]
    ok, r, t = drive.solve_pose(model, points, camera)
    assert ok and t[2, 0] > 0
    assert abs(t[2, 0] - 0.6) < 0.02
    # A previous frame's mirror solution behind the camera is not kept.
    ok, r, t = drive.solve_pose(model, points, camera, np.array([[0.0], [0.0], [0.0]]), np.array([[0.0], [0.0], [-0.6]]))
    assert ok and t[2, 0] > 0
