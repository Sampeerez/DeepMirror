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
    assert drive.face_hole((520, 640), mesh, fringe=False)[195, 320] > drive.face_hole((520, 640), mesh, fringe=True)[195, 320]   # no fringe: the swap goes further up the forehead


def test_edges_are_soft_and_take_their_colour_from_inside():
    import torch

    drive = _drive()
    image = torch.zeros(64, 64, 4)
    image[..., :3] = torch.tensor([40, 60, 90]) / 255   # the head's colour
    image[16:48, 16:48, 3] = 1.0
    image[14:16, 16:48, 3] = 0.03          # next to nothing: gone
    image[12:14, 16:48, 3] = 0.0
    image[16:18, 16:48, :3] = 0.98         # opaque on the outline, carrying the photo's white background
    out = drive.clean_edges(image)
    assert out[15, 30, 3] < 0.05
    assert out[17, 30, :3].max() < 0.6     # the white is gone
    assert out[30, 30, 3] == 1.0


def test_gaussians_turn_with_the_head():
    import cv2
    import torch

    drive = _drive()
    rotation = cv2.Rodrigues(np.array([0.3, -0.5, 0.2]))[0]
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    turned = drive.turned_quaternions(identity, torch.from_numpy(rotation).float())[0].numpy()
    w, x, y, z = turned
    back = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    assert np.abs(back - rotation).max() < 1e-4
    half = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]   # facing the camera: a half turn
    turned = drive.turned_quaternions(identity, torch.from_numpy(half).float())[0].numpy()
    assert abs(abs(turned[1]) - 1.0) < 1e-3


def _hair_motion(drive):
    import torch

    rng = np.random.default_rng(2)
    xyz = torch.from_numpy(rng.uniform(-0.1, 0.1, (2000, 3))).float()
    reach = torch.from_numpy(rng.uniform(0.0, 0.08, 2000)).float()
    return drive.HairMotion(xyz, reach), reach


def test_hair_stays_put_on_a_still_head_and_swings_when_it_moves():
    drive = _drive()
    motion, reach = _hair_motion(drive)
    still = np.eye(3)
    for frame in range(300):
        motion.step(frame / 30, still, np.array([0.0, 0.0, 0.6]))
    assert np.abs(motion.offset).max() < 1e-3         # a still head: the hair as built
    for frame in range(300, 330):
        motion.step(frame / 30, still, np.array([0.2 * np.sin(frame / 2), 0.0, 0.6]))
    assert np.abs(motion.offset).max() > 1e-3         # it swings
    assert np.abs(motion.offset).max() <= drive.HAIR_REACH   # and never runs away
    moved = motion.displacement("cpu").norm(dim=1)
    assert moved[reach < drive.HAIR_ROOT].max() < 1e-6   # what sits on the scalp stays on it
    for frame in range(330, 800):
        motion.step(frame / 30, still, np.array([0.0, 0.0, 0.6]))
    assert np.abs(motion.offset).max() < 1e-3         # and it settles again


def test_tilting_the_head_alone_keeps_the_hair_stable():
    import cv2

    drive = _drive()
    motion, _ = _hair_motion(drive)
    tilted = cv2.Rodrigues(np.array([0.0, 0.0, 0.5]))[0]
    for frame in range(400):
        motion.step(frame / 30, tilted, np.array([0.0, 0.0, 0.6]))
    assert np.isfinite(motion.offset).all()
    assert np.abs(motion.offset).max() <= drive.HAIR_REACH


def test_the_render_grows_while_there_is_room_and_shrinks_at_once():
    drive = _drive()
    budget = drive.RenderBudget()
    assert budget.limit == drive.RENDER_STEPS[drive.RENDER_START]
    for _ in range(60 * len(drive.RENDER_STEPS)):
        budget.update(1.0, budget.limit, voice=True)
    assert budget.limit == drive.RENDER_STEPS[-1]    # a fast GPU: the sharpest
    budget.update(50.0, budget.limit, voice=True)
    assert budget.limit == drive.RENDER_STEPS[-2]    # over budget: down at once
    limit = budget.limit
    for _ in range(200):
        budget.update(1.0, 300, voice=True)
    assert budget.limit == limit                     # a small head on screen does not raise it


def test_a_reused_3d_layer_moves_with_the_face():
    from backend.video.avatar import Layer
    from backend.video.live import LiveFaceSession

    layer = Layer(100, 50, np.full((40, 40, 4), 255, np.uint8))
    before = np.array([[120, 60], [150, 60], [135, 75], [125, 85], [145, 85]], np.float32)
    moved = LiveFaceSession._moved(layer, before, before + np.array([10.0, 5.0], np.float32))
    assert abs(moved.x - 110) <= 1 and abs(moved.y - 55) <= 1
    assert LiveFaceSession._moved(layer, None, before) is layer


def test_the_background_plate_learns_and_starts_again_when_the_picture_changes():
    from backend.video import hair

    frame = np.full((480, 640, 3), 40, np.uint8)
    points = np.array([[280, 220], [340, 220], [310, 255], [285, 290], [335, 290]], np.float32)
    known = np.zeros((hair.SIZE // 4, hair.SIZE // 4), np.float32)
    known[:40, :] = 1.0
    own = hair.OwnHead(0.0, known, np.zeros((hair.SIZE, hair.SIZE), np.float32))
    plate = hair.Plate()
    plate.update(frame, points, own)
    assert plate.seen is not None and plate.seen.any()
    _, matrix = hair.head_crop(frame, points)
    found = plate.in_crop(matrix)
    assert found is not None and abs(float(found[0][40, 256, 0]) - 40) < 2
    plate.update(np.full_like(frame, 200), points, own)    # another picture altogether
    assert abs(float(plate.colour[plate.seen > 0].mean()) - 200) < 2


def test_the_white_fringe_takes_its_neighbours_colour():
    import torch

    drive = _drive()
    xyz = torch.rand(1000, 3)
    rgb = torch.full((1000, 3), 0.3)
    rgb[:20] = 0.97
    fixed = drive.without_white_fringe(xyz, rgb)
    assert fixed[:20].max() < 0.4 and (fixed[20:] == rgb[20:]).all()


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
    ok, _, t = drive.solve_pose(model, points, camera)
    assert ok and t[2, 0] > 0
    assert abs(t[2, 0] - 0.6) < 0.02
    # A previous frame's mirror solution behind the camera is not kept.
    ok, _r, t = drive.solve_pose(model, points, camera, np.array([[0.0], [0.0], [0.0]]), np.array([[0.0], [0.0], [-0.6]]))
    assert ok and t[2, 0] > 0
