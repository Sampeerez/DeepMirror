"""Any camera, any shape: the real picture found within its frame, nothing stretched, the head fitted to the speaker."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from backend.video.engine import Tracked
from backend.video.harmony import _measure
from backend.video.live import PICTURE_EVERY, PICTURE_SURE, LiveFaceSession, Picture, picture_box
from backend.video.worker import fitted, stats_line

from .test_head3d import _drive


def _in_bars(width=1280, height=720, box=(437, 0, 842, 720), level=120):
    frame = np.zeros((height, width, 3), np.uint8)
    frame[box[1]:box[3], box[0]:box[2]] = level
    return frame


class TestThePicture:
    def test_bars_at_any_side_are_found(self):
        for box in ((437, 0, 842, 720), (0, 90, 1280, 630), (100, 60, 1180, 660)):
            found = picture_box(_in_bars(box=box))
            assert found is not None
            assert all(abs(a - b) <= 4 for a, b in zip(found, box, strict=True))

    def test_a_frame_without_bars_is_the_picture(self):
        assert picture_box(np.full((480, 640, 3), 90, np.uint8)) is None
        assert picture_box(np.full((1280, 720, 3), 90, np.uint8)) is None

    def test_a_dark_room_is_not_a_picture_in_bars(self):
        assert picture_box(np.zeros((720, 1280, 3), np.uint8)) is None
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[300:340, 620:660] = 200     # one lamp in the dark
        assert picture_box(frame) is None

    def test_a_small_logo_on_a_bar_changes_nothing(self):
        frame = _in_bars()
        frame[700:706, 20:24] = 255
        found = picture_box(frame)
        assert found is not None and abs(found[0] - 437) <= 4

    def test_it_is_taken_at_once_and_a_dark_moment_does_not_move_it(self):
        picture = Picture()
        assert picture.update(_in_bars()) is False      # the first frame: nothing to start over
        assert picture.box is not None
        before = picture.box
        for _ in range(PICTURE_EVERY * (PICTURE_SURE - 1)):
            picture.update(np.zeros((720, 1280, 3), np.uint8))
        assert picture.box == before
        for _ in range(PICTURE_EVERY):                   # back: nothing happened
            picture.update(_in_bars())
        assert picture.box == before

    def test_a_new_shape_is_taken_when_it_lasts(self):
        picture = Picture()
        picture.update(_in_bars())
        moved = [picture.update(np.full((720, 1280, 3), 120, np.uint8)) for _ in range(PICTURE_EVERY * (PICTURE_SURE + 1))]
        assert sum(moved) == 1
        assert picture.box is None


def test_the_session_works_on_the_picture_alone_and_gives_back_the_frame():
    seen = []

    def track(frame, t):
        seen.append(frame.shape)
        return Tracked(None, None)

    engine = SimpleNamespace(hair_from_3d=False, wants_head=False, identity=None, remote=None, reset_head=lambda: None,
                             new_picture=lambda: None, track=track, _own=None)
    session = LiveFaceSession(engine)  # type: ignore[arg-type]
    try:
        frame = _in_bars()
        session.submit(0.0, frame)
        deadline = time.monotonic() + 2.0
        out = session.ready()
        while not out and time.monotonic() < deadline:
            out = session.ready()
    finally:
        session.close()
    assert abs(seen[0][1] - 405) <= 8 and seen[0][0] == 720
    assert out[0][1].shape == frame.shape
    assert (out[0][1][:, :400] == 0).all() and (out[0][1][:, 860:] == 0).all()
    assert (out[0][1][:, 640] == 120).all()


class TestVirtualCameraFit:
    def test_nothing_is_stretched(self):
        out = fitted(np.full((1280, 720, 3), 200, np.uint8), (1280, 720))   # an upright phone
        assert out.shape == (720, 1280, 3)
        assert (out[:, :400] == 0).all() and (out[:, -400:] == 0).all()
        assert (out[360, 640] == 200).all()

    def test_the_same_shape_is_only_scaled_and_the_same_size_is_untouched(self):
        same = np.zeros((720, 1280, 3), np.uint8)
        assert fitted(same, (1280, 720)) is same
        assert fitted(np.full((1080, 1920, 3), 9, np.uint8), (1280, 720)).min() == 9


def test_a_session_leaves_one_readable_line_in_the_log():
    line = stats_line({"fps_out": 29.5, "head_fit": 0.93, "layers": (10, 2, 0), "picture": (437, 0, 842, 720)}, "3d", 3, True)
    for part in ("engine=3d", "fps_out=29.5", "head_fit=0.93", "layers=(10, 2, 0)", "picture=(437, 0, 842, 720)",
                 "dropped=3", "virtual_camera=True"):
        assert part in line


class TestHeadFit:
    def test_the_face_stays_and_the_sides_top_and_back_come_in(self):
        drive = _drive()
        import torch
        origin = torch.tensor([0.0, 0.03, -0.01])
        half = 0.075
        xyz = torch.tensor([[0.02, 0.0, 0.05],      # the face
                            [half, 0.0, -0.01],      # an ear
                            [0.0, 0.13, 0.0],        # the top of the hair
                            [0.0, 0.03, -0.11]])     # the back of the head
        out = drive.fitted_head(xyz, origin, 0.9, half)
        assert out[0].allclose(xyz[0])
        assert abs(float(out[1, 0]) - 0.9 * half) < 1e-5
        assert abs(float(out[2, 1] - origin[1]) - 0.9 * 0.1) < 1e-5
        assert abs(float(out[3, 2] - origin[2]) + 0.9 * 0.1) < 1e-5
        assert drive.fitted_head(xyz, origin, 1.0, half) is xyz

    def test_it_waits_for_many_measures_ignores_a_bad_one_and_never_jumps(self):
        drive = _drive()
        fit = drive.HeadFit()
        value = 1.0
        for _ in range(drive.FIT_SAMPLES // 3 - 1):
            value = fit.update(0.9, value)
        assert value == 1.0
        value = fit.update(0.9, value)
        assert 0.99 < value < 1.0
        value = fit.update(0.3, value)    # a bad frame among good ones
        assert value > 0.98
        for _ in range(600):
            value = fit.update(0.9, value)
        assert abs(value - 0.9) < 0.01

    def test_it_has_limits(self):
        drive = _drive()
        fit = drive.HeadFit()
        value = 1.0
        for _ in range(2000):
            value = fit.update(0.3, value)
        assert abs(value - drive.FIT_LOWEST) < 0.01


def test_a_large_patch_of_another_colour_leaves_the_hair():
    drive = _drive()
    import torch
    generator = torch.Generator().manual_seed(0)
    xyz = torch.rand(500, 3, generator=generator)
    rgb = torch.tensor([0.16, 0.12, 0.10]) + torch.rand(500, 3, generator=generator) * 0.01
    rgb[:75] = torch.tensor([0.35, 0.04, 0.08])     # 15 % of the hair: crimson
    fixed = drive.without_stray_hair(xyz, rgb, torch.ones(500, dtype=torch.bool))
    assert (fixed[:75, 0] - fixed[:75, 1]).abs().max() < 0.08


def test_thin_hair_gaussians_grow_towards_their_neighbours_and_the_rest_stay():
    drive = _drive()
    import torch
    xyz = torch.stack(torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij"), -1).reshape(-1, 2) * 0.01
    xyz = torch.cat([xyz, torch.zeros(len(xyz), 1)], 1)
    scaling = torch.full((25, 3), 1e-4)
    chosen = torch.ones(25, dtype=torch.bool)
    chosen[0] = False
    out = drive.filled_splats(xyz, scaling, chosen)
    assert out[0].equal(scaling[0])
    assert (out[1:] > 1e-3).all() and (out[1:] < 0.01).all()


def test_a_small_render_is_enlarged_before_the_outline_gets_its_contrast():
    drive = _drive()
    import torch
    image = torch.zeros(64, 64, 4)
    image[..., :3] = 0.5
    image[16:48, 16:48, 3] = 1.0
    out = drive.clean_edges(image, 128)
    assert out.shape == (128, 128, 4)
    assert drive.clean_edges(image).shape == (64, 64, 4)
    # One step of the small render's pixels would be two equal pixels here: the ramp is smooth.
    assert len(torch.unique((out[64, :, 3] * 255).round())) > len(torch.unique((drive.clean_edges(image)[32, :, 3] * 255).round()))


def test_the_tint_near_the_face_is_the_whole_faces():
    """A brow or an eyelid that differs in colour moves the brightness near it, never the tint."""
    under = np.full((72, 72, 3), (100, 110, 150), np.uint8)
    image = np.full((72, 72, 3), (100, 110, 150), np.uint8)
    under[20:30, 20:50] = (60, 140, 60)       # something green under one place of the face
    field = _measure(under, image, np.ones((72, 72), np.uint8))
    assert field is not None
    tint = field - field.mean(2)[..., None]
    assert np.ptp(tint, axis=(0, 1)).max() < 1e-5


def test_a_very_tall_phone_is_found_too():
    found = picture_box(_in_bars(box=(478, 0, 802, 720)))     # 20:9 upright: a quarter of the frame's width
    assert found is not None and abs(found[0] - 478) <= 4 and abs(found[2] - 802) <= 4
