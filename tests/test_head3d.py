"""The 3D head: build status, the live session around the drive process, and the setting."""

from __future__ import annotations

import numpy as np

from backend.video.avatar import HEAD_FILE, HeadBuilder, LamInstall, Layer, LiveAvatarSession, paste


class FakeDriver:
    """The drive process: remembers the frames sent; layers come back when the test says."""

    def __init__(self) -> None:
        self.sent: list[int] = []
        self.timings = {"track_ms": 9.0, "render_ms": 3.0}
        self.on_layer = lambda seq, layer: None

    def submit(self, seq: int, frame: np.ndarray) -> bool:
        self.sent.append(seq)
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, np.uint8)


def white_layer() -> Layer:
    image = np.zeros((4, 4, 4), np.uint8)
    image[..., :3] = 255
    image[..., 3] = 255
    return Layer(2, 2, image)


def test_paste_puts_the_layer_where_it_sits_and_nowhere_else():
    out = paste(frame(10), white_layer())
    assert (out[2:6, 2:6] == 255).all()
    assert (out[:2] == 10).all() and (out[6:] == 10).all()


def test_frames_wait_for_their_head_and_leave_in_order():
    driver, clock = FakeDriver(), Clock()
    session = LiveAvatarSession(driver, max_wait=0.3, in_flight=2, clock=clock)  # type: ignore[arg-type]
    for t in range(3):
        session.submit(t / 30, frame(10))
    assert driver.sent == [0, 1]          # two with the driver; the third gets the newest head
    assert session.ready() == []          # frame 0 waits for its head
    driver.on_layer(1, white_layer())
    assert session.ready() == []          # still in order: frame 0 first
    driver.on_layer(0, None)              # no face in frame 0
    out = session.ready()
    assert [t for t, _ in out] == [0, 1 / 30, 2 / 30]
    assert (out[0][1] == 10).all()                       # no head: the camera frame
    assert (out[1][1][2:6, 2:6] == 255).all()            # its own head
    assert (out[2][1][2:6, 2:6] == 255).all()            # the newest head, a frame old


def test_a_head_that_never_comes_back_does_not_hold_the_video():
    driver, clock = FakeDriver(), Clock()
    session = LiveAvatarSession(driver, max_wait=0.3, clock=clock)  # type: ignore[arg-type]
    session.submit(0.0, frame(10))
    assert session.ready() == []
    clock.now = 0.31
    assert len(session.ready()) == 1
    driver.on_layer(0, white_layer())   # too late: dropped, not kept forever
    assert session._layers == {}


def test_build_status_without_the_install(tmp_path):
    builder = HeadBuilder(tmp_path, lambda: tmp_path / "parser.onnx", lambda: False,
                          install=LamInstall(tmp_path / "lam"))
    assert builder.available is False
    assert builder.status("0123456789ab") == {"state": "unavailable"}
    (tmp_path / "0123456789ab").mkdir()
    (tmp_path / "0123456789ab" / HEAD_FILE).write_bytes(b"x")
    assert builder.status("0123456789ab") == {"state": "ready"}   # built earlier: still usable


def test_3d_is_a_live_setting_and_status_says_if_it_is_installed(client):
    status = client.get("/api/video/").json()
    assert status["head3d"] is False
    assert client.patch("/api/video/settings", json={"mode": "3d"}).json()["settings"]["mode"] == "3d"
    assert client.post("/api/video/faces/0123456789ab/head3d").status_code == 404
