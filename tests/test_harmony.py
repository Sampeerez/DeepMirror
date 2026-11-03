"""One light on the worn head: relit by the face it dresses, smoothly, and within limits."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from backend.video import hair, harmony
from backend.video.harmony import Harmony


def _head(side: int = 200, colour=(120, 150, 200)) -> np.ndarray:
    image = np.zeros((side, side, 4), np.uint8)
    image[..., :3] = colour
    image[..., 3] = 255
    return image


def _face(side: int = 200) -> np.ndarray:
    """The middle of the layer, at a quarter of its size."""
    face = np.zeros((side // 4, side // 4), np.uint8)
    face[side // 12:-side // 12, side // 12:-side // 12] = 1
    return face


def test_the_head_takes_the_light_of_the_face_under_it():
    head = _head()
    under = np.full((200, 200, 3), (60, 90, 160), np.uint8)   # dim and pink
    colour = Harmony().relit(under, head, _face())
    assert np.abs(colour[100, 100].astype(int) - (60, 90, 160)).max() <= 3
    # The same light on what is not skin, far from the face: the hair stays dark with the same tint.
    head[:20] = (30, 30, 30, 255)
    colour = Harmony().relit(under, head, _face())
    assert colour[5, 100, 0] < 30 and colour[5, 100, 2] > colour[5, 100, 0]


def test_a_face_only_on_one_side_lights_that_side_more_and_the_seam_closes():
    head = _head()
    under = np.full((200, 200, 3), (100, 100, 100), np.uint8)
    under[:, :100] = (60, 60, 60)
    colour = Harmony().relit(under, head, _face())
    # Just outside the face the head already looks like what the face shows there.
    assert colour[100, 40, 1].astype(int) < colour[100, 160, 1].astype(int)


def test_eyes_and_mouth_do_not_count():
    head = _head()
    under = np.full((200, 200, 3), (120, 150, 200), np.uint8)
    under[80:100, 80:120] = 10   # a dark eye under the same light
    colour = Harmony().relit(under, head, _face())
    assert np.abs(colour[30, 30].astype(int) - (120, 150, 200)).max() <= 3


def test_a_wrong_measure_never_paints_the_head_black():
    under = np.zeros((200, 200, 3), np.uint8)
    colour = Harmony().relit(under, _head(colour=(200, 200, 200)), _face())
    assert colour.min() >= 0.3 * 200


def test_the_light_changes_gradually_and_is_kept_while_the_face_is_covered(monkeypatch):
    monkeypatch.setattr(harmony, "EVERY", 1)
    light = Harmony()
    head = _head(colour=(100, 100, 100))
    light.relit(np.full((200, 200, 3), 100, np.uint8), head, _face())
    second = light.relit(np.full((200, 200, 3), 50, np.uint8), head, _face())
    assert 55 < second[100, 100, 0] < 95          # part of the way, not a jump
    kept = light.relit(np.full((200, 200, 3), 50, np.uint8), head, np.zeros((50, 50), np.uint8))
    assert kept[100, 100, 0] == second[100, 100, 0]
    light.reset()
    fresh = light.relit(np.full((200, 200, 3), 50, np.uint8), head, np.zeros((50, 50), np.uint8))
    assert fresh[100, 100, 0] == 100              # nothing known yet: the head as it is


def test_the_light_is_measured_only_every_few_frames(monkeypatch):
    calls = []
    real = harmony._measure
    monkeypatch.setattr(harmony, "_measure", lambda *args: calls.append(1) or real(*args))
    light = Harmony()
    for _ in range(harmony.EVERY * 2):
        light.relit(np.full((200, 200, 3), 50, np.uint8), _head(), _face())
    assert len(calls) == 2


def test_the_layer_goes_on_with_the_colour_it_was_given():
    frame = np.zeros((40, 40, 3), np.uint8)
    image = np.full((10, 10, 4), 255, np.uint8)
    out = hair.over(frame, image, 5, 5, colour=np.full((10, 10, 3), 77, np.uint8))
    assert out[8, 8, 0] == 77 and out[0, 0, 0] == 0
    assert hair.over(frame, image, 5, 5)[8, 8, 0] == 255


def test_a_layer_moved_to_where_the_face_is_takes_its_face_mask_along():
    from backend.video.avatar import Layer
    from backend.video.live import LiveFaceSession

    mask = np.zeros((25, 25), np.uint8)
    mask[10:15, 10:15] = 1
    layer = Layer(100, 50, np.full((100, 100, 4), 255, np.uint8), face=mask)
    before = np.array([[120, 60], [150, 60], [135, 75], [125, 85], [145, 85]], np.float32)
    moved = LiveFaceSession._moved(layer, before, before + np.array([40.0, 20.0], np.float32))
    assert moved.face is not None and moved.face.sum() > 0
    assert moved.face.shape == (moved.image.shape[0] // 4, moved.image.shape[1] // 4)


def test_the_photos_hair_takes_the_faces_light():
    from backend.video.hair import HairLayer, toned

    image = np.full((64, 64, 3), (120, 150, 200), np.uint8)
    image[:20] = (30, 40, 50)   # the hair
    skin = np.zeros((64, 64), np.float32)
    skin[20:40] = 1.0
    layer = HairLayer(image, np.ones((64, 64), np.float32), (0.0, 0.0), skin)
    face = np.full((64, 64, 3), (60, 75, 200), np.float32)
    lit = toned(layer, face, np.ones((64, 64), np.float32))
    assert abs(int(lit[30, 30, 1]) - 75) <= 3     # the forehead like the face
    assert lit[5, 5, 1] < image[5, 5, 1]          # the hair dimmed by the same light


def test_the_virtual_camera_waits_for_a_first_frame_before_keeping_time(monkeypatch):
    """pyvirtualcam fails if asked to sleep before anything was sent: that killed the thread."""
    import backend.video.worker as worker

    class Camera:
        device = "test"
        fps = 200

        def __init__(self) -> None:
            self.sent = 0
            self.got_frame = threading.Event()

        def send(self, _frame) -> None:
            self.sent += 1
            self.got_frame.set()

        def sleep_until_next_frame(self) -> None:
            assert self.sent, "asked to keep time before any frame"
            time.sleep(0.005)

        def close(self) -> None:
            pass

    camera = Camera()
    virtual = worker._VirtualCamera.__new__(worker._VirtualCamera)
    virtual.camera, virtual.device, virtual.size, virtual.delay = camera, "test", (8, 8), 0.0
    virtual._queue, virtual._lock, virtual._stop = [], threading.Lock(), threading.Event()
    failures = []
    monkeypatch.setattr(threading, "excepthook", lambda args: failures.append(args.exc_value))
    virtual._thread = threading.Thread(target=virtual._run, daemon=True)
    virtual._thread.start()
    time.sleep(0.05)      # no frame yet
    assert virtual._thread.is_alive()
    virtual.push(time.monotonic(), np.zeros((8, 8, 3), np.uint8))
    assert camera.got_frame.wait(2)
    virtual.close()
    assert not failures


def test_cv2_still_multiplies_fixed_point_as_the_light_needs():
    picture = np.full((4, 4, 3), 200, np.uint8)
    ratio = np.full((4, 4, 3), harmony.FIXED // 2, np.uint8)
    assert cv2.multiply(picture, ratio, scale=1 / harmony.FIXED)[0, 0, 0] == 100
