"""The clone engine: chosen by what was measured, one level down when it does not keep up."""

from __future__ import annotations

from collections import deque

import numpy as np

from backend.video import head_engines as he
from backend.video.personalive import drive


def test_auto_takes_the_best_that_fits():
    all_of_them = {"personalive", "head", "3d", "hair"}
    assert he.choose("auto", all_of_them, {"head_ms": 15.0}).engine == "head"
    assert he.choose("auto", all_of_them, {"head_ms": 300.0}).engine == "3d"
    assert he.choose("auto", {"head", "hair"}, {"head_ms": 300.0}).engine == "hair"


def test_auto_says_why_and_never_relies_on_what_was_not_measured():
    choice = he.choose("auto", {"head", "3d", "hair"}, {"head_ms": 300.0})
    assert choice.engine == "3d" and "300 ms" in choice.reason
    assert he.choose("auto", {"head", "3d", "hair"}, {}).engine == "3d"


def test_an_engine_that_did_not_keep_up_is_skipped():
    choice = he.choose("auto", {"head", "3d", "hair"}, {"head_ms": 15.0}, {"head"})
    assert choice.engine == "3d" and "could not keep up" in choice.reason
    assert he.choose("auto", {"3d", "hair"}, {}, {"3d"}).engine == "hair"


def test_an_engine_chosen_by_hand_stays_if_it_is_there():
    assert he.choose("head", {"head", "hair"}, {"head_ms": 300.0}).engine == "head"
    assert he.choose("personalive", {"personalive", "hair"}, {}).engine == "personalive"
    assert he.choose("face", {"hair"}, {}).engine == "face"
    missing = he.choose("personalive", {"head", "hair"}, {})
    assert missing.engine == "hair" and "not ready" in missing.reason


def test_levels_go_down_to_the_last():
    assert [he.next_level(level) for level in he.LEVELS] == ["3d", "hair", None]
    assert he.next_level("personalive") is None
    assert he.next_level("face") is None


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def test_watch_goes_down_only_if_the_video_stays_behind():
    clock = Clock()
    watch = he.EngineWatch(clock)
    assert not watch.update(30.0, 12.0, 0)
    clock.now = 5.0
    assert not watch.update(30.0, 29.0, 0)   # it recovered: the count starts again
    clock.now = 6.0
    assert not watch.update(30.0, 12.0, 0)
    clock.now = 6.0 + he.WATCH_SECONDS + 1.0
    assert watch.update(30.0, 12.0, 0)
    assert "12 of 30" in watch.reason


def test_watch_blames_nobody_for_a_slow_camera():
    clock = Clock()
    watch = he.EngineWatch(clock)
    for second in range(60):
        clock.now = float(second)
        assert not watch.update(6.0, 2.0, 0)


def test_watch_goes_down_for_the_voice():
    clock = Clock()
    watch = he.EngineWatch(clock)
    late, went_down = 0.0, False
    for second in range(30):
        clock.now = float(second)
        late += 2.0   # 40% of its chunks
        went_down = watch.update(30.0, 30.0, late)
        if went_down:
            break
    assert went_down and "voice" in watch.reason


def test_a_few_late_chunks_are_tolerated():
    clock = Clock()
    watch = he.EngineWatch(clock)
    for second in range(120):
        clock.now = float(second)
        assert not watch.update(30.0, 30.0, second // 10)   # what the GTX 1660 gives with the hybrid


def test_measured_values_are_kept_per_gpu(tmp_path):
    cache = tmp_path / "cache" / "head_engines.json"
    assert he.load_measured(cache, "A") == {}
    he.store_measured(cache, {"head_ms": 12.0}, "A")
    he.store_measured(cache, {"personalive_ms": 70.0}, "A")
    he.store_measured(cache, {"head_ms": 300.0}, "B")
    assert he.load_measured(cache, "A") == {"head_ms": 12.0, "personalive_ms": 70.0}
    assert he.load_measured(cache, "B") == {"head_ms": 300.0}


# --- A render per frame ------------------------------------------------------------------------


def _layer(t):
    from backend.video.engine import HeadLayer

    return HeadLayer(t, np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2)), np.zeros((2, 2)))


def _session(heads_per_second, layers, remote=None):
    from types import SimpleNamespace

    from backend.video.live import LiveFaceSession

    session = LiveFaceSession.__new__(LiveFaceSession)
    session.engine = SimpleNamespace(wants_head=True, remote=remote)
    session._heads_done = deque(layers, maxlen=8)
    session._head_times = deque([0.0, 1.0 / max(heads_per_second, 1e-3)], maxlen=10)
    return session


def test_half_precision_only_if_it_gives_the_same_picture():
    picture = np.full((8, 8, 3), 120, np.uint8)
    assert he.same_picture(picture, picture + 2)
    assert not he.same_picture(picture, np.zeros((8, 8, 3), np.uint8))   # NaNs come out flat
    assert not he.same_picture(picture, None)


def test_what_the_photo_does_not_show_is_not_worn():
    from types import SimpleNamespace

    import cv2

    from backend.video import head as heads
    from backend.video.engine import FaceEngine
    from backend.video.faces import TEMPLATES

    engine = FaceEngine.__new__(FaceEngine)
    size = heads.OUTPUT
    target = TEMPLATES[heads.HEAD_TEMPLATE] * size
    # A photo that ends a little above the eyes: the crop's top has nothing of it.
    photo = np.zeros((size, size, 3), np.uint8)[int(target[0][1]) - 60:]
    points = target - (0, int(target[0][1]) - 60)
    engine.portrait = photo
    engine._head = SimpleNamespace(set_source=lambda image, found: None)
    engine._head_for = None
    engine._photo_inside = None
    engine.models = SimpleNamespace(detector=SimpleNamespace(detect=lambda image: [
        SimpleNamespace(points=points, size=100, score=1.0)]))

    class Arbiter:
        def video(self, *_, **__):
            return FakeGrant()

    engine.arbiter = Arbiter()
    assert engine._prepare_head() is engine._head
    inside = engine._photo_inside
    assert inside is not None and inside[:20].max() < 0.01 and inside[size // 2, size // 2] > 0.99
    assert cv2.countNonZero((inside > 0.5).astype(np.uint8)) > 0


def test_a_frame_takes_the_render_nearest_in_time():
    session = _session(30.0, [_layer(1.0), _layer(1.1), _layer(1.2)])
    assert session._head_for(1.09).t == 1.1
    assert _session(30.0, [])._head_for(1.0) is None


def test_a_frame_waits_for_its_own_render_only_on_a_gpu_that_keeps_up(monkeypatch):
    from types import SimpleNamespace

    from backend.video import live

    monkeypatch.setattr(live.time, "monotonic", lambda: 1.0)
    record = SimpleNamespace(t=2.0, tracked=SimpleNamespace(face=object()))
    fast = _session(30.0, [_layer(1.9)])
    assert fast._waits_for_head(record, 0.01)
    assert not fast._waits_for_head(record, live.HEAD_WAIT + 0.01)   # never for long
    assert not _session(30.0, [_layer(2.0)])._waits_for_head(record, 0.01)   # it is there
    assert not _session(1.5, [_layer(1.9)])._waits_for_head(record, 0.01)   # this GPU would hold all up
    # PersonaLive comes in fours, later and slower: it may wait more, from fewer per second.
    generated = SimpleNamespace(wait=0.6, keeps_up=8.0)
    assert _session(10.0, [_layer(1.9)], generated)._waits_for_head(record, 0.4)


def test_the_generated_head_is_moved_onto_the_speakers_face():
    from types import SimpleNamespace

    import cv2

    from backend.video import head as heads
    from backend.video.engine import FaceEngine
    from backend.video.faces import TEMPLATES

    target = TEMPLATES[heads.HEAD_TEMPLATE] * heads.OUTPUT
    picture = np.zeros((heads.OUTPUT, heads.OUTPUT, 3), np.uint8)
    away = np.array([30.0, -20.0], np.float32)
    cv2.circle(picture, tuple(int(v) for v in target[2] + away), 6, (255, 255, 255), -1)
    engine = FaceEngine.__new__(FaceEngine)
    found = SimpleNamespace(points=target + away, size=100, score=1.0)
    engine.models = SimpleNamespace(detector=SimpleNamespace(detect=lambda image: [found]))
    moved = engine._on_the_face(picture)
    ys, xs = np.nonzero(moved[..., 0] > 128)
    assert abs(xs.mean() - target[2][0]) < 1.5 and abs(ys.mean() - target[2][1]) < 1.5
    engine.models = SimpleNamespace(detector=SimpleNamespace(detect=lambda image: []))
    assert engine._on_the_face(picture) is picture   # no face found in it: as it is


# --- PersonaLive's process, with a stand-in for the model ---------------------------------------


class Pipe:
    def __init__(self, incoming: list[tuple]) -> None:
        self.incoming = list(incoming)
        self.sent: list[tuple] = []

    def recv(self) -> tuple:
        return self.incoming.pop(0)

    def send(self, message: tuple) -> None:
        self.sent.append(message)


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.chunks: list[int] = []

    def reset(self) -> None:
        self.calls.append("reset")

    def fuse_reference(self, image) -> None:
        self.calls.append("fuse")

    def process_input(self, images):
        self.chunks.append(len(images))
        return np.zeros((len(images), 8, 8, 3), np.float32) + 0.5


class FakeGrant:
    def __enter__(self):
        return True

    def __exit__(self, *_):
        return False


class FakeArbiter:
    def __init__(self) -> None:
        self.asked = 0

    def video(self, _ms, timeout=0.0):
        self.asked += 1
        return FakeGrant()


CROP = np.zeros((8, 8, 3), np.uint8)


def test_drive_answers_in_fours_with_each_frames_number_and_time():
    frames = [("frame", seq, seq / 16.0, CROP) for seq in range(9)]
    pipe = Pipe([("portrait", CROP), *frames, ("stop",)])
    pipeline, arbiter = FakePipeline(), FakeArbiter()
    drive.serve(pipe, pipeline, lambda crops: crops, arbiter, lambda crop: crop)
    assert pipe.sent[0] == ("portrait_ready",)
    answers = [message for message in pipe.sent if message[0] == "frames"]
    assert [[seq for seq, _, _ in answer[1]] for answer in answers] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    seq, t, image = answers[1][1][1]
    assert seq == 5 and t == 5 / 16.0 and image.dtype == np.uint8 and image[0, 0, 0] == 127
    assert pipeline.chunks == [4, 4] and arbiter.asked == 3   # the photo and two chunks, through the arbiter


def test_drive_gives_back_each_chunk_with_the_frames_it_shows():
    # What comes out belongs to the frames handed in three chunks ago; the first three are its fade in.
    frames = [("frame", seq, seq / 16.0, CROP) for seq in range(20)]
    pipe = Pipe([("portrait", CROP), *frames, ("portrait", CROP), *frames[:12], ("stop",)])
    pipeline = FakePipeline()
    pipeline.temporal_adaptive_step = 4
    drive.serve(pipe, pipeline, lambda crops: crops, FakeArbiter(), lambda crop: crop)
    answers = [message for message in pipe.sent if message[0] in ("frames", "portrait_ready")]
    assert answers[0] == ("portrait_ready",) and answers[3] == ("portrait_ready",)
    assert [seq for seq, _, _ in answers[1][1]] == [0, 1, 2, 3]
    assert [seq for seq, _, _ in answers[2][1]] == [4, 5, 6, 7]
    assert len(answers) == 4   # a new photo starts its fade in again


def test_drive_says_when_it_gives_nans():
    class Broken(FakePipeline):
        def process_input(self, images):
            return np.full((len(images), 8, 8, 3), np.nan, np.float32)

    pipe = Pipe([("portrait", CROP), ("measure",), ("stop",)])
    drive.serve(pipe, Broken(), lambda crops: crops, FakeArbiter(), lambda crop: crop)
    assert pipe.sent[1] == ("measured", None)


def test_personalive_is_never_installed_without_being_chosen(tmp_path, monkeypatch):
    from backend.video.personalive.client import PersonaLiveInstaller

    installer = PersonaLiveInstaller(tmp_path)
    started = []
    monkeypatch.setattr(installer, "start", lambda: started.append(True))
    monkeypatch.setattr(installer, "why_not", lambda: None)
    monkeypatch.setattr(installer.install.__class__, "available", lambda self: False)
    installer.start_if_needed()
    assert not started


def test_drive_ignores_frames_before_a_photo_and_starts_again_with_a_new_one():
    pipe = Pipe([("frame", 0, 0.0, CROP), ("portrait", CROP), ("frame", 1, 0.1, CROP), ("portrait", CROP),
                 *[("frame", seq, seq / 16.0, CROP) for seq in range(2, 6)], ("stop",)])
    pipeline = FakePipeline()
    drive.serve(pipe, pipeline, lambda crops: crops, FakeArbiter(), lambda crop: crop)
    answers = [message for message in pipe.sent if message[0] == "frames"]
    assert [[seq for seq, _, _ in answer[1]] for answer in answers] == [[2, 3, 4, 5]]
    assert pipeline.calls == ["reset", "fuse", "reset", "fuse"]


def test_drive_measures_per_frame_and_leaves_the_photo_ready():
    pipe = Pipe([("measure",), ("portrait", CROP), ("measure",), ("stop",)])
    pipeline = FakePipeline()
    drive.serve(pipe, pipeline, lambda crops: crops, FakeArbiter(), lambda crop: crop)
    assert pipe.sent[0] == ("measured", None)   # no photo yet
    assert pipe.sent[2][0] == "measured" and pipe.sent[2][1] >= 0.0
    assert len(pipeline.chunks) == drive.MEASURE_CHUNKS and pipeline.calls[-2:] == ["reset", "fuse"]


def test_tensor_format_is_what_the_model_takes():
    import torch

    crop = np.zeros((4, 4, 3), np.uint8)
    crop[..., 0] = 255   # blue in BGR
    tensor = drive.to_tensor([crop, crop], torch, "cpu")
    assert tuple(tensor.shape) == (2, 3, 4, 4)
    assert float(tensor[0, 2, 0, 0]) == 1.0 and float(tensor[0, 0, 0, 0]) == -1.0   # blue last in RGB
    back = drive.to_crops(np.ones((1, 4, 4, 3), np.float32) * np.array([0.0, 0.0, 1.0], np.float32))
    assert back[0][0, 0].tolist() == [255, 0, 0]
