"""Face clone: GPU sharing with the voice, tracking, model choice, blending, library and API."""

from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest

from backend.compute_arbiter import GpuArbiter
from backend.video.blend import box_mask, paste
from backend.video.engine import OneEuro, choose_swapper, keys_per_second
from backend.video.faces import alignment
from backend.video.library import FaceLibrary, FaceNotFoundError

from .conftest import ROOT


@pytest.fixture
def arbiter():
    shared = GpuArbiter()
    yield shared
    shared.close()


class TestArbiter:
    def test_without_the_voice_the_face_has_the_gpu(self, arbiter):
        assert arbiter.gap() is None and arbiter.free_share() == 1.0
        with arbiter.video(50, timeout=0.0) as granted:
            assert granted

    def test_a_swap_never_starts_while_the_voice_renders(self, arbiter):
        arbiter.chunk_arrived()
        with arbiter.voice(), arbiter.video(10, timeout=0.02) as granted:
            assert not granted

    def test_a_swap_fits_only_before_the_next_render(self, arbiter):
        period = 0.1
        for _ in range(4):   # chunks every 100 ms, rendering 40 ms after a 10 ms delay
            arbiter.chunk_arrived()
            time.sleep(0.01)
            with arbiter.voice():
                time.sleep(0.04)
            time.sleep(period - 0.05)
        assert 0.09 < arbiter.period < 0.12
        gap = arbiter.gap()
        assert gap is not None and 30 < gap[0] < 70
        arbiter.chunk_arrived()
        time.sleep(0.01)
        with arbiter.voice():
            time.sleep(0.04)
        # About 50 ms left before the next render: 20 ms fits, 80 ms never does.
        with arbiter.video(20, timeout=0.0) as small:
            assert small
        with arbiter.video(80, timeout=0.05) as large:
            assert not large

    def test_the_voice_waits_for_a_running_swap(self, arbiter):
        started = threading.Event()

        def swap():
            with arbiter.video(30, timeout=0.0):
                started.set()
                time.sleep(0.05)

        thread = threading.Thread(target=swap)
        thread.start()
        started.wait()
        begin = time.monotonic()
        with arbiter.voice():
            waited = time.monotonic() - begin
        thread.join()
        assert waited >= 0.03

    def test_another_process_sees_the_same_state(self, arbiter):
        other = GpuArbiter(arbiter.name)
        arbiter.chunk_arrived()
        with arbiter.voice(), other.video(5, timeout=0.0) as granted:
            assert not granted
        other.close()


def test_one_euro_is_steady_when_still_and_follows_movement():
    smoother = OneEuro()
    rng = np.random.default_rng(0)
    # A 200 px face: 1 px of detector jitter, then a head turn of 10 px a frame (1.5 faces a second).
    still = [smoother(np.array([100.0]) + rng.normal(0, 1.0, 1), i / 30, 200.0)[0] for i in range(60)]
    assert np.std(still[20:]) < 0.5        # the jitter is mostly gone
    moved = [smoother(np.array([100.0 + 10 * i]), (60 + i) / 30, 200.0)[0] for i in range(15)]
    assert moved[-1] > 100 + 10 * 14 - 25  # the move is followed within a few frames


COSTS = {"hyperswap_1b_256": 43.0, "inswapper_128": 68.0, "ghost_1_256": 29.0}   # GTX 1660 SUPER, ms


class TestSwapperChoice:
    def test_one_swap_per_voice_gap_counts(self):
        assert keys_per_second(43, (66, 171)) == pytest.approx(1000 / 171)
        assert keys_per_second(29, (66, 171)) == pytest.approx(1000 / 171)   # two would need 70 ms
        assert keys_per_second(43, None) == pytest.approx(1000 / 43)

    def test_the_best_model_that_fits_wins(self):
        available = list(COSTS)
        assert choose_swapper(COSTS, available, "auto", None) == "hyperswap_1b_256"
        assert choose_swapper(COSTS, available, "auto", (66, 171)) == "hyperswap_1b_256"
        assert choose_swapper(COSTS, available, "auto", (40, 171)) == "ghost_1_256"
        assert choose_swapper(COSTS, available, "inswapper_128", (40, 171)) == "inswapper_128"


def test_paste_blends_only_inside_the_mask():
    frame = np.zeros((200, 200, 3), np.uint8)
    points = np.array([[80, 90], [120, 90], [100, 110], [85, 130], [115, 130]], np.float32)
    matrix = alignment(points, "arcface_128", 128)
    face = np.full((128, 128, 3), 255, np.uint8)
    out = paste(frame, face, box_mask(128, 0.3), matrix)
    assert out[110, 100].mean() > 200           # the face centre is replaced
    assert out[5, 5].sum() == 0 and out[195, 195].sum() == 0   # the rest of the frame is untouched


class TestLibrary:
    def test_faces_are_stored_activated_and_deleted(self, tmp_path):
        library = FaceLibrary(tmp_path)
        face = library.add("Alex", [b"photo"], np.ones(512, np.float32), [{"face": True, "size": 200}], b"thumb")
        assert library.list()[0]["name"] == "Alex" and library.list()[0]["photos"] == 1
        library.set_active(face["id"])
        assert library.active() == face["id"]
        np.testing.assert_array_equal(library.identity(face["id"]), np.ones(512))
        library.delete(face["id"])
        assert library.list() == [] and library.active() is None

    def test_unknown_or_malformed_ids_are_rejected(self, tmp_path):
        library = FaceLibrary(tmp_path)
        for bad in ("../../etc", "nope", "0123456789ab"):
            with pytest.raises(FaceNotFoundError):
                library.identity(bad)


class TestApi:
    def test_status_lists_the_models_to_download(self, client):
        status = client.get("/api/video/").json()
        assert status["models_ready"] is False
        assert {asset["id"] for asset in status["assets"]} >= {"scrfd_2.5g", "hyperswap_1b_256", "arcface_w600k_r50"}
        assert all(asset["id"] not in {"hubert", "rmvpe"} for asset in status["assets"])

    def test_faces_need_the_models_first(self, client):
        response = client.post("/api/video/faces", files={"files": ("a.jpg", io.BytesIO(b"x"), "image/jpeg")})
        assert response.status_code == 409

    def test_settings_are_validated_and_kept(self, client):
        response = client.patch("/api/video/settings", json={"swapper": "ghost_1_256", "mouth": "on",
                                                             "enhance": "sometimes", "occlusion": False})
        settings = response.json()["settings"]
        assert settings == {"swapper": "ghost_1_256", "enhance": "auto", "mouth": "on", "occlusion": False}
        assert client.get("/api/video/").json()["settings"]["swapper"] == "ghost_1_256"

    def test_an_unknown_face_is_a_404(self, client):
        assert client.post("/api/video/faces/0123456789ab/activate").status_code == 404
        assert client.delete("/api/video/faces/0123456789ab").status_code == 404


FACE_DIR = ROOT / "models" / "assets" / "face"


@pytest.mark.skipif(not (FACE_DIR / "scrfd_2.5g.onnx").is_file(), reason="face models not downloaded")
def test_the_detector_finds_a_drawn_face_and_its_five_points():
    import cv2

    from backend.video.faces import FaceDetector

    # A plain synthetic face is not enough for SCRFD; use a generated one if the test assets have it.
    sample = ROOT / "tests" / "data" / "face.jpg"
    if not sample.is_file():
        pytest.skip("no sample face image")
    image = cv2.imread(str(sample))
    faces = FaceDetector(FACE_DIR / "scrfd_2.5g.onnx").detect(image)
    assert len(faces) == 1 and faces[0].points.shape == (5, 2)


def _small_video(path, seconds=1.0):
    import av

    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=10)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = out.add_stream("aac", rate=48000)
        audio.layout = "mono"
        for k in range(int(seconds * 10)):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), k * 20 % 255, np.uint8), format="bgr24")
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        tone = (0.2 * np.sin(np.arange(int(48000 * seconds)) / 48000 * 2 * np.pi * 220)).astype(np.float32)
        for start in range(0, len(tone), 1024):
            frame = av.AudioFrame.from_ndarray(tone[start:start + 1024].reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate, frame.pts = 48000, start
            for packet in audio.encode(frame):
                out.mux(packet)
        for packet in audio.encode(None):
            out.mux(packet)


class TestVideoFiles:
    def test_only_videos_are_accepted_and_a_face_needs_the_models(self, client, tmp_path):
        bad = client.post("/api/convert/video", files={"file": ("a.wav", b"x", "audio/wav")})
        assert bad.status_code == 400
        _small_video(tmp_path / "clip.mp4")
        face = client.post("/api/convert/video", files={"file": ("clip.mp4", (tmp_path / "clip.mp4").read_bytes())},
                           data={"face": "on"})
        assert face.status_code == 409

    def test_a_video_keeps_its_length_and_its_sound(self, client, tmp_path):
        import av

        from backend.video.files import read_audio

        _small_video(tmp_path / "clip.mp4", 1.0)
        job = client.post("/api/convert/video", files={"file": ("clip.mp4", (tmp_path / "clip.mp4").read_bytes())},
                          data={"face": "off", "use_model": "false"}).json()
        for _ in range(100):
            state = client.get(f"/api/convert/video/{job['id']}").json()
            if state["state"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert state["state"] == "done", state
        response = client.get(f"/api/convert/video/{job['id']}/file")
        (tmp_path / "out.mp4").write_bytes(response.content)
        with av.open(str(tmp_path / "out.mp4")) as result:
            assert result.streams.video and result.streams.audio
        audio = read_audio(tmp_path / "out.mp4")
        assert audio is not None and abs(len(audio) / 48000 - 1.0) < 0.1
