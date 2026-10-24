"""HTTP and WebSocket endpoints of the assembled app."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf
from starlette.websockets import WebSocketDisconnect

from backend.audio.protocol import decode_frame, encode_frame

ORIGIN = {"origin": "http://127.0.0.1:5173"}


def wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(int(seconds * rate)) / rate
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate, format="WAV")
    return buffer.getvalue()


class TestSystem:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_config_describes_stream_device_and_assets(self, client):
        config = client.get("/api/config").json()
        # The test app runs on CPU, so it advertises the realtime-safe CPU stream.
        assert config["sample_rate"] == 48000 and config["chunk_size"] == 16384
        assert config["stream"]["lookahead_seconds"] == 0.1 and config["stream"]["lookback_seconds"] == 0.5
        assert config["device"]["device"] == "cpu"
        assert {asset["id"] for asset in config["assets"]} == {"hubert", "rmvpe", "vad"}
        assert config["default_f0_method"] == "pm"  # no accelerator and no RMVPE weights here
        assert any(method["id"] == "rmvpe" for method in config["f0_methods"])


class TestCors:
    def test_allows_a_configured_origin(self, client):
        response = client.get("/health", headers=ORIGIN)
        assert response.headers["access-control-allow-origin"] == ORIGIN["origin"]

    def test_does_not_echo_an_unlisted_origin(self, client):
        response = client.get("/health", headers={"origin": "https://evil.example"})
        assert "access-control-allow-origin" not in response.headers


class TestWebSocket:
    def test_rejects_a_cross_site_origin(self, client):
        with pytest.raises(WebSocketDisconnect), client.websocket_connect(
                "/ws/audio", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()

    @pytest.mark.parametrize("config", [{"sample_rate": 100}, {"chunk_size": 10**7}])
    def test_rejects_out_of_range_transport(self, client, config):
        with client.websocket_connect("/ws/audio", headers=ORIGIN) as ws:
            ws.send_text(json.dumps(config))
            with pytest.raises(WebSocketDisconnect):
                ws.receive_bytes()

    def test_frames_are_answered_with_converted_audio(self, client, fake_voice):
        with client.websocket_connect("/ws/audio", headers=ORIGIN) as ws:
            ws.send_text(json.dumps({"sample_rate": 48000, "chunk_size": 4096}))
            chunk = (0.1 * np.random.default_rng(0).standard_normal(4096)).astype(np.float32)
            ws.send_bytes(encode_frame(chunk, 7))
            audio, sequence, server_time = decode_frame(ws.receive_bytes())
        assert sequence == 7 and len(audio) == 4096 and server_time > 0
        assert fake_voice.calls == 1

    def test_bypass_over_the_wire_returns_the_input(self, client, fake_voice):
        with client.websocket_connect("/ws/audio", headers=ORIGIN) as ws:
            ws.send_text(json.dumps({"sample_rate": 48000, "chunk_size": 1024}))
            ws.send_text(json.dumps({"bypass": True}))
            chunk = np.linspace(-0.5, 0.5, 1024, dtype=np.float32)
            ws.send_bytes(encode_frame(chunk, 1))
            audio, _, _ = decode_frame(ws.receive_bytes())
        np.testing.assert_array_equal(audio, chunk)
        assert fake_voice.calls == 0


class TestConvert:
    def test_dsp_render_returns_an_inline_wav(self, client):
        response = client.post("/api/convert/", files={"file": ("voz ñ.wav", wav_bytes(), "audio/wav")},
                               data={"pitch_shift": "3"})
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("inline;")
        audio, rate = sf.read(io.BytesIO(response.content))
        assert rate == 16000 and len(audio) == 8000

    def test_model_render_keeps_the_file_length(self, client, fake_voice):
        response = client.post("/api/convert/", files={"file": ("a.wav", wav_bytes(1.0, 44100), "audio/wav")})
        audio, rate = sf.read(io.BytesIO(response.content))
        assert response.status_code == 200 and rate == 44100 and len(audio) == 44100
        assert fake_voice.calls == 1

    def test_empty_and_undecodable_uploads_are_rejected(self, client):
        assert client.post("/api/convert/", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400
        assert client.post("/api/convert/", files={"file": ("a.wav", b"nope", "audio/wav")}).status_code == 400


class TestModels:
    def test_upload_list_and_delete(self, client, settings):
        upload = client.post("/api/models/upload", files={"file": ("My Voice!.pth", b"weights", "application/octet-stream")})
        assert upload.status_code == 200 and upload.json()["name"] == "My_Voice_.pth"
        client.post("/api/models/upload", files={"file": ("My_Voice_.index", b"idx", "application/octet-stream")})
        models = client.get("/api/models/").json()
        assert [(m["name"], m["has_index"]) for m in models] == [("My_Voice_.pth", True)]
        assert client.delete("/api/models/My_Voice_.pth").status_code == 200
        assert client.get("/api/models/").json() == []

    def test_unsupported_types_and_missing_models(self, client):
        response = client.post("/api/models/upload", files={"file": ("x.onnx", b"1", "application/octet-stream")})
        assert response.status_code == 400
        assert client.post("/api/models/missing.pth/activate").status_code == 404
        assert client.delete("/api/models/missing.pth").status_code == 404

    def test_activation_errors_are_reported(self, client):
        client.post("/api/models/upload", files={"file": ("broken.pth", b"not a checkpoint", "application/octet-stream")})
        response = client.post("/api/models/broken.pth/activate")
        assert response.status_code == 500 and "HuBERT" in response.json()["detail"]

    def test_active_model_reports_the_fake_voice(self, client, fake_voice):
        assert client.get("/api/models/active").json()["name"] == "voice.pth"


class TestPresets:
    def test_save_list_and_delete(self, client):
        saved = client.post("/api/presets/", json={"name": "Mine", "settings": {"pitch_shift": 3, "index_rate": 2}})
        assert saved.status_code == 200
        preset = saved.json()
        assert preset["id"] == "user-mine" and preset["settings"]["index_rate"] == 1.0
        listed = client.get("/api/presets/").json()
        assert [p["id"] for p in listed["user"]] == ["user-mine"] and listed["builtin"]
        assert client.delete("/api/presets/user-mine").status_code == 200
        assert client.delete("/api/presets/user-mine").status_code == 404


class TestModelExport:
    def test_download_zips_the_voice_and_its_index(self, client, settings):
        import zipfile as zf

        (settings.MODELS_DIR / "Mi_Voz.pth").write_bytes(b"weights")
        (settings.MODELS_DIR / "Mi_Voz.index").write_bytes(b"index")
        response = client.get("/api/models/Mi_Voz.pth/download")
        assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
        assert "Mi_Voz.zip" in response.headers["content-disposition"]
        with zf.ZipFile(io.BytesIO(response.content)) as archive:
            assert sorted(archive.namelist()) == ["Mi_Voz.index", "Mi_Voz.pth"]
            assert archive.read("Mi_Voz.pth") == b"weights"

    def test_unknown_or_path_like_names_are_404(self, client):
        assert client.get("/api/models/missing.pth/download").status_code == 404
        assert client.get("/api/models/..%2F..%2Fpyproject.toml/download").status_code == 404
