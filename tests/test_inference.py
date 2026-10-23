"""Inference engine: checkpoints, retrieval index, pitch, conversion settings, and
equivalence with the legacy fairseq/rvc stack (golden references, local only)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.inference.checkpoints import CheckpointError, fold_weight_norm, load_rvc_checkpoint
from backend.inference.pipeline import ConversionSettings, mix_rms, sliding_rms
from backend.inference.pitch import (
    fit_frames,
    method_capabilities,
    normalize_method,
    suppress_octave_spikes,
    to_rvc_pitch,
)
from backend.inference.retrieval import FeatureIndex, IndexFormatError, read_index

from .conftest import GOLDEN, ROOT

# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def _checkpoint(**extra) -> dict:
    return {"config": [0] * 18, "weight": {"emb_g.weight": torch.zeros(2, 4)}, "version": "v2", "f0": 1, **extra}


class _Evil:
    def __reduce__(self):
        return (print, ("pwned",))


class TestCheckpointLoading:
    def test_plain_checkpoint_loads_safely(self, tmp_path):
        path = tmp_path / "voice.pth"
        torch.save(_checkpoint(), path)
        assert load_rvc_checkpoint(path)["version"] == "v2"

    def test_numpy_values_are_allowed(self, tmp_path):
        path = tmp_path / "voice.pth"
        torch.save(_checkpoint(sr=np.int64(40000), extra=np.arange(3)), path)
        assert int(load_rvc_checkpoint(path)["sr"]) == 40000

    def test_code_executing_pickles_are_refused(self, tmp_path):
        path = tmp_path / "evil.pth"
        torch.save(_checkpoint(payload=_Evil()), path)
        with pytest.raises(CheckpointError, match="OVC_RVC_ALLOW_UNSAFE_CHECKPOINTS"):
            load_rvc_checkpoint(path)

    def test_training_checkpoints_are_explained(self, tmp_path):
        path = tmp_path / "G_1000.pth"
        torch.save({"model": {}, "iteration": 1}, path)
        with pytest.raises(CheckpointError, match="exported"):
            load_rvc_checkpoint(path)


@pytest.mark.filterwarnings("ignore::FutureWarning")  # legacy weight_norm makes the RVC key layout
def test_fold_weight_norm_matches_torch_remove_weight_norm():
    conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 5, 4))
    positional = torch.nn.utils.weight_norm(torch.nn.Conv1d(6, 6, 8, groups=2), dim=2)
    for module in (conv, positional):
        folded = fold_weight_norm(module.state_dict())
        torch.nn.utils.remove_weight_norm(module)
        torch.testing.assert_close(folded["weight"], module.weight.detach())


# --------------------------------------------------------------------------
# Retrieval index (faiss on-disk format)
# --------------------------------------------------------------------------


def _write_ivf_flat(path: Path, centroids: np.ndarray, lists: list[np.ndarray]) -> np.ndarray:
    """Write an ``IndexIVFFlat`` exactly as faiss does; returns vectors in id order."""
    d = centroids.shape[1]
    vectors = np.concatenate(lists)
    header = lambda ntotal: struct.pack("<iqqq?i", d, ntotal, 1 << 20, 1 << 20, True, 1)  # noqa: E731
    with path.open("wb") as out:
        out.write(b"IwFl" + header(len(vectors)) + struct.pack("<QQ", len(centroids), 1))
        out.write(b"IxF2" + header(len(centroids)) + struct.pack("<Q", centroids.size))
        out.write(centroids.astype("<f4").tobytes())
        out.write(struct.pack("<bQ", 0, 0))  # no direct map
        out.write(b"ilar" + struct.pack("<QQ", len(lists), d * 4) + b"full")
        out.write(struct.pack("<Q", len(lists)) + np.array([len(x) for x in lists], "<u8").tobytes())
        next_id = 0
        for block in lists:
            out.write(block.astype("<f4").tobytes())
            out.write(np.arange(next_id, next_id + len(block), dtype="<i8").tobytes())
            next_id += len(block)
    return vectors


class TestRetrieval:
    def test_parses_ivf_flat_and_finds_exact_neighbours(self, tmp_path):
        rng = np.random.default_rng(0)
        lists = [rng.standard_normal((n, 8)).astype(np.float32) + i * 10 for i, n in enumerate((5, 0, 7))]
        vectors = _write_ivf_flat(tmp_path / "v.index", rng.standard_normal((3, 8)).astype(np.float32), lists)
        data = read_index(tmp_path / "v.index")
        np.testing.assert_array_equal(data.vectors, vectors)
        index = FeatureIndex(data, torch.device("cpu"))
        distances, rows = index.search(torch.from_numpy(vectors[[2, 9]]), k=3)
        assert rows[:, 0].tolist() == [2, 9]
        assert float(distances[:, 0].max()) < 1e-4

    def test_blend_returns_the_query_for_a_stored_vector(self, tmp_path):
        rng = np.random.default_rng(1)
        vectors = _write_ivf_flat(tmp_path / "v.index", np.zeros((1, 4), np.float32),
                                  [rng.standard_normal((20, 4)).astype(np.float32)])
        index = FeatureIndex(read_index(tmp_path / "v.index"), torch.device("cpu"))
        blended = index.blend(torch.from_numpy(vectors[:3]))
        torch.testing.assert_close(blended, torch.from_numpy(vectors[:3]), atol=1e-3, rtol=0)

    def test_rejects_unknown_formats(self, tmp_path):
        (tmp_path / "bad.index").write_bytes(b"IxPQ" + b"\x00" * 64)
        with pytest.raises(IndexFormatError):
            read_index(tmp_path / "bad.index")


# --------------------------------------------------------------------------
# Pitch
# --------------------------------------------------------------------------


class TestPitch:
    def test_single_frame_octave_jump_is_replaced(self):
        assert np.allclose(suppress_octave_spikes(np.array([200, 200, 400, 200, 200], np.float32)), 200)

    def test_glides_and_unvoiced_edges_are_kept(self):
        glide = np.array([200, 220, 245, 270, 300], np.float32)
        np.testing.assert_array_equal(suppress_octave_spikes(glide), glide)
        edges = np.array([0, 0, 200, 0, 400, 400], np.float32)
        np.testing.assert_array_equal(suppress_octave_spikes(edges), edges)

    def test_coarse_pitch_bins(self):
        coarse, f0 = to_rvc_pitch(np.array([0.0, 50.0, 1100.0, 440.0]), 12)
        assert coarse[0] == 1 and coarse[2] == 255 and 1 < coarse[3] < 255
        np.testing.assert_allclose(f0[3], 880.0)

    def test_fit_frames_centres_short_tracks(self):
        np.testing.assert_array_equal(fit_frames(np.ones(2), 5), [0, 0, 1, 1, 0])

    def test_method_names_and_capabilities(self, tmp_path):
        assert normalize_method(" Crepe_Tiny ") == "crepe-tiny"
        with pytest.raises(ValueError):
            normalize_method("rmvpe-onnx")
        caps = {item["id"]: item for item in method_capabilities(tmp_path / "missing.pt")}
        assert caps["pm"]["available"] and not caps["rmvpe"]["available"]
        assert caps["hybrid[crepe+rmvpe]"]["realtime"] is False


# --------------------------------------------------------------------------
# Conversion settings and loudness
# --------------------------------------------------------------------------


class TestConversionSettings:
    def test_payload_values_are_clamped_and_typed(self):
        s = ConversionSettings().with_payload({"index_rate": 3, "filter_radius": "5", "protect": -1,
                                               "pitch_shift": 99, "f0_method": "RMVPE"})
        assert (s.index_rate, s.filter_radius, s.protect, s.pitch_shift, s.f0_method) == (1.0, 5, 0.0, 24.0, "rmvpe")
        assert isinstance(s.filter_radius, int)

    @pytest.mark.parametrize("value", [None, True, "x", float("nan"), float("inf")])
    def test_unusable_values_keep_the_current_setting(self, value):
        assert ConversionSettings(index_rate=0.4).with_payload({"index_rate": value}).index_rate == 0.4

    def test_unknown_f0_method_is_ignored(self):
        assert ConversionSettings(f0_method="pm").with_payload({"f0_method": "nope"}).f0_method == "pm"


def test_sliding_rms_is_shift_invariant():
    rng = np.random.default_rng(2)
    audio = (rng.standard_normal(32000) * np.linspace(0.1, 1, 32000)).astype(np.float32)
    whole = sliding_rms(audio, 16000, len(audio) // 160, 0.2).numpy()
    shifted = sliding_rms(audio[4800:], 16000, (len(audio) - 4800) // 160, 0.2).numpy()
    np.testing.assert_allclose(shifted[20:-20], whole[30 + 20:-20], rtol=0.02)


def test_rms_mix_follows_the_source_envelope():
    source = np.concatenate([np.full(16000, 0.5), np.full(16000, 0.05)]).astype(np.float32)
    converted = np.full(40000 * 2, 0.2, dtype=np.float32)
    mixed = mix_rms(source, 16000, converted, 40000, rate=0.0)
    assert mixed[:20000].mean() > 4 * mixed[-20000:].mean()


# --------------------------------------------------------------------------
# Equivalence with the legacy stack (needs data/golden + the HuBERT/RMVPE assets)
# --------------------------------------------------------------------------

ASSETS = ROOT / "models" / "assets"
needs_golden = pytest.mark.skipif(
    not (GOLDEN / "hubert.npz").exists() or not (ASSETS / "hubert_base.pt").exists(),
    reason="golden references or model assets not available on this machine")


@needs_golden
class TestLegacyEquivalence:
    def test_hubert_features(self):
        from backend.inference.hubert import load_hubert

        golden = np.load(GOLDEN / "hubert.npz")
        audio = torch.from_numpy(np.load(GOLDEN / "input_16k.npy"))[None]
        model = load_hubert(ASSETS / "hubert_base.pt", torch.device("cpu"), False)
        for layer in (9, 12):
            ours = model.extract_features(audio, layer).numpy()
            reference = golden[f"layer{layer}"]
            assert np.abs(ours - reference).max() < 1e-3
            cosine = (ours * reference).sum() / np.linalg.norm(ours) / np.linalg.norm(reference)
            assert cosine > 0.99999

    def test_rmvpe_pitch(self):
        from backend.inference.rmvpe import Rmvpe

        f0 = Rmvpe(ASSETS / "rmvpe.pt", torch.device("cpu"), False).infer(np.load(GOLDEN / "input_16k.npy"))
        np.testing.assert_allclose(f0, np.load(GOLDEN / "f0_rmvpe.npz")["f0"], atol=1e-3)

    @pytest.mark.parametrize("variant", ["v1_f1", "v1_f0", "v2_f1", "v2_f0"])
    def test_synthesizer_variants(self, variant):
        from backend.inference.voice_model import VoiceModel

        golden = np.load(GOLDEN / f"synth_{variant}.npz")
        voice = VoiceModel.load(GOLDEN / f"synth_{variant}.pth", torch.device("cpu"), False)
        phone = torch.from_numpy(golden["phone"])
        pitch = torch.from_numpy(golden["pitch"]) if voice.use_pitch else None
        pitchf = torch.from_numpy(golden["pitchf"]) if voice.use_pitch else None
        torch.manual_seed(0)
        audio = voice.synthesizer.infer(phone, torch.tensor([phone.shape[1]]), pitch, pitchf, torch.tensor([3]))
        assert np.abs(audio.numpy() - golden["audio"]).max() < 1e-5


def test_onnx_decoder_matches_torch_output_shape_and_level(tmp_path):
    from backend.inference.onnx_decoder import onnx_decoder
    from backend.inference.synthesizer import Synthesizer

    torch.manual_seed(0)
    config = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512,
              [16, 16, 4, 4], 4, 256, 40000]
    synthesizer = Synthesizer(config, version="v2", use_pitch=True).eval()
    decode = onnx_decoder(synthesizer, tmp_path / "decoder.onnx")
    assert decode is not None and (tmp_path / "decoder.onnx").is_file()
    z, f0, g = torch.randn(1, 192, 33), torch.full((1, 33), 140.0), torch.randn(1, 256, 1)
    with torch.inference_mode():
        reference = synthesizer.dec(z, f0, g)
    fast = decode(z, f0, g)
    assert fast.shape == reference.shape == (1, 1, 33 * 400)
    assert abs(float(fast.std()) - float(reference.std())) < 0.1 * float(reference.std()) + 1e-4
