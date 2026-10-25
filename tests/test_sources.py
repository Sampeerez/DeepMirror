"""Where the downloaded pieces come from: newest VB-CABLE, the DeepFilterNet3 export, built assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend import assets as assets_module
from backend.assets import Asset, AssetStore
from backend.virtual_mic import MAC_PACKAGE, WINDOWS_PACKAGE, newest_package
from tests.conftest import ROOT

PAGE = """
<a href="https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip">old</a>
<a href="https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack47.zip">new</a>
<a href="https://evil.example.com/Download_CABLE/VBCABLE_Driver_Pack99.zip">elsewhere</a>
<a href="https://download.vb-audio.com/Download_MAC/VBCable_MACDriver_Pack110.zip">mac</a>
"""


class TestVbCablePackage:
    def test_the_newest_official_package_is_chosen(self):
        windows = newest_package(PAGE, WINDOWS_PACKAGE)
        assert windows.filename == "VBCABLE_Driver_Pack47.zip"
        assert windows.url.startswith("https://download.vb-audio.com/")
        assert windows.sha256 == ""          # unknown: the installer signature is checked instead
        assert newest_package(PAGE, MAC_PACKAGE).filename == "VBCable_MACDriver_Pack110.zip"

    def test_without_a_newer_link_the_known_package_is_kept(self):
        assert newest_package("<html>nothing</html>", WINDOWS_PACKAGE) is WINDOWS_PACKAGE
        older = PAGE.replace("Pack47", "Pack44").replace("Pack110", "Pack100")
        assert newest_package(older, WINDOWS_PACKAGE) is WINDOWS_PACKAGE
        assert newest_package(older, MAC_PACKAGE) is MAC_PACKAGE


class TestBuiltAssets:
    def test_a_downloaded_asset_is_built_before_it_is_ready(self, tmp_path, monkeypatch):
        payload = b"checkpoint"
        import hashlib

        asset = Asset("denoiser", "source.zip", "https://example.invalid/source.zip",
                      hashlib.sha256(payload).hexdigest(), len(payload), False, product="built.onnx")
        monkeypatch.setattr(assets_module, "ASSETS", (asset,))
        built = []

        def build(item, source, target):
            built.append(source.read_bytes())
            target.write_bytes(b"onnx")

        monkeypatch.setattr(assets_module, "build_product", build)
        (tmp_path / "source.zip").write_bytes(payload)
        store = AssetStore(tmp_path)
        store.prepare()
        assert not store.ready("denoiser")                 # downloaded, not built yet
        store._download_missing()
        assert store.ready("denoiser") and built == [payload]
        assert store.path("denoiser") == tmp_path / "built.onnx"
        store = AssetStore(tmp_path)
        store.prepare()
        assert store.ready("denoiser")                     # built once, ready at the next start

    def test_a_tampered_download_is_never_built(self, tmp_path, monkeypatch):
        asset = Asset("denoiser", "source.zip", "https://example.invalid/source.zip", "0" * 64, 4, False,
                      product="built.onnx")
        monkeypatch.setattr(assets_module, "ASSETS", (asset,))
        monkeypatch.setattr(assets_module, "build_product", lambda *a: pytest.fail("built a tampered file"))
        (tmp_path / "source.zip").write_bytes(b"evil")
        store = AssetStore(tmp_path)
        store.prepare()
        store._download_missing()
        assert store.status()[0]["state"] == "error"


CHECKPOINT = ROOT / "models" / "assets" / "DeepFilterNet3.zip"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="the official DeepFilterNet3 checkpoint is not downloaded")
def test_the_official_checkpoint_exports_to_a_working_stream(tmp_path: Path):
    import onnxruntime as ort

    from backend.audio.deepfilter import export_streaming_onnx
    from backend.audio.deepfilter.network import HOP, initial_state

    target = export_streaming_onnx(CHECKPOINT, tmp_path / "dfn.onnx")      # also checks ONNX == PyTorch
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    noise = (0.02 * rng.standard_normal(HOP * 300)).astype(np.float32)
    state, out = initial_state(), []
    for k in range(len(noise) // HOP):
        hop, state, _ = session.run(None, {"input_frame": noise[k * HOP:(k + 1) * HOP], "states": state,
                                           "atten_lim_db": np.array(100.0, np.float32)})
        out.append(np.asarray(hop))
    tail = np.concatenate(out)[HOP * 100:]
    assert 10 * np.log10(np.mean(tail ** 2) / np.mean(noise ** 2)) < -20      # steady noise removed
