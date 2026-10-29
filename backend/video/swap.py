"""Face swappers and the face enhancer.

Every swapper takes an aligned crop of the face in the frame and an identity
embedding of the face to wear, and returns the crop with the new face.
Measured on a GTX 1660 SUPER (fp32, ONNX Runtime CUDA): GHOST 29 ms,
HyperSwap 43 ms (stored in fp32, see convert.py), inswapper 68 ms per face.
Identity kept (ArcFace similarity to the face worn, on generated faces):
inswapper 0.80-0.90, GHOST 0.76-0.82, HyperSwap 0.71-0.79, SimSwap 0.58-0.66
(left out); HyperSwap is the sharpest (256 px) and masks hands and hair itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.video.faces import crop
from backend.video.runtime import infer, session


@dataclass(frozen=True, slots=True)
class SwapperSpec:
    id: str
    label: str
    file: str
    kind: str                  # hyperswap | inswapper | ghost
    template: str
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    converter: str | None = None   # model mapping ArcFace to the swapper's own identity space
    quality: int = 0           # higher is better (identity and detail, FaceFusion's and our own comparison)


SWAPPERS = {
    spec.id: spec for spec in (
        SwapperSpec("hyperswap_1b_256", "HyperSwap 1b", "hyperswap_1b_256_fp32.onnx", "hyperswap", "arcface_128", 256,
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), quality=4),
        SwapperSpec("inswapper_128", "inswapper", "inswapper_128.onnx", "inswapper", "arcface_128", 128,
                    (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), quality=3),
        SwapperSpec("ghost_1_256", "GHOST", "ghost_1_256.onnx", "ghost", "arcface_112_v1", 256,
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "crossface_ghost.onnx", quality=2),
    )
}


class FaceSwapper:
    def __init__(self, spec: SwapperSpec, directory: Path, gpu: bool) -> None:
        self.spec = spec
        self.session = session(directory / spec.file, gpu=gpu)
        self._converter = session(directory / spec.converter, gpu=False, threads=1) if spec.converter else None
        self._projection: np.ndarray | None = None
        if spec.kind == "inswapper":   # inswapper's identity projection is its last initializer
            import onnx
            from onnx import numpy_helper

            graph = onnx.load(str(directory / spec.file), load_external_data=False).graph
            self._projection = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)
        self._inputs = {i.name for i in self.session.get_inputs()}

    def identity(self, embedding: np.ndarray) -> np.ndarray:
        """The swapper's own conditioning vector for an ArcFace embedding."""
        vector = embedding.reshape(1, -1).astype(np.float32)
        if self._projection is not None:
            vector = vector @ self._projection
            return (vector / np.linalg.norm(vector)).astype(np.float32)
        if self._converter is not None:
            converted = infer(self._converter, {self._converter.get_inputs()[0].name: vector})[0]
            return converted.astype(np.float32)
        return (vector / np.linalg.norm(vector)).astype(np.float32)

    def swap(self, frame: np.ndarray, points: np.ndarray, identity: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """``(swapped crop, original crop, matrix, face mask or None)`` for the face at ``points``.

        HyperSwap also returns where the face really is (hands and hair in front
        are left out), which is used as an occlusion mask at no extra cost.
        """
        spec = self.spec
        original, matrix = crop(frame, points, spec.template, spec.size)
        mean, std = np.array(spec.mean, np.float32), np.array(spec.std, np.float32)
        blob = ((original[:, :, ::-1].astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
        feeds = {"target": blob.astype(np.float32), "source": identity}
        outputs = infer(self.session, {name: feeds[name] for name in self._inputs})
        result = outputs[0][0].transpose(1, 2, 0)
        if spec.kind in ("hyperswap", "ghost"):
            result = result * std + mean
        swapped = (np.clip(result, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)
        mask = None
        if spec.kind == "hyperswap" and len(outputs) > 1:
            mask = np.clip(outputs[1][0, 0], 0, 1).astype(np.float32)
        return swapped, original, matrix, mask


class FaceEnhancer:
    """GPEN-BFR 256: restores detail (eyes, teeth, skin) on the swapped face."""

    def __init__(self, path: Path, gpu: bool) -> None:
        self.session = session(path, gpu=gpu)
        self.size = self.session.get_inputs()[0].shape[2]

    def enhance(self, face: np.ndarray, blend: float = 0.8) -> np.ndarray:
        resized = cv2.resize(face, (self.size, self.size), interpolation=cv2.INTER_AREA)
        blob = ((resized[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = infer(self.session, {self.session.get_inputs()[0].name: blob})[0][0].transpose(1, 2, 0)
        restored = (np.clip((out + 1) / 2, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)
        restored = cv2.resize(restored, face.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        return cv2.addWeighted(restored, blend, face, 1 - blend, 0)


def match_colour(swapped: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift the swapped face's colour statistics (LAB, inside the mask) onto the original's lighting."""
    weight = mask > 0.5
    if weight.sum() < 64:
        return swapped
    source = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_mean, s_std = source[weight].mean(0), source[weight].std(0) + 1e-3
    t_mean, t_std = target[weight].mean(0), target[weight].std(0) + 1e-3
    # Only the lighting moves: luminance fully, colour cast half way (keeps the new skin tone).
    gain = np.array([t_std[0] / s_std[0], 1.0, 1.0], np.float32)
    shift = np.array([t_mean[0] - s_mean[0] * gain[0], 0.5 * (t_mean[1] - s_mean[1]),
                      0.5 * (t_mean[2] - s_mean[2])], np.float32)
    adjusted = np.clip(source * gain + shift, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
