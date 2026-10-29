"""Face parsing (BiSeNet, FaceFusion's release): which pixel of a face crop is skin, hair, an eye...

Used for masks that follow the face's real outline instead of a soft box (no halo
around the swap), and for the whole head: the hair, ears and neck of the head
being worn, and where the speaker's own hair is.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.video.runtime import cpu_threads, infer, session

SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# BiSeNet's 19 classes (CelebAMask-HQ).
SKIN, BROW_L, BROW_R, EYE_L, EYE_R, GLASSES, EAR_L, EAR_R, EARRING, NOSE, MOUTH, LIP_U, LIP_L, NECK, \
    NECKLACE, CLOTH, HAIR, HAT = range(1, 19)
FACE = (SKIN, BROW_L, BROW_R, EYE_L, EYE_R, GLASSES, NOSE, MOUTH, LIP_U, LIP_L)
HEAD = (*FACE, EAR_L, EAR_R, EARRING, HAIR, HAT)


class FaceParser:
    def __init__(self, path: Path, gpu: bool) -> None:
        self.session = session(path, gpu=gpu, threads=cpu_threads())

    def labels(self, crop: np.ndarray) -> np.ndarray:
        """The class of every pixel of ``crop`` (BGR), at the crop's own size."""
        size = crop.shape[:2][::-1]
        rgb = cv2.resize(crop, (SIZE, SIZE))[:, :, ::-1].astype(np.float32) / 255.0
        blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = infer(self.session, {self.session.get_inputs()[0].name: blob})[0][0]
        labels = logits.argmax(0).astype(np.uint8)
        return cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)


def region(labels: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    """1 where ``labels`` is one of ``classes`` (float32)."""
    return np.isin(labels, classes).astype(np.float32)


def feather(mask: np.ndarray, erode: float = 0.02, soft: float = 0.035) -> np.ndarray:
    """A hard region turned into a seamless blend: pulled in by ``erode`` and faded over ``soft``
    (both in units of the mask size), so the edge falls inside the face, not on its outline.
    """
    size = mask.shape[0]
    hard = (mask > 0.5).astype(np.uint8)
    inside = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    fade = max(1.0, soft * size)
    return np.clip((inside - erode * size) / fade, 0.0, 1.0).astype(np.float32)
