"""The worn person's hair, on every frame, where the speaker's head is.

Face swappers (inswapper, HyperSwap, GHOST, and so Deep-Live-Cam and FaceFusion) replace the
inside of the face only. Here the hair (and hat) of the face's photo is cut out once with the
face parser and kept in the head crop; every frame places it with that frame's own five points,
so it is always on the head: no older pose, no second face. The speaker's own hair where the new
hair does not reach is painted over with the background around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.video import parser as parsing
from backend.video.faces import alignment
from backend.video.head import HEAD_TEMPLATE, OUTPUT
from backend.video.parser import FaceParser, feather, region

SIZE = OUTPUT
# A frontal photo's hair still fits this much head turn (in eye distances, see engine.head_pose),
# and is gone past the second value.
HAIR_FITS = 0.16
HAIR_GONE = 0.30
# Where the forehead (kept with the hair) begins: this far above the eyes, in eye distances.
FOREHEAD_ABOVE_EYES = 0.55


@dataclass(slots=True)
class HairLayer:
    image: np.ndarray            # uint8 SIZE x SIZE x 3, the photo's head crop
    alpha: np.ndarray            # float32 0..1: its hair, hat and forehead
    pose: tuple[float, float]    # the photo's head turn
    skin: np.ndarray | None = None   # float32 0..1: the forehead part of ``alpha`` (toned to the face)


@dataclass(slots=True)
class OwnHead:
    """The speaker's own hair as last parsed, ready to paint over (in the head crop: it moves with
    each frame's points). The masks are worked out once per parse; a frame only needs the colour."""

    t: float
    known: np.ndarray            # float32 SIZE/4: background pixels the fill takes its colour from
    paint: np.ndarray            # float32 SIZE: where the fill goes (the hair, grown and faded)


def head_crop(frame: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = alignment(points, HEAD_TEMPLATE, SIZE)
    crop = cv2.warpAffine(frame, matrix, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return crop, matrix


def _border_fade(size: int, margin: float = 0.06) -> np.ndarray:
    """1 inside the crop, fading to 0 at its edges (a layer never shows a straight cut)."""
    ramp = np.clip(np.minimum(np.arange(size), np.arange(size)[::-1]) / (size * margin), 0, 1).astype(np.float32)
    return np.minimum(ramp[:, None], ramp[None, :])


def build_hair(portrait: np.ndarray, points: np.ndarray, parser: FaceParser, pose: tuple[float, float]
               ) -> HairLayer | None:
    """The photo's hair layer; None when the photo shows no hair (a bald head keeps the speaker's)."""
    crop, matrix = head_crop(portrait, points)
    labels = parser.labels(crop)
    hair = region(labels, (parsing.HAIR, parsing.HAT))
    if hair.mean() < 0.01:
        return None
    # The forehead goes with the hair (it hardly moves when talking): the hairline is the photo's,
    # and the face swap below it never shows the speaker's own hairline through a gap.
    eyes = cv2.transform(points[:2].reshape(1, -1, 2).astype(np.float32), matrix).reshape(-1, 2)
    brow = float(eyes[:, 1].mean() - FOREHEAD_ABOVE_EYES * np.linalg.norm(eyes[1] - eyes[0]))
    rows = np.arange(SIZE, dtype=np.float32)[:, None]
    above = np.clip((brow - rows) / (SIZE * 0.10), 0, 1)   # a long fade into the face below
    forehead = region(labels, (parsing.SKIN,)) * above
    alpha = feather(np.clip(hair + (forehead > 0.5), 0, 1), erode=0.004, soft=0.018)
    alpha = np.maximum(alpha * np.where(forehead > 0, above, 1.0), 0) * _border_fade(SIZE)
    skin = np.minimum(alpha, feather((forehead > 0.5).astype(np.float32), erode=0.0, soft=0.02))
    return HairLayer(crop, alpha.astype(np.float32), pose, skin.astype(np.float32))


def parse_own(crop: np.ndarray, parser: FaceParser, t: float) -> OwnHead:
    labels = parser.labels(crop)
    hair = region(labels, (parsing.HAIR, parsing.HAT))
    head = region(labels, (*parsing.HEAD, parsing.NECK))
    ring = max(9, int(SIZE * 0.04)) | 1
    # Hair right next to the face (sideburns, the temples) stays: painted over with the
    # background it would show as bright bands beside the cheeks; it reads as part of the hair.
    near = max(9, int(SIZE * 0.07)) | 1
    face = cv2.dilate(region(labels, parsing.FACE), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near, near)))
    hair = hair * (1.0 - face)
    grown = cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring, ring)))
    known = 1.0 - np.clip(cv2.dilate(head, np.ones((5, 5), np.uint8)) + grown, 0, 1)
    paint = np.maximum(grown, feather(cv2.dilate(grown, np.ones((ring, ring), np.uint8)), erode=0.0, soft=0.03))
    small = cv2.resize(known, (SIZE // 4, SIZE // 4), interpolation=cv2.INTER_AREA)
    return OwnHead(t, small.astype(np.float32), paint.astype(np.float32))


def hair_weight(hair: HairLayer, pose: tuple[float, float]) -> float:
    drift = float(np.hypot(pose[0] - hair.pose[0], pose[1] - hair.pose[1]))
    return float(np.clip((HAIR_GONE - drift) / (HAIR_GONE - HAIR_FITS), 0.0, 1.0))


def cover(crop: np.ndarray, own: OwnHead) -> np.ndarray:
    """``crop`` (this frame's head crop) as it would look without the speaker's hair: a
    normalised blur of the background around it (smooth, no streaks). Paste with ``own.paint``."""
    small = cv2.resize(crop, (SIZE // 4, SIZE // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    known = own.known
    total = float(known.sum())
    mean = (small * known[:, :, None]).sum((0, 1)) / total if total > 0 else small.mean((0, 1))
    filled = np.broadcast_to(mean, small.shape).astype(np.float32)
    # Coarse to fine: each scale fills where it has enough background nearby, so far from the
    # background the colour fades to its average instead of to black.
    for sigma in (SIZE // 24, SIZE // 48, SIZE // 96):
        weight = cv2.GaussianBlur(known, (0, 0), max(1, sigma))
        colour = cv2.GaussianBlur(small * known[:, :, None], (0, 0), max(1, sigma))
        trust = np.clip(weight / 0.2, 0, 1)[:, :, None]
        filled = filled * (1 - trust) + colour / np.maximum(weight, 1e-6)[:, :, None] * trust
    filled = np.clip(filled, 0, 255).astype(np.uint8)
    return cv2.resize(filled, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def toned(hair: HairLayer, face: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The hair layer's image with its forehead in the swapped face's skin tone (LAB means),
    so the forehead and the face below it are one skin. ``face`` and ``mask``: a keyframe's crop."""
    if hair.skin is None:
        return hair.image
    weight = mask > 0.5
    skin = hair.skin > 0.5
    if weight.sum() < 64 or skin.sum() < 64:
        return hair.image
    target = cv2.cvtColor(np.clip(face, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[weight].mean(0)
    lab = cv2.cvtColor(hair.image, cv2.COLOR_BGR2LAB).astype(np.float32)
    shift = target - lab[skin].mean(0)
    shifted = cv2.cvtColor(np.clip(lab + shift, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return cv2.blendLinear(shifted, hair.image, hair.skin, 1.0 - hair.skin)
