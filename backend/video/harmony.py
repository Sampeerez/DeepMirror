"""One light on the whole head: the worn head relit, on every frame, by the face it dresses.

The head and the swapped face come out of different models, lit differently. Each was matched to the
camera on its own, and they still disagreed (a pale forehead over a darker, pinker face). So the
head is compared with what is really under it, where both show the same thing: inside the face. The
ratio between them is the light falling on the head, and it multiplies skin, ears and hair alike, as
a real light would: dark hair stays dark and takes the same tint as the face.
"""

from __future__ import annotations

import cv2
import numpy as np

SIDE = 72              # the light is measured on a copy this long (it is smooth)...
EVERY = 3              # ...every so many frames (it changes slowly, and this is on the way of every frame)
LOWEST, HIGHEST = np.log(0.35), np.log(1.8)   # the ratio's limits: a wrong measure never paints the head black
FOLLOW = 0.5           # share of a new measure taken (no flicker)
OUTLIER = 0.45         # how far (log) from the light so far a pixel may be and still count: not eyes, not teeth
MIN_PIXELS = 24        # of the face, below which nothing is measured
REACHES = (0.14, 0.05, 0.025)   # how far each finer correction spreads, in copy widths: the finest closes the seam
FIXED = 128            # the ratio's 1.0 when applied
MOST = 0.25            # the most a correction may move the light (log): a lagging swap must not draw a line
SURE = 0.12            # share of measured pixels around, from which a correction is fully trusted


class Harmony:
    """Follows the light over time. ``reset`` for a new face or a new session."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._field: np.ndarray | None = None
        self._ratio: np.ndarray | None = None   # the field as uint8, applied to each frame
        self._age = 0

    def relit(self, under: np.ndarray, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """``image``'s colour (BGR, of its BGRA, straight alpha) in the light of ``under``, the picture
        below it, with the face already on. ``face`` (0 or 1, any smaller size): where both show the face."""
        colour = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self._age -= 1
        if self._age <= 0:
            self._follow(under, image, face)
            self._age = EVERY
        if self._ratio is None:
            return colour
        # In fixed point (uint8, 1.0 is FIXED): much faster than floats, and this runs on every frame.
        ratio = cv2.resize(self._ratio, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        return cv2.multiply(colour, ratio, scale=1 / FIXED)

    def _follow(self, under: np.ndarray, image: np.ndarray, face: np.ndarray) -> None:
        height, width = image.shape[:2]
        scale = SIDE / max(height, width)
        small = (max(1, round(width * scale)), max(1, round(height * scale)))
        measured = _measure(cv2.resize(under, small, interpolation=cv2.INTER_AREA),
                            cv2.resize(image, small, interpolation=cv2.INTER_AREA)[..., :3],
                            cv2.resize(face, small, interpolation=cv2.INTER_NEAREST))
        if measured is None:   # no face to measure on (covered for a moment): the last light stays
            return
        if self._field is not None:
            measured = cv2.resize(self._field, small, interpolation=cv2.INTER_LINEAR) * (1 - FOLLOW) + measured * FOLLOW
        self._field = measured
        self._ratio = np.clip(np.exp(measured) * FIXED, 0, 255).astype(np.uint8)


def _measure(under: np.ndarray, image: np.ndarray, face: np.ndarray) -> np.ndarray | None:
    """The log ratio between the two, everywhere: the overall light, corrected scale by scale from
    what the face shows, each correction fading out away from the face."""
    ours, theirs = image.astype(np.float32), under.astype(np.float32)
    # The head is rendered in full under its hole: only an empty render is left out.
    inside = (face > 0) & (ours.sum(2) > 45)
    if int(inside.sum()) < MIN_PIXELS:
        return None
    log_ratio = np.log((theirs + 4.0) / (ours + 4.0))
    field = np.full(ours.shape, np.median(log_ratio[inside], axis=0), np.float32)
    for reach in REACHES:
        rest = log_ratio - field
        # Only what agrees with the light so far counts: eyes, brows and mouth never sit in the same place.
        weight = (inside & (np.abs(rest).max(2) < OUTLIER)).astype(np.float32)
        sigma = max(1.0, ours.shape[1] * reach)
        spread = cv2.GaussianBlur(weight, (0, 0), sigma)
        # Brightness only: the tint of a place near the face is the whole face's (per channel, whatever
        # differs there, a brow or an eyelid, came out as a patch of colour on the forehead).
        weighted = np.asarray(rest.mean(2) * weight, np.float32)
        local = cv2.GaussianBlur(weighted, (0, 0), sigma) / np.maximum(spread, 1e-3)
        field += (np.clip(local, -MOST, MOST) * np.clip(spread / SURE, 0, 1))[..., None]
    return np.clip(field, LOWEST, HIGHEST)
