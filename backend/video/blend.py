"""Putting the new face back into the frame: soft masks, the real mouth, occlusion, paste-back."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import cv2
import numpy as np

from backend.video.runtime import cpu_threads, infer, session

# Landmark indices (68-point scheme): the outer lip line.
MOUTH = list(range(48, 60))


@cache
def box_mask(size: int, blur: float = 0.3, padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
             ) -> np.ndarray:
    """A soft-edged rectangle over the aligned crop (FaceFusion's box mask): no hard seams at the border."""
    blur_px = max(1, int(size * 0.5 * blur))
    top, right, bottom, left = (int(size * p) for p in padding)
    mask = np.ones((size, size), np.float32)
    edge = max(blur_px // 2, 1)
    mask[: max(edge, top)] = 0
    mask[-max(edge, bottom):] = 0
    mask[:, : max(edge, left)] = 0
    mask[:, -max(edge, right):] = 0
    if blur_px > 1:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_px * 0.25)
    mask.flags.writeable = False
    return mask


def mouth_mask(landmarks: np.ndarray, matrix: np.ndarray, size: int, grow: float = 0.15) -> np.ndarray:
    """1 over the speaker's own lips (in crop space), feathered: the real mouth keeps every syllable."""
    lips = cv2.transform(landmarks[MOUTH].reshape(1, -1, 2), matrix).reshape(-1, 2)
    centre = lips.mean(0)
    lips = centre + (lips - centre) * (1 + grow)
    mask = np.zeros((size, size), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(lips.astype(np.int32)), 1.0)
    width = float(np.ptp(lips[:, 0])) or 1.0
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, width * 0.12))


class Occluder:
    """XSeg: where hands, glasses or hair cover the face, the face underneath is not replaced."""

    def __init__(self, path: Path, gpu: bool) -> None:
        self.session = session(path, gpu=gpu, threads=cpu_threads())

    def mask(self, face: np.ndarray) -> np.ndarray:
        size = face.shape[0]
        blob = cv2.resize(face, (256, 256)).astype(np.float32)[None] / 255.0
        out = infer(self.session, {self.session.get_inputs()[0].name: blob})[0][0, :, :, 0]
        out = cv2.GaussianBlur(np.clip(out, 0, 1), (0, 0), 5) * 2 - 1   # erode the uncertain edge
        return cv2.resize(np.clip(out, 0, 1), (size, size))


def paste(frame: np.ndarray, face: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``frame`` with ``face`` (aligned crop) blended in through ``mask``; only the face region is touched."""
    size = face.shape[0]
    inverse = cv2.invertAffineTransform(matrix)
    corners = cv2.transform(np.array([[[0, 0], [size, 0], [size, size], [0, size]]], np.float32), inverse)[0]
    height, width = frame.shape[:2]
    x1, y1 = np.floor(corners.min(0)).astype(int)
    x2, y2 = np.ceil(corners.max(0)).astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return frame
    local = inverse.copy()
    local[:, 2] -= (x1, y1)
    region = (x2 - x1, y2 - y1)
    warped = cv2.warpAffine(face, local, region, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(mask, local, region, flags=cv2.INTER_LINEAR, borderValue=0.0)[:, :, None]
    out = frame.copy()
    patch = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (patch + (warped.astype(np.float32) - patch) * alpha).astype(np.uint8)
    return out
