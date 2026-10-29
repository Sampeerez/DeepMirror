"""Finding and describing faces: detection (SCRFD), 68 landmarks, identity (ArcFace), alignment.

Templates are the five-point positions (eyes, nose tip, mouth corners) each
model was trained on, in units of the crop size; they match FaceFusion's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.video.runtime import cpu_threads, infer, session

TEMPLATES = {
    "arcface_112_v1": np.array([[0.35473214, 0.45658929], [0.64526786, 0.45658929], [0.50000000, 0.61154464],
                                [0.37913393, 0.77687500], [0.62086607, 0.77687500]], np.float32),
    "arcface_112_v2": np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
                                [0.37097589, 0.82469196], [0.63151696, 0.82325089]], np.float32),
    "arcface_128": np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                             [0.38710391, 0.72160547], [0.61507734, 0.72034453]], np.float32),
    "ffhq_512": np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                          [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32),
}


@dataclass(slots=True)
class Face:
    box: np.ndarray                  # x1, y1, x2, y2 in frame pixels
    points: np.ndarray               # 5 x 2: eyes, nose tip, mouth corners
    score: float
    landmarks: np.ndarray | None = None   # 68 x 2, when computed
    embedding: np.ndarray | None = field(default=None, repr=False)   # ArcFace, unit length

    @property
    def size(self) -> float:
        return float(max(self.box[2] - self.box[0], self.box[3] - self.box[1]))


def alignment(points: np.ndarray, template: str, size: int) -> np.ndarray:
    """Similarity transform (2 x 3) taking ``points`` onto the template scaled to ``size``."""
    target = TEMPLATES[template] * size
    matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), target, method=cv2.LMEDS)
    if matrix is None:   # degenerate points: fall back to the least-squares fit
        matrix = cv2.estimateAffinePartial2D(points.astype(np.float32), target)[0]
    return matrix.astype(np.float32)


def crop(frame: np.ndarray, points: np.ndarray, template: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """The aligned ``size`` x ``size`` face and the matrix that made it."""
    matrix = alignment(points, template, size)
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE), matrix


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    keep = cv2.dnn.NMSBoxes([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes.tolist()],
                            scores.tolist(), 0.0, threshold)
    return [int(i) for i in np.array(keep).reshape(-1)]


class FaceDetector:
    """SCRFD 2.5G: boxes and five points. Light enough for the CPU at camera rate."""

    STRIDES = (8, 16, 32)

    def __init__(self, path: Path, size: int = 320, gpu: bool = False) -> None:
        self.session = session(path, gpu=gpu, threads=2)
        self.size = size
        self._anchors: dict[tuple[int, int, int], np.ndarray] = {}

    def _centres(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height // stride, width // stride)
        if (stride, *key) not in self._anchors:
            ys, xs = np.mgrid[: key[0], : key[1]]
            centres = (np.stack([xs, ys], -1).reshape(-1, 2) * stride).astype(np.float32)
            self._anchors[(stride, *key)] = np.repeat(centres, 2, axis=0)   # two anchors per cell
        return self._anchors[(stride, *key)]

    def detect(self, frame: np.ndarray, threshold: float = 0.5) -> list[Face]:
        faces = self._detect(frame, threshold, 1.0)
        # A face filling the whole picture (leaning into the camera) is found at half scale.
        return faces or self._detect(frame, threshold, 0.5)

    def detect_near(self, frame: np.ndarray, face: Face, threshold: float = 0.5) -> list[Face]:
        """Faces in a square twice the size of ``face`` around it: faster, and finer points."""
        height, width = frame.shape[:2]
        centre = (face.box[:2] + face.box[2:]) / 2
        half = face.size
        x1, y1 = int(max(centre[0] - half, 0)), int(max(centre[1] - half, 0))
        x2, y2 = int(min(centre[0] + half, width)), int(min(centre[1] + half, height))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return []
        found = self._detect(frame[y1:y2, x1:x2], threshold, 0.8)
        offset = np.array([x1, y1], np.float32)
        for item in found:
            item.box = item.box + np.tile(offset, 2)
            item.points = item.points + offset
        return found

    def _detect(self, frame: np.ndarray, threshold: float, zoom: float) -> list[Face]:
        height, width = frame.shape[:2]
        scale = self.size * zoom / max(height, width)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = ((canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outputs = infer(self.session, {self.session.get_inputs()[0].name: blob})
        boxes, points, scores = [], [], []
        for index, stride in enumerate(self.STRIDES):
            score = outputs[index].reshape(-1)
            keep = score >= threshold
            if not keep.any():
                continue
            centres = self._centres(self.size, self.size, stride)[keep]
            distance = outputs[index + 3][keep] * stride
            marks = outputs[index + 6][keep].reshape(-1, 5, 2) * stride
            boxes.append(np.concatenate([centres - distance[:, :2], centres + distance[:, 2:]], 1))
            points.append(marks + centres[:, None, :])
            scores.append(score[keep])
        if not boxes:
            return []
        box, point, score = np.concatenate(boxes) / scale, np.concatenate(points) / scale, np.concatenate(scores)
        return [Face(box[i], point[i], float(score[i])) for i in _nms(box, score, 0.4)]


class FaceLandmarker:
    """68 points from the five (FaceFusion's fan_68_5): mouth and jaw shape for masks, almost free."""

    def __init__(self, path: Path) -> None:
        self.session = session(path, gpu=False, threads=1)

    def landmarks(self, points: np.ndarray) -> np.ndarray:
        matrix = alignment(points, "ffhq_512", 1)
        normalized = cv2.transform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)
        result = infer(self.session, {self.session.get_inputs()[0].name: normalized[None].astype(np.float32)})[0]
        return cv2.transform(result.reshape(1, -1, 2), cv2.invertAffineTransform(matrix)).reshape(-1, 2)


class FaceRecognizer:
    """ArcFace w600k R50 identity embedding (what every swapper is conditioned on)."""

    def __init__(self, path: Path, gpu: bool) -> None:
        self.session = session(path, gpu=gpu, threads=cpu_threads())

    def embedding(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        face, _ = crop(frame, points, "arcface_112_v2", 112)
        blob = ((face[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        vector = infer(self.session, {self.session.get_inputs()[0].name: blob})[0][0]
        return (vector / max(np.linalg.norm(vector), 1e-9)).astype(np.float32)


def main_face(faces: list[Face], previous: Face | None = None) -> Face | None:
    """The face to follow: the one nearest the last one, else the largest."""
    if not faces:
        return None
    if previous is not None:
        centre = (previous.box[:2] + previous.box[2:]) / 2
        nearest = min(faces, key=lambda f: float(np.linalg.norm((f.box[:2] + f.box[2:]) / 2 - centre)))
        if np.linalg.norm((nearest.box[:2] + nearest.box[2:]) / 2 - centre) < 0.5 * previous.size:
            return nearest
    return max(faces, key=lambda f: f.size * f.score)
