"""Training data: source audio -> clean, normalised segments -> model features.

Mirrors RVC's preprocessing: 48 Hz high-pass, cuts at pauses, overlapping
3.7 s windows, and a soft peak normalisation that keeps 25% of the original
dynamics. Clipped windows (peak > 2.5 after the high-pass) are dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from backend.audio.resampling import resample
from backend.inference.hubert import HubertModel
from backend.inference.pitch import to_rvc_pitch
from backend.inference.rmvpe import Rmvpe

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 3.7
OVERLAP_SECONDS = 0.3
MIN_SEGMENT_SECONDS = 0.6
SILENCE_DB = -42.0
MIN_PAUSE_SECONDS = 0.4
KEPT_PAUSE_SECONDS = 0.25
# Speech is brought to this level before slicing, so the pause detector
# behaves the same for quiet and loud recordings.
TARGET_SPEECH_DB = -20.0
MAX_PEAK = 0.95

Progress = Callable[[float], None]


@dataclass(slots=True)
class Segment:
    """One training example. ``phone`` is 50 frames/s, pitch tracks are 100 frames/s."""

    audio: np.ndarray            # float32 at the model sample rate
    phone: np.ndarray | None = None   # float16 [frames50, 768]
    pitch: np.ndarray | None = None   # int64 coarse bins [frames100]
    pitchf: np.ndarray | None = None  # float32 Hz [frames100]


def audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(rate)


def _frame_rms_db(audio: np.ndarray, rate: int, seconds: float = 0.05) -> np.ndarray:
    frame = max(1, int(rate * seconds))
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0)
    power = np.mean(np.square(audio[: count * frame].astype(np.float64)).reshape(count, frame), axis=1)
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def normalise_loudness(audio: np.ndarray, rate: int) -> np.ndarray:
    """Scale so the louder half of the frames (the speech) sits near ``TARGET_SPEECH_DB``."""
    levels = _frame_rms_db(audio, rate)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if len(levels) == 0 or peak < 1e-5:
        return audio
    speech_db = float(np.median(levels[levels >= np.median(levels)]))
    gain = min(10 ** ((TARGET_SPEECH_DB - speech_db) / 20), MAX_PEAK / peak)
    return (audio * gain).astype(np.float32)


def assess_recording(audio: np.ndarray, rate: int) -> dict:
    """Facts about a source recording that predict training quality."""
    levels = _frame_rms_db(audio, rate)
    clipped = float(np.mean(np.abs(audio) >= 0.999)) * 100 if audio.size else 0.0
    if len(levels) == 0:
        return {"clipped_percent": round(clipped, 3), "noise_floor_db": None, "snr_db": None}
    floor = float(np.percentile(levels, 10))
    speech = float(np.percentile(levels, 90))
    return {"clipped_percent": round(clipped, 3), "noise_floor_db": round(floor, 1), "snr_db": round(speech - floor, 1)}


def quality_warnings(report: dict) -> list[str]:
    """Plain-language advice for recordings that will train a weaker voice."""
    warnings = []
    speech = report.get("speech_seconds") or 0.0
    if speech < 180:
        warnings.append(f"Only {speech / 60:.1f} min of speech after removing pauses: 5 min or more gives a "
                        "much steadier voice.")
    if (report.get("clipped_percent") or 0.0) > 0.1:
        warnings.append("Some of the audio is clipped (distorted peaks). Record a little quieter if you can.")
    snr = report.get("snr_db")
    if snr is not None and snr < 25:
        warnings.append("There is noticeable background noise. A quieter room gives a cleaner voice.")
    voiced = report.get("voiced_percent")
    if voiced is not None and voiced < 35:
        warnings.append("Little voiced speech was found. Use spoken or sung voice, without music.")
    return warnings


def split_on_pauses(audio: np.ndarray, rate: int) -> list[np.ndarray]:
    """Cut at pauses longer than ``MIN_PAUSE_SECONDS``, keeping a short margin of silence."""
    hop = max(1, int(rate * 0.015))
    window = 4 * hop
    power = np.convolve(audio.astype(np.float64) ** 2, np.ones(window) / window, mode="same")[::hop]
    silent = np.sqrt(np.maximum(power, 0)) < 10 ** (SILENCE_DB / 20)
    min_run = int(MIN_PAUSE_SECONDS * rate / hop)
    keep = int(KEPT_PAUSE_SECONDS * rate / hop)

    chunks, start, run_start = [], 0, None
    for frame, is_silent in enumerate([*silent.tolist(), False]):
        if is_silent and run_start is None:
            run_start = frame
        elif not is_silent and run_start is not None:
            if frame - run_start >= min_run:
                end = run_start + keep if run_start > 0 else 0
                if end > start:
                    chunks.append(audio[start * hop:end * hop])
                start = max(frame - keep, end)
            run_start = None
    chunks.append(audio[start * hop:])
    return [chunk for chunk in chunks if len(chunk) >= MIN_SEGMENT_SECONDS * rate]


def windows(chunk: np.ndarray, rate: int) -> list[np.ndarray]:
    """Overlapping ``WINDOW_SECONDS`` windows; the tail joins the last window (up to 4 s)."""
    size, step = int(WINDOW_SECONDS * rate), int((WINDOW_SECONDS - OVERLAP_SECONDS) * rate)
    pieces, start = [], 0
    while len(chunk) - start > size + int(OVERLAP_SECONDS * rate):
        pieces.append(chunk[start:start + size])
        start += step
    pieces.append(chunk[start:])
    return [piece for piece in pieces if len(piece) >= MIN_SEGMENT_SECONDS * rate]


def normalise(piece: np.ndarray) -> np.ndarray | None:
    peak = float(np.abs(piece).max()) if piece.size else 0.0
    if peak > 2.5 or peak < 1e-4:
        return None
    return (piece / peak * (0.9 * 0.75) + 0.25 * piece).astype(np.float32)


def prepare_segments(sources: list[Path], rate: int, progress: Progress | None = None,
                     report: dict | None = None) -> list[Segment]:
    """Clean, slice and normalise ``sources``; ``report`` (if given) receives recording facts."""
    high_pass = signal.butter(N=5, Wn=48, btype="high", fs=rate, output="sos")
    segments: list[Segment] = []
    assessments = []
    for number, path in enumerate(sources):
        audio, source_rate = load_mono(path)
        assessments.append((len(audio) / source_rate, assess_recording(audio, source_rate)))
        audio = np.asarray(signal.sosfilt(high_pass, resample(audio, source_rate, rate)), dtype=np.float32)
        audio = normalise_loudness(audio, rate)
        for chunk in split_on_pauses(audio, rate):
            for piece in windows(chunk, rate):
                normalised = normalise(piece)
                if normalised is not None:
                    segments.append(Segment(normalised))
        if progress:
            progress((number + 1) / len(sources))
    if report is not None:
        total = sum(seconds for seconds, _ in assessments) or 1.0

        def weighted(key: str) -> float | None:
            values = [(seconds, facts[key]) for seconds, facts in assessments if facts.get(key) is not None]
            return round(sum(s * v for s, v in values) / sum(s for s, _ in values), 2) if values else None

        report.update(source_seconds=round(total, 1), clipped_percent=weighted("clipped_percent"),
                      noise_floor_db=weighted("noise_floor_db"), snr_db=weighted("snr_db"),
                      speech_seconds=round(_unique_seconds(segments, rate), 1))
    logger.info("Prepared %d segments (%.1f min) from %d file(s)", len(segments),
                sum(len(s.audio) for s in segments) / rate / 60, len(sources))
    return segments


def _unique_seconds(segments: list[Segment], rate: int) -> float:
    """Speech duration without counting the overlap between consecutive windows twice."""
    return sum(max(0.0, len(s.audio) / rate - OVERLAP_SECONDS) for s in segments) + (OVERLAP_SECONDS if segments else 0)


def voiced_percent(segments: list[Segment]) -> float | None:
    tracks = [s.pitchf for s in segments if s.pitchf is not None]
    if not tracks:
        return None
    frames = np.concatenate(tracks)
    return round(float(np.mean(frames > 0)) * 100, 1)


@torch.no_grad()
def extract_features(segments: list[Segment], rate: int, hubert: HubertModel, rmvpe: Rmvpe,
                     device: torch.device, half: bool, progress: Progress | None = None) -> None:
    """HuBERT v2 content features and RMVPE pitch for every segment, in place."""
    dtype = torch.float16 if half else torch.float32
    for number, segment in enumerate(segments):
        audio16 = resample(segment.audio, rate, 16000)
        waveform = torch.from_numpy(audio16).to(device, dtype)[None]
        segment.phone = hubert.content(waveform, "v2")[0].float().cpu().numpy().astype(np.float16)
        coarse, f0 = to_rvc_pitch(rmvpe.infer(audio16), 0.0)
        segment.pitch, segment.pitchf = coarse, f0
        if progress and (number % 8 == 0 or number == len(segments) - 1):
            progress((number + 1) / len(segments))


def save_segments(segments: list[Segment], path: Path) -> None:
    """Store segments as concatenated arrays plus lengths (plain ``.npz``, no pickles)."""
    def packed(name: str) -> dict[str, np.ndarray]:
        arrays = [getattr(segment, name) for segment in segments]
        return {name: np.concatenate(arrays), f"{name}_lengths": np.array([len(a) for a in arrays])}

    fields = ["audio"] + (["phone", "pitch", "pitchf"] if segments and segments[0].phone is not None else [])
    partial = path.with_name(path.stem + ".part.npz")
    arrays: dict[str, np.ndarray] = {k: v for name in fields for k, v in packed(name).items()}
    with partial.open("wb") as handle:
        np.savez(handle, **arrays)  # type: ignore[arg-type]  # numpy stubs reserve the allow_pickle keyword
    partial.replace(path)


def load_segments(path: Path) -> list[Segment]:
    with np.load(path, allow_pickle=False) as data:
        def unpack(name: str) -> list[np.ndarray] | None:
            if name not in data:
                return None
            return np.split(data[name], np.cumsum(data[f"{name}_lengths"])[:-1])

        audio = unpack("audio") or []
        phone, pitch, pitchf = unpack("phone"), unpack("pitch"), unpack("pitchf")
    return [Segment(audio[i], phone[i] if phone else None, pitch[i] if pitch else None,
                    pitchf[i] if pitchf else None) for i in range(len(audio))]
