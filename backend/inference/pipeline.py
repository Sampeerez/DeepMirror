"""RVC voice conversion: 16 kHz speech in, target-voice speech out.

A faithful, readable re-implementation of upstream RVC's ``Pipeline``:

1. 48 Hz high-pass (removes rumble/DC the content encoder was never trained on)
2. reflect padding (``pad_seconds`` per side) so model edges settle
3. F0 over the padded signal, shifted and quantised for the pitch embedding
4. HuBERT content features, optionally blended with the voice's index
5. ``protect``: keep raw features on unvoiced frames (consonants, breaths)
6. synthesis, edge trimming, and RMS envelope mixing with the input

Long inputs (files) are split at the quietest points so memory stays bounded,
exactly as upstream does. Output is float32 at the voice's sample rate; no
int16 quantisation or per-call peak normalisation, so consecutive streaming
windows keep a consistent level.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from scipy import ndimage, signal
from torch.nn import functional as F

from backend.inference.device import ComputeDevice
from backend.inference.hubert import HubertModel
from backend.inference.pitch import (
    HOP,
    SAMPLE_RATE,
    PitchExtractor,
    expand_intonation,
    normalize_method,
    to_rvc_pitch,
)
from backend.inference.retrieval import FeatureIndex
from backend.inference.voice_model import VoiceModel

_HIGH_PASS = signal.butter(N=5, Wn=48, btype="high", fs=SAMPLE_RATE, output="sos")


@dataclass(frozen=True, slots=True)
class ConversionSettings:
    """Per-request conversion controls (RVC's standard knobs)."""

    pitch_shift: float = 0.0       # semitones
    f0_method: str = "rmvpe"
    index_rate: float = 0.75       # 0 = raw content features, 1 = fully retrieved
    filter_radius: int = 3         # median filter for Harvest/DIO
    rms_mix_rate: float = 0.25     # 1 = keep model loudness, 0 = follow the input envelope
    protect: float = 0.33          # < 0.5 keeps raw features on unvoiced frames
    crepe_hop_length: int = 160
    rmvpe_threshold: float = 0.03
    speaker_id: int = 0
    # Melody: 1 keeps the speaker's pitch movement, > 1 widens it around the centre.
    pitch_expressiveness: float = 1.0
    pitch_center_hz: float | None = None   # the speaker's typical pitch (None: the clip's median)

    def with_payload(self, payload: dict[str, Any]) -> ConversionSettings:
        """Apply the valid, clamped fields of a client payload; ignore the rest.

        Client keys: ``pitch_shift``, ``f0_method``, ``index_rate``,
        ``filter_radius``, ``rms_mix_rate``, ``protect``, ``crepe_hop_length``,
        ``speaker_id``. ``None`` means "keep the current value".
        """
        changes: dict[str, Any] = {}
        for name, (cast, low, high) in CONTROL_RANGES.items():
            value = clamped_number(payload.get(name), cast, low, high)
            if value is not None:
                changes[name] = value
        method = payload.get("f0_method")
        if isinstance(method, str):
            with contextlib.suppress(ValueError):
                changes["f0_method"] = normalize_method(method)
        return replace(self, **changes) if changes else self


# Client-facing control -> (type, minimum, maximum).
CONTROL_RANGES: dict[str, tuple[type, float, float]] = {
    "pitch_shift": (float, -24.0, 24.0),
    "index_rate": (float, 0.0, 1.0),
    "filter_radius": (int, 0, 7),
    "rms_mix_rate": (float, 0.0, 1.0),
    "protect": (float, 0.0, 0.5),
    "crepe_hop_length": (int, 64, 512),
    "speaker_id": (int, 0, 10_000),
    "pitch_expressiveness": (float, 0.5, 2.0),
}


def clamped_number(value: Any, cast: type, low: float, high: float) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = cast(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(number, float) and not math.isfinite(number):
        return None
    return cast(min(max(number, low), high))


def rms_envelope(audio: np.ndarray, sample_rate: int, length: int) -> torch.Tensor:
    """Half-second RMS envelope (centred frames) linearly stretched to ``length``."""
    frame = sample_rate // 2 * 2
    hop = sample_rate // 2
    padded = np.pad(audio.astype(np.float64), frame // 2)
    count = 1 + (len(padded) - frame) // hop
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=(count, frame), strides=(padded.strides[0] * hop, padded.strides[0]))
    rms = torch.from_numpy(np.sqrt(np.mean(windows ** 2, axis=1)).astype(np.float32))
    return F.interpolate(rms[None, None], size=length, mode="linear").squeeze()


def sliding_rms(audio: np.ndarray, sample_rate: int, length: int, frame_seconds: float) -> torch.Tensor:
    """Shift-invariant RMS (moving window), sampled every 10 ms and stretched to ``length``.

    Unlike :func:`rms_envelope`, whose frames are anchored to the buffer start,
    this gives the same envelope for the same audio wherever it sits in a
    buffer, so overlapping streaming windows agree on the loudness they apply.
    """
    hop = max(1, sample_rate // 100)
    frame = max(hop, int(frame_seconds * sample_rate))
    power = ndimage.uniform_filter1d(audio.astype(np.float64) ** 2, frame, mode="nearest")
    rms = torch.from_numpy(np.sqrt(np.maximum(power[::hop], 0.0)).astype(np.float32))
    return F.interpolate(rms[None, None], size=length, mode="linear").squeeze()


def mix_rms(source: np.ndarray, source_rate: int, converted: np.ndarray, converted_rate: int,
            rate: float, frame_seconds: float | None = None) -> np.ndarray:
    """Blend the converted loudness envelope towards the source's (upstream ``change_rms``).

    ``frame_seconds=None`` reproduces upstream exactly (half-second frames
    anchored to the buffer); a value selects :func:`sliding_rms`.
    """
    if frame_seconds is None:
        source_env = rms_envelope(source, source_rate, len(converted))
        converted_env = rms_envelope(converted, converted_rate, len(converted))
    else:
        source_env = sliding_rms(source, source_rate, len(converted), frame_seconds)
        converted_env = sliding_rms(converted, converted_rate, len(converted), frame_seconds)
    converted_env = torch.clamp_min(converted_env, 1e-6)
    gain = torch.pow(source_env, 1 - rate) * torch.pow(converted_env, rate - 1)
    return (converted * gain.numpy()).astype(np.float32)


class VoiceConverter:
    """Shared HuBERT + pitch extractor, plus the active voice and its index."""

    def __init__(self, compute: ComputeDevice, hubert: HubertModel, pitch: PitchExtractor,
                 voice: VoiceModel, index: FeatureIndex | None) -> None:
        self.compute = compute
        self.hubert = hubert
        self.pitch = pitch
        self.voice = voice
        self.index = index
        self._dtype = torch.float16 if compute.half else torch.float32
        self._lock = threading.Lock()

    @property
    def output_rate(self) -> int:
        return self.voice.sample_rate

    def convert(self, audio: np.ndarray, settings: ConversionSettings, *, pad_seconds: float | None = None,
                split_long_input: bool = True, rms_frame_seconds: float | None = None,
                tail_seconds: float | None = None,
                on_f0: Callable[[np.ndarray], None] | None = None) -> np.ndarray:
        """Convert 16 kHz mono ``audio``; returns ``len(audio) * sr / 16000`` samples (± one frame).

        ``on_f0`` receives the unshifted pitch track (Hz, 100 frames/s, aligned
        with ``audio``) before synthesis. Thread-safe: concurrent streams and
        file renders take turns on the device.
        """
        with self._lock:
            return self._convert(audio, settings, pad_seconds, split_long_input, rms_frame_seconds, tail_seconds,
                                 on_f0)

    def estimate_f0(self, audio: np.ndarray, settings: ConversionSettings) -> np.ndarray:
        """Unshifted pitch track (Hz, 100 frames/s) of 16 kHz ``audio``."""
        with self._lock:
            return self.pitch.estimate(settings.f0_method, audio, len(audio) // HOP,
                                       filter_radius=settings.filter_radius,
                                       crepe_hop_length=settings.crepe_hop_length,
                                       rmvpe_threshold=settings.rmvpe_threshold)

    def _convert(self, audio: np.ndarray, settings: ConversionSettings, pad_seconds: float | None,
                 split_long_input: bool, rms_frame_seconds: float | None,
                 tail_seconds: float | None, on_f0: Callable[[np.ndarray], None] | None) -> np.ndarray:
        voice = self.voice
        pad_seconds = self.compute.offline_pad_seconds if pad_seconds is None else pad_seconds
        t_pad = round(SAMPLE_RATE * pad_seconds / HOP) * HOP
        t_pad_target = t_pad // HOP * voice.synthesizer.hop_length

        audio = signal.sosfiltfilt(_HIGH_PASS, audio.astype(np.float64)).astype(np.float32)
        cuts = self._split_points(audio) if split_long_input else []
        padded = np.pad(audio, (t_pad, t_pad), mode="reflect")
        frames = len(padded) // HOP

        pitch = pitchf = None
        if voice.use_pitch:
            f0 = self.pitch.estimate(settings.f0_method, padded, frames, filter_radius=settings.filter_radius,
                                     crepe_hop_length=settings.crepe_hop_length,
                                     rmvpe_threshold=settings.rmvpe_threshold)
            if on_f0 is not None:
                pad_frames = t_pad // HOP
                on_f0(f0[pad_frames:frames - pad_frames])
            f0 = expand_intonation(f0, settings.pitch_expressiveness, settings.pitch_center_hz)
            coarse, shifted = to_rvc_pitch(f0, settings.pitch_shift)
            pitch = torch.from_numpy(coarse[:frames]).to(self.compute.device)[None]
            pitchf = torch.from_numpy(shifted[:frames]).to(self.compute.device)[None]

        speaker = torch.tensor([int(np.clip(settings.speaker_id, 0, voice.speaker_count - 1))],
                               device=self.compute.device)
        segments = []
        start = 0
        for cut in [*cuts, None]:
            end = None if cut is None else cut // HOP * HOP + 2 * t_pad + HOP
            frame_slice = slice(start // HOP, None if end is None else (end - HOP) // HOP)
            rendered = self._synthesize(
                padded[start:end],
                None if pitch is None else pitch[:, frame_slice],
                None if pitchf is None else pitchf[:, frame_slice],
                speaker, settings,
                None if tail_seconds is None else int(tail_seconds * 100) + t_pad // HOP)
            segments.append(rendered[t_pad_target:len(rendered) - t_pad_target] if t_pad_target else rendered)
            if cut is not None:
                start = cut // HOP * HOP
        output = np.concatenate(segments)
        if settings.rms_mix_rate != 1:
            output = mix_rms(audio, SAMPLE_RATE, output, voice.sample_rate, settings.rms_mix_rate,
                             rms_frame_seconds)
        return output

    def _split_points(self, audio: np.ndarray) -> list[int]:
        """Quietest sample near every ``center_seconds`` boundary of long inputs."""
        compute = self.compute
        if len(audio) + HOP <= SAMPLE_RATE * compute.max_seconds:
            return []
        padded = np.pad(audio, (HOP // 2, HOP // 2), mode="reflect")
        energy = np.convolve(np.abs(padded), np.ones(HOP), mode="valid")[: len(audio)]
        query = SAMPLE_RATE * compute.query_seconds
        cuts = []
        for t in range(SAMPLE_RATE * compute.center_seconds, len(audio), SAMPLE_RATE * compute.center_seconds):
            window = energy[t - query:t + query]
            cuts.append(t - query + int(np.argmin(window)))
        return cuts

    def _synthesize(self, audio: np.ndarray, pitch: torch.Tensor | None, pitchf: torch.Tensor | None,
                    speaker: torch.Tensor, settings: ConversionSettings,
                    tail_frames: int | None = None) -> np.ndarray:
        """Render one padded segment; with ``tail_frames`` only its end is vocoded.

        The result always spans the whole segment (the skipped start is
        silence), so callers keep the same sample alignment either way.
        """
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).to(self.compute.device, self._dtype)[None]
        features = self.hubert.content(waveform, self.voice.version)
        use_protect = settings.protect < 0.5 and pitchf is not None
        raw = features.clone() if use_protect else None

        if self.index is not None and settings.index_rate > 0:
            retrieved = self.index.blend(features[0]).unsqueeze(0)
            features = retrieved * settings.index_rate + (1 - settings.index_rate) * features

        features = F.interpolate(features.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        frames = len(audio) // HOP
        if features.shape[1] < frames:
            frames = features.shape[1]
            if pitch is not None and pitchf is not None:
                pitch, pitchf = pitch[:, :frames], pitchf[:, :frames]

        if use_protect and raw is not None and pitchf is not None:
            raw = F.interpolate(raw.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            mask = torch.where(pitchf >= 1, 1.0, settings.protect).unsqueeze(-1)[:, : features.shape[1]]
            if mask.shape[1] < features.shape[1]:
                mask = F.pad(mask, (0, 0, 0, features.shape[1] - mask.shape[1]), value=1.0)
            features = (features * mask + raw * (1 - mask)).to(raw.dtype)

        lengths = torch.tensor([frames], device=self.compute.device)
        audio_out = self.voice.synthesizer.infer(features, lengths, pitch, pitchf, speaker, tail_frames)
        rendered = audio_out[0, 0].float().cpu().numpy()
        full_length = features.shape[1] * self.voice.synthesizer.hop_length
        if len(rendered) < full_length:
            rendered = np.concatenate([np.zeros(full_length - len(rendered), np.float32), rendered])
        return rendered
