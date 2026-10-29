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
import time
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
    correct_pitch,
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
    # Fast loudness movement of the input (trills, closures, stops) put back on the
    # voice: 0 = off, 1 = all of it. See transfer_articulation.
    articulation: float = 1.0
    # Autotune: 0 = natural pitch, 1 = every note snapped in tune.
    pitch_correction: float = 0.0

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
    "articulation": (float, 0.0, 1.0),
    "pitch_correction": (float, 0.0, 1.0),
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


# Articulation transfer: envelopes on a 2.5 ms grid, fast (8 ms) against slow (60 ms).
ARTICULATION_GRID_HZ = 400
ARTICULATION_FAST_SECONDS = 0.008
ARTICULATION_SLOW_SECONDS = 0.3
ARTICULATION_CUT_DB, ARTICULATION_BOOST_DB = -30.0, 6.0
ARTICULATION_FLOOR_DB = 45.0          # below the loudest moment: the room, left alone
# Only where the input trills: its loudness moves at 18-34 Hz (a rolled r) more than at
# speech rate (2-12 Hz). Speech stays below 0 dB here, a rolled r sits around +10 dB.
TRILL_BAND_HZ, SPEECH_BAND_HZ = (18.0, 34.0), (2.0, 12.0)
TRILL_START_DB, TRILL_FULL_DB = -1.0, 3.0
TRILL_WINDOW_SECONDS = 0.15


def trill_weight(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """0..1 per grid frame: how clearly the input is trilling (fast and slow envelopes from _fast_and_slow)."""
    if len(fast) < 32:
        return np.zeros(len(fast))
    movement = 20 * np.log10(fast / slow)
    rate = ARTICULATION_GRID_HZ
    trill = signal.sosfiltfilt(signal.butter(2, TRILL_BAND_HZ, "band", fs=rate, output="sos"), movement)
    speech = signal.sosfiltfilt(signal.butter(2, SPEECH_BAND_HZ, "band", fs=rate, output="sos"), movement)
    window = max(1, round(TRILL_WINDOW_SECONDS * rate))
    ratio_db = 10 * np.log10((ndimage.uniform_filter1d(trill ** 2, window) + 1e-6)
                             / (ndimage.uniform_filter1d(speech ** 2, window) + 1e-6))
    return np.clip((ratio_db - TRILL_START_DB) / (TRILL_FULL_DB - TRILL_START_DB), 0.0, 1.0)


def _fast_and_slow(audio: np.ndarray, rate: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
    hop = rate // ARTICULATION_GRID_HZ
    power = audio.astype(np.float64) ** 2
    fast = ndimage.uniform_filter1d(power, max(1, round(ARTICULATION_FAST_SECONDS * rate)), mode="nearest")
    slow = ndimage.uniform_filter1d(power, max(1, round(ARTICULATION_SLOW_SECONDS * rate)), mode="nearest")
    fast, slow = np.sqrt(fast[::hop] + 1e-14), np.sqrt(slow[::hop] + 1e-14)
    if len(fast) < frames:
        fast, slow = np.pad(fast, (0, frames - len(fast)), mode="edge"), np.pad(slow, (0, frames - len(slow)), mode="edge")
    return fast[:frames], slow[:frames]


def transfer_articulation(source: np.ndarray, source_rate: int, converted: np.ndarray, converted_rate: int,
                          strength: float = 1.0) -> np.ndarray:
    """Put the source's fast loudness movement (trills, closures, stops) back on the voice.

    HuBERT describes the input 50 times a second, so the model cannot reproduce
    an amplitude movement faster than about 25 Hz: a rolled "rrrr" (closures
    at 20-30 Hz) comes out as a steady vowel, and short stops are smeared. The
    model keeps its own loudness (slow envelope); only the ratio of fast to
    slow envelope is taken from the source, limited to -30..+6 dB.

    Only applied where the input trills (trill_weight): on ordinary speech the
    model's own envelope sounds more natural and closer to the voice (measured
    on the user's takes: speaker similarity 0.50 -> 0.55, DNSMOS 2.4 -> 3.1).
    """
    if strength <= 0 or len(converted) == 0 or len(source) == 0:
        return converted
    frames = len(converted) // (converted_rate // ARTICULATION_GRID_HZ)
    if frames < 4:
        return converted
    in_fast, in_slow = _fast_and_slow(source, source_rate, frames)
    weight = trill_weight(in_fast, in_slow)
    if weight.max() <= 0:
        return converted
    out_fast, out_slow = _fast_and_slow(converted, converted_rate, frames)
    gain_db = 20 * np.log10((in_fast / in_slow) / (out_fast / out_slow))
    gain_db = np.clip(gain_db * strength * weight, ARTICULATION_CUT_DB, ARTICULATION_BOOST_DB)
    loud = 20 * np.log10(in_slow)
    gain_db[loud < loud.max() - ARTICULATION_FLOOR_DB] = 0.0
    gain_db = ndimage.uniform_filter1d(gain_db, 3, mode="nearest")
    hop = converted_rate // ARTICULATION_GRID_HZ
    positions = (np.arange(len(converted)) - hop / 2) / hop
    gain = 10 ** (np.interp(positions, np.arange(frames), gain_db) / 20)
    return (converted * gain).astype(np.float32)


class VoiceConverter:
    """Shared HuBERT + pitch extractor, plus the active voice and its index."""

    def __init__(self, compute: ComputeDevice, hubert: HubertModel, pitch: PitchExtractor,
                 voice: VoiceModel, index: FeatureIndex | None, lock: threading.Lock | None = None) -> None:
        self.compute = compute
        self.hubert = hubert
        self.pitch = pitch
        self.voice = voice
        self.index = index
        self._dtype = torch.float16 if compute.half else torch.float32
        # Converters sharing one device pass one lock, so their renders take turns.
        self._lock = lock or threading.Lock()
        # Milliseconds spent in each stage of the last conversion (for the stream health log).
        self.timings: dict[str, float] = {}

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
        started = time.perf_counter()
        if voice.use_pitch:
            # Tracked on the real audio only: a mirrored pad confuses the tracker near the
            # window's end (a rolled r kept 39% of its voicing, 98% without the pad).
            pad_frames = t_pad // HOP
            inner = frames - 2 * pad_frames
            f0 = self.pitch.estimate(settings.f0_method, audio, inner, filter_radius=settings.filter_radius,
                                     crepe_hop_length=settings.crepe_hop_length,
                                     rmvpe_threshold=settings.rmvpe_threshold)
            if on_f0 is not None:
                on_f0(f0)
            f0 = np.pad(f0, (pad_frames, frames - inner - pad_frames), mode="edge") if len(f0) else f0
            f0 = expand_intonation(f0, settings.pitch_expressiveness, settings.pitch_center_hz)
            f0 = correct_pitch(f0, settings.pitch_correction, settings.pitch_shift)
            coarse, shifted = to_rvc_pitch(f0, settings.pitch_shift)
            pitch = torch.from_numpy(coarse[:frames]).to(self.compute.device)[None]
            pitchf = torch.from_numpy(shifted[:frames]).to(self.compute.device)[None]
        pitch_done = time.perf_counter()

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
        self.timings = {"pitch": (pitch_done - started) * 1000.0,
                        "voice synthesis": (time.perf_counter() - pitch_done) * 1000.0}
        if settings.rms_mix_rate != 1:
            output = mix_rms(audio, SAMPLE_RATE, output, voice.sample_rate, settings.rms_mix_rate,
                             rms_frame_seconds)
        if settings.articulation > 0:
            output = transfer_articulation(audio, SAMPLE_RATE, output, voice.sample_rate, settings.articulation)
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
