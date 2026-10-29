"""Converting a whole video file: the face (in the face process), the voice (here), then one MP4.

The audio is converted with the same render as the audio Converter, so the
voice keeps its length; video and audio are muxed on the file's own clock,
which keeps lips and voice together.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KEEP_SECONDS = 3600          # finished files are deleted after an hour
AUDIO_RATE = 48000


@dataclass(slots=True)
class VideoJob:
    id: str
    name: str
    folder: Path
    state: str = "queued"     # queued | face | voice | mux | done | error
    progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)

    def describe(self) -> dict:
        return {"id": self.id, "name": self.name, "state": self.state, "progress": round(self.progress, 3),
                "error": self.error}

    @property
    def output(self) -> Path:
        return self.folder / "output.mp4"


def read_audio(path: Path) -> np.ndarray | None:
    """The file's sound as mono float32 at 48 kHz, or None when it has none."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
        chunks = []
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else None


def mux(video: Path, audio: np.ndarray | None, target: Path) -> None:
    """``video``'s stream (copied, not re-encoded) with ``audio`` as AAC, into ``target``."""
    import av

    with av.open(str(video)) as source, av.open(str(target), "w") as out:
        in_stream = source.streams.video[0]
        out_video = out.add_stream_from_template(in_stream)
        out_audio = None
        if audio is not None:
            out_audio = out.add_stream("aac", rate=AUDIO_RATE)
            out_audio.layout = "mono"
            out_audio.bit_rate = 192000
        for packet in source.demux(in_stream):
            if packet.dts is None:
                continue
            packet.stream = out_video
            out.mux(packet)
        if out_audio is not None and audio is not None:
            step = 1024
            for start in range(0, len(audio), step):
                block = np.ascontiguousarray(audio[start:start + step]).reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(block, format="flt", layout="mono")
                frame.sample_rate = AUDIO_RATE
                frame.pts = start
                for packet in out_audio.encode(frame):
                    out.mux(packet)
            for packet in out_audio.encode(None):
                out.mux(packet)


class VideoJobs:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "convert"
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> VideoJob | None:
        return self._jobs.get(job_id)

    def _clean(self) -> None:
        now = time.time()
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.state in ("done", "error") and now - job.created > KEEP_SECONDS:
                    shutil.rmtree(job.folder, ignore_errors=True)
                    del self._jobs[job_id]

    def start(self, name: str, data: bytes, suffix: str,
              swap: Callable[[Path, Path, Callable[[float], None]], int] | None,
              voice: Callable[[np.ndarray], np.ndarray] | None) -> VideoJob:
        """Save the upload and convert it in the background: ``swap`` the face, ``voice`` the sound."""
        self._clean()
        job_id = uuid.uuid4().hex[:12]
        folder = self.root / job_id
        folder.mkdir(parents=True)
        source = folder / f"input{suffix}"
        source.write_bytes(data)
        job = VideoJob(job_id, name, folder)
        with self._lock:
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job, source, swap, voice), name=f"video-{job_id}",
                         daemon=True).start()
        return job

    def _run(self, job: VideoJob, source: Path, swap, voice) -> None:
        try:
            frames = job.folder / "frames.mp4"
            if swap is not None:
                job.state = "face"

                def report(done: float) -> None:
                    job.progress = 0.85 * done

                swap(source, frames, report)
            else:
                frames = source
            job.state, job.progress = "voice", 0.85
            audio = read_audio(source)
            if audio is not None and voice is not None:
                audio = voice(audio)
            job.state, job.progress = "mux", 0.95
            mux(frames, audio, job.output)
            job.state, job.progress = "done", 1.0
            logger.info("Converted video %s", job.name)
        except Exception as exc:
            logger.exception("Video conversion failed")
            job.state, job.error = "error", f"{type(exc).__name__}: {exc}"
