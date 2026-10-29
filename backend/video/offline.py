"""Whole video files: every frame swapped (no keyframe shortcut), written as H.264.

Runs in the face process. The voice of the file is converted in the server
and muxed afterwards (see backend.video.files).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from backend.video.engine import FaceEngine

# Offline there is no hurry: wait for GPU gaps as long as it takes (the live voice still comes first).
SWAP_WAIT_SECONDS = 30.0


def swap_video(engine: FaceEngine, source: Path, target: Path, on_progress: Callable[[float], None]) -> int:
    """Write ``source``'s frames with the active face to ``target`` (video only); returns the frame count."""
    import av

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or 30
        total = stream.frames or 0
        duration = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else 0.0
        with av.open(str(target), "w") as out:
            encoder = out.add_stream("libx264", rate=rate)
            encoder.width = stream.codec_context.width // 2 * 2
            encoder.height = stream.codec_context.height // 2 * 2
            encoder.pix_fmt = "yuv420p"
            encoder.options = {"crf": "17", "preset": "veryfast"}
            count = 0
            for frame in container.decode(stream):
                image = frame.to_ndarray(format="bgr24")[: encoder.height, : encoder.width]
                t = float(frame.time or count / float(rate))
                tracked = engine.track(image, t)
                result = image
                if engine.identity is not None and tracked.face is not None:
                    key = None
                    while key is None:   # every frame gets its own swap
                        key = engine.keyframe(image, tracked, t, 30.0, wait=SWAP_WAIT_SECONDS)
                    result = engine.compose(image, tracked, t, key, None, 30.0)
                video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(result, dtype=np.uint8), format="bgr24")
                for packet in encoder.encode(video_frame):
                    out.mux(packet)
                count += 1
                if count % 10 == 0:
                    done = count / total if total else (t / duration if duration else 0.0)
                    on_progress(min(done, 0.999))
            for packet in encoder.encode():
                out.mux(packet)
    return count
