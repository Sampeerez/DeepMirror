"""Binary audio frame format shared by the browser and the server.

Layout (little-endian)::

    bytes 0-3   uint32 sequence number
    bytes 4-7   uint32 reserved; responses carry server time in 1/100 ms
    bytes 8-N   float32 mono PCM
"""

from __future__ import annotations

import struct

import numpy as np

HEADER_FORMAT = "<II"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def decode_frame(data: bytes) -> tuple[np.ndarray, int, int]:
    """Return ``(samples, sequence, reserved)``; raises ``ValueError`` on malformed input."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Frame too short: {len(data)} bytes (minimum {HEADER_SIZE})")
    sequence, reserved = struct.unpack_from(HEADER_FORMAT, data)
    payload = memoryview(data)[HEADER_SIZE:]
    if len(payload) % 4:
        raise ValueError(f"Payload size {len(payload)} is not a multiple of 4 (float32 alignment)")
    return np.frombuffer(payload, dtype="<f4").astype(np.float32), sequence, reserved


def encode_frame(samples: np.ndarray, sequence: int, reserved: int = 0) -> bytes:
    header = struct.pack(HEADER_FORMAT, sequence & 0xFFFFFFFF, reserved & 0xFFFFFFFF)
    return header + np.asarray(samples, dtype="<f4").tobytes()
