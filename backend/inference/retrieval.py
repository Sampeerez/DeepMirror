"""Feature retrieval from RVC ``.index`` files without faiss.

RVC trains an ``IndexIVFFlat`` (or ``IndexFlatL2``) over the speaker's HuBERT
features and blends each input frame with its 8 nearest training frames,
which is what makes the converted timbre match the target voice. This module
reads the faiss on-disk format directly and searches with torch:

* ``exact`` search (default) compares against every stored vector — better
  neighbours than IVF with ``nprobe=1``, and fast on a GPU;
* ``ivf`` search reproduces faiss' approximate result (nearest coarse
  centroids, then only their inverted lists).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

logger = logging.getLogger(__name__)

_NEIGHBOURS = 8


class IndexFormatError(ValueError):
    """The file is not a faiss index layout this reader supports."""


@dataclass(slots=True)
class IndexData:
    vectors: np.ndarray            # [ntotal, d] float32, stored order
    centroids: np.ndarray | None   # [nlist, d] for IVF indexes
    list_offsets: np.ndarray | None  # [nlist + 1] start of each inverted list in ``vectors``
    ids: np.ndarray                # [ntotal] faiss id of each stored row
    nprobe: int


class _Reader:
    """Sequential little-endian reader over an open binary file."""

    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle

    def _exact(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise IndexFormatError("index file is truncated")
        return data

    def take(self, fmt: str) -> tuple:
        return struct.unpack(fmt, self._exact(struct.calcsize(fmt)))

    def fourcc(self) -> bytes:
        return self._exact(4)

    def array(self, dtype: type[np.generic], count: int) -> np.ndarray:
        values = np.fromfile(self.handle, dtype=np.dtype(dtype).newbyteorder("<"), count=int(count))
        if values.size != int(count):
            raise IndexFormatError("index file is truncated")
        return values

    def header(self) -> tuple[int, int, int]:
        d, ntotal = self.take("<iq")
        self.take("<qq?")  # two dummy fields + is_trained
        (metric,) = self.take("<i")
        if metric > 1:
            self.take("<f")
        return d, ntotal, metric


def _read_flat(reader: _Reader) -> tuple[int, np.ndarray]:
    d, ntotal, metric = reader.header()
    if metric != 1:
        raise IndexFormatError("only L2 indexes are supported")
    (count,) = reader.take("<Q")
    if count != d * ntotal:
        raise IndexFormatError("flat index size mismatch")
    return d, reader.array(np.float32, count).reshape(ntotal, d)


def read_index(path: Path) -> IndexData:
    """Parse an ``IndexIVFFlat`` (``IwFl``) or ``IndexFlatL2`` (``IxF2``) file."""
    with path.open("rb") as handle:
        reader = _Reader(handle)
        kind = reader.fourcc()
        if kind == b"IxF2":
            _, vectors = _read_flat(reader)
            return IndexData(vectors, None, None, np.arange(len(vectors), dtype=np.int64), 1)
        if kind != b"IwFl":
            raise IndexFormatError(f"unsupported index type {kind!r}")

        d, ntotal, metric = reader.header()
        if metric != 1:
            raise IndexFormatError("only L2 indexes are supported")
        nlist, nprobe = reader.take("<QQ")
        if reader.fourcc() != b"IxF2":
            raise IndexFormatError("IVF quantizer must be a flat L2 index")
        _, centroids = _read_flat(reader)
        (direct_map_type,) = reader.take("<b")
        (direct_map_size,) = reader.take("<Q")
        reader.array(np.int64, direct_map_size)
        if direct_map_type == 2:
            raise IndexFormatError("hashtable direct maps are not supported")

        if reader.fourcc() != b"ilar":
            raise IndexFormatError("only array inverted lists are supported")
        lists, code_size = reader.take("<QQ")
        if lists != nlist or code_size != d * 4:
            raise IndexFormatError("inverted list layout mismatch")
        layout = reader.fourcc()
        sizes = np.zeros(nlist, dtype=np.int64)
        if layout == b"full":
            (count,) = reader.take("<Q")
            sizes[:] = reader.array(np.uint64, count)
        elif layout == b"sprs":
            (count,) = reader.take("<Q")
            pairs = reader.array(np.uint64, count).reshape(-1, 2)
            sizes[pairs[:, 0].astype(np.int64)] = pairs[:, 1]
        else:
            raise IndexFormatError(f"unknown inverted list layout {layout!r}")

        vectors = np.empty((ntotal, d), dtype=np.float32)
        ids = np.empty(ntotal, dtype=np.int64)
        offsets = np.zeros(nlist + 1, dtype=np.int64)
        position = 0
        for list_number in range(nlist):
            size = int(sizes[list_number])
            offsets[list_number] = position
            if size:
                vectors[position:position + size] = reader.array(np.float32, size * d).reshape(size, d)
                ids[position:position + size] = reader.array(np.int64, size)
                position += size
        offsets[nlist] = position
        if position != ntotal:
            raise IndexFormatError("inverted lists do not add up to ntotal")
        return IndexData(vectors, centroids, offsets, ids, int(nprobe))


class FeatureIndex:
    """Nearest-neighbour feature blending for one voice model."""

    def __init__(self, data: IndexData, device: torch.device, exact: bool = True) -> None:
        self.exact = exact or data.centroids is None
        self.nprobe = data.nprobe
        self.device = device
        self.vectors = torch.from_numpy(data.vectors).to(device)
        self._vector_norms = (self.vectors * self.vectors).sum(dim=1)
        self.centroids = torch.from_numpy(data.centroids).to(device) if data.centroids is not None else None
        self.list_offsets = data.list_offsets

    @classmethod
    def load(cls, path: Path, device: torch.device, exact: bool = True) -> FeatureIndex:
        data = read_index(path)
        logger.info("Loaded index %s: %d vectors x %d dims", path.name, *data.vectors.shape)
        return cls(data, device, exact)

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, queries: torch.Tensor, k: int = _NEIGHBOURS) -> tuple[torch.Tensor, torch.Tensor]:
        """Squared L2 distances and row numbers of the ``k`` nearest stored vectors.

        Candidates are ranked with the fast ``|q|^2 - 2 q.v + |v|^2`` expansion,
        whose cancellation error is visible on large vectors; the distances of
        the chosen neighbours (which set the blend weights) are then recomputed
        exactly.
        """
        queries = queries.float().to(self.device)
        if self.exact:
            _, rows = self._exact(queries, self.vectors, self._vector_norms, k)
        else:
            _, rows = self._ivf(queries, k)
        neighbours = self.vectors[rows.clamp_min(0)]
        distances = (neighbours - queries.unsqueeze(1)).square().sum(dim=-1)
        distances = torch.where(rows < 0, torch.full_like(distances, float("inf")), distances)
        return distances, rows

    @staticmethod
    def _exact(queries: torch.Tensor, vectors: torch.Tensor, norms: torch.Tensor,
               k: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(k, vectors.shape[0])
        block = 65536
        query_norms = (queries * queries).sum(dim=1, keepdim=True)
        best_d = torch.empty((queries.shape[0], 0), device=queries.device)
        best_i = torch.empty((queries.shape[0], 0), dtype=torch.long, device=queries.device)
        for start in range(0, vectors.shape[0], block):
            part = vectors[start:start + block]
            distances = (query_norms - 2.0 * queries @ part.T + norms[start:start + block]).clamp_min_(0)
            d, i = torch.topk(distances, min(k, part.shape[0]), dim=1, largest=False)
            merged_d = torch.cat([best_d, d], dim=1)
            merged_i = torch.cat([best_i, i + start], dim=1)
            best_d, order = torch.topk(merged_d, min(k, merged_d.shape[1]), dim=1, largest=False)
            best_i = torch.gather(merged_i, 1, order)
        return best_d, best_i

    def _ivf(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.centroids is not None and self.list_offsets is not None
        centroid_norms = (self.centroids * self.centroids).sum(dim=1)
        _, probes = self._exact(queries, self.centroids, centroid_norms, self.nprobe)
        distances = torch.full((queries.shape[0], k), float("inf"), device=self.device)
        rows = torch.full((queries.shape[0], k), -1, dtype=torch.long, device=self.device)
        for q in range(queries.shape[0]):
            candidates = torch.cat([
                torch.arange(int(self.list_offsets[c]), int(self.list_offsets[c + 1]), device=self.device)
                for c in probes[q].tolist()
            ])
            if candidates.numel() == 0:
                continue
            d, i = self._exact(queries[q:q + 1], self.vectors[candidates], self._vector_norms[candidates], k)
            distances[q, :d.shape[1]] = d[0]
            rows[q, :i.shape[1]] = candidates[i[0]]
        return distances, rows

    def blend(self, features: torch.Tensor, k: int = _NEIGHBOURS) -> torch.Tensor:
        """Inverse-square-distance weighted mean of each frame's neighbours.

        ``features`` is ``[frames, dims]``; the result has the same shape and dtype.
        """
        distances, rows = self.search(features, k)
        weights = 1.0 / distances.clamp_min(1e-12).square()
        weights = weights / weights.sum(dim=1, keepdim=True)
        neighbours = self.vectors[rows.clamp_min(0)]
        blended = (neighbours * weights.unsqueeze(-1)).sum(dim=1)
        return blended.to(features.dtype).to(features.device)


def write_ivf_flat(path: Path, centroids: np.ndarray, vectors: np.ndarray, assignments: np.ndarray) -> None:
    """Write an ``IndexIVFFlat`` (L2, nprobe=1) exactly as faiss serialises it.

    ``assignments[i]`` is the inverted list of ``vectors[i]``. The file loads
    in faiss itself and in other RVC tools, not only in :func:`read_index`.
    """
    d = vectors.shape[1]
    nlist = len(centroids)

    def header(ntotal: int) -> bytes:
        return struct.pack("<iqqq?i", d, ntotal, 1 << 20, 1 << 20, True, 1)

    order = np.argsort(assignments, kind="stable")
    sizes = np.bincount(assignments, minlength=nlist).astype("<u8")
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("wb") as out:
        out.write(b"IwFl" + header(len(vectors)) + struct.pack("<QQ", nlist, 1))
        out.write(b"IxF2" + header(nlist) + struct.pack("<Q", centroids.size))
        out.write(np.ascontiguousarray(centroids, dtype="<f4").tobytes())
        out.write(struct.pack("<bQ", 0, 0))  # no direct map
        out.write(b"ilar" + struct.pack("<QQ", nlist, d * 4) + b"full")
        out.write(struct.pack("<Q", nlist) + sizes.tobytes())
        start = 0
        for size in sizes.tolist():
            rows = order[start:start + size]
            if size:
                out.write(np.ascontiguousarray(vectors[rows], dtype="<f4").tobytes())
                out.write(rows.astype("<i8").tobytes())
            start += size
    partial.replace(path)
