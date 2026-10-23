"""Build a voice's retrieval index from its training features (RVC's ``train_index``)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch

from backend.inference.retrieval import write_ivf_flat

logger = logging.getLogger(__name__)

# Above this many feature frames, RVC replaces them with k-means centroids.
MAX_VECTORS = 200_000
REDUCED_VECTORS = 10_000


def kmeans(vectors: torch.Tensor, clusters: int, iterations: int = 25, seed: int = 0) -> torch.Tensor:
    """Lloyd's k-means with k-means++-style seeding on a random subset; returns centroids."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = vectors.shape[0]
    clusters = min(clusters, count)
    centroids = vectors[torch.randperm(count, generator=generator)[:1].to(vectors.device)]
    sample = vectors[torch.randperm(count, generator=generator)[: min(count, clusters * 40)].to(vectors.device)]
    while centroids.shape[0] < clusters:  # seed proportionally to squared distance
        distance = torch.cdist(sample, centroids).min(dim=1).values.square()
        picks = torch.multinomial(distance + 1e-12, min(clusters - centroids.shape[0], 64), replacement=False,
                                  generator=None)
        centroids = torch.cat([centroids, sample[picks]])
    for _ in range(iterations):
        assignment = assign(vectors, centroids)
        sums = torch.zeros_like(centroids).index_add_(0, assignment, vectors)
        counts = torch.bincount(assignment, minlength=clusters).unsqueeze(1)
        empty = counts.squeeze(1) == 0
        centroids = torch.where(empty.unsqueeze(1), centroids, sums / counts.clamp_min(1))
    return centroids


def assign(vectors: torch.Tensor, centroids: torch.Tensor, block: int = 65536) -> torch.Tensor:
    return torch.cat([torch.cdist(vectors[i:i + block], centroids).argmin(dim=1)
                      for i in range(0, vectors.shape[0], block)])


def build_index(features: list[np.ndarray], path: Path, device: torch.device) -> int:
    """Write ``path`` from per-segment ``[frames, dims]`` features; returns the stored vector count."""
    vectors = torch.from_numpy(np.concatenate(features).astype(np.float32)).to(device)
    vectors = vectors[torch.randperm(vectors.shape[0], device=device)]
    if vectors.shape[0] > MAX_VECTORS:
        logger.info("Reducing %d feature frames to %d centroids", vectors.shape[0], REDUCED_VECTORS)
        vectors = kmeans(vectors, REDUCED_VECTORS, iterations=15)
    count = vectors.shape[0]
    lists = max(1, min(int(16 * math.sqrt(count)), count // 39))
    centroids = kmeans(vectors, lists)
    assignments = assign(vectors, centroids)
    write_ivf_flat(path, centroids.cpu().numpy(), vectors.cpu().numpy(), assignments.cpu().numpy())
    logger.info("Wrote index %s: %d vectors in %d lists", path.name, count, lists)
    return count
