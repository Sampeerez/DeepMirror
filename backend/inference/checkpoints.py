"""Safe loading of PyTorch checkpoints and weight-norm folding.

RVC ``.pth`` files and the HuBERT checkpoint are Python pickles. Unpickling an
arbitrary file runs whatever code its author embedded, so:

* RVC checkpoints load with ``weights_only=True`` (plus data-only numpy
  symbols). Falling back to a full unpickle needs an explicit operator opt-in.
* The HuBERT checkpoint embeds fairseq/omegaconf configuration objects. It is
  accepted only when its SHA-256 matches the published file, and even then it
  is read with a restricted unpickler that turns every non-tensor class into
  an inert placeholder, so no foreign code can run.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import logging
import pickle
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

UNSAFE_CHECKPOINT_HINT = (
    "Only .pth files from a source you trust should ever be loaded that way. "
    "If you trust this one, restart with OVC_RVC_ALLOW_UNSAFE_CHECKPOINTS=true."
)


class CheckpointError(RuntimeError):
    """A checkpoint could not be loaded safely or has an unexpected layout."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy_safe_globals() -> list[Any]:
    """Data-only numpy symbols that some RVC checkpoints pickle alongside tensors."""
    resolved: list[Any] = []
    for module_name, attr in (
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
    ):
        try:
            resolved.append(getattr(importlib.import_module(module_name), attr))
        except (ImportError, AttributeError):
            continue
    try:
        dtypes = importlib.import_module("numpy.dtypes")
    except ImportError:
        return resolved
    resolved.extend(
        value for name in dir(dtypes)
        if name.endswith("DType") and isinstance(value := getattr(dtypes, name), type)
    )
    return resolved


def load_rvc_checkpoint(path: Path, *, allow_unsafe: bool = False) -> dict:
    """Load an RVC voice checkpoint, refusing code-executing pickles by default."""
    try:
        with torch.serialization.safe_globals(_numpy_safe_globals()):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if not allow_unsafe:
            logger.error("Refusing to unpickle %s: %s", path.name, safe_error)
            raise CheckpointError(
                f"{path.name} could not be loaded safely. {UNSAFE_CHECKPOINT_HINT}"
            ) from safe_error
        logger.warning(
            "OVC_RVC_ALLOW_UNSAFE_CHECKPOINTS is set: unpickling %s with full "
            "privileges. Safe load failed with: %s", path.name, safe_error,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "weight" not in checkpoint:
        raise CheckpointError(
            f"{path.name} is not an RVC inference checkpoint (expected 'config' and 'weight'). "
            "Training checkpoints such as G_*.pth must be exported first."
        )
    if "emb_g.weight" not in checkpoint["weight"]:
        raise CheckpointError(f"{path.name} is missing emb_g.weight and cannot be used.")
    return checkpoint


class _Placeholder:
    """Stands in for any non-tensor class found in a trusted checkpoint."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __call__(self, *args: Any, **kwargs: Any) -> _Placeholder:
        return _Placeholder(*args)


class _RestrictedUnpickler(pickle.Unpickler):
    """Resolves only torch storage/tensor rebuild helpers; everything else is inert."""

    _ALLOWED_PREFIXES = ("torch._utils", "torch.storage", "torch._tensor", "collections")

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name.endswith("Storage"):
            return getattr(torch, name)
        if module.startswith(self._ALLOWED_PREFIXES):
            return super().find_class(module, name)
        if module == "torch" and name in {"Size", "device", "float32", "float16"}:
            return getattr(torch, name)
        return _Placeholder


class _RestrictedPickleModule:
    """``pickle_module`` shim so ``torch.load`` uses the restricted unpickler."""

    Unpickler = _RestrictedUnpickler
    __name__ = "restricted_pickle"

    @staticmethod
    def load(file: io.BufferedIOBase, **kwargs: Any) -> Any:
        return _RestrictedUnpickler(file, **kwargs).load()


def load_trusted_state_dict(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    """Return the ``model`` state dict of a hash-pinned fairseq-style checkpoint."""
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise CheckpointError(
            f"{path.name} does not match the expected checksum; delete it so it can be downloaded again."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False,
                            pickle_module=_RestrictedPickleModule)  # type: ignore[arg-type]
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise CheckpointError(f"{path.name} has no model weights.")
    return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}


def fold_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace ``weight_g``/``weight_v`` pairs with the effective ``weight``.

    Equivalent to ``torch.nn.utils.remove_weight_norm`` after loading. The norm
    dimension is the one axis on which ``weight_g`` is not broadcast (0 for
    most layers, 2 for HuBERT's positional convolution).
    """
    folded: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith("weight_v"):
            continue
        if not key.endswith("weight_g"):
            folded[key] = value
            continue
        prefix = key[: -len("_g")]
        g = value.float()
        v = state[prefix + "_v"].float()
        kept = [axis for axis, size in enumerate(g.shape) if size != 1] or [0]
        reduce_axes = [axis for axis in range(v.dim()) if axis not in kept]
        norm = torch.linalg.vector_norm(v, dim=reduce_axes, keepdim=True)
        folded[prefix] = (v * (g / norm)).to(value.dtype)
    return folded
