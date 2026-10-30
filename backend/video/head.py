"""The whole head: the photo of the face worn, animated by the speaker (LivePortrait).

A face swap only replaces the inside of the face; hair, beard, ears and the
head's outline stay the speaker's. Here the head *is* the photo's: LivePortrait
keeps the photo's 3D appearance and moves its implicit keypoints with the
speaker's pose and expressions (relative to the first frame seen), then warps
and renders it. Hair, beard and skin come from the photo, pixel for pixel.

One render costs about 300 ms on a GTX 1660 SUPER (fp32; fp16 is slower and
unstable there), far more than the gaps the live voice leaves on the GPU. So a
render is cut into *stages*, each short enough to fit one gap, and the caller
runs them one by one inside ``GpuArbiter.video`` (the voice never waits). Live,
the head is refreshed a few times a second and moved with the speaker's head on
every frame; the face swap on top keeps the eyes and mouth live. Files render
every frame.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.video.faces import TEMPLATES, alignment

logger = logging.getLogger(__name__)

INPUT = 256      # what the networks see
MASK_PARTS = 4   # the motion mask's convolution, in this many depth slices
LIP_KEYPOINTS = [6, 12, 14, 17, 19, 20]   # LivePortrait's lip keypoints
OUTPUT = 512     # what the generator draws
FILES = {
    "appearance": "liveportrait_appearance_feature_extractor.pth",
    "motion": "liveportrait_motion_extractor.pth",
    "warping": "liveportrait_warping_module.pth",
    "generator": "liveportrait_spade_generator.pth",
    "stitching": "liveportrait_stitching_retargeting_module.pth",
}
# LivePortrait's own config (src/config/models.yaml).
PARAMS = {
    "appearance": dict(image_channel=3, block_expansion=64, num_down_blocks=2, max_features=512,
                       reshape_channel=32, reshape_depth=16, num_resblocks=6),
    "motion": dict(num_kp=21, backbone="convnextv2_tiny"),
    "warping": dict(num_kp=21, block_expansion=64, max_features=512, num_down_blocks=2, reshape_channel=32,
                    estimate_occlusion_map=True,
                    dense_motion_params=dict(block_expansion=32, max_features=1024, num_blocks=5, reshape_depth=16,
                                             compress=4)),
    "generator": dict(upscale=2, block_expansion=64, max_features=512, num_down_blocks=2),
    "stitching": dict(input_size=126, hidden_sizes=[128, 128, 64], output_size=65),
}
# The head crop: FFHQ's five points pulled towards the centre, so the whole head and its hair fit
# (LivePortrait crops about 2.3 face widths around the face).
HEAD_TEMPLATE = "head"
TEMPLATES[HEAD_TEMPLATE] = ((TEMPLATES["ffhq_512"] - 0.5) * 0.76 + np.array([0.5, 0.54], np.float32)).astype(np.float32)


def ready(directory: Path) -> bool:
    return all((directory / name).is_file() for name in FILES.values())


@dataclass
class Source:
    """The photo, prepared once: its 3D appearance volume and keypoints."""

    feature: object          # torch tensor 1x32x16x64x64
    info: dict               # LivePortrait's keypoint info (degrees)
    rotation: object         # 1x3x3
    keypoints: object        # x_s: 1x21x3, the photo's posed keypoints


class HeadModels:
    """LivePortrait's five networks, loaded once (PyTorch, CUDA when there is one)."""

    def __init__(self, directory: Path, gpu: bool, place: Callable[[Callable[[], None]], None] | None = None) -> None:
        """``place`` runs each network's copy to the GPU (the face process passes one that waits for a
        gap in the voice: half a gigabyte uploaded at once would hold the voice up)."""
        import torch

        from backend.video.liveportrait.appearance_feature_extractor import AppearanceFeatureExtractor
        from backend.video.liveportrait.motion_extractor import MotionExtractor
        from backend.video.liveportrait.spade_generator import SPADEDecoder
        from backend.video.liveportrait.stitching_retargeting_network import StitchingRetargetingNetwork
        from backend.video.liveportrait.warping_network import WarpingNetwork

        self.torch = torch
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        # cuDNN's autotuner picks 3D convolutions with gigabytes of workspace: not next to the voice.
        torch.backends.cudnn.benchmark = False

        def load(cls, key: str, state: dict | None = None):
            model = cls(**PARAMS[key])
            if state is None:
                state = torch.load(directory / FILES[key], map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            moved: list = []
            (place or (lambda copy: copy()))(lambda: moved.append(model.to(self.device)))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return moved[0]

        self.appearance = load(AppearanceFeatureExtractor, "appearance")
        self.motion = load(MotionExtractor, "motion")
        self.warping = load(WarpingNetwork, "warping")
        self.generator = load(SPADEDecoder, "generator")
        checkpoint = torch.load(directory / FILES["stitching"], map_location="cpu", weights_only=True)
        shoulder = {k.replace("module.", ""): v for k, v in checkpoint["retarget_shoulder"].items()}
        self.stitching = load(StitchingRetargetingNetwork, "stitching", shoulder)

    def tensor(self, image: np.ndarray):
        """BGR uint8 HxWx3 -> 1x3x256x256 RGB in 0..1 on the device."""
        rgb = cv2.resize(image, (INPUT, INPUT), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        x = self.torch.from_numpy(np.ascontiguousarray(rgb)).float().div_(255.0)
        return x.permute(2, 0, 1)[None].to(self.device)


def _degrees(pred, torch):
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx = torch.arange(66, device=pred.device, dtype=torch.float32)
        return (torch.softmax(pred, dim=1) * idx).sum(1) * 3 - 97.5
    return pred


def _rotation(pitch, yaw, roll, torch):
    """LivePortrait's rotation matrix (degrees in, transposed out)."""
    x, y, z = (a.reshape(-1, 1) / 180 * math.pi for a in (pitch, yaw, roll))
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    rx = torch.cat([one, zero, zero, zero, torch.cos(x), -torch.sin(x), zero, torch.sin(x), torch.cos(x)], 1)
    ry = torch.cat([torch.cos(y), zero, torch.sin(y), zero, one, zero, -torch.sin(y), zero, torch.cos(y)], 1)
    rz = torch.cat([torch.cos(z), -torch.sin(z), zero, torch.sin(z), torch.cos(z), zero, zero, zero, one], 1)
    rot = rz.reshape(-1, 3, 3) @ ry.reshape(-1, 3, 3) @ rx.reshape(-1, 3, 3)
    return rot.permute(0, 2, 1)


class HeadRenderer:
    """Renders the photo's head in the speaker's pose and expression, in stages."""

    def __init__(self, models: HeadModels) -> None:
        self.models = models
        self.source: Source | None = None
        self._reference: dict | None = None      # the first driving frame's keypoint info
        self.result: np.ndarray | None = None     # the last finished render

    # --- The photo ---------------------------------------------------------

    def set_source(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Prepare the photo (its face's five points); returns its head crop at OUTPUT size."""
        torch, m = self.models.torch, self.models
        crop = cv2.warpAffine(image, alignment(points, HEAD_TEMPLATE, OUTPUT), (OUTPUT, OUTPUT),
                              flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        with torch.no_grad():
            x = m.tensor(crop)
            info = self._info(x)
            rotation = _rotation(info["pitch"], info["yaw"], info["roll"], torch)
            keypoints = self._posed(info, rotation)
            self.source = Source(m.appearance(x), info, rotation, keypoints)
        self._reference = None
        return crop

    def reset(self) -> None:
        """The next frame becomes the neutral reference (the head's rest pose and expression)."""
        self._reference = None

    def _info(self, x) -> dict:
        torch = self.models.torch
        info = self.models.motion(x)
        out = {k: v.float() for k, v in info.items()}
        for key in ("pitch", "yaw", "roll"):
            out[key] = _degrees(out[key], torch)[:, None]
        out["kp"] = out["kp"].reshape(1, -1, 3)
        out["exp"] = out["exp"].reshape(1, -1, 3)
        return out

    @staticmethod
    def _posed(info: dict, rotation):
        kp = info["kp"] @ rotation + info["exp"]
        kp = kp * info["scale"][..., None]
        kp[:, :, 0:2] = kp[:, :, 0:2] + info["t"][:, None, 0:2]
        return kp

    def _stitch(self, kp_source, kp_driving):
        torch = self.models.torch
        delta = self.models.stitching(torch.cat([kp_source.reshape(1, -1), kp_driving.reshape(1, -1)], 1))
        n = kp_source.shape[1]
        out = kp_driving + delta[..., : 3 * n].reshape(1, n, 3)
        out[..., :2] = out[..., :2] + delta[..., 3 * n: 3 * n + 2].reshape(1, 1, 2)
        return out

    # --- One render, in stages ----------------------------------------------

    def stages(self, frame: np.ndarray, points: np.ndarray) -> Iterator[Callable[[], None]]:
        """The steps of one render of the head posed like the face at ``points`` in ``frame``.

        Each step is a callable that queues some GPU work and waits for it; run them in order
        (each inside its own GPU grant). After the last one, ``result`` holds the head (BGR
        uint8, OUTPUT x OUTPUT, in the HEAD_TEMPLATE crop of ``points``).
        """
        torch, m = self.models.torch, self.models
        if self.source is None:
            return
        state: dict = {}
        source = self.source

        def sync() -> None:
            if m.device.type == "cuda":
                torch.cuda.synchronize(m.device)

        def motion() -> None:
            crop = cv2.warpAffine(frame, alignment(points, HEAD_TEMPLATE, INPUT), (INPUT, INPUT),
                                  flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
            info = self._info(m.tensor(crop))
            if self._reference is None:
                self._reference = info
            ref, src = self._reference, source.info
            r_d = _rotation(info["pitch"], info["yaw"], info["roll"], torch)
            r_ref = _rotation(ref["pitch"], ref["yaw"], ref["roll"], torch)
            rotation = (r_d @ r_ref.permute(0, 2, 1)) @ source.rotation
            expression = src["exp"] + (info["exp"] - ref["exp"])
            # The lips follow the speaker's own (as LivePortrait's lip region does): a photo that
            # smiles does not keep smiling while the speaker talks.
            expression[:, LIP_KEYPOINTS] = info["exp"][:, LIP_KEYPOINTS]
            scale = src["scale"] * (info["scale"] / ref["scale"])
            shift = src["t"] + (info["t"] - ref["t"])
            shift[..., 2] = 0
            driving = scale[..., None] * (src["kp"] @ rotation + expression) + shift[:, None, :]
            state["driving"] = self._stitch(source.keypoints, driving)
            sync()

        dense: Any = m.warping.dense_motion_network   # LivePortrait's, always built with one
        hourglass = dense.hourglass

        def deform() -> None:
            feature = torch.relu(dense.norm(dense.compress(source.feature)))
            sparse = dense.create_sparse_motions(feature, state["driving"], source.keypoints)
            deformed = dense.create_deformed_feature(feature, sparse)
            heatmap = dense.create_heatmap_representations(deformed, state["driving"], source.keypoints)
            bs, _, d, h, w = feature.shape
            state["sparse"] = sparse
            state["skips"] = [torch.cat([heatmap, deformed], dim=2).view(bs, -1, d, h, w)]
            sync()

        def down(block) -> Callable[[], None]:
            def run() -> None:
                state["skips"].append(block(state["skips"][-1]))
                sync()
            return run

        def up(block) -> Callable[[], None]:
            def run() -> None:
                skips = state["skips"]
                out = state.pop("up", None)
                out = block(skips.pop() if out is None else out)
                state["up"] = torch.cat([out, skips.pop()], dim=1)
                sync()
            return run

        def decode() -> None:
            decoder = hourglass.decoder
            state["prediction"] = torch.relu(decoder.norm(decoder.conv(state.pop("up"))))
            state["logits"] = []
            sync()

        # The motion mask is a 7x7x7 3D convolution (about 150 ms here): computed a few depth
        # slices at a time (each with the 3-voxel halo it needs), each part short enough for one
        # gap next to the voice. Same result as the whole convolution.
        conv: Any = dense.mask
        depth = PARAMS["warping"]["dense_motion_params"]["reshape_depth"]
        halo = int(conv.padding[0])
        parts = np.array_split(np.arange(depth), MASK_PARTS)

        def mask_part(slices: np.ndarray) -> Callable[[], None]:
            lo, hi = int(slices[0]), int(slices[-1]) + 1

            def run() -> None:
                if "padded" not in state:
                    state["padded"] = torch.nn.functional.pad(state["prediction"], (0, 0, 0, 0, halo, halo))
                state["logits"].append(conv3d_depth_slice(state["padded"], conv, lo, hi))
                sync()
            return run

        def mask() -> None:
            prediction = state.pop("prediction")
            state.pop("padded", None)
            weights = torch.softmax(torch.cat(state.pop("logits"), dim=2), dim=1).unsqueeze(2)
            sparse = state.pop("sparse").permute(0, 1, 5, 2, 3, 4)
            state["deformation"] = (sparse * weights).sum(dim=1).permute(0, 2, 3, 4, 1)
            bs, _, _, h, w = prediction.shape
            state["occlusion"] = torch.sigmoid(dense.occlusion(prediction.view(bs, -1, h, w)))
            sync()

        def warp() -> None:
            warping = m.warping
            out = torch.nn.functional.grid_sample(source.feature, state.pop("deformation"), align_corners=False)
            bs, c, d, h, w = out.shape
            out = warping.fourth(warping.third(out.view(bs, c * d, h, w)))
            state["feature"] = out * state.pop("occlusion")
            state["x"] = m.generator.fc(state["feature"])
            sync()

        generator = m.generator

        def middle(block) -> Callable[[], None]:
            def run() -> None:
                state["x"] = block(state["x"], state["feature"])
                sync()
            return run

        def upper(block) -> list[Callable[[], None]]:
            """An upsampling SPADE block in three parts (shortcut, first and second convolution)."""
            def shortcut() -> None:
                state["x"] = generator.up(state["x"])
                state["skip"] = block.shortcut(state["x"], state["feature"])
                sync()

            def first() -> None:
                state["dx"] = block.conv_0(block.actvn(block.norm_0(state.pop("x"), state["feature"])))
                sync()

            def second() -> None:
                dx = block.conv_1(block.actvn(block.norm_1(state.pop("dx"), state["feature"])))
                state["x"] = state.pop("skip") + dx
                sync()
            return [shortcut, first, second]

        def image() -> None:
            out = torch.sigmoid(generator.conv_img(torch.nn.functional.leaky_relu(state.pop("x"), 2e-1)))
            rgb = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            self.result = np.ascontiguousarray(rgb[:, :, ::-1])
            state.clear()

        steps: list[Callable[[], None]] = [motion, deform]
        steps += [down(block) for block in hourglass.encoder.down_blocks]
        steps += [up(block) for block in hourglass.decoder.up_blocks]
        steps += [decode, *(mask_part(part) for part in parts), mask, warp]
        steps += [middle(getattr(generator, f"G_middle_{i}")) for i in range(6)]
        steps += [*upper(generator.up_0), *upper(generator.up_1), image]
        for step in steps:
            yield _without_grad(step, torch)

    def render(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """One whole render at once (files, or when nothing else needs the GPU)."""
        for step in self.stages(frame, points):
            step()
        return self.result


def conv3d_depth_slice(padded, conv, lo: int, hi: int):
    """Output depths ``lo:hi`` of the 3D convolution ``conv`` over a volume already padded in depth
    by the kernel's half size: the same numbers as the whole convolution, a slice at a time."""
    import torch

    halo = int(conv.padding[0])
    return torch.nn.functional.conv3d(padded[:, :, lo: hi + 2 * halo], conv.weight, conv.bias,
                                      padding=(0, conv.padding[1], conv.padding[2]))


def _without_grad(step: Callable[[], None], torch) -> Callable[[], None]:
    def run() -> None:
        with torch.no_grad():
            step()
    return run
