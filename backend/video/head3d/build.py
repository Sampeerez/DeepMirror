"""Build a face's 3D head with LAM (runs in LAM's own environment, cwd = LAM's source folder).

    python build.py <face folder> <BiSeNet parser .onnx> <MediaPipe face_landmarker.task> [<GPEN .onnx>]

Picks the best photo for it (facing the camera, large, sharp, evenly lit, eyes open, mouth shut),
restores a small face's detail first (GPEN), tracks FLAME on it, builds the Gaussian head
(LAM-20K), renders it from the photo's own camera and drops the Gaussians that sit on clothes, the
neck or the background there (the speaker's own body stays in the picture). Writes ``head3d.pt``
(and ``head3d.jpg``, the head seen from the photo) into the face folder; prints
``progress <0..1> <stage>`` lines on the way.

The GPU is used one model at a time (the tracker's, then LAM's in half precision), so a build fits
next to the studio's own models on a 6 GB card.
"""

import gc
import math
import os
import sys
import tempfile
import time
from pathlib import Path

LAM_MODEL = "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
LAM_CONFIG = "./configs/inference/lam-20k-8gpu.yaml"
FLAME = "./model_zoo/human_parametric_models/flame_assets/flame"
NECK, NECKLACE, CLOTH = 14, 15, 16   # BiSeNet (CelebAMask-HQ) classes that are body, not head
SMALL_FACE = 300                     # pixels across: below this the photo is enlarged and restored first
# MediaPipe mesh points for the five FFHQ alignment points (eyes from their corners, nose, mouth corners).
EYE_RIGHT, EYE_LEFT, NOSE, MOUTH = (33, 133), (362, 263), 4, (61, 291)
FFHQ_256 = [[96.49, 119.97], [159.45, 120.10], [128.32, 157.01], [100.63, 185.71], [156.54, 185.58]]
HEAD_VERSION = 2


def load_prebuilt_plugins() -> None:
    """nvdiffrast's CUDA plugin, prebuilt (DEEPMIRROR_LAM_PLUGINS): nvdiffrast would otherwise
    compile it in every new process, which needs ninja and a C++/CUDA compiler on this computer."""
    folder = os.environ.get("DEEPMIRROR_LAM_PLUGINS")
    if not folder or not any(Path(folder).glob("nvdiffrast_plugin.*")):
        return
    import importlib

    import nvdiffrast.torch.ops as ops
    import torch  # noqa: F401  (first: the plugin links against its DLLs)

    sys.path.insert(0, folder)
    ops._cached_plugin[False] = importlib.import_module("nvdiffrast_plugin")


def progress(done: float, stage: str) -> None:
    print(f"progress {done:.2f} {stage}", flush=True)


def best_photo(photos: list[Path], landmarker_path: str):
    """``(path, image, five points, face width)`` of the photo best suited to a 3D head."""
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=landmarker_path),
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True, num_faces=1)
    candidates = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for path in photos:
            image = cv2.imread(str(path))
            if image is None:
                continue
            found = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            if not found.face_landmarks:
                continue
            height, width = image.shape[:2]
            mesh = np.array([[p.x * width, p.y * height] for p in found.face_landmarks[0]])
            matrix = found.facial_transformation_matrixes[0]
            yaw = math.degrees(math.atan2(-matrix[2][0], math.hypot(matrix[2][1], matrix[2][2])))
            pitch = math.degrees(math.atan2(matrix[2][1], matrix[2][2]))
            scores = {c.category_name: c.score for c in found.face_blendshapes[0]}
            x0, y0 = mesh.min(0).astype(int)
            x1, y1 = mesh.max(0).astype(int)
            face = cv2.cvtColor(image[max(y0, 0):y1, max(x0, 0):x1], cv2.COLOR_BGR2GRAY)
            if face.size == 0:
                continue
            size = float(x1 - x0)
            sharp = float(cv2.Laplacian(cv2.resize(face, (256, 256)), cv2.CV_64F).var())
            half = face.shape[1] // 2
            balance = abs(float(face[:, :half].mean()) - float(face[:, half:].mean())) / 255.0
            blink = max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))
            mouth = scores.get("jawOpen", 0.0)
            points = np.array([mesh[list(EYE_RIGHT)].mean(0), mesh[list(EYE_LEFT)].mean(0), mesh[NOSE],
                               mesh[MOUTH[0]], mesh[MOUTH[1]]], np.float32)
            # Lower is better: the turn (degrees), then penalties for shut eyes, an open mouth (the
            # head is built with the photo's expression: a neutral one animates best) and one-sided
            # light; sharpness only breaks near ties.
            cost = (abs(yaw) + 0.6 * abs(pitch) + 60 * max(0.0, blink - 0.25) + 120 * max(0.0, mouth - 0.08)
                    + 30 * balance - 2.0 * math.log1p(sharp) / math.log(10))
            candidates.append((cost, size, path, image, points))
    if not candidates:
        raise SystemExit("No face found in the photos")
    largest = max(c[1] for c in candidates)
    cost, size, path, image, points = min((c for c in candidates if c[1] >= 0.7 * largest), key=lambda c: c[0])
    return path, image, points, size


def restore_small_face(image, points, size: float, gpen_path: str | None):
    """A face under SMALL_FACE pixels: the photo enlarged so it is about 400 px across, and the face
    itself restored with GPEN (sharper eyes, skin and hair line for LAM's 512 px input)."""
    import cv2
    import numpy as np

    if size >= SMALL_FACE:
        return image
    scale = 400.0 / max(size, 1.0)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    if not gpen_path or not Path(gpen_path).is_file():
        return image
    import onnxruntime as ort

    points = points * scale
    matrix, _ = cv2.estimateAffinePartial2D(points, np.array(FFHQ_256, np.float32), method=cv2.LMEDS)
    if matrix is None:
        return image
    crop = cv2.warpAffine(image, matrix, (256, 256), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    session = ort.InferenceSession(gpen_path, providers=["CPUExecutionProvider"])
    blob = ((crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    out = session.run(None, {session.get_inputs()[0].name: blob})[0][0].transpose(1, 2, 0)
    restored = (np.clip((out + 1) / 2, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)
    # Back into the photo, faded towards the crop's edges (no seam).
    ramp = np.clip(np.minimum(np.arange(256), np.arange(256)[::-1]) / 40.0, 0, 1).astype(np.float32)
    mask = np.minimum(ramp[:, None], ramp[None, :])
    inverse = cv2.invertAffineTransform(matrix)
    height, width = image.shape[:2]
    back = cv2.warpAffine(restored, inverse, (width, height), flags=cv2.INTER_CUBIC)
    weight = cv2.warpAffine(mask, inverse, (width, height))[..., None] * 0.8
    return (back * weight + image * (1 - weight)).astype(np.uint8)


def live_extras(device: str = "cpu"):
    """What the live side needs besides the Gaussians, worked out once here: FLAME's ARKit
    blendshapes and its neck on the head's 20k points (so the live process starts at once)."""
    import pickle

    import numpy as np
    import torch
    from lam.models.rendering.flame_model.flame import FlameHeadSubdivided

    flame = FlameHeadSubdivided(
        300, 100, add_teeth=False, add_shoulder=False, flame_model_path=f"{FLAME}/flame2023.pkl",
        flame_lmk_embedding_path=f"{FLAME}/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path=f"{FLAME}/head_template_mesh.obj", flame_parts_path=f"{FLAME}/FLAME_masks.pkl",
        subdivide_num=1, teeth_bs_flag=False, oral_mesh_flag=False)
    arkit = torch.from_numpy(np.load(f"{FLAME}/flame_arkit_bs.npy").astype(np.float32))   # 52 x 5023 x 3
    with open(f"{FLAME}/FLAME_masks.pkl", "rb") as file:
        regions = pickle.load(file, encoding="latin1")
    neck = torch.zeros(arkit.shape[1], 1)
    neck[torch.as_tensor(np.concatenate([regions["neck"], regions["boundary"]]))] = 1
    with torch.no_grad():
        _, up, neck_up = flame.upsample_mesh_cpu(flame.v_template.float(),
                                                 [arkit.permute(1, 0, 2).reshape(arkit.shape[1], -1), neck])
    return up.reshape(-1, 52, 3).permute(1, 0, 2).contiguous().half(), neck_up[:, 0].contiguous().half()


def load_lam():
    """LAM-20K, ready on the GPU (image encoder and transformer in half precision), and its settings.
    (What LAM's app_lam.py does, without its web UI's dependencies.)"""
    from lam.models import ModelLAM
    from lam.models.encoders import dinov2_fusion_wrapper
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    # LAM's weights hold the whole image encoder: DINOv2's own (1.1 GB) need not be downloaded first.
    build = dinov2_fusion_wrapper.Dinov2FusionWrapper._build_dinov2
    dinov2_fusion_wrapper.Dinov2FusionWrapper._build_dinov2 = staticmethod(
        lambda name, modulation_dim=None, pretrained=True: build(name, modulation_dim=modulation_dim, pretrained=False))
    config = OmegaConf.load(LAM_CONFIG)
    model = ModelLAM(**config.model)
    weights = load_file(os.path.join(LAM_MODEL, "model.safetensors"), device="cpu")
    state = model.state_dict()
    missing = [name for name in state if name not in weights]
    if any(name.startswith("encoder.") for name in missing):
        raise SystemExit("LAM's weights lack part of the image encoder")
    for name, value in weights.items():
        if name in state and state[name].shape == value.shape:
            state[name].copy_(value)
    model.eval()
    model.encoder.half()
    model.transformer.half()
    sizes = {"source_size": config.dataset.source_image_res, "render_size": config.dataset.render_image.high}
    return model.to("cuda"), sizes


def main() -> None:
    folder, parser_path, landmarker_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    gpen_path = sys.argv[4] if len(sys.argv) > 4 else None
    sys.path.insert(0, os.getcwd())
    os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
    sys.argv = sys.argv[:1]   # LAM's modules read the command line too
    load_prebuilt_plugins()

    started = time.perf_counter()
    photos = sorted(folder.glob("photo_*.jpg"), key=lambda path: int(path.stem.split("_")[1]))
    progress(0.02, "photo")
    photo, image, points, size = best_photo(photos, landmarker_path)
    image = restore_small_face(image, points, size, gpen_path)

    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    work = Path(tempfile.mkdtemp(prefix="deepmirror-head3d-"))
    raw = work / "raw.png"
    cv2.imwrite(str(raw), image)

    # 1. FLAME on the photo (its own models on the GPU, freed before LAM loads).
    progress(0.08, "tracking")
    tracking = FlameTrackingSingleImage(
        output_dir=str(work / "tracking"),
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth", detect_iris_landmarks=False)
    if tracking.preprocess(str(raw)) != 0:
        raise SystemExit("Could not find the head in the photo")
    progress(0.30, "fitting")
    if tracking.optimize() != 0:
        raise SystemExit("Could not fit a head to the photo")
    code, tracked = tracking.export()
    if code != 0:
        raise SystemExit("Could not fit a head to the photo")
    del tracking
    gc.collect()
    torch.cuda.empty_cache()

    # 2. The Gaussian head (LAM): image encoder and transformer in half precision.
    progress(0.60, "loading")
    lam, sizes = load_lam()
    progress(0.75, "building")
    source, _, _, shape = preprocess_image(
        os.path.join(tracked, "images/00000_00.png"), mask_path=os.path.join(tracked, "fg_masks/00000_00.png"),
        intr=None, pad_ratio=0, bg_color=1., max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
        render_tgt_size=sizes["source_size"], multiply=14, need_mask=True, get_shape_param=True)
    # The photo's own view (its camera and FLAME pose): each Gaussian lands where it came from.
    seq = prepare_motion_seqs(
        os.path.join(tracked, "flame_param"), None, save_root=str(work), fps=30, bg_color=1., aspect_standard=1.0,
        enlarge_ratio=[1.0, 1, 0], render_image_res=sizes["render_size"], multiply=16, need_mask=False,
        vis_motion=False, shape_param=shape, test_sample=False, cross_id=False, src_driven=["src", "src"])
    seq["flame_params"]["betas"] = shape.unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        result = lam.infer_single_view(
            source.unsqueeze(0).to("cuda"), None, None, render_c2ws=seq["render_c2ws"][:, :1].cuda(),
            render_intrs=seq["render_intrs"][:, :1].cuda(), render_bg_colors=seq["render_bg_colors"][:, :1].cuda(),
            flame_params={k: (v if k == "betas" else v[:, :1]).cuda() for k, v in seq["flame_params"].items()})
    peak = torch.cuda.max_memory_allocated() / 2**30
    gaussians = result["cano_gs_lst"][0]
    del lam
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Only the head: Gaussians that land on clothes, the neck or the background in the photo go.
    progress(0.90, "cleaning")
    view = (result["comp_rgb"][0].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
    posed = result["3dgs"]
    while isinstance(posed, (list, tuple)):
        posed = posed[0]
    xyz = posed.xyz.detach().float().cpu().numpy()
    c2w, intr = seq["render_c2ws"][0, 0].numpy(), seq["render_intrs"][0, 0].numpy()
    camera = (np.linalg.inv(c2w) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    u = intr[0, 0] * camera[:, 0] / camera[:, 2] + intr[0, 2]
    v = intr[1, 1] * camera[:, 1] / camera[:, 2] + intr[1, 2]

    session = ort.InferenceSession(parser_path, providers=["CPUExecutionProvider"])
    rgb = cv2.resize(view, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
    blob = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
    labels = session.run(None, {session.get_inputs()[0].name: blob})[0][0].argmax(0).astype(np.uint8)
    labels = cv2.resize(labels, view.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    body = cv2.dilate(np.isin(labels, (NECK, NECKLACE, CLOTH)).astype(np.uint8), np.ones((5, 5), np.uint8))
    # The photo's own person mask (LAM's matting), a little wider: what falls outside is a floater.
    person = cv2.imread(os.path.join(tracked, "fg_masks/00000_00.png"), cv2.IMREAD_GRAYSCALE)
    person = cv2.resize(person, view.shape[1::-1], interpolation=cv2.INTER_LINEAR) > 64
    person = cv2.dilate(person.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    height, width = body.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    cut = v >= height - 1                      # below the photo's edge: shoulders
    iu, iv = u[inside].astype(int), v[inside].astype(int)
    cut[inside] |= (body[iv, iu] > 0) | ~person[iv, iu]

    keep = torch.from_numpy(~cut).to(gaussians.opacity.device).float()[:, None]
    arkit_up, neck_up = live_extras()
    head = {"xyz": gaussians.xyz, "opacity": gaussians.opacity * keep, "rotation": gaussians.rotation,
            "scaling": gaussians.scaling, "shs": gaussians.shs, "betas": shape.reshape(1, -1),
            "arkit": arkit_up, "neck": neck_up, "photo": photo.name, "version": HEAD_VERSION}
    partial = folder / "head3d.pt.part"
    torch.save({k: (t.detach().float().cpu() if torch.is_tensor(t) and k not in ("arkit", "neck")
                    else t.cpu() if torch.is_tensor(t) else t) for k, t in head.items()}, partial)
    partial.replace(folder / "head3d.pt")
    alpha = result["comp_mask"][0][..., 0].float().cpu().numpy()
    preview = (view * alpha[..., None] + 255 * (1 - alpha[..., None])).astype(np.uint8)
    cv2.imwrite(str(folder / "head3d.jpg"), cv2.resize(preview, (256, 256), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    progress(1.0, f"done in {time.perf_counter() - started:.0f} s from {photo.name}, peak {peak:.1f} GB, "
                  f"{int(cut.sum())} Gaussians dropped")


if __name__ == "__main__":
    main()
