"""Build a face's 3D head with LAM (runs in LAM's own environment, cwd = LAM's source folder).

    python build.py <face folder> <BiSeNet parser .onnx> <MediaPipe face_landmarker.task> [<GPEN .onnx>]

Picks the best photo for it (facing the camera, large, sharp, evenly lit, eyes open, mouth shut),
restores a small face's detail first (GPEN), builds the Gaussian head from it (LAM-20K), drops the
Gaussians that sit on clothes, the neck or the background there, then trains the head on every
photo of the face: FLAME is tracked on each (its camera, pose and expression) and the Gaussians'
colours, opacities, sizes and positions are fitted so the head matches each photo from its own
angle. The trained head is kept only if it fits the photos better. Writes ``head3d.pt`` (and
``head3d.jpg``: the head from three sides) into the face folder; prints ``progress <0..1> <stage>``
lines on the way. Under the studio each training step waits for a gap the live voice leaves.

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
FACE_CLASSES = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)   # skin, brows, eyes, glasses, nose, mouth, lips
HAIR = 17
SMALL_FACE = 300                     # pixels across: below this the photo is enlarged and restored first
# MediaPipe mesh points for the five FFHQ alignment points (eyes from their corners, nose, mouth corners).
EYE_RIGHT, EYE_LEFT, NOSE, MOUTH = (33, 133), (362, 263), 4, (61, 291)
FFHQ_256 = [[96.49, 119.97], [159.45, 120.10], [128.32, 157.01], [100.63, 185.71], [156.54, 185.58]]
HEAD_VERSION = 2
TRAIN_STEPS = 2000                   # training steps over all the photos (DEEPMIRROR_TRAIN_STEPS: more is finer)


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


def parse_head(session, image_rgb):
    """BiSeNet classes of an RGB image (0..255 uint8), at the image's size."""
    import cv2
    import numpy as np

    rgb = cv2.resize(image_rgb, (512, 512)).astype(np.float32) / 255.0
    blob = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
    labels = session.run(None, {session.get_inputs()[0].name: blob})[0][0].argmax(0).astype(np.uint8)
    return cv2.resize(labels, image_rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)


def gpu_arbiter():
    """The studio's GPU arbiter (the live voice comes first), when this runs under the studio."""
    import importlib.util

    path, name = os.environ.get("DEEPMIRROR_ARBITER_FILE"), os.environ.get("DEEPMIRROR_ARBITER_NAME")
    if not path or not name:
        return None
    try:
        spec = importlib.util.spec_from_file_location("compute_arbiter", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.GpuArbiter(name)
    except Exception:
        return None


def ssim_loss(a, b, mask):
    """1 - SSIM between two 3 x H x W images inside ``mask`` (1 x H x W), 11-pixel Gaussian window."""
    import torch
    import torch.nn.functional as F

    x = torch.arange(11, device=a.device, dtype=torch.float32) - 5
    g = torch.exp(-(x * x) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    window = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)

    def blur(t):
        return F.conv2d(t[None], window, padding=5, groups=3)[0]

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return ((1 - ssim) * mask).sum() / (mask.sum() * 3 + 1e-6)


def main() -> None:
    folder, parser_path, landmarker_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    gpen_path = sys.argv[4] if len(sys.argv) > 4 else None
    steps = int(os.environ.get("DEEPMIRROR_TRAIN_STEPS", TRAIN_STEPS))
    sys.path.insert(0, os.getcwd())
    os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
    sys.argv = sys.argv[:1]   # LAM's modules read the command line too
    load_prebuilt_plugins()

    started = time.perf_counter()
    photos = sorted(folder.glob("photo_*.jpg"), key=lambda path: int(path.stem.split("_")[1]))
    progress(0.01, "photo")
    photo, image, points, size = best_photo(photos, landmarker_path)
    image = restore_small_face(image, points, size, gpen_path)

    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from lam.models.rendering.gaussian_model import GaussianModel
    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    work = Path(tempfile.mkdtemp(prefix="deepmirror-head3d-"))
    session = ort.InferenceSession(parser_path, providers=["CPUExecutionProvider"])

    # 1. FLAME on every photo with a face (the chosen one first): each is a view to train on.
    progress(0.03, "tracking")
    tracking = FlameTrackingSingleImage(
        output_dir=str(work / "tracking"),
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth", detect_iris_landmarks=False)
    others = [p for p in photos if p != photo] if steps > 0 else []
    tracked_dirs: list[tuple[str, str]] = []
    for index, (name, picture) in enumerate([(photo.name, image)] + [(p.name, cv2.imread(str(p))) for p in others]):
        raw = work / f"view{index}.png"
        if picture is None:
            continue
        cv2.imwrite(str(raw), picture)
        try:
            ok = tracking.preprocess(str(raw)) == 0 and tracking.optimize() == 0
            code, tracked = tracking.export() if ok else (1, None)
        except Exception as exc:
            ok, code, tracked = False, 1, None
            print(f"  {name}: no fit ({type(exc).__name__})", flush=True)
        if ok and code == 0 and tracked:
            tracked_dirs.append((name, tracked))
        elif index == 0:
            raise SystemExit("Could not fit a head to the photo")
        progress(0.03 + 0.37 * (index + 1) / (len(others) + 1), f"tracking {index + 1}/{len(others) + 1}")
    del tracking
    gc.collect()
    torch.cuda.empty_cache()

    # 2. The Gaussian head (LAM) from the chosen photo: encoder and transformer in half precision.
    progress(0.42, "loading")
    lam, sizes = load_lam()
    progress(0.48, "building")
    tracked = tracked_dirs[0][1]
    source, _, _, shape = preprocess_image(
        os.path.join(tracked, "images/00000_00.png"), mask_path=os.path.join(tracked, "fg_masks/00000_00.png"),
        intr=None, pad_ratio=0, bg_color=1., max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
        render_tgt_size=sizes["source_size"], multiply=14, need_mask=True, get_shape_param=True)

    def view_of(directory: str) -> dict:
        """A photo's camera, FLAME pose and expression (with the chosen photo's head shape)."""
        seq = prepare_motion_seqs(
            os.path.join(directory, "flame_param"), None, save_root=str(work), fps=30, bg_color=1.,
            aspect_standard=1.0, enlarge_ratio=[1.0, 1, 0], render_image_res=sizes["render_size"], multiply=16,
            need_mask=False, vis_motion=False, shape_param=shape, test_sample=False, cross_id=False,
            src_driven=["src", "src"])
        seq["flame_params"]["betas"] = shape.unsqueeze(0)
        intr = seq["render_intrs"][:, :1].cuda()
        return {"flame": {k: (v if k == "betas" else v[:, :1]).cuda() for k, v in seq["flame_params"].items()},
                "c2w": seq["render_c2ws"][:, :1].cuda(), "intr": intr, "bg": seq["render_bg_colors"][:, :1].cuda(),
                "size": (int(intr[0, 0, 1, 2] * 2), int(intr[0, 0, 0, 2] * 2))}

    main_view = view_of(tracked)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        result = lam.infer_single_view(source.unsqueeze(0).to("cuda"), None, None, render_c2ws=main_view["c2w"],
                                       render_intrs=main_view["intr"], render_bg_colors=main_view["bg"],
                                       flame_params=main_view["flame"])
    peak = torch.cuda.max_memory_allocated() / 2**30
    gaussians = result["cano_gs_lst"][0]
    lam.encoder = lam.transformer = None   # only the renderer (FLAME) is needed from here on
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Only the head: Gaussians that land on clothes, the neck or the background in the photo go.
    progress(0.52, "cleaning")
    view = (result["comp_rgb"][0].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    posed = result["3dgs"]
    while isinstance(posed, (list, tuple)):
        posed = posed[0]
    xyz = posed.xyz.detach().float().cpu().numpy()
    c2w, intr = main_view["c2w"][0, 0].cpu().numpy(), main_view["intr"][0, 0].cpu().numpy()
    camera = (np.linalg.inv(c2w) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    u = intr[0, 0] * camera[:, 0] / camera[:, 2] + intr[0, 2]
    v = intr[1, 1] * camera[:, 1] / camera[:, 2] + intr[1, 2]
    labels = parse_head(session, view)
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
    keep = torch.from_numpy(~cut).cuda().float()[:, None]
    # The Gaussians of the hair (seen on the hair in the front photo, or behind it): the only ones
    # training may change; forehead, ears and face stay as LAM made them.
    on_hair = np.zeros(len(u), bool)
    on_hair[inside] = labels[iv, iu] == HAIR
    hair_only = torch.from_numpy(on_hair & ~cut).cuda()

    base = {"xyz": gaussians.xyz.detach().float(), "opacity": (gaussians.opacity.detach().float() * keep),
            "rotation": gaussians.rotation.detach().float(), "scaling": gaussians.scaling.detach().float(),
            "shs": gaussians.shs.detach().float(), "offset": gaussians.offset.detach().float()}
    renderer = lam.renderer
    query, _ = renderer.get_query_points(main_view["flame"], torch.device("cuda"))

    def render(attrs: dict, view: dict, shift=None) -> tuple:
        flame = dict(view["flame"])
        if shift is not None:
            flame["translation"] = flame["translation"] + shift
        model = GaussianModel(xyz=attrs["xyz"], opacity=attrs["opacity"], rotation=attrs["rotation"],
                              scaling=attrs["scaling"], shs=attrs["shs"], offset=attrs["offset"])
        out = renderer.forward_animate_gs([model], query, renderer.get_single_view_smpl_data(flame, 0), view["c2w"],
                                          view["intr"], view["size"][0], view["size"][1], view["bg"])
        return out["comp_rgb"][0, 0], out["comp_mask"][0, 0]

    # 4. Training on every photo: colours, opacities, sizes, turns and small moves of the Gaussians,
    # so the head matches each photo from its own angle (one photo shows the front, others the sides).
    views = []
    for index, (name, directory) in enumerate(tracked_dirs):
        try:
            item = view_of(directory)
        except Exception as exc:
            print(f"  {name}: skipped ({type(exc).__name__}: {exc})", flush=True)
            continue
        h, w = item["size"]
        target = cv2.cvtColor(cv2.imread(os.path.join(directory, "images/00000_00.png")), cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)
        fg = cv2.imread(os.path.join(directory, "fg_masks/00000_00.png"), cv2.IMREAD_GRAYSCALE)
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR) > 127
        head_labels = parse_head(session, target)
        head = fg & ~np.isin(head_labels, (0, NECK, NECKLACE, CLOTH))
        head = cv2.erode(head.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        # What the 3D head shows live: hair, ears and the head's outline (the face is the swap's).
        outer = head & ~np.isin(head_labels, FACE_CLASSES)
        item.update(name=name, weight=2.0 if index == 0 else 1.0,
                    target=torch.from_numpy(target).cuda().float().permute(2, 0, 1) / 255.0,
                    mask=torch.from_numpy(head).cuda().float()[None],
                    outer=torch.from_numpy(outer).cuda().float()[None])
        views.append(item)

    final = base
    trained_note = "not trained (one photo)"
    if steps > 0 and len(views) >= 3:
        # One side photo is held out: the training is kept only if the head also fits a photo it
        # never saw (fitting the training photos alone can mean ghosts between their angles).
        held = 1 + int(np.argmax([float(item["outer"].sum()) for item in views[1:]]))
        train_idx = [i for i in range(len(views)) if i != held]
        logit = lambda p: torch.log(p.clamp(1e-4, 1 - 1e-4) / (1 - p.clamp(1e-4, 1 - 1e-4)))  # noqa: E731
        rgb0, opacity0 = base["shs"].clone(), logit(base["opacity"])
        params = {"rgb": rgb0.clone().requires_grad_(True), "opacity": opacity0.clone().requires_grad_(True)}
        turns = torch.zeros(len(views), 1, 1, 3, device="cuda", requires_grad=True)
        shifts = torch.zeros(len(views), 1, 1, 3, device="cuda", requires_grad=True)

        def current() -> dict:
            return {**base, "opacity": torch.sigmoid(params["opacity"]) * keep, "shs": params["rgb"].clamp(0, 1)}

        def render_view(attrs: dict, index: int):
            item = dict(views[index])
            flame = dict(item["flame"])
            flame["rotation"] = flame["rotation"] + turns[index]
            item["flame"] = flame
            return render(attrs, item, shifts[index])

        def held_score(attrs: dict) -> float:
            with torch.no_grad():
                rgb, _ = render_view(attrs, held)
                item = views[held]
                error = (((rgb - item["target"]) ** 2) * item["outer"]).sum() / (item["outer"].sum() * 3 + 1e-6)
                return float(10 * torch.log10(1.0 / error.clamp(min=1e-10)))

        arbiter = gpu_arbiter()
        weights = np.array([views[i]["weight"] for i in train_idx])
        rng = np.random.default_rng(0)
        align_steps = max(100, steps // 5)

        def run(total: int, optimizer, loss_of, share: tuple[float, float], label: str) -> None:
            order = rng.choice(train_idx, size=total, p=weights / weights.sum())
            step_ms, done = 30.0, 0
            while done < total:
                gate = arbiter.video(step_ms + 5, timeout=1.0) if arbiter is not None else None
                granted = gate.__enter__() if gate is not None else True
                try:
                    if not granted:
                        continue
                    tick = time.perf_counter()
                    loss = loss_of(int(order[done]))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if params["rgb"].grad is not None:
                        params["rgb"].grad[~hair_only] = 0
                        params["opacity"].grad[~hair_only] = 0
                    optimizer.step()
                    torch.cuda.synchronize()
                    step_ms = 0.9 * step_ms + 0.1 * (time.perf_counter() - tick) * 1000
                    done += 1
                    if done % 50 == 0:
                        progress(share[0] + (share[1] - share[0]) * done / total, f"{label} {done}/{total}")
                finally:
                    if gate is not None:
                        gate.__exit__(None, None, None)

        frozen = {k: v.detach() for k, v in current().items()}

        # a) Each photo's own camera fit, refined: its turn and position, the head left as LAM made it.
        def align_loss(index: int):
            rgb, alpha = render_view(frozen, index)
            item = views[index]
            return (((rgb - item["target"]).abs() * item["mask"]).sum() / (item["mask"].sum() * 3 + 1e-6)
                    + 0.5 * (alpha - item["mask"]).abs().mean())

        run(align_steps, torch.optim.Adam([{"params": [turns], "lr": 2e-3}, {"params": [shifts], "lr": 2e-4}]),
            align_loss, (0.55, 0.65), "aligning")
        before = held_score(frozen)

        # b) The colours (and opacities) of the hair, ears and outline, kept close to LAM's own.
        def colour_loss(index: int):
            rgb, _ = render_view(current(), index)
            item = views[index]
            mask = item["outer"]
            l1 = ((rgb - item["target"]).abs() * mask).sum() / (mask.sum() * 3 + 1e-6)
            return (0.8 * l1 + 0.2 * ssim_loss(rgb, item["target"], mask)
                    + 2.0 * ((params["rgb"] - rgb0) ** 2).mean() + 0.01 * ((params["opacity"] - opacity0) ** 2).mean())

        turns.requires_grad_(False)
        shifts.requires_grad_(False)
        run(steps, torch.optim.Adam([{"params": [params["rgb"]], "lr": 2e-3},
                                     {"params": [params["opacity"]], "lr": 5e-3}]),
            colour_loss, (0.65, 0.93), "training")
        trained = {k: v.detach() for k, v in current().items()}
        after = held_score(trained)
        print(f"  held-out photo {views[held]['name']}: {before:.2f} -> {after:.2f} dB (hair, ears, outline)",
              flush=True)
        if after > before + 0.1:
            final = trained
            trained_note = f"trained on {len(train_idx)} photos (+{after - before:.1f} dB on an unseen one)"
        else:
            trained_note = f"training did not help an unseen photo ({after - before:+.1f} dB): LAM's head kept"
        if arbiter is not None:
            arbiter.close()

    # 5. The head, and a preview of it from three sides.
    progress(0.95, "saving")
    arkit_up, neck_up = live_extras()
    head = {"xyz": final["xyz"], "opacity": final["opacity"], "rotation": final["rotation"],
            "scaling": final["scaling"], "shs": final["shs"], "betas": shape.reshape(1, -1),
            "arkit": arkit_up, "neck": neck_up, "photo": photo.name, "views": len(views), "version": HEAD_VERSION}
    partial = folder / "head3d.pt.part"
    torch.save({k: (t.detach().float().cpu() if torch.is_tensor(t) and k not in ("arkit", "neck")
                    else t.cpu() if torch.is_tensor(t) else t) for k, t in head.items()}, partial)
    partial.replace(folder / "head3d.pt")
    tiles = []
    with torch.no_grad():
        for yaw in (-0.6, 0.0, 0.6):
            turned = dict(main_view)
            flame = dict(main_view["flame"])
            rotation = cv2.Rodrigues(flame["rotation"][0, 0].cpu().numpy().astype(np.float64))[0]
            spun = cv2.Rodrigues(cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0] @ rotation)[0].ravel()
            flame["rotation"] = torch.tensor(spun, dtype=torch.float32, device="cuda").view(1, 1, 3)
            turned["flame"] = flame
            rgb, alpha = render(final, turned)
            picture = rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            a = alpha.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            picture = ((picture * a + (1 - a)) * 255).astype(np.uint8)[..., ::-1]
            tiles.append(cv2.resize(picture, (256, 256), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(folder / "head3d.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    progress(1.0, f"done in {time.perf_counter() - started:.0f} s from {photo.name}, {trained_note}, "
                  f"peak {peak:.1f} GB, {int(cut.sum())} Gaussians dropped")


if __name__ == "__main__":
    main()
