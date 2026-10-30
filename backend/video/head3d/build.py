"""Build a face's 3D head with LAM (runs in LAM's own environment, cwd = LAM's source folder).

    python build.py <face folder> <BiSeNet parser .onnx> <MediaPipe face_landmarker.task>

Picks the most frontal photo, tracks FLAME on it, builds the Gaussian head (LAM-20K), renders it
from the photo's own camera and drops the Gaussians that sit on clothes or the neck there (the
speaker's own body stays in the picture). Writes ``head3d.pt`` (and ``head3d.jpg``, the head seen
from the photo) into the face folder; prints ``progress <0..1> <stage>`` lines on the way.
"""

import math
import os
import sys
import tempfile
import time
from pathlib import Path

LAM_MODEL = "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
LAM_CONFIG = "./configs/inference/lam-20k-8gpu.yaml"
NECK, NECKLACE, CLOTH = 14, 15, 16   # BiSeNet (CelebAMask-HQ) classes that are body, not head


def progress(done: float, stage: str) -> None:
    print(f"progress {done:.2f} {stage}", flush=True)


def frontal_photo(photos: list[Path], landmarker_path: str) -> Path:
    """The photo where the head turns least (MediaPipe's head pose), among those with a large face."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=landmarker_path),
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
            matrix = found.facial_transformation_matrixes[0]
            yaw = math.degrees(math.atan2(-matrix[2][0], math.hypot(matrix[2][1], matrix[2][2])))
            pitch = math.degrees(math.atan2(matrix[2][1], matrix[2][2]))
            xs = [p.x * image.shape[1] for p in found.face_landmarks[0]]
            candidates.append((abs(yaw) + 0.5 * abs(pitch), max(xs) - min(xs), path))
    if not candidates:
        raise SystemExit("No face found in the photos")
    largest = max(c[1] for c in candidates)
    return min((c for c in candidates if c[1] >= 0.75 * largest), key=lambda c: c[0])[2]


def main() -> None:
    folder, parser_path, landmarker_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    sys.path.insert(0, os.getcwd())
    os.environ.update({"APP_MODEL_NAME": LAM_MODEL, "APP_INFER": LAM_CONFIG, "NUMBA_THREADING_LAYER": "omp"})
    sys.argv = sys.argv[:1]   # LAM's modules read the command line too

    photos = sorted(folder.glob("photo_*.jpg"), key=lambda path: int(path.stem.split("_")[1]))
    progress(0.02, "photo")
    photo = frontal_photo(photos, landmarker_path)

    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from app_lam import _build_model, parse_configs
    from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
    from PIL import Image
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    started = time.perf_counter()
    progress(0.05, "loading")
    cfg, _ = parse_configs()
    lam = _build_model(cfg).to("cuda").eval()
    work = Path(tempfile.mkdtemp(prefix="deepmirror-head3d-"))
    tracking = FlameTrackingSingleImage(
        output_dir=str(work / "tracking"),
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth", detect_iris_landmarks=False)
    raw = work / "raw.png"
    Image.open(photo).convert("RGB").save(raw)
    progress(0.25, "tracking")
    if tracking.preprocess(str(raw)) != 0 or tracking.optimize() != 0:
        raise SystemExit("Could not fit a head to the photo")
    code, tracked = tracking.export()
    if code != 0:
        raise SystemExit("Could not fit a head to the photo")

    progress(0.70, "building")
    image, _, _, shape = preprocess_image(
        os.path.join(tracked, "images/00000_00.png"), mask_path=os.path.join(tracked, "fg_masks/00000_00.png"),
        intr=None, pad_ratio=0, bg_color=1., max_tgt_size=None, aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
        render_tgt_size=cfg.source_size, multiply=14, need_mask=True, get_shape_param=True)
    # The photo's own view (its camera and FLAME pose): each Gaussian lands where it came from.
    seq = prepare_motion_seqs(
        os.path.join(tracked, "flame_param"), None, save_root=str(work), fps=30, bg_color=1., aspect_standard=1.0,
        enlarge_ratio=[1.0, 1, 0], render_image_res=cfg.render_size, multiply=16, need_mask=False,
        vis_motion=False, shape_param=shape, test_sample=False, cross_id=False, src_driven=["src", "src"])
    seq["flame_params"]["betas"] = shape.unsqueeze(0)
    with torch.no_grad():
        result = lam.infer_single_view(
            image.unsqueeze(0).to("cuda"), None, None, render_c2ws=seq["render_c2ws"][:, :1].cuda(),
            render_intrs=seq["render_intrs"][:, :1].cuda(), render_bg_colors=seq["render_bg_colors"][:, :1].cuda(),
            flame_params={k: (v if k == "betas" else v[:, :1]).cuda() for k, v in seq["flame_params"].items()})

    progress(0.90, "cleaning")
    view = (result["comp_rgb"][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
    posed = result["3dgs"]
    while isinstance(posed, (list, tuple)):
        posed = posed[0]
    points = posed.xyz.detach().cpu().numpy()
    c2w, intr = seq["render_c2ws"][0, 0].numpy(), seq["render_intrs"][0, 0].numpy()
    camera = (np.linalg.inv(c2w) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    u = intr[0, 0] * camera[:, 0] / camera[:, 2] + intr[0, 2]
    v = intr[1, 1] * camera[:, 1] / camera[:, 2] + intr[1, 2]

    session = ort.InferenceSession(parser_path, providers=["CPUExecutionProvider"])
    rgb = cv2.resize(view, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
    blob = ((rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
    labels = session.run(None, {session.get_inputs()[0].name: blob})[0][0].argmax(0).astype(np.uint8)
    labels = cv2.resize(labels, view.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    body = cv2.dilate(np.isin(labels, (NECK, NECKLACE, CLOTH)).astype(np.uint8), np.ones((5, 5), np.uint8))
    height, width = body.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    cut = v >= height - 1                      # below the photo's edge: shoulders
    cut[inside] |= body[v[inside].astype(int), u[inside].astype(int)] > 0

    gaussians = result["cano_gs_lst"][0]
    keep = torch.from_numpy(~cut).to(gaussians.opacity.device).float()[:, None]
    head = {"xyz": gaussians.xyz, "opacity": gaussians.opacity * keep, "rotation": gaussians.rotation,
            "scaling": gaussians.scaling, "shs": gaussians.shs, "betas": shape.reshape(1, -1),
            "photo": photo.name, "version": 1}
    partial = folder / "head3d.pt.part"
    torch.save({k: (t.detach().cpu() if torch.is_tensor(t) else t) for k, t in head.items()}, partial)
    partial.replace(folder / "head3d.pt")
    alpha = result["comp_mask"][0][..., 0].cpu().numpy()
    preview = (view * alpha[..., None] + 255 * (1 - alpha[..., None])).astype(np.uint8)
    cv2.imwrite(str(folder / "head3d.jpg"), cv2.resize(preview, (256, 256)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    progress(1.0, f"done in {time.perf_counter() - started:.0f} s, {int(cut.sum())} body Gaussians dropped")


if __name__ == "__main__":
    main()
