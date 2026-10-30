"""Drive a built 3D head live (runs in LAM's own environment, cwd = LAM's source folder).

    python drive.py <compute_arbiter.py> <arbiter name> <face_landmarker.task>
    (connects back to the face process: address and key in DEEPMIRROR_HEAD3D)

For each camera frame from the face process: MediaPipe finds the face's landmarks and its 52 ARKit
blendshapes (they move the eyeballs too, so the gaze follows); the head's pose comes from those
landmarks (PnP against the head's own FLAME landmarks, through a One-Euro filter: steady when still,
no lag when moving); the Gaussians move with FLAME's ARKit blendshapes and that pose; a camera
turned to look at the head renders it at the size it has on screen, and the render is warped back
into the frame's own camera. The head's light is then matched to the speaker's skin in the frame.
What goes back is the head layer (BGRA, straight alpha) and where it sits in the frame.

Detection and rendering run on two threads, so one frame renders while the next one is tracked.
The GPU is shared through the voice's arbiter: a render starts only in a gap the voice leaves, and
the process runs below normal CPU priority. Messages carry only plain Python values and bytes (the
two environments have different NumPys).
"""

import importlib.util
import math
import os
import pickle
import queue
import sys
import threading
import time
from multiprocessing.connection import Client

import cv2
import numpy as np
import torch

ARKIT = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
         "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
         "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
         "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
         "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
         "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
         "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
         "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
         "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
         "noseSneerLeft", "noseSneerRight", "tongueOut"]   # the order of FLAME's ARKit blendshapes
# Blinks and the jaw follow at once; the rest is smoothed a little (MediaPipe's scores flicker).
FAST = torch.tensor([name.startswith(("eyeBlink", "jaw", "mouthClose", "mouthFunnel", "mouthPucker"))
                     for name in ARKIT])
# MediaPipe's mesh points for iBUG 68 points 17..67 (brows, nose, eyes, mouth; the jaw line is left
# out: on a turning head the visible outline is not a fixed point of the mesh).
MEDIAPIPE_68 = [71, 63, 105, 66, 107, 336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160,
                158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84,
                181, 78, 82, 13, 312, 308, 317, 14, 87]
# The face's outline and what is not skin inside it (for the light match).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOT_SKIN = [[70, 63, 105, 66, 107, 133, 144, 145, 153, 33], [336, 296, 334, 293, 300, 263, 373, 374, 380, 362],
            [61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181]]
FLAME = "./model_zoo/human_parametric_models/flame_assets/flame"
RENDER_MIN, RENDER_MAX = 256, 640   # render side, pixels: the head's size on screen, within these
HEAD_SPAN = 2.6        # the rendered square, in face widths (hair included)
FOCAL = 0.96           # assumed webcam focal length, in frame widths (~55 degrees across)
DETECT_WIDTH = 640     # MediaPipe reads a copy this wide (its landmarks are relative: no loss)
NECK_FADE = 0.012      # metres below the chin over which the photo's neck fades out
LIGHT_FOLLOW = 0.25    # share of the new light estimate taken each frame


class OneEuro:
    """One-Euro filter (Casiez et al. 2012): steady when still, little lag when moving."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx = None
        self.t = 0.0

    @staticmethod
    def _alpha(cutoff, dt: float):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class Head:
    """A built head (``head3d.pt``) and the FLAME model that moves it."""

    def __init__(self, path: str) -> None:
        from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
        from lam.models.rendering.flame_model.lbs import blend_shapes, vertices2landmarks

        flame = FlameHeadSubdivided(
            300, 100, add_teeth=False, add_shoulder=False, flame_model_path=f"{FLAME}/flame2023.pkl",
            flame_lmk_embedding_path=f"{FLAME}/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{FLAME}/head_template_mesh.obj", flame_parts_path=f"{FLAME}/FLAME_masks.pkl",
            subdivide_num=1, teeth_bs_flag=False, oral_mesh_flag=False)
        device = "cuda"
        data = torch.load(path, map_location="cpu")
        arkit = torch.from_numpy(np.load(f"{FLAME}/flame_arkit_bs.npy").astype(np.float32))   # 52 x 5023 x 3
        if "arkit" in data:   # built with them (version 2): nothing to work out here
            arkit_up, neck_up = data["arkit"].float(), data["neck"].float()[:, None]
        else:
            with open(f"{FLAME}/FLAME_masks.pkl", "rb") as file:
                regions = pickle.load(file, encoding="latin1")
            neck = torch.zeros(arkit.shape[1], 1)
            neck[torch.as_tensor(np.concatenate([regions["neck"], regions["boundary"]]))] = 1
            with torch.no_grad():   # onto the head's 20k points (on the CPU, before .cuda())
                _, up, neck_up = flame.upsample_mesh_cpu(flame.v_template.float(),
                                                         [arkit.permute(1, 0, 2).reshape(arkit.shape[1], -1), neck])
            arkit_up = up.reshape(-1, 52, 3).permute(1, 0, 2).contiguous()
        self.flame = flame.to(device)
        self.arkit = arkit.to(device)
        self.arkit_up = arkit_up.to(device)
        self.xyz = data["xyz"].float().to(device)
        if self.arkit_up.shape[1] != self.xyz.shape[0]:
            raise ValueError("This 3D head was built for another head model")
        self.rotation, self.scaling = data["rotation"].float().to(device), data["scaling"].float().to(device)
        self.rgb = data["shs"].float().reshape(self.xyz.shape[0], -1)[:, :3].contiguous().to(device)
        betas = data["betas"].float().reshape(1, -1).to(device)
        self.v_shaped = self.flame.v_template[None] + blend_shapes(
            torch.cat([betas, torch.zeros(1, 100, device=device)], 1), self.flame.shapedirs)
        self.root = self.flame.J_regressor[0] @ self.v_shaped[0]
        faces, bary = self.flame.full_lmk_faces_idx, self.flame.full_lmk_bary_coords
        chin = vertices2landmarks(self.v_shaped, self.flame.faces, faces[:, 8:9], bary[:, 8:9])[0, 0]
        below = ((chin[1] - self.xyz[:, 1]) / NECK_FADE).clamp(0, 1)[:, None]
        # Only the head: the photo's neck fades out (the speaker's own neck and body stay).
        # A clean cut, not a fade: half-transparent Gaussians would show as specks along the jaw.
        neck_cut = (1 - 2 * neck_up.to(device)).clamp(0, 1)
        self.opacity = (data["opacity"].float().to(device) * neck_cut * (1 - below)).contiguous()
        self._landmarks = (faces[:, 17:68], bary[:, 17:68], vertices2landmarks)

    @torch.no_grad()
    def landmarks(self, weights: torch.Tensor) -> np.ndarray:
        """The head's landmarks 17..67 (metres, FLAME space) with this expression."""
        faces, bary, to_landmarks = self._landmarks
        posed = self.v_shaped + torch.einsum("k,kvc->vc", weights, self.arkit)[None]
        return to_landmarks(posed, self.flame.faces, faces, bary)[0].double().cpu().numpy()

    @torch.no_grad()
    def render(self, weights: torch.Tensor, rotation: np.ndarray, translation: np.ndarray, look: np.ndarray,
               focal: float, size: int) -> torch.Tensor:
        """size x size x 4 (BGR, straight alpha, 0..1) of the head posed in the camera's frame,
        seen by a camera at the same place turned by ``look``."""
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        device = self.xyz.device
        R = torch.from_numpy(rotation).float().to(device)
        axis = torch.from_numpy(cv2.Rodrigues(rotation)[0].ravel()).float().to(device)
        pose = torch.cat([axis, torch.zeros(12, device=device)])[None]   # global; neck, jaw, eyes still
        transforms, _ = self.flame.get_transformed_mat(
            pose=pose, v_shaped=self.v_shaped, posedirs=self.flame.posedirs, parents=self.flame.parents,
            J_regressor=self.flame.J_regressor, pose2rot=True, dtype=torch.float32)
        moved = self.xyz + torch.einsum("k,kvc->vc", weights, self.arkit_up)
        moved = self.flame.skinning(v_posed=moved[None], A=transforms, lbs_weights=self.flame.lbs_weights_up,
                                    batch_size=1, num_joints=self.flame.joint_num, dtype=torch.float32,
                                    device=device)[0]
        # FLAME turns about its root joint; the pose turns about the origin.
        moved = moved + (torch.from_numpy(translation).float().to(device) - self.root + R @ self.root)

        view = torch.eye(4, device=device)
        view[:3, :3] = torch.from_numpy(look).float()
        tan = size / (2 * focal)
        near, far = 0.01, 100.0
        projection = torch.zeros(4, 4, device=device)
        projection[0, 0] = projection[1, 1] = 1 / tan
        projection[3, 2] = 1.0
        projection[2, 2] = far / (far - near)
        projection[2, 3] = -far * near / (far - near)
        settings = GaussianRasterizationSettings(
            image_height=size, image_width=size, tanfovx=tan, tanfovy=tan, bg=torch.zeros(3, device=device),
            scale_modifier=1.0, viewmatrix=view.T, projmatrix=view.T @ projection.T, sh_degree=0,
            campos=torch.zeros(3, device=device), prefiltered=False, debug=False)
        image, _, _, alpha = GaussianRasterizer(settings)(
            means3D=moved, means2D=torch.zeros_like(moved), shs=None, colors_precomp=self.rgb,
            opacities=self.opacity, scales=self.scaling, rotations=self.rotation, cov3D_precomp=None)
        bgr = image.flip(0) / alpha.clamp(min=1e-3)
        return torch.cat([bgr, alpha], 0).permute(1, 2, 0).clamp(0, 1)


class Light:
    """The head's light moved to the speaker's: LAB statistics of the skin inside the face (eyes,
    brows and mouth left out), in the frame and in the head layer; brightness and contrast fully,
    colour cast part way (the worn skin tone stays), followed smoothly over time."""

    def __init__(self) -> None:
        self.gain = 1.0
        self.shift = np.zeros(3, np.float32)

    def reset(self) -> None:
        self.gain, self.shift = 1.0, np.zeros(3, np.float32)

    def apply(self, layer: np.ndarray, frame: np.ndarray, mesh: np.ndarray) -> np.ndarray:
        """``layer``: BGRA over ``frame`` (the same region); ``mesh``: MediaPipe points in it."""
        skin = np.zeros(layer.shape[:2], np.uint8)
        cv2.fillConvexPoly(skin, cv2.convexHull(mesh[FACE_OVAL].astype(np.int32)), 1)
        for part in NOT_SKIN:
            cv2.fillConvexPoly(skin, cv2.convexHull(mesh[part].astype(np.int32)), 0)
        skin = cv2.erode(skin, np.ones((7, 7), np.uint8))
        both = (skin > 0) & (layer[..., 3] > 240)
        if both.sum() > 400:
            step = max(1, int(math.sqrt(both.sum() / 4000)))   # a few thousand pixels are plenty
            pick = both[::step, ::step]
            ours = cv2.cvtColor(layer[::step, ::step, :3], cv2.COLOR_BGR2LAB)[pick].astype(np.float32)
            theirs = cv2.cvtColor(frame[::step, ::step], cv2.COLOR_BGR2LAB)[pick].astype(np.float32)
            gain = float(np.clip((theirs[:, 0].std() + 1) / (ours[:, 0].std() + 1), 0.75, 1.35))
            shift = theirs.mean(0) - ours.mean(0) * np.array([gain, 1, 1], np.float32)
            shift[1:] *= 0.6
            self.gain += LIGHT_FOLLOW * (gain - self.gain)
            self.shift += LIGHT_FOLLOW * (shift - self.shift)
        lab = cv2.cvtColor(layer[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] *= self.gain
        lab += self.shift
        out = layer.copy()
        out[..., :3] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return out


def below_normal_priority() -> None:
    """The voice's process comes first on the CPU too."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    else:
        os.nice(5)


def main() -> None:
    arbiter_file, arbiter_name, landmarker_path = sys.argv[1:4]
    sys.path.insert(0, os.getcwd())
    below_normal_priority()
    host, port, key = os.environ["DEEPMIRROR_HEAD3D"].split(":")
    conn = Client((host, int(port)), authkey=bytes.fromhex(key))
    send_lock = threading.Lock()

    def send(*message) -> None:
        with send_lock:
            conn.send(message)

    spec = importlib.util.spec_from_file_location("compute_arbiter", arbiter_file)
    assert spec is not None and spec.loader is not None
    arbiter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arbiter_module)
    arbiter = arbiter_module.GpuArbiter(arbiter_name)

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_path), output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO, num_faces=1))
    torch.set_num_threads(1)
    cv2.setNumThreads(2)
    head: Head | None = None
    light = Light()
    tracked: queue.Queue = queue.Queue(maxsize=2)
    state: dict = {"rvec": None, "tvec": None, "weights": None, "stamp": 0,
                   "rotation": OneEuro(min_cutoff=1.2, beta=0.6), "position": OneEuro(min_cutoff=1.2, beta=4.0)}
    timings = {"track_ms": 0.0, "render_ms": 3.0, "render_size": 0}

    def reset() -> None:
        state["rvec"] = state["tvec"] = state["weights"] = None
        state["rotation"].x = state["position"].x = None
        light.reset()

    def track(seq: int, t: float, frame: np.ndarray) -> None:
        """MediaPipe and the pose for one frame; to the render thread."""
        if head is None:
            send("layer", seq, None, timings)
            return
        started = time.perf_counter()
        height, width = frame.shape[:2]
        scale = min(1.0, DETECT_WIDTH / width)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        state["stamp"] = max(state["stamp"] + 1, int(t * 1000))
        found = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), state["stamp"])
        if not found.face_landmarks:
            reset()
            send("layer", seq, None, timings)
            return
        mesh = np.array([[p.x * width, p.y * height] for p in found.face_landmarks[0]])
        points = mesh[MEDIAPIPE_68]
        scores = {c.category_name: c.score for c in found.face_blendshapes[0]}
        weights = torch.tensor([scores.get(name, 0.0) for name in ARKIT])
        if state["weights"] is not None:
            weights = torch.where(FAST, weights, 0.6 * weights + 0.4 * state["weights"])
        state["weights"] = weights
        weights = weights.to("cuda")
        focal = FOCAL * max(width, height)
        camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        model = head.landmarks(weights)
        guess = state["rvec"] is not None
        ok, rvec, tvec = cv2.solvePnP(model, points, camera, None, state["rvec"], state["tvec"],
                                      useExtrinsicGuess=guess, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            send("layer", seq, None, timings)
            return
        state["rvec"], state["tvec"] = rvec, tvec   # the raw fit: the next frame's starting guess
        smooth_r = state["rotation"](rvec.ravel(), t)
        smooth_t = state["position"](tvec.ravel(), t)
        timings["track_ms"] = 0.9 * timings["track_ms"] + 0.1 * (time.perf_counter() - started) * 1000
        tracked.put((seq, frame, mesh, weights, cv2.Rodrigues(smooth_r.reshape(3, 1))[0], smooth_t, model, points,
                     camera))

    def render_loop() -> None:
        while True:
            item = tracked.get()
            if item is None:
                return
            try:
                layer = place(*item[1:])
            except Exception as exc:
                send("log", 40, f"3D head render failed: {type(exc).__name__}: {exc}")
                layer = None
            send("layer", item[0], layer, timings)

    def place(frame, mesh, weights, rotation, translation, model, points, camera):
        """The head layer for one tracked frame: (x, y, width, height, BGRA bytes), or None."""
        height, width = frame.shape[:2]
        centre = rotation @ model.mean(0) + translation
        distance = float(np.linalg.norm(centre))
        span_px = HEAD_SPAN * float(np.ptp(points[:, 0]))
        span = span_px * centre[2] / camera[0, 0]
        # Rendered at about the size it has on screen: sharp, and no GPU spent on pixels never seen.
        size = int(min(RENDER_MAX, max(RENDER_MIN, round(span_px / 32) * 32)))
        # A camera at the same place turned to look at the head: the head fills the render.
        z = centre / distance
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        look = np.stack([x, np.cross(z, x), z])
        focal = size * distance / span
        with arbiter.video(timings["render_ms"] + 1.0, timeout=0.1) as granted:
            if not granted or head is None:
                return None
            started = time.perf_counter()
            image = (head.render(weights, rotation, translation, look, focal, size) * 255).to(torch.uint8)
            image = image.cpu().numpy()
            timings["render_ms"] = 0.9 * timings["render_ms"] + 0.1 * (time.perf_counter() - started) * 1000
            timings["render_size"] = size
        # Back into the frame's own camera (a pure turn of the camera: a homography).
        inner = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]])
        warp = camera @ look.T @ np.linalg.inv(inner)
        corners = cv2.perspectiveTransform(np.array([[[0, 0], [size, 0], [0, size], [size, size]]], np.float64), warp)[0]
        x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int), [width, height])
        if x1 <= x0 or y1 <= y0:
            return None
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        warped = cv2.warpPerspective(image, shift @ warp, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR)
        warped = light.apply(warped, frame[y0:y1, x0:x1], mesh - [x0, y0])
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0), warped.tobytes()

    renderer = threading.Thread(target=render_loop, name="head3d-render", daemon=True)
    renderer.start()
    send("ready")
    try:
        while True:
            message = conn.recv()
            kind = message[0]
            if kind == "stop":
                break
            if kind == "head":
                try:
                    head = Head(message[1]) if message[1] else None
                    reset()
                    send("loaded", message[1], None)
                except Exception as exc:
                    head = None
                    send("loaded", message[1], f"{type(exc).__name__}: {exc}")
            elif kind == "frame":
                _, seq, t, width, height, data = message
                frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
                try:
                    track(seq, t, frame)
                except Exception as exc:
                    send("log", 40, f"3D head tracking failed: {type(exc).__name__}: {exc}")
                    send("layer", seq, None, timings)
    except (EOFError, OSError, KeyboardInterrupt):
        pass
    finally:
        tracked.put(None)
        arbiter.close()


if __name__ == "__main__":
    main()
