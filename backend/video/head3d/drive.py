"""Drive a built 3D head live (runs in LAM's own environment, cwd = LAM's source folder).

    python drive.py <compute_arbiter.py> <arbiter name> <face_landmarker.task>
    (connects back to the face process: address and key in DEEPMIRROR_HEAD3D)

For each camera frame from the face process: MediaPipe finds the face's landmarks and its 52 ARKit
blendshapes (they move the eyeballs too, so the gaze follows); the head's pose comes from those
landmarks (PnP against the head's own FLAME landmarks, through a One-Euro filter: steady when still,
no lag when moving); the Gaussians move with FLAME's ARKit blendshapes and that pose; a camera
turned to look at the head renders it at the size it has on screen, and the render is warped back
into the frame's own camera. The head's light is then matched to the speaker's skin in the frame.
What goes back is the head layer (BGRA, straight alpha), where it sits in the frame, and the face's
five points (the studio paints the speaker's own hair over by them).

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
FRINGE_WHITE = 0.8     # a Gaussian's darkest channel above this may be the build's white, not the person's
HAIR_NODES = 32        # guide nodes of the hair's motion
HAIR_ROOT, HAIR_TIP = 0.004, 0.05    # metres from the scalp where a Gaussian starts to follow, and follows fully
HAIR_STIFF, HAIR_DAMP, HAIR_TIE = 60.0, 7.0, 25.0   # the springs (1/s^2, 1/s, 1/s^2)
HAIR_GRAVITY, HAIR_INERTIA = 0.25, 0.03             # how strongly gravity and the head's motion pull
HAIR_REACH = 0.025     # metres: the most a node moves
# The five points the studio places faces by (eyes, nose tip, mouth corners), from MediaPipe's mesh.
FIVE = [(33, 133), (362, 263), (4,), (61,), (291,)]
FLAME = "./model_zoo/human_parametric_models/flame_assets/flame"
RENDER_MIN = 256       # render side, pixels: the head's size on screen, down to this...
RENDER_STEPS = (256, 320, 384, 448, 512, 640, 768, 1024)   # ...and up to one of these, as far as the GPU has room
RENDER_START = 2       # (384)
RENDER_BUDGET_VOICE, RENDER_BUDGET = 9.0, 16.0    # ms a render may take with the voice streaming, and without
FRAME_MARGIN = 1.10    # the rendered square: the head's projected extent and this margin
EDGE_LOW, EDGE_HIGH = 0.08, 0.7   # alpha contrast at the head's outline: soft enough to keep fine strands
EDGE_DEPTH = 7         # pixels: how far inside the outline the fringe takes its colour from
SHARPEN = 0.35         # unsharp mask amount on the rendered head
HOLE_RISE = 0.04       # the face hole's top: just above the upper eyelids
HOLE_RISE_BARE = 0.42  # without a fringe: up the forehead (eye distances above the upper eyelids)
HOLE_INSET = 0.9       # above the nose the hole is this much of the face (the swap's fade lies on 3D skin)
HOLE_WIDEN = 1.18      # below the nose the hole is this much wider than the face
HOLE_SOFT = 0.006      # its fade, in face widths (short: a half-faded 3D forehead shows what is under it)
# The upper eyelids: the face hole's top is at their crease. Brows and forehead come from the 3D head
# (they move with its brow blendshapes): a face swap of a forehead under a fringe draws the fringe's
# shadow, and the worn person's brows are part of who they are.
BROWS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 362, 398, 384, 385, 386, 387, 388, 466, 263]
FOCAL = 0.96           # a webcam's usual focal length, in frame widths (~55 degrees across)
DETECT_WIDTH = 640     # MediaPipe reads a copy this wide (its landmarks are relative: no loss)
NECK_FADE = 0.012      # metres below the chin over which the photo's neck fades out
LIGHT_FOLLOW = 0.25    # share of the new light estimate taken each frame
LIGHT_EVERY = 4        # frames between light measures (the light hardly changes in between)
LIGHT_SIZE = 128       # the render's copy the light is measured on


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
        with torch.no_grad():
            scalp = self.flame.get_cano_verts(betas)[0]
        reach = (self.xyz - scalp).norm(dim=1) * (self.opacity[:, 0] > 0.05).float()   # how far from the scalp
        self.rgb = without_white_fringe(self.xyz, self.rgb)
        self.hair = HairMotion(self.xyz, reach)
        # The head's own extent (what is visible of it, hair included): the render is framed on it,
        # so no pose ever cuts the hair or the chin. Offsets from the root joint, 8 corners.
        # (Percentiles: a few stray Gaussians must not blow the frame up and leave the head a few
        # blurred pixels in it.)
        seen = self.xyz[self.opacity[:, 0] > 0.05]
        low = torch.quantile(seen, 0.005, dim=0) - 0.01
        high = torch.quantile(seen, 0.995, dim=0) + 0.01
        self.opacity = self.opacity * ((self.xyz >= low - 0.03) & (self.xyz <= high + 0.03)).all(1, keepdim=True)
        corners = torch.stack([torch.stack([x, y, z]) for x in (low[0], high[0]) for y in (low[1], high[1])
                               for z in (low[2], high[2])])
        self.corners = corners.cpu().double().numpy()

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
        sway = self.hair.displacement(device)
        if sway is not None:
            moved = moved + sway
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
            opacities=self.opacity, scales=self.scaling, rotations=turned_quaternions(self.rotation, R),
            cov3D_precomp=None)
        bgr = image.flip(0) / alpha.clamp(min=1e-3)
        return torch.cat([bgr, alpha], 0).permute(1, 2, 0).clamp(0, 1)


def without_white_fringe(xyz: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Colours with the build's white background taken out. LAM builds against white, so Gaussians
    around the hair's outline are nearly white (a pale line along it; a blond head's highlights stay
    below the limit). They take the colour of their nearest ordinary neighbours."""
    whiteness = rgb.min(dim=1).values
    white = whiteness > max(FRINGE_WHITE, float(torch.quantile(whiteness, 0.98)) if len(whiteness) else 1.0)
    white |= whiteness > 0.9
    plain = torch.nonzero(~white)[:, 0]
    if not white.any() or len(plain) < 16:
        return rgb
    fixed = rgb.clone()
    for chunk in torch.nonzero(white)[:, 0].split(2048):
        nearest = torch.cdist(xyz[chunk], xyz[plain]).topk(8, largest=False).indices
        fixed[chunk] = rgb[plain[nearest]].median(dim=1).values
    return fixed


def turned_quaternions(quaternions: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """The Gaussians' orientations (w, x, y, z) turned with the head. LAM moves their centres only:
    long thin Gaussians (hair strands) would keep pointing the same way while the head turns."""
    m = rotation
    w = torch.sqrt((1 + m[0, 0] + m[1, 1] + m[2, 2]).clamp(min=1e-8)) / 2
    if w < 1e-3:   # a half turn: the formula below has no w to divide by
        i = int(torch.argmax(torch.diagonal(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = torch.sqrt((1 + m[i, i] - m[j, j] - m[k, k]).clamp(min=1e-8)) * 2
        v = [0.0, 0.0, 0.0]
        v[i], v[j], v[k] = s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
        w, x, y, z = (m[k, j] - m[j, k]) / s, v[0], v[1], v[2]
    else:
        x, y, z = (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)
    q0, q1, q2, q3 = quaternions.unbind(-1)
    return torch.stack([w * q0 - x * q1 - y * q2 - z * q3, w * q1 + x * q0 + y * q3 - z * q2,
                        w * q2 - x * q3 + y * q0 + z * q1, w * q3 + x * q2 - y * q1 + z * q0], -1)


def solve_pose(model: np.ndarray, points: np.ndarray, camera: np.ndarray, rvec=None, tvec=None):
    """The head's pose from its landmarks, always in front of the camera. PnP has a mirror solution
    behind the camera that fits the points just as well (the head then renders inside out, huge
    and blurred), so it starts from a head facing the camera and refuses that solution."""
    width_m = float(np.ptp(model[:, 0])) or 0.1
    width_px = float(np.ptp(points[:, 0])) or 1.0
    facing = (np.array([[math.pi], [0.0], [0.0]]),   # FLAME looks along +z with y up; the camera along +z with y down
              np.array([[0.0], [0.0], [float(camera[0, 0]) * width_m / width_px]]))
    for start in ((rvec, tvec), facing) if rvec is not None else (facing,):
        ok, r, t = cv2.solvePnP(model, points, camera, None, start[0].copy(), start[1].copy(), useExtrinsicGuess=True,
                                flags=cv2.SOLVEPNP_ITERATIVE)
        if ok and t[2, 0] > 0:
            return True, r, t
    return False, None, None


def _ycc(bgr):
    """BGR (0..1, last axis) to Y, Cb, Cr: a linear transform, so it costs nothing on the GPU."""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    stack = torch.stack if torch.is_tensor(bgr) else np.stack
    return stack([y, 0.564 * (b - y), 0.713 * (r - y)], -1)


def _bgr(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    r, b = y + 1.403 * cr, y + 1.773 * cb
    stack = torch.stack if torch.is_tensor(ycc) else np.stack
    return stack([b, (y - 0.299 * r - 0.114 * b) / 0.587, r], -1)


class RenderBudget:
    """How large the head may render: as sharp as this GPU allows next to the voice. A step up only
    when the render is well under budget at the present limit, a step down at once when over it."""

    def __init__(self) -> None:
        self.step = RENDER_START
        self.calm = 0

    @property
    def limit(self) -> int:
        return RENDER_STEPS[self.step]

    def update(self, render_ms: float, size: int, voice: bool) -> None:
        allowed = RENDER_BUDGET_VOICE if voice else RENDER_BUDGET
        if render_ms > allowed and self.step > 0:
            self.step -= 1
            self.calm = 0
        elif size >= self.limit and render_ms < 0.55 * allowed and self.step < len(RENDER_STEPS) - 1:
            self.calm += 1
            if self.calm >= 60:
                self.step += 1
                self.calm = 0
        else:
            self.calm = 0


class HairMotion:
    """The hair's own motion: a few guide nodes among the hair Gaussians, each a damped spring pulled
    by the head's acceleration and by gravity as the head tilts (in the head's own frame), and
    tied to its neighbours. The further a Gaussian sits from the scalp, the more it follows them,
    so short hair barely moves and long hair sways. No training and no network: well under a
    millisecond a frame. (The idea of cage-driven hair in CompHairHead; motion from angular velocity,
    acceleration and relative gravity as in DynHair.)"""

    def __init__(self, xyz: torch.Tensor, reach: torch.Tensor, nodes: int = HAIR_NODES) -> None:
        device = xyz.device
        self.weight = ((reach - HAIR_ROOT) / HAIR_TIP).clamp(0, 1)[:, None]
        hair = self.weight[:, 0] > 0.02
        self.active = int(hair.sum()) >= nodes * 4
        if not self.active:
            return
        points = xyz[hair]
        # Guide nodes: well-spread hair Gaussians (farthest-point sampling).
        chosen = [int(torch.argmax(self.weight[hair, 0]))]
        distance = torch.full((len(points),), 1e9, device=device)
        for _ in range(nodes - 1):
            distance = torch.minimum(distance, ((points - points[chosen[-1]]) ** 2).sum(1))
            chosen.append(int(torch.argmax(distance)))
        centres = points[chosen]
        # Each hair Gaussian follows its nearest nodes, softly.
        near = torch.cdist(xyz, centres)
        values, self.index = near.topk(3, largest=False)
        soft = torch.softmax(-values / 0.015, dim=1)
        self.blend = soft * self.weight
        between = torch.cdist(centres, centres).cpu().numpy()
        self.neighbours = np.argsort(between, axis=1)[:, 1:5]
        # Tips swing more freely than what sits near the scalp.
        tip = self.weight[hair, 0][chosen].cpu().numpy()
        self.stiffness = HAIR_STIFF * (1.6 - 0.8 * tip)[:, None]
        self.offset = np.zeros((nodes, 3), np.float64)
        self.velocity = np.zeros((nodes, 3), np.float64)
        self.reset()

    def reset(self) -> None:
        self.last_t = None
        self.last_position = None
        self.last_velocity = None
        if self.active:
            self.offset[:] = 0.0
            self.velocity[:] = 0.0

    def step(self, t: float, rotation: np.ndarray, position: np.ndarray) -> None:
        """Advance to time ``t`` with the head at ``rotation``, ``position`` (camera frame, y down)."""
        if not self.active:
            return
        if self.last_t is None or t <= self.last_t or t - self.last_t > 0.3:
            self.last_t, self.last_position, self.last_velocity = t, position.copy(), np.zeros(3)
            self.offset *= 0.5
            self.velocity[:] = 0.0
            return
        dt = t - self.last_t
        velocity = (position - self.last_position) / dt
        acceleration = (velocity - self.last_velocity) / dt
        self.last_t, self.last_position, self.last_velocity = t, position.copy(), velocity
        # In the head's own frame: gravity (camera +y is down) and the head's acceleration the other way.
        drive = rotation.T @ (np.array([0.0, 9.81, 0.0]) * HAIR_GRAVITY - np.clip(acceleration, -30.0, 30.0))
        rest = rotation.T @ (np.array([0.0, 9.81, 0.0]) * HAIR_GRAVITY)   # hanging as in the photo: no shift
        force = (drive - rest) * HAIR_INERTIA
        tied = self.offset[self.neighbours].mean(1)
        for _ in range(max(1, int(dt / 0.012))):
            h = dt / max(1, int(dt / 0.012))
            pull = force - self.stiffness * self.offset - HAIR_DAMP * self.velocity + HAIR_TIE * (tied - self.offset)
            self.velocity += h * pull
            self.offset += h * self.velocity
        self.offset = np.clip(self.offset, -HAIR_REACH, HAIR_REACH)
        self.last_t = t

    def displacement(self, device) -> torch.Tensor | None:
        """Each Gaussian's displacement now (head frame), or None without hair to move."""
        if not self.active:
            return None
        offsets = torch.from_numpy(self.offset).float().to(device)
        return (offsets[self.index] * self.blend[..., None]).sum(1)


class Light:
    """The head's light moved to the speaker's: statistics of the skin inside the face (eyes, brows
    and mouth left out), in the frame and in the head's render; brightness and contrast fully, the
    colour cast part way and only on skin tones (the worn skin tone stays; dark hair keeps its own
    colour), followed smoothly over time. Measured on small copies, applied on the GPU."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.gain, self.shift, self.slope = 1.0, np.zeros(3, np.float32), 0.0

    @staticmethod
    def _skin(shape, mesh: np.ndarray) -> np.ndarray:
        skin = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(skin, cv2.convexHull(mesh[FACE_OVAL].astype(np.int32)), 1)
        for part in NOT_SKIN:
            cv2.fillConvexPoly(skin, cv2.convexHull(mesh[part].astype(np.int32)), 0)
        return cv2.erode(skin, np.ones((3, 3), np.uint8)) > 0

    def measure(self, small: np.ndarray, small_mesh: np.ndarray, frame: np.ndarray, mesh: np.ndarray) -> None:
        """``small``: the render (BGRA 0..1) at low resolution, ``small_mesh`` the face in it;
        ``frame`` and ``mesh``: the camera frame and the face in it."""
        ours_mask = self._skin(small.shape[:2], small_mesh) & (small[..., 3] > 0.95)
        x0, y0 = np.maximum(mesh.min(0).astype(int), 0)
        x1, y1 = mesh.max(0).astype(int)
        crop = frame[y0:y1, x0:x1]
        if ours_mask.sum() < 50 or crop.size == 0:
            return
        scale = min(1.0, 128.0 / max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        theirs_mask = self._skin(crop.shape[:2], (mesh - [x0, y0]) * scale)
        if theirs_mask.sum() < 50:
            return
        ours = _ycc(small[..., :3][ours_mask].astype(np.float32))
        theirs = _ycc(crop[theirs_mask].astype(np.float32) / 255.0)
        gain = float(np.clip((theirs[:, 0].std() + 0.004) / (ours[:, 0].std() + 0.004), 0.75, 1.35))
        shift = theirs.mean(0) - ours.mean(0) * np.array([gain, 1, 1], np.float32)
        shift[1:] *= 0.6
        self.gain += LIGHT_FOLLOW * (gain - self.gain)
        self.shift += LIGHT_FOLLOW * (shift - self.shift)
        # Where the light comes from: the speaker's skin brighter on one side than the head's is.
        def tilt(mask, picture):
            middle = (np.nonzero(mask)[1].min() + np.nonzero(mask)[1].max()) / 2
            xs = np.nonzero(mask)[1]
            luma = _ycc(picture)[..., 0][mask]
            return float(luma[xs > middle].mean() - luma[xs <= middle].mean()) if (xs > middle).any() and (xs <= middle).any() else 0.0

        slope = tilt(theirs_mask, crop.astype(np.float32) / 255.0) - tilt(ours_mask, small[..., :3].astype(np.float32))
        self.slope += LIGHT_FOLLOW * (float(np.clip(slope, -0.25, 0.25)) - self.slope)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """``image``: H x W x 4 (BGR, straight alpha, 0..1) on the GPU."""
        ycc = _ycc(image[..., :3])
        tone = ((ycc[..., :1] - 0.24) / 0.27).clamp(0, 1)
        shift = torch.as_tensor(self.shift, device=image.device)
        across = torch.linspace(-1.0, 1.0, image.shape[1], device=image.device)[None, :, None]
        ycc = torch.cat([ycc[..., :1] * self.gain + shift[0] + self.slope * across, ycc[..., 1:] + shift[1:] * tone], -1)
        return torch.cat([_bgr(ycc).clamp(0, 1), image[..., 3:]], -1)


def _gauss(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _blur(planes: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of C x H x W planes (separable)."""
    kernel = _gauss(sigma, planes.device)
    channels, size = planes.shape[0], kernel.numel()
    out = torch.nn.functional.conv2d(planes[None], kernel.view(1, 1, 1, size).expand(channels, 1, 1, size),
                                     padding=(0, size // 2), groups=channels)
    out = torch.nn.functional.conv2d(out, kernel.view(1, 1, size, 1).expand(channels, 1, size, 1),
                                     padding=(size // 2, 0), groups=channels)
    return out[0]


def clean_edges(image: torch.Tensor) -> torch.Tensor:
    """A crisp head outline: the Gaussians' soft fringe given contrast, the fringe's colour taken
    from just inside it (LAM's fringe carries the photo's background colour: the halo), and the
    detail lightly sharpened. ``image``: H x W x 4 (BGR, straight alpha, 0..1), on the GPU."""
    planes = image.permute(2, 0, 1)
    colour, alpha = planes[:3], planes[3:]
    shaped = ((alpha - EDGE_LOW) / (EDGE_HIGH - EDGE_LOW)).clamp(0, 1)
    shaped = shaped * shaped * (3 - 2 * shaped)
    # The fringe's colour: from well inside the outline (even opaque Gaussians on the outline carry
    # the white the head was built against).
    core = -torch.nn.functional.max_pool2d(-(alpha > 0.92).float()[None], EDGE_DEPTH, 1, EDGE_DEPTH // 2)[0]
    blurred = _blur(torch.cat([colour * core, core]), 3.0)
    weight = blurred[3:]
    inside = blurred[:3] / weight.clamp(min=1e-4)
    fringe = (1 - core) * (weight > 0.02).float()
    colour = colour * (1 - fringe) + inside * fringe
    colour = colour + SHARPEN * (colour - _blur(colour, 1.0))
    return torch.cat([colour.clamp(0, 1), shaped]).permute(1, 2, 0)


def face_hole(shape: tuple[int, int], mesh: np.ndarray, fringe: bool = True) -> np.ndarray:
    """0..1 where the face swap shows instead of the 3D head: the face itself (from just above the
    brows; the 3D forehead and hairline overlap the swap's top) and, below the nose, everything a
    little wider than the face down to the bottom (the photo's jaw and stubble never ring the
    speaker's own jaw). Faded at its edge; worked out at a quarter of the size (it is smooth)."""
    eyes = np.linalg.norm(mesh[33] - mesh[263]) or 1.0
    chin, between = mesh[152], (mesh[33] + mesh[263]) / 2
    up = (between - chin) / (np.linalg.norm(between - chin) or 1.0)
    side = np.array([up[1], -up[0]])
    # With a fringe the forehead's swap draws the fringe's shadow: the 3D head comes down to the
    # eyelids. Without one the swap, the speaker's own skin, goes up the forehead to the hairline.
    brows = mesh[BROWS] + up * eyes * (HOLE_RISE if fringe else HOLE_RISE_BARE)
    line = float(np.dot(brows.mean(0), up))
    oval = mesh[FACE_OVAL]
    # Above the nose the hole sits a little inside the face: the swap's soft edge then fades into
    # the 3D head's own skin (light-matched), never into what is painted over the speaker's hair.
    middle = oval.mean(0)
    upper = (oval - mesh[4]) @ up > 0
    inset = np.where(upper[:, None], middle + (oval - middle) * HOLE_INSET, oval)
    face = np.vstack([inset[(inset @ up) <= line], brows])
    # Below the nose: the face's width plus a margin, down past the bottom of the layer.
    nose = mesh[4]
    lower = oval[(oval - nose) @ up <= 0]
    centre = lower.mean(0)
    lower = centre + (lower - centre) * np.array([1.0, 1.0]) + np.outer((lower - centre) @ side, side) * (HOLE_WIDEN - 1)
    far = chin - up * eyes * 6
    below = np.vstack([lower, far + side * eyes * 2.2, far - side * eyes * 2.2])
    quarter = (max(1, shape[0] // 4), max(1, shape[1] // 4))
    mask = np.zeros(quarter, np.float32)
    for part in (face, below):
        cv2.fillConvexPoly(mask, cv2.convexHull((part / 4).astype(np.int32)), 1.0)
    soft = max(0.5, HOLE_SOFT * eyes / 0.4 / 4)   # eyes are about 0.4 face widths apart
    mask = cv2.GaussianBlur(mask, (0, 0), soft)
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def below_normal_priority() -> None:
    """The voice's process comes first on the CPU too."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p   # pointer-sized, or the pseudo-handle is cut
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)
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
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        # A face looking down or with its eyes shut is still a face (MediaPipe's 0.5 drops it).
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3))
    torch.set_num_threads(1)
    cv2.setNumThreads(2)
    head: Head | None = None
    light = Light()
    budget = RenderBudget()
    tracked: queue.Queue = queue.Queue(maxsize=2)
    state: dict = {"rvec": None, "tvec": None, "weights": None, "stamp": 0, "hole": True, "fringe": False,
                   "rotation": OneEuro(min_cutoff=1.2, beta=0.6), "position": OneEuro(min_cutoff=1.2, beta=4.0)}
    timings = {"track_ms": 0.0, "render_ms": 3.0, "render_size": 0}

    def reset() -> None:
        if head is not None:
            head.hair.reset()
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
        model = head.landmarks(weights)
        focal = FOCAL * max(width, height)
        camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        ok, rvec, tvec = solve_pose(model, points, camera, state["rvec"], state["tvec"])
        if not ok:
            send("layer", seq, None, timings)
            return
        state["rvec"], state["tvec"] = rvec, tvec   # the raw fit: the next frame's starting guess
        smooth_r = state["rotation"](rvec.ravel(), t)
        smooth_t = state["position"](tvec.ravel(), t)
        timings["track_ms"] = 0.9 * timings["track_ms"] + 0.1 * (time.perf_counter() - started) * 1000
        rotation = cv2.Rodrigues(smooth_r.reshape(3, 1))[0]
        tracked.put((seq, t, frame, mesh, weights, rotation, smooth_t, model, points, camera))

    def render_loop() -> None:
        while True:
            item = tracked.get()
            if item is None:
                return
            started = time.perf_counter()
            try:
                layer = place(*item[1:])
            except Exception as exc:
                send("log", 40, f"3D head render failed: {type(exc).__name__}: {exc}")
                layer = None
            timings["layer_ms"] = 0.9 * timings.get("layer_ms", 0.0) + 0.1 * (time.perf_counter() - started) * 1000
            send("layer", item[0], layer, timings)

    def place(t, frame, mesh, weights, rotation, translation, model, points, camera):
        """The head layer for one tracked frame: (x, y, width, height, BGRA bytes, five points), or None."""
        height, width = frame.shape[:2]
        head.hair.step(t, rotation, translation)
        # Framed on the head's own 3D extent (hair and chin included, with a margin for the jaw and
        # for expressions), so nothing of it is ever cut off.
        corners = (rotation @ head.corners.T).T + translation
        projected = (camera @ corners.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        low, high = projected.min(0), projected.max(0)
        side_px = float((high - low).max()) * FRAME_MARGIN
        middle = (low + high) / 2
        depth = float(corners[:, 2].mean())
        ray = np.linalg.inv(camera) @ np.array([middle[0], middle[1], 1.0])
        centre = ray * depth / ray[2]
        distance = float(np.linalg.norm(centre))
        span = side_px * depth / camera[0, 0]
        # Rendered at about the size it has on screen: sharp, and no GPU spent on pixels never seen.
        size = int(min(budget.limit, max(RENDER_MIN, round(side_px / 32) * 32)))
        # A camera at the same place turned to look at the head: the head fills the render.
        z = centre / distance
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        look = np.stack([x, np.cross(z, x), z])
        focal = size * distance / span
        with arbiter.video(timings["render_ms"] + 1.0, timeout=0.25) as granted:
            if not granted or head is None:
                return None
            started = time.perf_counter()
            image = clean_edges(head.render(weights, rotation, translation, look, focal, size))
            # Back into the frame's own camera (a pure turn of the camera: a homography).
            inner = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]])
            warp = camera @ look.T @ np.linalg.inv(inner)
            # The light: measured on a small copy of the render (and the face in it), applied here.
            small = None
            if timings.setdefault("frames", 0) % LIGHT_EVERY == 0:
                small = torch.nn.functional.interpolate(image.permute(2, 0, 1)[None], size=(LIGHT_SIZE, LIGHT_SIZE),
                                                        mode="area")[0].permute(1, 2, 0).cpu().numpy()
            timings["frames"] += 1
            image = (light.apply(image) * 255).to(torch.uint8).cpu().numpy()
            timings["render_ms"] = 0.9 * timings["render_ms"] + 0.1 * (time.perf_counter() - started) * 1000
            timings["render_size"] = size
            budget.update(timings["render_ms"], size, arbiter.gap() is not None)
        # The GPU's turn is over: the voice must never wait for CPU work done while holding it.
        if small is not None:
            in_render = cv2.perspectiveTransform(mesh[None].astype(np.float64), np.linalg.inv(warp))[0]
            light.measure(small, in_render * (LIGHT_SIZE / size), frame, mesh)
        quad = cv2.perspectiveTransform(np.array([[[0, 0], [size, 0], [0, size], [size, size]]], np.float64), warp)[0]
        x0, y0 = np.maximum(np.floor(quad.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(quad.max(0)).astype(int), [width, height])
        if x1 <= x0 or y1 <= y0:
            return None
        shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        warped = cv2.warpPerspective(image, shift @ warp, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR)
        local = mesh - [x0, y0]
        if state["hole"]:   # the face swap fills the face
            warped[..., 3] = (warped[..., 3] * (1.0 - face_hole(warped.shape[:2], local, state["fringe"]))).astype(np.uint8)
        five = [[float(v) for v in mesh[list(group)].mean(0)] for group in FIVE]
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0), warped.tobytes(), five

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
            elif kind == "hole":   # whether the face is left to the face swap (the default)
                state["hole"] = bool(message[1])
            elif kind == "fringe":  # whether the speaker's hair falls over their forehead
                state["fringe"] = bool(message[1])
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
