"""How natural is the worn head? A benchmark with a ground truth, for any GPU.

    uv run python scripts/bench_head.py <video of one person> <out folder> [--engine 3d|hair|face|head|personalive]
        [--frames 300] [--photos 8] [--steps 0] [--tag name] [--face id] [--light]

The head is built from frames of the video and worn over the same person in the same video, so the
real frame is what a perfect clone would show. Per head turn it reports how well the hair matches
(silhouette IoU, colour PSNR), how far the hair lags behind the face when the head moves, and the
frame rate, keyframes and GPU memory. Writes ``report.json`` and ``sheet.jpg`` (real | worn, at the
widest turns and fastest moves).

``--light`` puts the speaker in a dim, pink, uneven room while the head keeps the photos' light: the
worn head must end up in the room's. ``--face id`` wears a face of the library instead of the video's.
Then it also reports the seam: the biggest colour step going up the forehead, forehead against
cheeks, and how much the colour flickers from frame to frame (each next to the real frame's own).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.compute_arbiter import GpuArbiter  # noqa: E402
from backend.video import hair as hairs  # noqa: E402
from backend.video import parser as parsing  # noqa: E402
from backend.video.avatar import HEAD_FILE, HeadBuilder, HeadDriver, LamInstall  # noqa: E402
from backend.video.engine import FaceEngine, FaceModels, VideoSettings, head_pose  # noqa: E402
from backend.video.faces import alignment  # noqa: E402
from backend.video.live import LiveFaceSession, picture_box  # noqa: E402
from backend.video.personalive.client import PersonaLiveHead, PersonaLiveInstall  # noqa: E402

FACE_DIR = ROOT / "models" / "assets" / "face"
YAW_BINS = (0.0, 0.1, 0.2, 0.35, 10.0)   # head_pose units: straight on, slight, turned, far


def dim_room(frame: np.ndarray) -> np.ndarray:
    """``frame`` as a dim room with a pink lamp to one side would show it."""
    height, width = frame.shape[:2]
    across = np.linspace(0.82, 1.12, width, dtype=np.float32)[None, :, None]
    down = np.linspace(1.08, 0.92, height, dtype=np.float32)[:, None, None]
    tint = np.array([0.74, 0.62, 0.86], np.float32)   # BGR
    return np.clip(frame.astype(np.float32) * tint * across * down, 0, 255).astype(np.uint8)


def read_video(path: Path, limit: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        sys.exit(f"Could not read {path}")
    return frames, fps


def build_face(frames: list[np.ndarray], count: int, models: FaceModels, keep: Path) -> tuple[Path, np.ndarray]:
    """A face folder made of frames of the video, with its 3D head; reused if already built."""
    folder = keep / "face"
    folder.mkdir(parents=True, exist_ok=True)
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, count).astype(int)]
    for index, picture in enumerate(picks):
        cv2.imwrite(str(folder / f"photo_{index}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
    identity, _ = models.source_embedding(picks)
    return folder, np.asarray(identity)


def build_head(folder: Path, steps: int) -> None:
    if (folder / HEAD_FILE).is_file():
        return
    builder = HeadBuilder(folder.parent, lambda: FACE_DIR / "bisenet_resnet_34.onnx", lambda: False,
                          log_dir=folder.parent)
    builder.request(folder.name, steps=steps)
    last = None
    while True:
        state = builder.status(folder.name)
        if state != last:
            print(" ", state, flush=True)
            last = state
        if state["state"] in ("ready", "failed"):
            break
        time.sleep(1.0)
    if state["state"] == "failed":
        sys.exit(f"The head did not build: {state.get('error')}")


def run_live(frames: list[np.ndarray], fps: float, session: LiveFaceSession) -> tuple[dict[int, np.ndarray], list[dict], float]:
    """Frames fed at camera rate, as the face process does; what comes out, by frame number."""
    outputs: dict[int, np.ndarray] = {}
    stats: list[dict] = []
    start = time.monotonic()

    def drain() -> None:
        for t, image in session.ready():
            outputs[round(t * fps)] = image

    for index, frame in enumerate(frames):
        while time.monotonic() - start < index / fps:
            time.sleep(0.001)
        session.submit(index / fps, frame)
        drain()
        if index % 30 == 29:
            stats.append(session.stats())
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        drain()
        time.sleep(0.005)
    return outputs, stats, time.monotonic() - start


def run_every_frame(frames: list[np.ndarray], fps: float, engine: FaceEngine) -> tuple[dict[int, np.ndarray], float]:
    """Every frame its own head render and swap, however long that takes: the quality a GPU fast
    enough would show live, measurable on any GPU."""
    outputs: dict[int, np.ndarray] = {}
    engine.reset_head()
    times = []
    for index, frame in enumerate(frames):
        t = index / fps
        tracked = engine.track(frame, t)
        started = time.perf_counter()
        layer = engine.head_layer(frame, tracked, t, wait=30.0) if engine.wants_head else None
        times.append(time.perf_counter() - started)
        key = engine.keyframe(frame, tracked, t, 30.0, wait=30.0) if tracked.face is not None else None
        outputs[index] = engine.compose(frame, tracked, t, key, None, 30.0, head=layer)
    return outputs, float(np.median(times) * 1000.0)


def skin_colours(crop: np.ndarray, labels: np.ndarray, points: np.ndarray) -> dict | None:
    """Colours (LAB) up the middle of the forehead, from the brows to the hair, and on the cheeks."""
    eyes = points[:2]
    distance = float(np.linalg.norm(eyes[1] - eyes[0]))
    middle, level = float(eyes[:, 0].mean()), float(eyes[:, 1].mean())
    half = int(distance * 0.4)
    bottom = int(level - 0.45 * distance)
    x0, x1 = int(middle) - half, int(middle) + half
    if x0 < 0 or x1 > crop.shape[1] or bottom < 8:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin = labels == parsing.SKIN
    brows = np.nonzero(np.isin(labels[:, x0:x1], (parsing.BROW_L, parsing.BROW_R)).any(1))[0]   # from above the brows, whoever's they are
    if len(brows):
        bottom = min(bottom, int(brows.min()) - 3)
    rows = []
    for row in range(bottom, -1, -1):
        inside = skin[row, x0:x1]
        if inside.mean() < 0.5:
            break
        rows.append(lab[row, x0:x1][inside].mean(0))
    if len(rows) < 8:
        return None
    rows = rows[:max(8, int(len(rows) * 0.8))]   # not the hair's own shadow on the skin below it
    profile = cv2.blur(np.array(rows, np.float32), (1, 3))
    reach = max(2, int(distance * 0.06))
    steps = np.abs(profile[reach:] - profile[:-reach]).max(0)
    below = int(level + 0.55 * distance)
    size = int(0.2 * distance)
    cheeks = [lab[below:below + size, int(middle + side * distance * 0.5):int(middle + side * distance * 0.5) + size]
              for side in (-0.9, 0.5)]
    cheek = np.concatenate([part.reshape(-1, 3) for part in cheeks])
    if not len(cheek):
        return None
    forehead = profile[len(profile) // 2:].mean(0)
    return {"steps": steps, "against_cheeks": np.abs(forehead - cheek.mean(0)), "forehead": forehead}


def seam_report(frames: list[np.ndarray], outputs: dict[int, np.ndarray], models: FaceModels) -> dict:
    parser = models.parser
    assert parser is not None
    seen: dict[str, list] = {"worn": [], "real": []}
    for index, real in enumerate(frames):
        worn = outputs.get(index)
        found = models.detector.detect(real)
        if worn is None or not found:
            continue
        points = max(found, key=lambda face: face.size).points
        matrix = alignment(points, hairs.HEAD_TEMPLATE, hairs.SIZE)
        local = cv2.transform(points.reshape(1, -1, 2).astype(np.float32), matrix).reshape(-1, 2)
        for name, picture in (("worn", worn), ("real", real)):
            crop, _ = hairs.head_crop(picture, points)
            colours = skin_colours(crop, parser.labels(crop), local)
            if colours is not None:
                seen[name].append((index, colours))
    report = {}
    for name, rows in seen.items():
        if not rows:
            continue
        flicker = [np.abs(b[1]["forehead"] - a[1]["forehead"]) for a, b in itertools.pairwise(rows) if b[0] == a[0] + 1]
        report[name] = {"frames": len(rows),
                        "step_lab": np.round(np.median([r[1]["steps"] for r in rows], axis=0), 1).tolist(),
                        "forehead_vs_cheeks_lab": np.round(np.median([r[1]["against_cheeks"] for r in rows], axis=0), 1).tolist(),
                        "flicker_lab": np.round(np.mean(flicker, axis=0), 2).tolist() if flicker else None}
    return report


def measure(frames: list[np.ndarray], outputs: dict[int, np.ndarray], models: FaceModels, fps: float) -> dict:
    """Hair against the real frame, frame by frame, after the run (nothing here slows the run down)."""
    parser = models.parser
    assert parser is not None
    rows = []
    for index, real in enumerate(frames):
        worn = outputs.get(index)
        found = models.detector.detect(real)
        if worn is None or not found:
            continue
        points = max(found, key=lambda face: face.size).points
        crop_real, matrix = hairs.head_crop(real, points)
        crop_worn, _ = hairs.head_crop(worn, points)
        hair_real = np.isin(parser.labels(crop_real), (parsing.HAIR, parsing.HAT))
        hair_worn = np.isin(parser.labels(crop_worn), (parsing.HAIR, parsing.HAT))
        union = hair_real | hair_worn
        if union.sum() < 200:
            continue
        error = ((crop_real.astype(np.float32) - crop_worn.astype(np.float32)) ** 2)[union].mean()
        scale = float(np.linalg.norm(matrix[0, :2])) or 1.0
        centre_real = np.array(np.nonzero(hair_real))[::-1].mean(1) if hair_real.any() else None
        centre_worn = np.array(np.nonzero(hair_worn))[::-1].mean(1) if hair_worn.any() else None
        rows.append({"frame": index, "yaw": abs(head_pose(points)[0]), "nose": points[2].tolist(),
                     "iou": float((hair_real & hair_worn).sum() / union.sum()),
                     "psnr": float(10 * np.log10(255.0 ** 2 / max(float(error), 1e-6))),
                     "offset": None if centre_real is None or centre_worn is None
                     else ((centre_worn - centre_real) / scale).tolist()})
    by_yaw = {}
    for low, high in itertools.pairwise(YAW_BINS):
        picked = [row for row in rows if low <= row["yaw"] < high]
        if picked:
            by_yaw[f"{low}-{high}"] = {"frames": len(picked), "iou": round(float(np.mean([r["iou"] for r in picked])), 3),
                                       "psnr": round(float(np.mean([r["psnr"] for r in picked])), 2)}
    # The lag: how far the hair sits behind the real hair, per pixel per second the head moves.
    speeds, offsets = [], []
    for previous, row in itertools.pairwise(rows):
        if row["offset"] is None or row["frame"] != previous["frame"] + 1:
            continue
        speeds.append(np.array(row["nose"]) - np.array(previous["nose"]))
        offsets.append(row["offset"])
    lag_ms = None
    if len(speeds) > 20:
        velocity = np.array(speeds).ravel() * fps
        offset = np.array(offsets).ravel()
        offset = offset - offset.mean()
        lag_ms = round(float(-(velocity @ offset) / max(float(velocity @ velocity), 1e-6) * 1000.0), 1)
    return {"frames_measured": len(rows), "iou": round(float(np.mean([r["iou"] for r in rows])), 3) if rows else None,
            "psnr": round(float(np.mean([r["psnr"] for r in rows])), 2) if rows else None,
            "by_yaw": by_yaw, "hair_lag_ms": lag_ms,
            "drift_px": round(float(np.mean([np.linalg.norm(r["offset"]) for r in rows if r["offset"]])), 1) if rows else None,
            "rows": rows}


def bars_changed(frames: list[np.ndarray], outputs: dict[int, np.ndarray]) -> int:
    """Pixels painted on the camera's black bars, over the whole run (none should be)."""
    box = picture_box(frames[0])
    if box is None:
        return 0
    outside = np.ones(frames[0].shape[:2], bool)
    outside[box[1]:box[3], box[0]:box[2]] = False
    return sum(int((np.abs(outputs[i].astype(np.int16) - frames[i])[outside].max(1) > 8).sum()) for i in outputs if i < len(frames))


def sheet(frames: list[np.ndarray], outputs: dict[int, np.ndarray], rows: list[dict], path: Path) -> None:
    if not rows:
        return
    by_turn = sorted(rows, key=lambda row: row["yaw"])
    moves = [(np.linalg.norm(np.array(b["nose"]) - np.array(a["nose"])), b["frame"]) for a, b in itertools.pairwise(rows)]
    picks = [by_turn[0]["frame"], by_turn[len(by_turn) // 2]["frame"], by_turn[-1]["frame"], max(moves)[1] if moves else 0]
    box = picture_box(frames[0])
    x0, y0, x1, y1 = box if box is not None else (0, 0, frames[0].shape[1], frames[0].shape[0])
    tiles = []
    for index in picks:
        pair = np.hstack([frames[index][y0:y1, x0:x1], outputs[index][y0:y1, x0:x1]])
        tiles.append(cv2.resize(pair, (960, int(960 * pair.shape[0] / pair.shape[1]))))
    cv2.imwrite(str(path), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Close up: the head as worn, large enough to judge its edges, ears and hair.
    close = []
    for index in picks:
        found = rows[0] if not rows else min(rows, key=lambda row: abs(row["frame"] - index))
        x, y = np.array(found["nose"], int)
        half = outputs[index].shape[0] // 2
        patch = outputs[index][max(0, y - half):y + half, max(0, x - half):x + half]
        close.append(cv2.resize(patch, (480, 480)))
    cv2.imwrite(str(path.with_name(path.stem + "_close.jpg")), np.vstack([np.hstack(close[:2]), np.hstack(close[2:])]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--engine", default="3d", choices=["3d", "hair", "face", "head", "personalive"])
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--photos", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0, help="3D head training steps (0: LAM's head as built)")
    parser.add_argument("--every-frame", action="store_true",
                        help="whole head: each frame waits for its own render (quality, not speed)")
    parser.add_argument("--tag", default="", help="a name for this run's files")
    parser.add_argument("--rebuild", action="store_true", help="build the 3D head again")
    parser.add_argument("--face", default="", help="wear this face of the library (data/faces/<id>)")
    parser.add_argument("--light", action="store_true", help="the speaker in a dim, pink, uneven room")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(args.video, args.frames)
    arbiter = GpuArbiter()
    models = FaceModels(FACE_DIR, True)
    models.measure(args.out / "swap_ms.json", arbiter)
    if args.face:
        folder = ROOT / "data" / "faces" / args.face
        identity = np.load(folder / "identity.npy")
    else:
        folder, identity = build_face(frames, args.photos, models, args.out)
    if args.light:
        frames = [dim_room(frame) for frame in frames]
    if args.engine == "3d" and not args.face:
        if args.rebuild:
            (folder / HEAD_FILE).unlink(missing_ok=True)
        build_head(folder, args.steps)
    mode = {"3d": "hair", "personalive": "head"}.get(args.engine, args.engine)
    engine = FaceEngine(models, arbiter, VideoSettings(mode=mode))
    engine.set_identity(identity)
    engine.set_portrait(cv2.imdecode(np.fromfile(folder / "photo_0.jpg", np.uint8), cv2.IMREAD_COLOR))
    engine.hair_photos = [str(path) for path in sorted(folder.glob("photo_*.jpg"))]
    engine.plan()
    driver = None
    if args.engine == "3d":
        driver = HeadDriver(LamInstall.default(), arbiter.name, lambda seq, layer: None, log_path=args.out / "drive.log")
        if not driver.wait_ready(180):
            sys.exit("The 3D head process did not start")
        while not driver.use(folder / HEAD_FILE) and driver.error is None:
            time.sleep(0.05)
        if driver.error:
            sys.exit(f"The 3D head did not load: {driver.error}")
    head_ms = None
    generated = None
    if args.engine == "personalive":
        generated = PersonaLiveHead(PersonaLiveInstall.default(), arbiter.name, engine, args.out / "personalive.log")
        engine.remote = generated
        head_ms = generated.measure(engine.portrait)
        if head_ms is None:
            sys.exit(f"PersonaLive did not run: {generated.error or 'see personalive.log (NaN or too slow)'}")
    if args.every_frame:
        started = time.monotonic()
        outputs, head_ms = run_every_frame(frames, fps, engine)
        stats, elapsed = [], time.monotonic() - started
    else:
        session = LiveFaceSession(engine, head3d=driver)
        outputs, stats, elapsed = run_live(frames, fps, session)
        session.close()
    if driver is not None:
        driver.close()
    if generated is not None:
        generated.close()
    metrics = measure(frames, outputs, models, fps)
    metrics["bars_changed"] = bars_changed(frames, outputs)
    rows = metrics.pop("rows")
    metrics["seam"] = seam_report(frames, outputs, models)
    tail = stats[2:] or stats

    def median(key: str):
        values = [s[key] for s in tail if isinstance(s.get(key), int | float)]
        return round(float(np.median(values)), 1) if values else None

    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    report = {"video": args.video.name, "engine": args.engine, "gpu": gpu, "frames_in": len(frames),
              "frames_out": len(outputs), "fps_out": round(len(outputs) / max(elapsed, 1e-3), 1),
              "keys_per_second": median("keys_per_second"), "track_ms": median("track_ms"),
              "compose_ms": median("compose_ms"), "render_ms": median("render_ms"),
              "render_size": median("render_size"), "head_fit": tail[-1].get("head_fit") if tail else None, "head_render_ms": head_ms, **metrics}
    tag = f"_{args.tag}" if args.tag else ""
    (args.out / f"report{tag}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    sheet(frames, outputs, rows, args.out / f"sheet{tag}.jpg")
    print(json.dumps(report, indent=1))
    arbiter.close()


if __name__ == "__main__":
    main()
