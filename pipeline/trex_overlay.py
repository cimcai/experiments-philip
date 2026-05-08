"""Render TRex tracks (NPZ per individual) onto the smoke clip frames as
overlay PNGs and an mp4. Mirror smoke_overlay.py's visual style so we can
compare side-by-side with SAM.
"""

from __future__ import annotations

import glob
import math
import re
from pathlib import Path

import cv2
import numpy as np

WORKTREE = Path(__file__).resolve().parent.parent
CLIP = WORKTREE / "pipeline" / "smoke_clip.mp4"
DATA = WORKTREE / "pipeline" / "trex_out" / "data"
OUT = WORKTREE / "pipeline" / "trex_overlays"
FPS = 15.0


def load_tracks() -> dict[int, dict]:
    """Return {id: {'frames': np.array, 'x': np.array, 'y': np.array, 'angle': np.array}}.

    Filters out IDs with zero valid samples.
    """
    files = sorted(glob.glob(str(DATA / "smoke_clip_id*.npz")),
                   key=lambda p: int(re.search(r"id(\d+)\.npz", p).group(1)))
    tracks: dict[int, dict] = {}
    for fp in files:
        d = np.load(fp)
        x = d["X#wcentroid"]
        y = d["Y#wcentroid"]
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        oid = int(re.search(r"id(\d+)\.npz", fp).group(1))
        tracks[oid] = {
            "frames": d["frame"][valid].astype(int),
            "x": x[valid],
            "y": y[valid],
            "angle": d["ANGLE"][valid] if "ANGLE" in d else np.zeros(int(valid.sum())),
            "n": int(valid.sum()),
        }
    return tracks


def color_for_id(oid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(oid * 9176 + 17)
    h = float(rng.uniform(0, 180))
    hsv = np.uint8([[[int(h), 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_frame(img: np.ndarray, tracks: dict[int, dict], frame_idx: int,
               min_track_len: int) -> np.ndarray:
    out = img.copy()
    n = 0
    for oid, t in tracks.items():
        if t["n"] < min_track_len:
            continue
        idx = np.where(t["frames"] == frame_idx)[0]
        if not len(idx):
            continue
        i = int(idx[0])
        col = color_for_id(oid)
        x, y = int(t["x"][i]), int(t["y"][i])
        ang = float(t["angle"][i])
        cv2.circle(out, (x, y), 4, col, 1, cv2.LINE_AA)
        if math.isfinite(ang):
            dx = int(round(10 * math.cos(ang)))
            dy = int(round(10 * math.sin(ang)))
            cv2.line(out, (x, y), (x + dx, y + dy), col, 1, cv2.LINE_AA)
        n += 1
    cv2.putText(out, f"frame {frame_idx}  shown={n}  (min_track_len={min_track_len})",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tracks = load_tracks()
    print(f"loaded {len(tracks)} non-empty tracks")
    lens = np.array([t["n"] for t in tracks.values()])
    print(f"track lengths: median={int(np.median(lens))} p90={int(np.quantile(lens,0.9))} max={int(lens.max())}")

    # Render two variants: all tracks vs only "real" tracks (>=20 frames)
    for min_len, label in [(0, "all"), (20, "min20"), (50, "min50")]:
        cap = cv2.VideoCapture(str(CLIP))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_mp4 = OUT / f"trex_overlay_{label}.mp4"
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            writer.write(draw_frame(img, tracks, idx, min_track_len=min_len))
            idx += 1
        cap.release()
        writer.release()
        n_shown = sum(1 for t in tracks.values() if t["n"] >= min_len)
        print(f"  {label}: {n_shown} tracks shown  → {out_mp4.name}")

    # Stills at frames 0, 36, 72
    cap = cv2.VideoCapture(str(CLIP))
    for tgt in [0, 36, 72]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
        ok, img = cap.read()
        if not ok:
            break
        for min_len, label in [(0, "all"), (20, "min20")]:
            o = draw_frame(img, tracks, tgt, min_track_len=min_len)
            cv2.imwrite(str(OUT / f"frame_{tgt:03d}_{label}.png"), o)
    cap.release()
    print(f"\nstills + overlays in {OUT}/")


if __name__ == "__main__":
    main()
