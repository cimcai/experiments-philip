"""Detect and track ants in ants_full.mp4.

Detection: crop top half (ants region), mask the red T-shape and the
"WONDER WORLD" watermark, adaptive threshold dark-on-light, connected
components, ellipse fit.

Tracking: frame-to-frame Hungarian assignment with a small constant-velocity
prior. Births create new IDs; tracks lost for too many frames are terminated.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

VIDEO = Path(__file__).parent.parent / "ants_full.mp4"
OUT_DIR = Path(__file__).parent

# Detection params (tuned on full-res frame at 1920x1080)
ADAPT_BLOCK = 31
ADAPT_C = 8
MIN_AREA = 12
MAX_AREA = 400  # singletons; bigger blobs are merges
WATERMARK_BOX = (1640, 0, 1920, 60)  # x0,y0,x1,y1 in top crop coords
RED_HSV_RANGES = [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))]
BG_DIFF_THRESH = 12  # min |gray - background| for a pixel to count as "moving"
N_BG_SAMPLES = 80  # frames sampled across whole video for median background

# Linker params
MAX_MATCH_DIST = 30.0  # px; baseline cap on association distance
MAX_LOST = 30  # tracker frames a track can be unmatched before termination (~1s at 30fps)
LOST_DIST_GROWTH = 1.5  # extra px of association radius per frame lost
HARD_TELEPORT_CAP = 50.0  # absolute upper bound on per-frame association distance
VEL_BLEND = 0.6  # exponential blend for velocity update
VEL_DECAY = 0.85  # per-frame velocity decay while a track is lost


@dataclass
class Track:
    tid: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0
    area: int = 0
    last_seen: int = 0
    lost: int = 0
    history: list = field(default_factory=list)  # for diagnostic overlay


def compute_background(video_path: Path, n_samples: int = N_BG_SAMPLES) -> np.ndarray:
    """Median grayscale background across uniformly sampled frames.

    Static features (printed dots, walls) are present in every frame and
    survive the median. The T-shape moves across the video so it appears
    at any one position only briefly and is filtered out. Ants are
    constantly moving and never dominate any pixel.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    grays = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            continue
        top = f[: f.shape[0] // 2, :]
        grays.append(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY))
    cap.release()
    bg = np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)
    return bg


def detect(frame: np.ndarray, bg_gray: np.ndarray) -> np.ndarray:
    """Return Nx4 array: x, y, theta, area."""
    h, w = frame.shape[:2]
    top = frame[: h // 2, :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    red_mask = np.zeros(top.shape[:2], dtype=np.uint8)
    for lo, hi in RED_HSV_RANGES:
        red_mask |= cv2.inRange(hsv, lo, hi)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    # Foreground = pixels that differ from the long-term median background.
    # Static dots, walls, and barriers are part of the background and do not fire.
    diff = cv2.absdiff(gray, bg_gray)
    fg_mask = (diff > BG_DIFF_THRESH).astype(np.uint8) * 255
    # Dark-on-light adaptive threshold for ant body recognition.
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C
    )
    th = cv2.bitwise_and(th, fg_mask)
    # Suppress only the red T-shape pixels themselves (not a halo). Ants
    # adjacent to or on top of the T get to keep their dark pixels because
    # an ant's body is dark, not red.
    th[red_mask > 0] = 0
    x0, y0, x1, y1 = WATERMARK_BOX
    th[y0:y1, x0:x1] = 0
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < MIN_AREA or a > MAX_AREA:
            continue
        # Ellipse from second moments — restrict to component bbox so we
        # don't scan the whole image per blob.
        bx, by, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        sub = lab[by:by + bh, bx:bx + bw] == i
        ys, xs = np.where(sub)
        if xs.size < 5:
            theta = 0.0
        else:
            mx, my = xs.mean(), ys.mean()
            dx = xs - mx
            dy = ys - my
            cxx = (dx * dx).mean()
            cyy = (dy * dy).mean()
            cxy = (dx * dy).mean()
            theta = 0.5 * math.atan2(2 * cxy, cxx - cyy)
        cx, cy = cents[i]
        out.append((cx, cy, theta, int(a)))
    if not out:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(out, dtype=np.float32)


class Tracker:
    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self.next_id = 0

    def step(self, frame_idx: int, dets: np.ndarray) -> list[tuple[int, float, float, float, int]]:
        active = list(self.tracks.values())
        if dets.shape[0] == 0:
            for t in active:
                t.lost += 1
            self._reap()
            return []
        if not active:
            for d in dets:
                t = self._birth(frame_idx, d)
            return [(t.tid, t.x, t.y, t.theta, t.area) for t in self.tracks.values()]

        # Predict next position. Lost tracks coast on decayed velocity so they
        # don't drift off the screen but still move along their last heading.
        pred = np.empty((len(active), 2), dtype=np.float32)
        per_radius = np.empty(len(active), dtype=np.float32)
        for i, t in enumerate(active):
            decay = VEL_DECAY ** t.lost
            pred[i, 0] = t.x + t.vx * decay
            pred[i, 1] = t.y + t.vy * decay
            per_radius[i] = min(MAX_MATCH_DIST + LOST_DIST_GROWTH * t.lost, HARD_TELEPORT_CAP)
        det_xy = dets[:, :2]
        # Cost matrix: euclidean distance, capped per-track by per_radius.
        diff = pred[:, None, :] - det_xy[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        radius = per_radius[:, None]
        big = float(per_radius.max()) * 4 + 1e3
        cost = np.where(dist <= radius, dist, big)
        ti, di = linear_sum_assignment(cost)

        matched_t = set()
        matched_d = set()
        for ri, ci in zip(ti, di):
            if cost[ri, ci] >= big:
                continue
            t = active[ri]
            cx, cy, th, area = dets[ci]
            new_vx = cx - t.x
            new_vy = cy - t.y
            t.vx = VEL_BLEND * new_vx + (1 - VEL_BLEND) * t.vx
            t.vy = VEL_BLEND * new_vy + (1 - VEL_BLEND) * t.vy
            t.x, t.y, t.theta, t.area = float(cx), float(cy), float(th), int(area)
            t.last_seen = frame_idx
            t.lost = 0
            t.history.append((frame_idx, t.x, t.y))
            matched_t.add(ri)
            matched_d.add(ci)

        for ri, t in enumerate(active):
            if ri not in matched_t:
                t.lost += 1
        for di_ in range(dets.shape[0]):
            if di_ not in matched_d:
                self._birth(frame_idx, dets[di_])

        self._reap()
        return [(t.tid, t.x, t.y, t.theta, t.area) for t in self.tracks.values() if t.lost == 0]

    def _birth(self, frame_idx: int, d: np.ndarray) -> Track:
        cx, cy, th, area = d
        t = Track(
            tid=self.next_id,
            x=float(cx),
            y=float(cy),
            theta=float(th),
            area=int(area),
            last_seen=frame_idx,
            history=[(frame_idx, float(cx), float(cy))],
        )
        self.tracks[self.next_id] = t
        self.next_id += 1
        return t

    def _reap(self) -> None:
        dead = [tid for tid, t in self.tracks.items() if t.lost > MAX_LOST]
        for tid in dead:
            del self.tracks[tid]


def run(start: int = 0, end: int | None = None, frame_step: int = 1, write_overlay: Path | None = None):
    bg_path = OUT_DIR / "background.png"
    if bg_path.exists():
        bg_gray = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
        print(f"loaded cached background from {bg_path}")
    else:
        print(f"computing median background from {N_BG_SAMPLES} samples...")
        bg_gray = compute_background(VIDEO)
        cv2.imwrite(str(bg_path), bg_gray)
        print(f"wrote {bg_path}")

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total
    end = min(end, total)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    tracker = Tracker()
    rows: list[tuple] = []
    overlay_writer = None
    if write_overlay is not None:
        # Match the cropped top half size.
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("could not read first frame")
        h, w = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(str(write_overlay), fourcc, fps, (w, h // 2))

    pbar = tqdm(total=(end - start) // max(frame_step, 1))
    f = start
    t_start = time.time()
    while f < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (f - start) % frame_step == 0:
            dets = detect(frame, bg_gray)
            active = tracker.step(f, dets)
            t_ms = 1000.0 * f / fps
            for tid, x, y, theta, area in active:
                rows.append((f, t_ms, tid, x, y, theta, area))
            if overlay_writer is not None:
                top = frame[: frame.shape[0] // 2, :].copy()
                # Draw history trails (last ~30 frames worth) and current dot.
                for t in tracker.tracks.values():
                    if t.lost > 0:
                        continue
                    h_ = t.history[-25:]
                    if len(h_) >= 2:
                        pts = np.array([(int(p[1]), int(p[2])) for p in h_], dtype=np.int32)
                        cv2.polylines(top, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(top, (int(t.x), int(t.y)), 4, (0, 255, 0), 1)
                    # Heading line
                    dx = math.cos(t.theta) * 6
                    dy = math.sin(t.theta) * 6
                    cv2.line(top, (int(t.x - dx), int(t.y - dy)), (int(t.x + dx), int(t.y + dy)), (255, 200, 0), 1)
                cv2.putText(top, f"f={f}  tracks={len([1 for t in tracker.tracks.values() if t.lost == 0])}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                overlay_writer.write(top)
            pbar.update(1)
        f += 1
    pbar.close()
    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()

    elapsed = time.time() - t_start
    n_frames_done = (f - start) // max(frame_step, 1)
    print(f"\nelapsed: {elapsed:.1f}s   frames processed: {n_frames_done}   "
          f"per-frame: {1000*elapsed/max(n_frames_done,1):.1f}ms")
    print(f"unique track IDs created: {tracker.next_id}")
    print(f"rows in trajectories: {len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "tracks.parquet"))
    ap.add_argument("--overlay", type=str, default=None)
    args = ap.parse_args()

    overlay = Path(args.overlay) if args.overlay else None
    rows = run(start=args.start, end=args.end, frame_step=args.step, write_overlay=overlay)
    df = pd.DataFrame(rows, columns=["frame", "t_ms", "ant_id", "x", "y", "theta", "area"])
    df.to_parquet(args.out)
    print(f"wrote {args.out}: {len(df)} rows, {df['ant_id'].nunique()} unique IDs")
    # Per-track length stats
    if not df.empty:
        lens = df.groupby("ant_id").size()
        print(f"track length: median={int(lens.median())}, p90={int(lens.quantile(0.9))}, max={int(lens.max())}")


if __name__ == "__main__":
    main()
