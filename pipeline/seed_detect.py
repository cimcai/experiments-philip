"""Local OpenCV detector that produces seed boxes for SAM 2.1.

This is a stripped-down classical detector — no tracking, no IDs, no
post-processing — just per-frame ant bounding boxes. SAM 2.1 takes those
boxes at the seed frame(s) and propagates identities itself.

Usage:
    python -m pipeline.seed_detect ants_full.mp4 --frame 4800 --out seed.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# Same params as the classical detect() — proven to work on this footage.
ADAPT_BLOCK = 31
ADAPT_C = 8
MIN_AREA = 12
MAX_AREA = 400
WATERMARK_BOX = (1640, 0, 1920, 60)  # x0,y0,x1,y1 in top crop coords (1920×540)
RED_HSV_RANGES = [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))]
BG_DIFF_THRESH = 12
N_BG_SAMPLES = 80


def compute_background(video_path: Path, n_samples: int = N_BG_SAMPLES) -> np.ndarray:
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
    return np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)


def detect_boxes(top: np.ndarray, bg_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) bounding boxes for ants in the top crop."""
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    red_mask = np.zeros(top.shape[:2], dtype=np.uint8)
    for lo, hi in RED_HSV_RANGES:
        red_mask |= cv2.inRange(hsv, lo, hi)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_gray)
    fg_mask = (diff > BG_DIFF_THRESH).astype(np.uint8) * 255
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C
    )
    th = cv2.bitwise_and(th, fg_mask)
    th[red_mask > 0] = 0
    x0, y0, x1, y1 = WATERMARK_BOX
    th[y0:y1, x0:x1] = 0
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < MIN_AREA or a > MAX_AREA:
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Inflate slightly so SAM 2.1 has context around the ant body.
        bx = max(0, bx - 2)
        by = max(0, by - 2)
        bw = min(top.shape[1] - bx, bw + 4)
        bh = min(top.shape[0] - by, bh + 4)
        boxes.append((bx, by, bw, bh))
    return boxes


def detect_red_t_box(top: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest-red-blob bounding box for the T-shape (separate object track)."""
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    mask = np.zeros(top.shape[:2], dtype=np.uint8)
    for lo, hi in RED_HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return (int(x), int(y), int(w), int(h))


def seed_at_frame(video_path: Path, frame_idx: int, bg_gray: np.ndarray) -> dict:
    """Return seed prompt dict for a single frame.

    Output: {"frame": int, "ants": [(x,y,w,h), ...], "load": (x,y,w,h) | None,
             "frame_w": int, "frame_h": int}
    Coordinates are in the top-crop coordinate space (1920×540 for the source).
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx}")
    top = f[: f.shape[0] // 2, :]
    boxes = detect_boxes(top, bg_gray)
    load = detect_red_t_box(top)
    return {
        "frame": frame_idx,
        "ants": boxes,
        "load": load,
        "frame_w": top.shape[1],
        "frame_h": top.shape[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("seed.json"))
    ap.add_argument("--bg", type=Path, default=Path("background.npy"),
                    help="cache file for the median background")
    args = ap.parse_args()

    if args.bg.exists():
        bg_gray = np.load(args.bg)
        print(f"loaded cached bg from {args.bg}")
    else:
        bg_gray = compute_background(args.video)
        np.save(args.bg, bg_gray)
        print(f"wrote {args.bg}")

    seed = seed_at_frame(args.video, args.frame, bg_gray)
    args.out.write_text(json.dumps(seed, indent=2))
    print(f"frame {args.frame}: {len(seed['ants'])} ants, "
          f"load={'found' if seed['load'] else 'missing'}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
