"""Track the red T-shape (the load) frame-by-frame.

Output: pipeline/loads.parquet with one row per (sampled) frame:
    frame, t_ms, lx, ly, lw, lh, l_theta, l_vx, l_vy, l_omega
where (lx, ly) is the load center, (lw, lh) is the OBB extent, l_theta is
the OBB angle in radians, and (l_vx, l_vy, l_omega) are first-difference
velocities.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

VIDEO = Path(__file__).parent.parent / "ants_full.mp4"
OUT = Path(__file__).parent / "loads.parquet"
RED_HSV_RANGES = [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))]
FRAME_STEP = 2  # match run.py


def red_obb(top: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """Largest-red-blob oriented bounding box: (cx, cy, w, h, theta_rad)."""
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    mask = np.zeros(top.shape[:2], dtype=np.uint8)
    for lo, hi in RED_HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:  # T should be much larger
        return None
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(c)
    # cv2.minAreaRect returns angle in [-90, 0); normalize to a long-axis angle.
    if w < h:
        w, h = h, w
        angle_deg += 90.0
    return float(cx), float(cy), float(w), float(h), math.radians(angle_deg)


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows: list[tuple] = []
    last: tuple[float, float, float, float] | None = None  # (t_ms, lx, ly, l_theta)
    pbar = tqdm(total=total // FRAME_STEP)
    f = 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % FRAME_STEP == 0:
            top = frame[: frame.shape[0] // 2, :]
            obb = red_obb(top)
            if obb is not None:
                cx, cy, w, h, theta = obb
                t_ms = 1000.0 * f / fps
                vx = vy = omega = 0.0
                if last is not None:
                    dt = (t_ms - last[0]) / 1000.0
                    if dt > 0:
                        vx = (cx - last[1]) / dt
                        vy = (cy - last[2]) / dt
                        # Unwrap angle delta to nearest π/2 (T-shape is not symmetric so π would be wrong, but minAreaRect already restricts to a 180° range).
                        dth = theta - last[3]
                        while dth > math.pi / 2:
                            dth -= math.pi
                        while dth < -math.pi / 2:
                            dth += math.pi
                        omega = dth / dt
                rows.append((f, t_ms, cx, cy, w, h, theta, vx, vy, omega))
                last = (t_ms, cx, cy, theta)
            pbar.update(1)
        f += 1
    pbar.close()
    cap.release()

    df = pd.DataFrame(rows, columns=["frame", "t_ms", "lx", "ly", "lw", "lh",
                                     "l_theta", "l_vx", "l_vy", "l_omega"])
    df.to_parquet(OUT)
    elapsed = time.time() - t_start
    print(f"\nelapsed: {elapsed:.1f}s   frames with load: {len(df)}")
    print(f"wrote {OUT}: {len(df)} rows")
    print(f"load extent (median): w={int(df.lw.median())} h={int(df.lh.median())} px")
    print(f"speed (median): {float(np.hypot(df.l_vx, df.l_vy).median()):.1f} px/s")


if __name__ == "__main__":
    main()
