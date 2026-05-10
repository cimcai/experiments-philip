"""Render ants-philip's existing tracks_clean.parquet onto the same 5s smoke clip
for direct comparison with SAM and TRex. Reads from the sibling repo."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys

WORKTREE = Path(__file__).resolve().parent.parent
CLIP = WORKTREE / "pipeline" / "smoke_clip.mp4"
DEFAULT_TRACKS = Path("/Users/theosechopoulos/Development/cimc/ants-philip/pipeline/tracks_clean.parquet")
TRACKS = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "aphilip_overlays"
OUT = WORKTREE / "pipeline" / OUT_NAME
FPS = 15.0
START_S = 80.0
SRC_FPS = 60


def color_for_id(oid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(oid * 9176 + 17)
    h = float(rng.uniform(0, 180))
    hsv = np.uint8([[[int(h), 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw(img: np.ndarray, df_f: pd.DataFrame, label: str) -> np.ndarray:
    out = img.copy()
    for _, r in df_f.iterrows():
        col = color_for_id(int(r.ant_id))
        x, y = int(r.x), int(r.y)
        cv2.circle(out, (x, y), 4, col, 1, cv2.LINE_AA)
        ang = float(r.theta)
        dx = int(round(10 * math.cos(ang)))
        dy = int(round(10 * math.sin(ang)))
        cv2.line(out, (x, y), (x + dx, y + dy), col, 1, cv2.LINE_AA)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(TRACKS)
    print(f"loaded {len(df):,} rows from {TRACKS.name}")

    # Smoke clip starts at 80s in the source video. Source is 60fps; smoke clip is 15fps.
    # tracks_clean.parquet uses source-frame numbers (0..12650).
    # Map clip frame i (at 15fps) → source frame round(80*60 + i*60/15) = 4800 + i*4.
    seed_src = int(round(START_S * SRC_FPS))
    src_step = int(round(SRC_FPS / FPS))  # 4

    # Stills + overlay video
    cap = cv2.VideoCapture(str(CLIP))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_mp4 = OUT / "aphilip_overlay.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        src_frame = seed_src + idx * src_step
        df_f = df[df.frame == src_frame]
        ov = draw(img, df_f, f"ants-philip frame={src_frame}  ants={len(df_f)}")
        writer.write(ov)
        if idx in (0, 36, 72):
            cv2.imwrite(str(OUT / f"frame_{idx:03d}_src{src_frame}.png"), ov)
        idx += 1
    cap.release()
    writer.release()
    print(f"wrote {out_mp4}  ({idx} frames)")


if __name__ == "__main__":
    main()
