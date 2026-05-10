"""Step 4: side-by-side overlay comparing baseline ant_id colors vs resolved
gid colors on the same 5s smoke clip. Identity preservation should show as
ants keeping their color across frames where the baseline reshuffles.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parent.parent
CLIP = WORKTREE / "pipeline" / "smoke_clip.mp4"
RESOLVED = WORKTREE / "pipeline" / "tracks_resolved.parquet"
OUT_DIR = WORKTREE / "pipeline" / "resolved_overlays"
FPS = 15.0
START_S = 80.0
SRC_FPS = 60


def color_for_id(oid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(oid) * 9176 + 17)
    h = float(rng.uniform(0, 180))
    hsv = np.uint8([[[int(h), 240, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw(img: np.ndarray, df_f: pd.DataFrame, color_col: str, label: str) -> np.ndarray:
    out = img.copy()
    for _, r in df_f.iterrows():
        col = color_for_id(int(r[color_col]))
        x, y = int(r.x), int(r.y)
        cv2.circle(out, (x, y), 4, col, 1, cv2.LINE_AA)
        ang = float(r.theta)
        dx = int(round(10 * math.cos(ang)))
        dy = int(round(10 * math.sin(ang)))
        cv2.line(out, (x, y), (x + dx, y + dy), col, 1, cv2.LINE_AA)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(RESOLVED)
    print(f"loaded {len(df):,} rows  ant_ids={df.ant_id.nunique()}  gids={df.gid.nunique()}")

    seed_src = int(round(START_S * SRC_FPS))
    src_step = int(round(SRC_FPS / FPS))

    cap = cv2.VideoCapture(str(CLIP))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_mp4 = OUT_DIR / "resolved_vs_baseline.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, 2 * h))
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        src_frame = seed_src + idx * src_step
        df_f = df[df.frame == src_frame]
        top = draw(img, df_f, "ant_id",
                   f"baseline (ant_id) frame={src_frame} N={len(df_f)}")
        bot = draw(img, df_f, "gid",
                   f"resolved (gid) frame={src_frame} N={len(df_f)}")
        stacked = np.vstack([top, bot])
        writer.write(stacked)
        if idx in (0, 36, 72):
            cv2.imwrite(str(OUT_DIR / f"frame_{idx:03d}.png"), stacked)
        idx += 1
    cap.release()
    writer.release()
    print(f"wrote {out_mp4}  ({idx} frames)")


if __name__ == "__main__":
    main()
