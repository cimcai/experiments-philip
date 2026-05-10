"""Throwaway sanity script: render a few frames from the smoke clip with
SAM-derived ant centroids, theta directions, and the load mask drawn on top.

Run after sam2_modal.py::smoke_test produces pipeline/smoke_tracks.parquet.

Usage:
    uv run python pipeline/smoke_overlay.py
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WORKTREE = Path(__file__).resolve().parent.parent
VIDEO = WORKTREE / "ants_full.mp4"
TRACKS = WORKTREE / "pipeline" / "smoke_tracks.parquet"
OUT = WORKTREE / "pipeline" / "smoke_overlays"

# Must match sam2_modal.py::smoke_test defaults.
START_S = 80.0
DUR_S = 5.0
FPS = 15.0


def regen_chunk(tmpdir: Path) -> Path:
    chunk = tmpdir / "chunk.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{START_S:.3f}", "-t", f"{DUR_S:.3f}",
            "-i", str(VIDEO),
            "-vf", f"crop=iw:ih/2:0:0,fps={FPS:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-an", str(chunk),
        ],
        check=True,
    )
    return chunk


def read_frame(chunk: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(chunk))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read chunk frame {frame_idx}")
    return f


def color_for_id(oid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(oid * 9176 + 17)
    h = float(rng.uniform(0, 180))
    hsv = np.uint8([[[int(h), 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_overlay(img: np.ndarray, df_chunk: pd.DataFrame, frame_offset: int) -> np.ndarray:
    out = img.copy()
    rel = df_chunk[df_chunk.frame == frame_offset]
    for _, r in rel.iterrows():
        col = color_for_id(int(r.obj_id)) if r.kind == "ant" else (0, 0, 255)
        x, y = int(r.x), int(r.y)
        if r.kind == "ant":
            cv2.circle(out, (x, y), 4, col, 1, cv2.LINE_AA)
            dx = int(round(10 * math.cos(float(r.theta))))
            dy = int(round(10 * math.sin(float(r.theta))))
            cv2.line(out, (x, y), (x + dx, y + dy), col, 1, cv2.LINE_AA)
            cv2.putText(out, str(int(r.obj_id)), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        else:
            cv2.drawMarker(out, (x, y), col, cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.putText(out, "load", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    cv2.putText(out, f"frame {frame_offset}  ants={int((rel.kind=='ant').sum())}  "
                     f"load={'yes' if (rel.kind=='load').any() else 'no'}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    if not TRACKS.exists():
        raise SystemExit(f"missing {TRACKS}; run sam2_modal.py::smoke_test first")
    df = pd.read_parquet(TRACKS)
    print(f"loaded {len(df):,} rows  frames {df.frame.min()}→{df.frame.max()}  "
          f"obj_ids={df.obj_id.nunique()}  kinds={df.kind.unique().tolist()}")

    OUT.mkdir(exist_ok=True)
    src_fps = 60
    seed_frame_global = int(round(START_S * src_fps))
    chunk_frames = int(round(DUR_S * FPS))

    with tempfile.TemporaryDirectory() as td:
        chunk = regen_chunk(Path(td))

        # Stills at start/mid/end for quick eyeballing.
        for chunk_idx in [0, chunk_frames // 2, chunk_frames - 1]:
            global_frame = seed_frame_global + chunk_idx
            img = read_frame(chunk, chunk_idx)
            overlay = draw_overlay(img, df, global_frame)
            out_path = OUT / f"frame_{chunk_idx:03d}_global_{global_frame}.png"
            cv2.imwrite(str(out_path), overlay)

        # Full overlay video — easier to scrub than stills.
        cap = cv2.VideoCapture(str(chunk))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_mp4 = OUT / "smoke_overlay.mp4"
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        chunk_idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            overlay = draw_overlay(img, df, seed_frame_global + chunk_idx)
            writer.write(overlay)
            chunk_idx += 1
        cap.release()
        writer.release()
        print(f"wrote {out_mp4} ({chunk_idx} frames)")

    print(f"\noverlays in {OUT}/")


if __name__ == "__main__":
    main()
