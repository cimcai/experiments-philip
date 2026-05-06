"""Pack tracks + load track into a compact binary format for the HTML viewer.

Output (in pipeline/viewer/):
  - data.bin   : binary blob with detection records
  - meta.json  : sizes/offsets, fps, video dims, ant summary table

The viewer fetches data.bin once and builds in-memory indexes.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "viewer"
OUT.mkdir(exist_ok=True)
TRACKS = Path(__file__).parent / "tracks_clean.parquet"
LOADS = Path(__file__).parent / "loads.parquet"

# Video geometry (the source ants_full.mp4 is 1920x1080; our cropped top half is
# 1920x540 at the source, but the viewer mp4 is scaled to 1280x360).
SRC_W, SRC_H = 1920, 540
VIEWER_W, VIEWER_H = 1280, 360
SCALE = VIEWER_W / SRC_W


def main() -> None:
    tracks = pd.read_parquet(TRACKS).sort_values(["frame", "ant_id"]).reset_index(drop=True)
    loads = pd.read_parquet(LOADS).set_index("frame").sort_index()

    # Build detection records — one row per (frame, ant_id) we tracked.
    # Layout per record (16 bytes): ant_id u32, x f32, y f32, theta f32
    n_rec = len(tracks)
    frames_present = sorted(tracks.frame.unique().tolist())
    n_frames = len(frames_present)

    # Frame index: for each present frame, the record offset where its detections start.
    frame_offsets = np.zeros(n_frames + 1, dtype=np.uint32)
    frame_numbers = np.array(frames_present, dtype=np.uint32)
    cur = 0
    frame_to_idx = {f: i for i, f in enumerate(frames_present)}
    for f, group in tracks.groupby("frame", sort=True):
        i = frame_to_idx[f]
        frame_offsets[i] = cur
        cur += len(group)
    frame_offsets[-1] = cur

    # Detection arrays.
    ant_ids = tracks.ant_id.to_numpy(np.uint32)
    xs = (tracks.x.to_numpy() * SCALE).astype(np.float32)
    ys = (tracks.y.to_numpy() * SCALE).astype(np.float32)
    thetas = tracks.theta.to_numpy(np.float32)

    # Load-track arrays aligned to frames_present (NaN if frame missing in loads).
    lx = np.full(n_frames, np.nan, dtype=np.float32)
    ly = np.full(n_frames, np.nan, dtype=np.float32)
    lw = np.full(n_frames, np.nan, dtype=np.float32)
    lh = np.full(n_frames, np.nan, dtype=np.float32)
    lth = np.full(n_frames, np.nan, dtype=np.float32)
    for i, f in enumerate(frames_present):
        if f in loads.index:
            r = loads.loc[f]
            lx[i] = r.lx * SCALE
            ly[i] = r.ly * SCALE
            lw[i] = r.lw * SCALE
            lh[i] = r.lh * SCALE
            lth[i] = r.l_theta

    # Per-ant summary: id, length, total_path_px, first_frame, last_frame.
    g = tracks.groupby("ant_id")
    summary = pd.DataFrame({
        "id": g.size().index.to_numpy(np.int64),
        "n": g.size().to_numpy(),
        "first": g.frame.min().to_numpy(),
        "last": g.frame.max().to_numpy(),
    })
    # Path length
    def path_len(grp):
        if len(grp) < 2:
            return 0.0
        dx = np.diff(grp.x.values); dy = np.diff(grp.y.values)
        return float(np.sqrt(dx * dx + dy * dy).sum())
    summary["path"] = g.apply(path_len, include_groups=False).to_numpy() * SCALE
    summary = summary.sort_values("n", ascending=False).reset_index(drop=True)

    # Pack everything into one binary blob.
    sections: dict[str, bytes] = {
        "frame_numbers": frame_numbers.tobytes(),
        "frame_offsets": frame_offsets.tobytes(),
        "ant_ids": ant_ids.tobytes(),
        "xs": xs.tobytes(),
        "ys": ys.tobytes(),
        "thetas": thetas.tobytes(),
        "lx": lx.tobytes(),
        "ly": ly.tobytes(),
        "lw": lw.tobytes(),
        "lh": lh.tobytes(),
        "lth": lth.tobytes(),
    }
    layout = {}
    blob = bytearray()
    for name, b in sections.items():
        layout[name] = {"offset": len(blob), "length": len(b)}
        blob.extend(b)

    (OUT / "data.bin").write_bytes(bytes(blob))

    meta = {
        "video_url": "ants.mp4",
        "video_w": VIEWER_W,
        "video_h": VIEWER_H,
        "src_w": SRC_W,
        "src_h": SRC_H,
        "scale": SCALE,
        "fps": 60.0,
        "frame_step": 2,  # ant tracks are at every other frame
        "n_frames": n_frames,
        "n_records": n_rec,
        "n_ants": int(summary.id.nunique()),
        "layout": layout,
        "ant_summary": summary.head(2000).to_dict(orient="records"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta))

    print(f"frames present: {n_frames:,}")
    print(f"records: {n_rec:,}")
    print(f"ants: {meta['n_ants']:,} (top 2000 listed in meta)")
    print(f"data.bin: {len(blob)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
