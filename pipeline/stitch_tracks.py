"""Stitch fragmented ant tracks together.

The online tracker terminates a track once it's been lost for MAX_LOST frames.
When the same ant reappears later it gets a fresh ant_id and a new color in
the viewer. This post-processing pass tries to merge `(track_A.end, track_B.start)`
pairs that look like the same physical ant resuming after an occlusion:

  - B's first frame is within STITCH_GAP_MAX tracker frames after A's last frame
  - extrapolating A by its trailing velocity to B's first frame lands within
    STITCH_DIST_MAX pixels of B's first position
  - among contending A's, the closest one wins; greedy single-pass

Output: pipeline/tracks_stitched.parquet (overwrites pipeline/tracks.parquet on
disk if --inplace is passed, but by default the original file is preserved).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# How far in time to look for a continuation (in tracker-frame units; tracker
# samples every other video frame so this is at 30 fps effective).
STITCH_GAP_MAX = 60          # ~2 s at effective 30 fps
STITCH_DIST_MAX = 60.0       # pixels (full-res frame coords; the viewer scales)
STITCH_TAIL_FRAMES = 6       # frames at the tail used to estimate exit velocity


def per_track_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ant_id", "frame"])
    g = df.groupby("ant_id", sort=False)
    first = g.first()
    last = g.last()
    n = g.size().rename("n")

    # Trailing velocity per track: average over the last STITCH_TAIL_FRAMES.
    def tail_vel(grp):
        if len(grp) < 2:
            return pd.Series({"vx": 0.0, "vy": 0.0})
        tail = grp.tail(STITCH_TAIL_FRAMES)
        dt = tail.frame.iloc[-1] - tail.frame.iloc[0]
        if dt <= 0:
            return pd.Series({"vx": 0.0, "vy": 0.0})
        return pd.Series({
            "vx": (tail.x.iloc[-1] - tail.x.iloc[0]) / dt,
            "vy": (tail.y.iloc[-1] - tail.y.iloc[0]) / dt,
        })
    vels = g.apply(tail_vel, include_groups=False)

    s = pd.DataFrame({
        "first_frame": first.frame,
        "last_frame": last.frame,
        "first_x": first.x, "first_y": first.y,
        "last_x": last.x,  "last_y": last.y,
        "n": n,
    }).join(vels)
    return s


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Always merge the later-starting track into the earlier-starting one
        # so the surviving id is the one whose track started first.
        self.parent[rb] = ra
        return True


def stitch(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s = per_track_summary(df)

    # Index of (start_frame, ant_id) sorted by start_frame for fast bisect.
    starts = s.reset_index()[["ant_id", "first_frame", "first_x", "first_y"]].sort_values(
        "first_frame"
    ).reset_index(drop=True)
    start_frames = starts.first_frame.to_numpy()

    # Iterate ends sorted by last_frame so earlier-ending tracks claim B's first.
    ends = s.reset_index().sort_values("last_frame").reset_index(drop=True)

    uf = UnionFind(s.index.tolist())
    claimed: set[int] = set()
    matches = []
    for _, a in ends.iterrows():
        f_a = int(a.last_frame)
        lo = np.searchsorted(start_frames, f_a + 1, side="left")
        hi = np.searchsorted(start_frames, f_a + STITCH_GAP_MAX + 1, side="left")
        if lo == hi:
            continue
        cand = starts.iloc[lo:hi]
        best_id = None
        best_d = STITCH_DIST_MAX + 1.0
        for _, b in cand.iterrows():
            bid = int(b.ant_id)
            if bid in claimed:
                continue
            # Don't stitch a track to itself or to anything already in the same
            # union (chained merges would double-claim B otherwise).
            if uf.find(bid) == uf.find(int(a.ant_id)):
                continue
            dt = int(b.first_frame) - f_a
            ax = float(a.last_x) + float(a.vx) * dt
            ay = float(a.last_y) + float(a.vy) * dt
            d = float(np.hypot(ax - float(b.first_x), ay - float(b.first_y)))
            if d < best_d:
                best_d = d
                best_id = bid
        if best_id is not None:
            uf.union(int(a.ant_id), best_id)
            claimed.add(best_id)
            matches.append((int(a.ant_id), best_id, best_d))

    # Remap ant_ids via union-find roots and renumber contiguously.
    roots = df["ant_id"].map(uf.find)
    unique = sorted(roots.unique())
    remap = {old: new for new, old in enumerate(unique)}
    df = df.copy()
    df["ant_id"] = roots.map(remap).astype(np.int64)

    info = {
        "matches": len(matches),
        "ids_before": int(s.shape[0]),
        "ids_after": int(df["ant_id"].nunique()),
    }
    return df, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="pipeline/tracks.parquet")
    ap.add_argument("--out", default="pipeline/tracks_stitched.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"loaded {len(df):,} rows · {df.ant_id.nunique():,} ids")
    out, info = stitch(df)
    print(f"stitched {info['matches']:,} pairs")
    print(f"ids: {info['ids_before']:,} -> {info['ids_after']:,}")
    out.to_parquet(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
