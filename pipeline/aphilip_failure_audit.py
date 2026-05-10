"""Audit ants-philip's tracks_clean.parquet for failure modes that would
matter for the behavioral analysis — specifically:
  1. Track length distribution (fragmentation)
  2. Births/deaths by location (left-edge births = expected; mid-frame births = ID swaps)
  3. Per-frame count consistency
  4. Velocity jumps (high-velocity frames = likely ID swap)
  5. Cluster zone vs free zone behavior (proxy for ID stability under occlusion)

The load track also matters — we expect ants pile on it, so the cluster zone
is roughly the load OBB plus margin. We use loads.parquet to define it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

APHILIP = Path("/Users/theosechopoulos/Development/cimc/ants-philip/pipeline")
TRACKS = Path(sys.argv[1]) if len(sys.argv) > 1 else APHILIP / "tracks_clean.parquet"
LOADS = APHILIP / "loads.parquet"

# Source frame coords: top crop is 1920x540
FRAME_W, FRAME_H = 1920, 540
LEFT_EDGE_PX = 100  # ants entering from left within this strip = expected new IDs
CLUSTER_RADIUS_PX = 80  # within this many px of load center = "near-load" zone


def main() -> None:
    print(f"reading {TRACKS}")
    df = pd.read_parquet(TRACKS)
    print(f"  {len(df):,} rows  {df.frame.nunique()} frames  {df.ant_id.nunique()} unique ant_ids")
    print(f"  src frame range: {df.frame.min()}-{df.frame.max()}")

    print(f"\nreading {LOADS}")
    loads = pd.read_parquet(LOADS).set_index("frame").sort_index()
    print(f"  {len(loads):,} rows  cols={list(loads.columns)}")

    # ---- 1. Track length distribution ----
    print("\n=== 1. TRACK LENGTH DISTRIBUTION ===")
    track_len = df.groupby("ant_id").size()
    print(f"  total unique IDs: {len(track_len):,}")
    print(f"  median length: {int(track_len.median())} frames "
          f"({track_len.median()/60:.1f}s @ 60fps)")
    print(f"  p10/p50/p90: {int(track_len.quantile(0.1))}/{int(track_len.quantile(0.5))}/"
          f"{int(track_len.quantile(0.9))}")
    print(f"  shortest 10%: tracks <= {int(track_len.quantile(0.1))} frames")
    print(f"  IDs lasting < 30 frames (0.5s): {(track_len < 30).sum()} "
          f"({(track_len < 30).mean():.1%})")
    print(f"  IDs lasting > 600 frames (10s): {(track_len > 600).sum()} "
          f"({(track_len > 600).mean():.1%})")
    print(f"  longest track: {int(track_len.max())} frames "
          f"({track_len.max()/60:.1f}s) — ID {track_len.idxmax()}")

    # ---- 2. Births/deaths by location ----
    print("\n=== 2. BIRTHS & DEATHS BY LOCATION ===")
    firsts = df.sort_values(["ant_id", "frame"]).groupby("ant_id").first()
    lasts = df.sort_values(["ant_id", "frame"]).groupby("ant_id").last()
    n = len(firsts)

    left_births = (firsts.x < LEFT_EDGE_PX).sum()
    right_births = (firsts.x > FRAME_W - LEFT_EDGE_PX).sum()
    mid_births = n - left_births - right_births
    print(f"  total births: {n:,}")
    print(f"  on left edge (x<{LEFT_EDGE_PX}):  {left_births:,} ({left_births/n:.1%})  "
          f"← expected for new ants entering")
    print(f"  on right edge (x>{FRAME_W-LEFT_EDGE_PX}): {right_births:,} ({right_births/n:.1%})")
    print(f"  in middle of frame: {mid_births:,} ({mid_births/n:.1%})  "
          f"← suggests ID swap / re-entry after occlusion")

    left_deaths = (lasts.x < LEFT_EDGE_PX).sum()
    right_deaths = (lasts.x > FRAME_W - LEFT_EDGE_PX).sum()
    mid_deaths = n - left_deaths - right_deaths
    print(f"\n  total deaths: {n:,}")
    print(f"  on left edge:  {left_deaths:,} ({left_deaths/n:.1%})")
    print(f"  on right edge: {right_deaths:,} ({right_deaths/n:.1%})  ← exits")
    print(f"  in middle of frame: {mid_deaths:,} ({mid_deaths/n:.1%})  "
          f"← suggests ID lost mid-track")

    # ---- 3. Per-frame count consistency ----
    print("\n=== 3. PER-FRAME COUNT ===")
    per_f = df.groupby("frame").size()
    print(f"  median per-frame ant count: {int(per_f.median())}")
    print(f"  min/max: {int(per_f.min())}/{int(per_f.max())}")
    print(f"  std/median: {per_f.std()/per_f.median():.2f}  "
          f"(<0.20 = stable population)")

    # ---- 4. Velocity jumps ----
    print("\n=== 4. VELOCITY JUMPS (high frame-to-frame motion = likely ID swap) ===")
    df_sorted = df.sort_values(["ant_id", "frame"])
    df_sorted["dx"] = df_sorted.groupby("ant_id").x.diff()
    df_sorted["dy"] = df_sorted.groupby("ant_id").y.diff()
    df_sorted["df"] = df_sorted.groupby("ant_id").frame.diff()
    moved = df_sorted.dropna(subset=["dx", "dy"])
    moved = moved[moved.df > 0]
    moved["speed"] = np.sqrt(moved.dx ** 2 + moved.dy ** 2) / moved.df
    print(f"  speed (px per source-frame): "
          f"median={moved.speed.median():.2f}  p90={moved.speed.quantile(0.9):.2f}  "
          f"p99={moved.speed.quantile(0.99):.2f}  max={moved.speed.max():.1f}")
    fast = (moved.speed > 20).sum()
    print(f"  frames with speed > 20 px/f (~ant width): {fast:,} ({fast/len(moved):.2%})  "
          f"← outliers are candidate ID swaps")
    very_fast = (moved.speed > 50).sum()
    print(f"  frames with speed > 50 px/f (3x ant width): {very_fast:,} ({very_fast/len(moved):.2%})")

    # ---- 5. Near-load vs far-from-load behavior ----
    print("\n=== 5. NEAR-LOAD vs FAR-FROM-LOAD ===")
    # Join loads with tracks
    tracks_with_load = df.merge(
        loads[["lx", "ly"]].reset_index(),
        on="frame", how="inner"
    )
    tracks_with_load["d_to_load"] = np.sqrt(
        (tracks_with_load.x - tracks_with_load.lx) ** 2
        + (tracks_with_load.y - tracks_with_load.ly) ** 2
    )
    near = tracks_with_load[tracks_with_load.d_to_load < CLUSTER_RADIUS_PX]
    far = tracks_with_load[tracks_with_load.d_to_load >= CLUSTER_RADIUS_PX]
    print(f"  rows near load (<{CLUSTER_RADIUS_PX}px): {len(near):,} "
          f"({len(near)/len(tracks_with_load):.1%})")
    print(f"  rows far from load: {len(far):,}")

    # Average track length when near load vs far
    # Approximation: count rows per ant_id in each region
    near_lens = near.groupby("ant_id").size()
    far_lens = far.groupby("ant_id").size()
    print(f"  median consecutive-frame stay near load: {int(near_lens.median())} "
          f"(rows/ant in near zone)")
    print(f"  median stay far from load: {int(far_lens.median())}")

    # IDs that ONLY appear near the load (suspicious — likely fragments born inside cluster)
    only_near = set(near.ant_id) - set(far.ant_id)
    only_far = set(far.ant_id) - set(near.ant_id)
    both = set(near.ant_id) & set(far.ant_id)
    print(f"\n  IDs ONLY observed near load (likely cluster-fragment): {len(only_near):,}")
    print(f"  IDs ONLY observed far from load: {len(only_far):,}")
    print(f"  IDs observed both: {len(both):,}")

    # ---- 6. Track survival per minute of video ----
    print("\n=== 6. SUMMARY ===")
    real_ant_population = int(per_f.median())
    n_unique_ids = len(track_len)
    duration_min = (df.frame.max() - df.frame.min()) / 60 / 60
    print(f"  Real ant population (median): {real_ant_population}")
    print(f"  Unique tracked IDs over {duration_min:.1f} min: {n_unique_ids:,}")
    print(f"  Ratio: {n_unique_ids / real_ant_population:.1f}x more IDs than ants")
    print(f"    Lower bound on ID switches: ~{n_unique_ids - real_ant_population:,}")
    print(f"    (If every ant had exactly 1 ID: ratio would be 1. Higher = more fragmentation.)")


if __name__ == "__main__":
    main()
