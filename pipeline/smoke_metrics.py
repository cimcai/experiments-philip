"""Print decision-gate metrics from smoke_tracks.parquet.

Reports per-ant counts, track-length distribution, theta jitter, and load
coverage so we can apply the Phase 1.5 PASS / MARGINAL / FAIL rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = Path(__file__).resolve().parent / "smoke_tracks.parquet"


def main() -> None:
    tracks = Path(sys.argv[1]) if len(sys.argv) > 1 else TRACKS
    print(f"reading {tracks}")
    df = pd.read_parquet(tracks)
    print(f"rows: {len(df):,}  frames: {df.frame.nunique()}  "
          f"obj_ids: {df.obj_id.nunique()}  kinds: {df.kind.unique().tolist()}")

    ants = df[df.kind == "ant"].copy()
    load = df[df.kind == "load"].copy()

    # Per-frame ant count
    per_f = ants.groupby("frame").size()
    print("\nPER-FRAME ANT COUNT")
    print(f"  median={int(per_f.median())}  min={int(per_f.min())}  max={int(per_f.max())}  "
          f"std={per_f.std():.1f}")
    print(f"  frames with detections: {len(per_f)}/{df.frame.nunique()}")

    # Per-track length
    track_len = ants.groupby("obj_id").size()
    seeded_at_first = ants[ants.frame == ants.frame.min()].obj_id.nunique()
    print("\nTRACK LENGTH (frames per obj_id)")
    print(f"  ants seeded at first frame: {seeded_at_first}")
    print(f"  total unique ant obj_ids: {len(track_len)}")
    print(f"  track length median={int(track_len.median())}  "
          f"min={int(track_len.min())}  max={int(track_len.max())}  "
          f"mean={track_len.mean():.1f}")
    survival_rate = (track_len >= per_f.shape[0]).mean()
    print(f"  fraction of tracks surviving full chunk: {survival_rate:.2%}")

    # Theta jitter
    print("\nTHETA JITTER (consecutive-frame |Δθ| in radians)")
    ants_sorted = ants.sort_values(["obj_id", "frame"])
    deltas = []
    for _, grp in ants_sorted.groupby("obj_id"):
        if len(grp) < 2:
            continue
        dt = np.diff(grp.theta.to_numpy())
        # Wrap into [-pi/2, pi/2] since theta from PCA is direction-ambiguous (pi-period).
        dt = np.mod(dt + np.pi/2, np.pi) - np.pi/2
        deltas.append(np.abs(dt))
    if deltas:
        all_d = np.concatenate(deltas)
        print(f"  median={np.median(all_d):.3f}  p90={np.quantile(all_d,0.9):.3f}  "
              f"max={all_d.max():.3f}")
    else:
        print("  (insufficient data)")

    # Area distribution
    print("\nMASK AREA (pixels)")
    print(f"  ant median={ants.area.median():.0f}  p10={ants.area.quantile(0.1):.0f}  "
          f"p90={ants.area.quantile(0.9):.0f}")
    if len(load):
        print(f"  load median={load.area.median():.0f}  min={load.area.min():.0f}  "
              f"max={load.area.max():.0f}")

    # Load coverage
    print("\nLOAD COVERAGE")
    n_frames = df.frame.nunique()
    load_frames = load.frame.nunique()
    print(f"  load present in {load_frames}/{n_frames} frames "
          f"({load_frames/max(n_frames,1):.1%})")

    # Decision summary
    print("\n--- DECISION GATE ---")
    median_track = int(track_len.median())
    pass_track = median_track >= 50
    pass_count = (per_f.std() / max(per_f.median(), 1)) < 0.20
    pass_load = (load_frames / max(n_frames, 1)) >= 0.95
    pass_jitter = bool(deltas) and float(np.median(np.concatenate(deltas))) < 0.5

    flags = {
        "track length median > 50": pass_track,
        "per-frame count stable (std/median < 20%)": pass_count,
        "load present in >=95% of frames": pass_load,
        "theta jitter median < 0.5 rad": pass_jitter,
    }
    for k, v in flags.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    n_pass = sum(flags.values())
    if n_pass == 4:
        print("\n=> Phase 1: PASS  (proceed to Phase 2A)")
    elif n_pass >= 2:
        print("\n=> Phase 1: MARGINAL  (consider 2x upsample, or fall back to 2B)")
    else:
        print("\n=> Phase 1: FAIL  (proceed to Phase 2B tracklet-graph)")


if __name__ == "__main__":
    main()
