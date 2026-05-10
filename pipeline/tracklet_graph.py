"""Step 2 of the identity solver: build the tracklet graph.

Nodes:
  • tracklet — one per ant_id, with birth/death frames + positions + velocity
  • event — one per cluster event (merge or split)

Edges:
  • merge_in: tracklet -> merge_event (the tracklet died into this merge)
  • split_out: split_event -> tracklet (this tracklet was born from this split)
  • continuation: tracklet_A -> tracklet_B (A's death extrapolates to B's birth,
      neither involved in a cluster event of the relevant kind)

Velocity-extrapolation continuation is the same idea as stitch_tracks.py but
runs after cluster events so we don't double-bind. Output is a JSON graph
with adjacency lists, easy to inspect by eye.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TRACKS = Path(
    "/Users/theosechopoulos/Development/cimc/ants-philip/pipeline/tracks_step1_maxlost60_clean.parquet"
)
DEFAULT_EVENTS = Path(__file__).resolve().parent / "cluster_events.parquet"
DEFAULT_OUT = Path(__file__).resolve().parent / "tracklet_graph.json"

# Continuation parameters (same spirit as stitch_tracks.py but for tracks
# unrelated to detected cluster events).
CONT_MAX_GAP_FRAMES = 180       # max source-frame gap (3s) — catch more 1-1 swaps
CONT_MAX_DIST_PX = 100.0        # max distance after velocity extrapolation
EVENT_BIND_RADIUS_PX = 35.0     # tracklet endpoint must be within this of event center
EVENT_BIND_FRAME_TOL = 2        # event frame can differ from endpoint frame by this much


def build_tracklets(df: pd.DataFrame) -> dict[int, dict]:
    """Compute per-tracklet summary stats."""
    tracklets: dict[int, dict] = {}
    df = df.sort_values(["ant_id", "frame"])
    for ant_id, grp in df.groupby("ant_id"):
        x = grp.x.to_numpy()
        y = grp.y.to_numpy()
        f = grp.frame.to_numpy()
        # Mean velocity over the last min(20, len) samples
        k = min(20, len(grp))
        if k >= 2:
            df_dt = max(int(f[-1] - f[-k]), 1)
            vx = float(x[-1] - x[-k]) / df_dt
            vy = float(y[-1] - y[-k]) / df_dt
        else:
            vx = vy = 0.0
        # Mean velocity at start
        if k >= 2:
            df_dt = max(int(f[k - 1] - f[0]), 1)
            vx0 = float(x[k - 1] - x[0]) / df_dt
            vy0 = float(y[k - 1] - y[0]) / df_dt
        else:
            vx0 = vy0 = 0.0
        tracklets[int(ant_id)] = {
            "ant_id": int(ant_id),
            "first_frame": int(f[0]),
            "last_frame": int(f[-1]),
            "first_x": float(x[0]),
            "first_y": float(y[0]),
            "last_x": float(x[-1]),
            "last_y": float(y[-1]),
            "vx_end": vx,
            "vy_end": vy,
            "vx_start": vx0,
            "vy_start": vy0,
            "n_frames": int(len(grp)),
            "area_mean": float(grp.area.mean()),
        }
    return tracklets


def bind_events_to_tracklets(events: pd.DataFrame, tracklets: dict[int, dict],
                             ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Associate each event with its constituent tracklet endpoints.

    Returns:
      merge_in: list of (tracklet_id, event_id) where tracklet died into a merge
      split_out: list of (event_id, tracklet_id) where tracklet was born from a split
    """
    merge_in: list[tuple[int, int]] = []
    split_out: list[tuple[int, int]] = []

    for _, ev in events.iterrows():
        ev_frame = int(ev.frame)
        ev_x, ev_y = float(ev.x), float(ev.y)
        member_ids = list(ev.member_ids)
        for tid in member_ids:
            t = tracklets.get(int(tid))
            if t is None:
                continue
            if ev.kind == "merge":
                # tracklet should die at/near the event frame
                d_frame = abs(t["last_frame"] - ev_frame)
                d_pos = ((t["last_x"] - ev_x) ** 2 + (t["last_y"] - ev_y) ** 2) ** 0.5
                if d_frame <= EVENT_BIND_FRAME_TOL and d_pos <= EVENT_BIND_RADIUS_PX * 3:
                    merge_in.append((int(tid), int(ev.event_id)))
            else:  # split
                d_frame = abs(t["first_frame"] - ev_frame)
                d_pos = ((t["first_x"] - ev_x) ** 2 + (t["first_y"] - ev_y) ** 2) ** 0.5
                if d_frame <= EVENT_BIND_FRAME_TOL and d_pos <= EVENT_BIND_RADIUS_PX * 3:
                    split_out.append((int(ev.event_id), int(tid)))
    return merge_in, split_out


def find_continuation_edges(tracklets: dict[int, dict],
                            tids_in_merge: set[int],
                            tids_in_split: set[int]) -> list[tuple[int, int, dict]]:
    """Greedy: for each non-merged terminus, find a non-split birth that fits."""
    # Bucket tracklet starts by first_frame for fast lookup
    starts_by_frame: dict[int, list[int]] = {}
    for tid, t in tracklets.items():
        if tid in tids_in_split:
            continue  # already explained by a split event
        starts_by_frame.setdefault(t["first_frame"], []).append(tid)

    edges: list[tuple[int, int, dict]] = []
    used_starts: set[int] = set()
    # Sort by death frame so earlier deaths get first pick of stitchable births
    dying = sorted(
        ((tid, t) for tid, t in tracklets.items() if tid not in tids_in_merge),
        key=lambda kv: kv[1]["last_frame"],
    )
    for tid, t in dying:
        best = None
        best_score = float("inf")
        # Check candidate births in the gap window
        for gap in range(1, CONT_MAX_GAP_FRAMES + 1):
            target_frame = t["last_frame"] + gap
            for cand_tid in starts_by_frame.get(target_frame, []):
                if cand_tid == tid or cand_tid in used_starts:
                    continue
                c = tracklets[cand_tid]
                # Extrapolate dying tracklet's last position by velocity
                px = t["last_x"] + t["vx_end"] * gap
                py = t["last_y"] + t["vy_end"] * gap
                d = ((c["first_x"] - px) ** 2 + (c["first_y"] - py) ** 2) ** 0.5
                if d > CONT_MAX_DIST_PX:
                    continue
                # Score combines distance and gap (prefer smaller gaps and tighter fits)
                score = d + 0.3 * gap
                if score < best_score:
                    best_score = score
                    best = (cand_tid, gap, d)
        if best is not None:
            cand_tid, gap, d = best
            edges.append((tid, cand_tid, {"gap": gap, "dist": d}))
            used_starts.add(cand_tid)
    return edges


def main() -> None:
    tracks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS
    events_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_EVENTS
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUT

    print(f"reading tracks: {tracks_path}")
    tracks_df = pd.read_parquet(tracks_path)
    print(f"  {len(tracks_df):,} rows  {tracks_df.ant_id.nunique()} tracklets")

    print(f"reading events: {events_path}")
    events_df = pd.read_parquet(events_path)
    print(f"  {len(events_df)} events")

    print("\nbuilding tracklet summaries...")
    tracklets = build_tracklets(tracks_df)
    print(f"  {len(tracklets)} tracklet nodes")

    print("\nbinding tracklets to cluster events...")
    merge_in, split_out = bind_events_to_tracklets(events_df, tracklets)
    print(f"  merge_in edges: {len(merge_in)}  (tracklets dying into merges)")
    print(f"  split_out edges: {len(split_out)}  (tracklets born from splits)")

    tids_in_merge = {tid for tid, _ in merge_in}
    tids_in_split = {tid for _, tid in split_out}
    print(f"  tracklets bound to a merge: {len(tids_in_merge)}")
    print(f"  tracklets bound to a split: {len(tids_in_split)}")

    print("\nfinding continuation edges (gap stitching for non-cluster cases)...")
    cont_edges = find_continuation_edges(tracklets, tids_in_merge, tids_in_split)
    print(f"  {len(cont_edges)} continuation edges")

    # Stats by gap
    if cont_edges:
        gaps = np.array([e[2]["gap"] for e in cont_edges])
        dists = np.array([e[2]["dist"] for e in cont_edges])
        print(f"  gap distribution: median={int(np.median(gaps))} "
              f"p90={int(np.quantile(gaps, 0.9))} max={int(gaps.max())}")
        print(f"  dist distribution: median={np.median(dists):.1f} "
              f"p90={np.quantile(dists, 0.9):.1f} max={dists.max():.1f}")

    graph = {
        "metadata": {
            "tracks_path": str(tracks_path),
            "events_path": str(events_path),
            "n_tracklets": len(tracklets),
            "n_events": len(events_df),
            "n_merge_in": len(merge_in),
            "n_split_out": len(split_out),
            "n_continuation": len(cont_edges),
            "params": {
                "cont_max_gap_frames": CONT_MAX_GAP_FRAMES,
                "cont_max_dist_px": CONT_MAX_DIST_PX,
                "event_bind_radius_px": EVENT_BIND_RADIUS_PX,
                "event_bind_frame_tol": EVENT_BIND_FRAME_TOL,
            },
        },
        "tracklets": list(tracklets.values()),
        "events": [
            {
                "event_id": int(r.event_id), "frame": int(r.frame), "kind": r.kind,
                "x": float(r.x), "y": float(r.y), "n_members": int(r.n_members),
                "suspicious": bool(r.suspicious),
            }
            for r in events_df.itertuples(index=False)
        ],
        "edges": {
            "merge_in": [{"tracklet_id": tid, "event_id": eid}
                         for tid, eid in merge_in],
            "split_out": [{"event_id": eid, "tracklet_id": tid}
                          for eid, tid in split_out],
            "continuation": [{"src_tracklet_id": a, "dst_tracklet_id": b, **d}
                             for a, b, d in cont_edges],
        },
    }
    out_path.write_text(json.dumps(graph, indent=2))
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}  ({size_mb:.1f} MB)")

    # ---- Quick coverage stats ----
    print("\n=== COVERAGE ===")
    n_tids = len(tracklets)
    n_explained = len(tids_in_merge | tids_in_split | {a for a, _, _ in cont_edges} | {b for _, b, _ in cont_edges})
    print(f"  tracklets with at least one in/out edge: {n_explained}/{n_tids} "
          f"({n_explained / n_tids:.1%})")
    n_isolated = n_tids - n_explained
    print(f"  isolated tracklets (no edges): {n_isolated}")


if __name__ == "__main__":
    main()
