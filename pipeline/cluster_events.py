"""Step 1 of the identity solver: detect cluster events from existing per-frame tracks.

A cluster event is a frame where multiple ants either disappear together
(merge) or appear together (split). The tracker doesn't preserve identity
through these events; the tracklet graph (next step) will reconnect them.

V1 strategy — purely from track endpoints, no re-detection:
  • For each frame F, collect deaths (last_frame == F) and births (first_frame == F)
  • Spatially cluster within each list using a small radius (R ≈ 30 px)
  • Any cluster of >= 2 endpoints at the same frame is an event
  • Filter out edge births/deaths (ants entering/leaving the arena)

Output: cluster_events.parquet with one row per event, plus member ant_ids.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DEFAULT_TRACKS = Path(
    "/Users/theosechopoulos/Development/cimc/ants-philip/pipeline/tracks_step1_maxlost60_clean.parquet"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "cluster_events.parquet"

FRAME_W, FRAME_H = 1920, 540
EDGE_PX = 100             # exclude births/deaths within this many px of left/right edge
CLUSTER_RADIUS = 30.0     # spatial radius (px) for grouping simultaneous endpoints
MIN_MEMBERS = 2           # cluster must have >= this many endpoints to count
TIME_WINDOW = 2           # also collapse events within ±this many frames


def cluster_endpoints(frame_idx: int, points: np.ndarray, ids: np.ndarray,
                      radius: float, min_members: int) -> list[dict]:
    """Greedy spatial clustering: nearest-pair merge until no pair within radius.

    Returns one dict per cluster with size >= min_members.
    """
    if len(points) < min_members:
        return []
    n = len(points)
    cluster_of = np.arange(n)  # union-find
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    for i, j in pairs:
        ri = i
        while cluster_of[ri] != ri:
            ri = cluster_of[ri]
        rj = j
        while cluster_of[rj] != rj:
            rj = cluster_of[rj]
        if ri != rj:
            cluster_of[max(ri, rj)] = min(ri, rj)
    # Path compression
    for i in range(n):
        r = i
        while cluster_of[r] != r:
            r = cluster_of[r]
        cluster_of[i] = r

    out: list[dict] = []
    for c in np.unique(cluster_of):
        members = np.where(cluster_of == c)[0]
        if len(members) < min_members:
            continue
        cx = float(points[members, 0].mean())
        cy = float(points[members, 1].mean())
        out.append({
            "frame": int(frame_idx),
            "x": cx,
            "y": cy,
            "n_members": int(len(members)),
            "member_ids": [int(ids[m]) for m in members],
        })
    return out


def main() -> None:
    tracks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACKS
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    print(f"reading {tracks_path}")
    df = pd.read_parquet(tracks_path)
    print(f"  {len(df):,} rows, {df.ant_id.nunique()} ant_ids, "
          f"frames {df.frame.min()}-{df.frame.max()}")

    df = df.sort_values(["ant_id", "frame"])

    # Track endpoints. Exclude births at first frame (initial state) and deaths
    # at last frame (video ended) — those are not real cluster events.
    f_min = int(df.frame.min())
    f_max = int(df.frame.max())
    firsts = df.groupby("ant_id").first().reset_index()
    lasts = df.groupby("ant_id").last().reset_index()
    print(f"  total births: {len(firsts):,}  deaths: {len(lasts):,}")
    firsts = firsts[firsts.frame > f_min]
    lasts = lasts[lasts.frame < f_max]
    print(f"  after dropping video-boundary endpoints: "
          f"births={len(firsts):,}  deaths={len(lasts):,}")

    # Filter edge endpoints (entering/leaving via left/right edge)
    def keep_central(g: pd.DataFrame) -> pd.DataFrame:
        return g[(g.x >= EDGE_PX) & (g.x <= FRAME_W - EDGE_PX)]

    firsts_c = keep_central(firsts)
    lasts_c = keep_central(lasts)
    print(f"  central births: {len(firsts_c):,}  central deaths: {len(lasts_c):,}")

    # ---- Detect merges (clusters of simultaneous deaths) ----
    print(f"\ndetecting merge events (radius={CLUSTER_RADIUS}px, min_members={MIN_MEMBERS})...")
    merges: list[dict] = []
    for f, group in lasts_c.groupby("frame"):
        pts = group[["x", "y"]].to_numpy(dtype=np.float64)
        ids = group.ant_id.to_numpy()
        for c in cluster_endpoints(int(f), pts, ids, CLUSTER_RADIUS, MIN_MEMBERS):
            c["kind"] = "merge"
            merges.append(c)
    print(f"  {len(merges)} raw merge events")

    # ---- Detect splits (clusters of simultaneous births) ----
    print(f"detecting split events...")
    splits: list[dict] = []
    for f, group in firsts_c.groupby("frame"):
        pts = group[["x", "y"]].to_numpy(dtype=np.float64)
        ids = group.ant_id.to_numpy()
        for c in cluster_endpoints(int(f), pts, ids, CLUSTER_RADIUS, MIN_MEMBERS):
            c["kind"] = "split"
            splits.append(c)
    print(f"  {len(splits)} raw split events")

    # ---- Collapse close-in-time events at same location (e.g. merge over 2 frames) ----
    def collapse_temporal(events: list[dict]) -> list[dict]:
        """Merge events of the same kind within TIME_WINDOW frames and CLUSTER_RADIUS px."""
        if not events:
            return events
        events = sorted(events, key=lambda e: (e["frame"], e["x"]))
        out: list[dict] = []
        used = [False] * len(events)
        for i, e in enumerate(events):
            if used[i]:
                continue
            group = [e]
            used[i] = True
            for j in range(i + 1, len(events)):
                if used[j]:
                    continue
                f = events[j]
                if f["frame"] - e["frame"] > TIME_WINDOW:
                    break
                d = ((f["x"] - e["x"]) ** 2 + (f["y"] - e["y"]) ** 2) ** 0.5
                if d <= CLUSTER_RADIUS:
                    group.append(f)
                    used[j] = True
            merged_ids: list[int] = []
            for g in group:
                merged_ids.extend(g["member_ids"])
            out.append({
                "frame": int(round(np.mean([g["frame"] for g in group]))),
                "x": float(np.mean([g["x"] for g in group])),
                "y": float(np.mean([g["y"] for g in group])),
                "n_members": len(merged_ids),
                "member_ids": merged_ids,
                "kind": e["kind"],
            })
        return out

    merges = collapse_temporal(merges)
    splits = collapse_temporal(splits)
    print(f"\nafter temporal collapse:  merges={len(merges)}  splits={len(splits)}")

    # ---- Flag likely overlay artifacts ----
    # Heuristic: large events (>= 6 members) clustered in time AND space with
    # other large events tend to be picture-in-picture overlay artifacts.
    SUSPICIOUS_THRESHOLD = 6  # n_members
    rows: list[dict] = []
    for ev_id, ev in enumerate(merges + splits):
        rows.append({
            "event_id": ev_id,
            "frame": ev["frame"],
            "kind": ev["kind"],
            "x": ev["x"],
            "y": ev["y"],
            "n_members": ev["n_members"],
            "member_ids": ev["member_ids"],
            "suspicious": bool(ev["n_members"] >= SUSPICIOUS_THRESHOLD),
        })
    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    n_susp = int(out.suspicious.sum())
    print(f"\nwrote {out_path}: {len(out)} events  ({n_susp} flagged suspicious)")

    # ---- Summary ----
    print("\n=== SUMMARY ===")
    if len(out):
        print(f"total events: {len(out):,}  (merges={len(merges)}, splits={len(splits)})")
        for kind in ("merge", "split"):
            sub = out[out.kind == kind]
            if not len(sub):
                continue
            print(f"\n{kind}:  count={len(sub):,}")
            print(f"  member count distribution:")
            counts = sub.n_members.value_counts().sort_index()
            for k, v in counts.items():
                print(f"    {int(k):3d}-way: {int(v):,}")
            total = int(sub.n_members.sum())
            print(f"  total ants involved (rough): {total:,}")
    else:
        print("no cluster events detected — check radius/threshold")


if __name__ == "__main__":
    main()
