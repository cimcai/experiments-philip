"""Step 3 of plan A: solve global identity assignment using the tracklet graph.

Strategy (greedy, count-conserving):
1. Continuation edges directly merge two tracklets into one global identity.
2. For each merge event:
     - Tracklets dying into it pool their identities into the cluster's "member set".
3. For each split event (in temporal order):
     - The cluster's member set has K identities accumulated so far.
     - K_out tracklets emerge; assign each emergent tracklet to the best-matching
       member identity by motion continuity (predicted exit position).
     - If K_out < K: surplus identities stay in the cluster (it didn't fully break up).
     - If K_out > K: surplus emergent tracklets get fresh identities (we lost track
       of who entered).
4. Tracklets not covered by any of the above retain their own ant_id as identity.

Output: tracks_resolved.parquet with a `gid` column (global identity) added.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_GRAPH = Path(__file__).resolve().parent / "tracklet_graph.json"
DEFAULT_TRACKS = Path(
    "/Users/theosechopoulos/Development/cimc/ants-philip/pipeline/tracks_step1_maxlost60_clean.parquet"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "tracks_resolved.parquet"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, a: int) -> int:
        if self.parent.setdefault(a, a) == a:
            return a
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[max(ra, rb)] = min(ra, rb)


def main() -> None:
    graph_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRAPH
    tracks_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TRACKS
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUT

    print(f"reading graph: {graph_path}")
    g = json.loads(graph_path.read_text())
    tracklets = {t["ant_id"]: t for t in g["tracklets"]}
    events = {e["event_id"]: e for e in g["events"]}
    cont_edges = g["edges"]["continuation"]
    merge_in = g["edges"]["merge_in"]
    split_out = g["edges"]["split_out"]
    print(f"  {len(tracklets)} tracklets  {len(events)} events  "
          f"{len(cont_edges)} cont  {len(merge_in)} merge_in  {len(split_out)} split_out")

    # 1. Continuation edges: union-find merge
    uf = UnionFind()
    for tid in tracklets:
        uf.find(tid)  # initialize
    for e in cont_edges:
        uf.union(e["src_tracklet_id"], e["dst_tracklet_id"])
    n_components_after_cont = len({uf.find(t) for t in tracklets})
    print(f"\nafter continuation merges: {n_components_after_cont} components")

    # 2. For each merge event: collect identities entering
    # 3. For each split event in temporal order: assign emergent tracklets
    # We process events sorted by frame, but for "merge then later split" semantics,
    # we maintain per-event "member identities".
    in_by_event: dict[int, list[int]] = defaultdict(list)
    for e in merge_in:
        in_by_event[e["event_id"]].append(e["tracklet_id"])
    out_by_event: dict[int, list[int]] = defaultdict(list)
    for e in split_out:
        out_by_event[e["event_id"]].append(e["tracklet_id"])

    # For now: each merge event creates a virtual "cluster_id" that acts as a
    # bag of identities. Multiple merges can stack (cluster grows). Multiple
    # splits empty the bag. We model the cluster as a separate union-find
    # bucket that represents *whichever identity emerged*.
    #
    # Simplification: union the event's incoming tracklet identities. When the
    # event has outgoing tracklets, union all of them into the same bucket.
    # This trades the count-conservation niceness for a coarser equivalence
    # class, but yields a consistent identity for ants that pass through a
    # cluster together. Granular per-ant assignment inside a cluster is left
    # for a future iteration.
    suspicious_ev = {ev_id for ev_id, e in events.items() if e.get("suspicious")}
    print(f"  excluding {len(suspicious_ev)} suspicious events from union")
    # NOTE: we do NOT directly union the in-list of a merge or out-list of a
    # split. Those tracklets coexist in time before the event; unioning them
    # would put multiple concurrent positions under one gid. Only the
    # merge→split bridge carries identity through a cluster event below.
    # Bridge: if a merge and a later split share location, link their *single*
    # in-tracklet to *single* out-tracklet (greedy 1:1). For 2-way pairs we
    # assign by velocity continuity. For larger or unequal counts we abstain
    # to avoid wrong identity assignments.
    bridges = 0
    for m_id, m in events.items():
        if m["kind"] != "merge" or m_id in suspicious_ev:
            continue
        m_in = in_by_event.get(m_id, [])
        if not m_in:
            continue
        for s_id, s in events.items():
            if s["kind"] != "split" or s_id in suspicious_ev:
                continue
            s_out = out_by_event.get(s_id, [])
            if not s_out:
                continue
            if not (0 < s["frame"] - m["frame"] <= 200):
                continue
            d = ((m["x"] - s["x"]) ** 2 + (m["y"] - s["y"]) ** 2) ** 0.5
            if d > 60:
                continue
            if len(m_in) != len(s_out):
                continue  # mismatched counts — count not conserved, abstain
            if len(m_in) == 1:
                uf.union(m_in[0], s_out[0])
                bridges += 1
            elif len(m_in) == 2:
                # Match by motion: use end velocity of incoming, start velocity of outgoing
                inA, inB = tracklets[m_in[0]], tracklets[m_in[1]]
                outA, outB = tracklets[s_out[0]], tracklets[s_out[1]]
                def vsim(p, q):
                    return (p["vx_end"] - q["vx_start"]) ** 2 + (p["vy_end"] - q["vy_start"]) ** 2
                # Two pairings: (A->A, B->B) vs (A->B, B->A); pick lower total
                cost_aa = vsim(inA, outA) + vsim(inB, outB)
                cost_ab = vsim(inA, outB) + vsim(inB, outA)
                if cost_aa <= cost_ab:
                    uf.union(m_in[0], s_out[0]); uf.union(m_in[1], s_out[1])
                else:
                    uf.union(m_in[0], s_out[1]); uf.union(m_in[1], s_out[0])
                bridges += 2
    print(f"merge-split bridges (count-conserved passthrough): {bridges}")

    # Compute final mapping
    gid_of: dict[int, int] = {}
    root_to_gid: dict[int, int] = {}
    next_gid = 0
    for tid in sorted(tracklets):
        r = uf.find(tid)
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        gid_of[tid] = root_to_gid[r]
    n_components = len(root_to_gid)
    print(f"\nfinal components / global IDs: {n_components}")
    print(f"  tracklets per gid: median={np.median([np.sum(np.array(list(gid_of.values()))==g) for g in range(min(100, n_components))]):.0f}")

    # Distribution
    gid_arr = np.array(list(gid_of.values()))
    sizes = np.bincount(gid_arr)
    print(f"  gid sizes: median={int(np.median(sizes))} max={int(sizes.max())} "
          f"singletons={int((sizes == 1).sum())}")

    # Apply to tracks
    print(f"\nreading tracks: {tracks_path}")
    df = pd.read_parquet(tracks_path)
    df["gid"] = df.ant_id.map(gid_of).astype("int32")
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {len(df):,} rows  unique gids={df.gid.nunique()}")

    # Final stats
    print("\n=== RESOLUTION SUMMARY ===")
    print(f"  before: {len(tracklets):,} unique ant_ids")
    print(f"  after:  {df.gid.nunique():,} unique gids")
    print(f"  reduction: {1 - df.gid.nunique() / len(tracklets):.1%}")


if __name__ == "__main__":
    main()
