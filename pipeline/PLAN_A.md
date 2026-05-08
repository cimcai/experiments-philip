# Plan A — Tracklet graph + count conservation

## Context

Best baseline: `tracks_step1_maxlost60_clean.parquet` (step=1, MAX_LOST=60, then stitched). 7,429 unique IDs over 211 s of video, 412 real ants per frame → **18× fragmentation**. The dominant failure mode (88.8% of births mid-frame, 91.2% of deaths mid-frame) is cluster events: ants pile onto the T-shape, merge into a single blob, then disperse — and the per-frame Hungarian tracker can't preserve identity through that.

Behavioral science question: *what rules do ants follow when joining/leaving the pile?* That makes cluster boundaries (entries and exits) the highest-value moments to recover.

## What the tracklet graph adds

The current pipeline emits a stream of (frame, ant_id, x, y, theta, area) — implicitly assuming each detection corresponds to one ant. When clusters form, that breaks:

- N ants approach each other, their masks merge → only one detection survives
- Hungarian sees N-1 unmatched tracks → kills them
- One detection grows in area as more ants pile on
- When an ant peels off, a new detection appears nearby → Hungarian gives it a new ID

Tracklet graph reframes detections into:
- **Nodes** = tracklets (continuous Hungarian matches between birth, merge, split, or death)
- **Edges** = "tracklet B is plausibly the same ant as tracklet A": continuation (gap), child of merge, parent of split
- **Identity** = a global ID assigned by walking the graph subject to count conservation

When 6 ants pile on the T and the cluster persists 5 s before one peels off, the cluster blob is *one* node in the tracklet graph. Identity is preserved at boundaries (the 6 → 5 + 1 split is graph-resolvable) without trying to disentangle ants inside the pile.

## Scope (one-day target)

### 1. Detect cluster events from existing tracks (read-only)

Input: `tracks_step1_maxlost60_clean.parquet`. For each frame:
- Identify "detection blobs" by *area* — anything with area > 1.5× median ant area (~150 px²) is likely a multi-ant blob
- For each large blob, count nearby tracks (within blob radius); call this its "membership count"
- A **merge event** = a frame where K tracks die simultaneously near a single growing blob; record (event_frame, blob_id, dead_tids, location, area_jump)
- A **split event** = K tracks born simultaneously near a shrinking blob

This requires no re-tracking. Output: `cluster_events.parquet` with cols `[event_frame, kind, blob_track_id, member_count, dead_or_born_tids, x, y, area]`.

### 2. Build the tracklet graph

Nodes: every ant_id in the cleaned tracks (7,429 of them) + every detected cluster blob.

Edges:
- **Continuation**: parent track ends within 30 frames of child track start, distance <= velocity-extrapolated radius. (This is essentially what `stitch_tracks.py` does — port that scoring.)
- **Merge child**: track ends within 5 frames of a merge event, within blob radius
- **Split parent**: track starts within 5 frames of a split event, within blob radius
- **Cluster persistence**: each cluster blob is a track itself; it has membership count over time

Output: `tracklet_graph.json` (or networkx pickle) — for inspection.

### 3. Solve identity assignment

Greedy graph walk:
1. Initialize: every long track (>10 s) gets a unique global identity. Short tracks default to "fragment".
2. Apply continuation edges: merge tracklets joined by gaps (this overlaps with stitcher; reuse).
3. For each cluster event in temporal order:
   - Merge: at frame F, K identified tracks enter cluster C. Stamp the cluster's "member set" with their identities.
   - Split: at frame F, the cluster has member set {a, b, c, d, e, f}. K' tracks emerge. Assign each emergent track to whichever identity in the member set best matches by motion continuity (predicted exit velocity + time-in-cluster prior).
   - Conserve count: emergent tracks count must equal departed tracks count + birth from outside; if mismatch, leave the surplus as fragments (don't fabricate).
4. For tracks not yet identified (orphans): propagate from continuation edges; fall back to "fragment" if none.

Output: `tracks_resolved.parquet` with cols `[frame, t_ms, ant_id_global, x, y, theta, area, in_cluster_id, confidence]`.

### 4. Audit + visualize

Re-run the failure audit on `tracks_resolved.parquet`:
- Fragmentation ratio (target: <5×, vs current 18×)
- Mid-frame births (target: <50%, vs current 89%)
- IDs that traverse cluster events (we want the count to be **non-zero and growing** — that's the new capability)

Render an overlay video that highlights:
- Cluster blobs (translucent shaded regions)
- Identity preservation through cluster events (color stays on each global ID)

### 5. Behavioral primitives (stretch goal)

Once identity preserved, compute per-(global-ant) features:
- Time spent in cluster vs free
- Approach/depart velocity at cluster boundaries
- Position relative to load center (which side did this ant push from?)

Output: `ant_summary.parquet` for downstream behavioral analysis.

## Files to add

All new under `ants-philip-sam3/pipeline/`:
- `cluster_events.py` — step 1
- `tracklet_graph.py` — step 2
- `identity_solver.py` — step 3
- `audit_resolved.py` — step 4 (extends `aphilip_failure_audit.py`)
- `behavior_primitives.py` — step 5

Each runs independently from CLI; orchestrator `plan_a.py` chains them.

## Files to reuse

- `pipeline/aphilip_failure_audit.py` — for evaluation
- `ants-philip/pipeline/stitch_tracks.py:60-130` — velocity extrapolation logic for continuation edges
- `ants-philip/pipeline/run.py:140-190` — Tracker class understanding, NOT re-running

## Risks

- **Cluster blob detection is noisy**: choosing the area threshold matters. Will likely need tuning on a few cluster events.
- **Count conservation can be wrong**: if SAM-merged ants are off by 1, the solver propagates errors. Mitigation: low-confidence flag on cluster-emerged identities; track them separately.
- **Long-cluster events**: if the pile persists 30+ seconds, motion-continuity-based exit assignment becomes near-random. Acceptable to flag those identities as "uncertain".

## Verification

End-to-end pass:
1. `tracks_resolved.parquet` exists and has > 1,000 IDs with `>=10 s` length (currently 2,526)
2. Mid-frame birth rate drops below 50%
3. Visualize 3 cluster events: verify entering ants keep their colors after exiting
4. Run on the existing viewer (export_viewer.py) and scrub through cluster events
