# ants-philip

Multi-ant tracking and identity recovery from top-down cooperative-behavior footage.

![Viewer demo](docs/viewer-demo.gif)

The pipeline takes a top-down video of many ants (here: ~400 ants pushing a T-shaped load through barriers, from a [Wonder World clip](https://www.youtube.com/watch?v=j9xnhmFA7Ao)) and produces per-ant trajectories with global identities preserved through cluster events. A browser viewer renders the resolved tracks frame-by-frame for analysis.

## Result

| Metric                          | Baseline (Hungarian + stitcher) | + Identity solver |
|---------------------------------|---------------------------------|-------------------|
| Unique tracked IDs / 211 s      | 7,429                           | **4,635** (−38%)  |
| Median track length             | 3.6 s                           | **8.9 s**         |
| Tracks > 10 s                   | 20.5%                           | **47.5%**         |
| Mid-frame births (= ID swaps)   | 90.6%                           | 88.0%             |
| Real ant population (per frame) | 412                             | 412               |

The remaining fragmentation lives almost entirely in cluster events where count conservation isn't unambiguous (more than 2 ants entering a pile and a different number leaving). The solver abstains rather than guess wrong; see [`pipeline/IDENTITY_SOLVER.md`](pipeline/IDENTITY_SOLVER.md) for the methodology.

## Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| 1     | [`pipeline/run.py`](pipeline/run.py) | Background-subtraction detector + Hungarian linker with constant-velocity prior. Emits per-frame `(x, y, θ, area, ant_id)`. |
| 2     | [`pipeline/stitch_tracks.py`](pipeline/stitch_tracks.py) | Velocity-extrapolation pass that merges fragmented tracks across short occlusions. |
| 3a    | [`pipeline/cluster_events.py`](pipeline/cluster_events.py) | Detects cluster events (multiple ants disappearing/appearing together). |
| 3b    | [`pipeline/tracklet_graph.py`](pipeline/tracklet_graph.py) | Builds a graph: nodes = tracklets + events; edges = continuation, merge-in, split-out. |
| 3c    | [`pipeline/identity_solver.py`](pipeline/identity_solver.py) | Greedy union-find solver with count-conserved 1:1 / 2:2 cluster passthrough bridges. |
| 4     | [`pipeline/export_viewer.py`](pipeline/export_viewer.py) → [`pipeline/viewer/`](pipeline/viewer/) | Packs resolved tracks into a compact binary blob; static-served HTML viewer reads it. |

[`pipeline/load_track.py`](pipeline/load_track.py) tracks the red T-shape (the "load") separately. [`pipeline/run_step1_max60.py`](pipeline/run_step1_max60.py) is a thin wrapper that runs stage 1 at 60 fps with a longer grace window — the configuration that fed the identity solver above.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cimcai/ants-philip.git
cd ants-philip
uv sync
```

The lockfile pins all dependencies. `imageio-ffmpeg` provides a bundled ffmpeg binary, so no system-level ffmpeg install is needed.

## Get the data

The 65 MB source video and a pre-computed resolved-tracks parquet are hosted as release assets on the [`theosech/ants-philip`](https://github.com/theosech/ants-philip/releases/tag/v0.1.0) fork:

```bash
./scripts/fetch_data.sh
```

This downloads:
- `ants_full.mp4` (65 MB) → repo root
- `pipeline/tracks_clean.parquet` (80 MB) — pre-computed resolved tracks (4,635 gids), so you can skip the slow CV stages and view results immediately.

## Run

**Fast path — viewer only** (uses the fetched pre-computed tracks):

```bash
uv run python pipeline/export_viewer.py
python3 pipeline/viewer/serve.py
# open http://localhost:8765
```

**Full pipeline** (re-derives `tracks_clean.parquet` from the source video; ~10–30 min for stages 1–2):

```bash
# Stage 1: detect + track (default --step 2 → 30 fps effective)
uv run python -m pipeline.run

# Stage 1 alt: --step 1 + MAX_LOST=60 (the configuration tuned for the identity solver)
uv run python -m pipeline.run_step1_max60

# Stage 2: stitch tracks across short gaps
uv run python -m pipeline.stitch_tracks --in pipeline/tracks_step1_maxlost60.parquet --out pipeline/tracks_step1_maxlost60_clean.parquet

# Stage 3: identity solver (a-c chained)
uv run python pipeline/cluster_events.py
uv run python pipeline/tracklet_graph.py
uv run python pipeline/identity_solver.py

# Wire the resolved tracks into the viewer and serve
uv run python -c "import pandas as pd; \
  df = pd.read_parquet('pipeline/tracks_resolved.parquet'); \
  df[['frame','t_ms','gid','x','y','theta','area']].rename(columns={'gid':'ant_id'}).to_parquet('pipeline/tracks_clean.parquet', index=False)"
uv run python pipeline/export_viewer.py
python3 pipeline/viewer/serve.py
```

The audit script reproduces the result-table metrics on any tracks parquet:

```bash
uv run python pipeline/aphilip_failure_audit.py pipeline/tracks_clean.parquet
```

## Viewer keyboard

| Key            | Action                                |
|----------------|---------------------------------------|
| `Space`        | play / pause                          |
| `←` / `→`      | step one frame; `Shift+→` jumps 10    |
| `F`            | toggle force-arrow overlay            |
| `L`            | toggle load (T-shape) overlay         |
| `Esc`          | unhide all ants                       |
| `/`            | focus search                          |

Click any row (or any ant on the canvas) to toggle visibility. `select all` / `unselect all` operate on the currently filtered list.

## Run on your own footage

Drop a top-down video at `ants_full.mp4` and the pipeline should run as-is for similar scale. Tunables worth checking:

- [`pipeline/run.py:35-44`](pipeline/run.py) — `BG_DIFF_THRESH`, `N_BG_SAMPLES`, `MIN_AREA`, `MAX_AREA`, `MAX_LOST`, distance caps.
- [`pipeline/cluster_events.py`](pipeline/cluster_events.py) — `EDGE_PX`, `CLUSTER_RADIUS`, `MIN_MEMBERS` (cluster-event detection thresholds).
- [`pipeline/tracklet_graph.py`](pipeline/tracklet_graph.py) — `CONT_MAX_GAP_FRAMES`, `CONT_MAX_DIST_PX` (continuation-edge generosity).

For very different scales (much smaller / larger ants, very different fps, different colored load), parameters will need re-tuning. The audit script is the recommended yardstick — re-run after each tunable change and watch fragmentation ratio + median track length.

## Repository layout

```
ants-philip/
├── ants_full.mp4              # source video (release asset, gitignored)
├── pipeline/
│   ├── run.py                 # detect + track
│   ├── run_step1_max60.py     # step=1 / MAX_LOST=60 wrapper
│   ├── load_track.py          # red-T (load) tracker
│   ├── stitch_tracks.py       # post-process stitcher
│   ├── cluster_events.py      # identity-solver step 1
│   ├── tracklet_graph.py      # identity-solver step 2
│   ├── identity_solver.py     # identity-solver step 3
│   ├── IDENTITY_SOLVER.md     # methodology doc
│   ├── aphilip_failure_audit.py
│   ├── *_overlay.py           # diagnostic visualizations
│   ├── export_viewer.py       # pack tracks for viewer
│   ├── seed_detect.py         # classical seed boxes (used by sam2_modal)
│   ├── sam2_modal.py          # SAM 2.1 video segmenter on Modal (reference; not used in pipeline)
│   ├── filter_grid_artifacts.py  # remove static-grid false positives
│   └── viewer/
│       ├── index.html         # browser viewer
│       └── serve.py           # range-request HTTP server
├── scripts/
│   └── fetch_data.sh          # download release assets
├── docs/
│   └── viewer-demo.gif        # this README's demo
├── pyproject.toml
├── uv.lock
└── LICENSE                    # MIT
```

## License

[MIT](LICENSE).
