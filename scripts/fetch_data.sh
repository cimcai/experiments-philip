#!/usr/bin/env bash
# Download the source video and a pre-computed resolved-tracks parquet
# from the GitHub release. Lets a fresh clone go straight to the viewer
# without rerunning the slow detector / tracker / identity-solver stages.

set -euo pipefail

REPO="theosech/ants-philip"
TAG="v0.1.0"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p pipeline

fetch() {
  local url="$1" dest="$2" min_bytes="$3"
  if [[ -f "$dest" ]] && (( $(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest") >= min_bytes )); then
    echo "  [skip] $dest already present"
    return
  fi
  echo "  [get]  $dest"
  curl -fL --progress-bar -o "$dest" "$url"
}

echo "Fetching release assets from ${REPO}@${TAG}…"
fetch "${BASE}/ants_full.mp4"       ants_full.mp4                       50000000
fetch "${BASE}/tracks_clean.parquet" pipeline/tracks_clean.parquet      50000000

cat <<'NOTE'

Done. Two ways to proceed:

  1. Skip the slow CV stages — go straight to the viewer:
       uv run python pipeline/export_viewer.py
       python3 pipeline/viewer/serve.py
       open http://localhost:8765

  2. Re-run the full pipeline (re-derives tracks_clean.parquet):
       uv run python -m pipeline.run
       uv run python -m pipeline.stitch_tracks --in pipeline/tracks.parquet --out pipeline/tracks_step.parquet
       uv run python pipeline/cluster_events.py
       uv run python pipeline/tracklet_graph.py
       uv run python pipeline/identity_solver.py
       uv run python pipeline/export_viewer.py
NOTE
