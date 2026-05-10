"""Identify and remove grid-dot false positives from tracks_clean.parquet.

The footage has a printed lattice of dark dots on the surface (used as
fiducials). The classical detector occasionally picks them up as "ants" that
never move. This filter:

1. Flags identities whose position has near-zero variance across their lifetime.
2. Confirms the flagged points lie on a regular lattice — each candidate must
   have multiple nearest-neighbors at the modal spacing among other stationary
   points. A briefly-resting real ant won't satisfy this.
3. Drops those identities and writes the filtered tracks back to
   tracks_clean.parquet. The original is backed up to
   tracks_clean_with_grid.parquet on the first run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

OUT = Path(__file__).parent
SRC = OUT / "tracks_clean.parquet"
BAK = OUT / "tracks_clean_with_grid.parquet"

# Stationarity threshold: a real ant's median position-std is ~90 px across
# its lifetime; grid-dot centroids wobble a few pixels at most due to threshold
# noise on the dot's edges. 8 px keeps clean separation while catching dots
# whose centroids drift more on certain frames.
STD_PX = 8.0
# Some grid dots are detected only intermittently (the threshold edge flickers
# with lighting), creating many short tracklets at the same fixed location.
# Drop the minimum lifetime so each flash is independently scrutinized — the
# lattice-phase check is the real false-positive guard.
MIN_LIFETIME = 8  # frames

# Lattice tolerance: nearest-neighbor distances within this fraction of the
# detected grid spacing are considered "on the grid".
NN_TOL = 0.15
MIN_GRID_NEIGHBORS = 2  # need at least this many NN at grid spacing

# Grid spacing search: the real grid spacing is found via histogram-mode of
# pairwise distances between stationary candidates, restricted to plausible
# fiducial spacings. This avoids being dragged down by tightly-clustered
# T-pile candidates (real ants resting on the load).
GRID_SPACING_MIN_PX = 50
GRID_SPACING_MAX_PX = 200
GRID_HIST_BIN_PX = 1.0

# Lattice alignment: a candidate must lie within this many pixels of the nearest
# lattice intersection in BOTH x and y. The lattice phase (offset) is recovered
# from the dominant histogram peak of (mx mod spacing) and (my mod spacing).
# 14 px accommodates phase drift across the frame. Real ants in the T-pile
# cluster have x-phases in 25-66, well outside this tolerance from x_offset=78.5.
LATTICE_TOL_PX = 14.0


def main() -> None:
    df = pd.read_parquet(SRC)
    n_ids_in = df.ant_id.nunique()
    print(f"input: {len(df):,} rows, {n_ids_in:,} identities")

    g = df.groupby("ant_id")
    stats = pd.DataFrame({
        "mx": g.x.mean(),
        "my": g.y.mean(),
        "sx": g.x.std().fillna(0),
        "sy": g.y.std().fillna(0),
        "n": g.size(),
    })
    stationary = stats[(stats.sx < STD_PX) & (stats.sy < STD_PX) & (stats.n >= MIN_LIFETIME)]
    print(f"stationary candidates: {len(stationary):,} "
          f"(sx,sy < {STD_PX} px, n >= {MIN_LIFETIME} f)")

    if len(stationary) < 4:
        print("too few stationary candidates to detect a grid; nothing to remove")
        return

    pts = stationary[["mx", "my"]].to_numpy()
    tree = cKDTree(pts)

    # Find the dominant grid spacing via histogram of pairwise distances within
    # the plausible fiducial range. Median would be skewed by the dense T-pile
    # cluster (resting ants very close to each other).
    pair_dists = tree.sparse_distance_matrix(tree, max_distance=GRID_SPACING_MAX_PX,
                                              output_type="coo_matrix").data
    pair_dists = pair_dists[(pair_dists >= GRID_SPACING_MIN_PX)
                            & (pair_dists <= GRID_SPACING_MAX_PX)]
    if len(pair_dists) == 0:
        print(f"no pairwise distances in [{GRID_SPACING_MIN_PX}, {GRID_SPACING_MAX_PX}] px range")
        return
    bins = np.arange(GRID_SPACING_MIN_PX, GRID_SPACING_MAX_PX + GRID_HIST_BIN_PX,
                      GRID_HIST_BIN_PX)
    counts, edges = np.histogram(pair_dists, bins=bins)
    peak_idx = int(np.argmax(counts))
    grid_spacing = float((edges[peak_idx] + edges[peak_idx + 1]) / 2)
    print(f"grid spacing (histogram peak): {grid_spacing:.1f} px "
          f"({counts[peak_idx]} pairs in {GRID_HIST_BIN_PX:.0f} px bin)")

    # For each candidate, count nearest neighbors at the grid spacing.
    k_check = min(9, len(pts))
    dists, _ = tree.query(pts, k=k_check)
    nn_dists = dists[:, 1:]  # drop self
    near_grid = np.abs(nn_dists - grid_spacing) / grid_spacing < NN_TOL
    has_grid_neighbors = near_grid.sum(axis=1) >= MIN_GRID_NEIGHBORS

    # Recover the lattice phase (offset) from the dominant peak of (pos mod
    # spacing). Real grid points share a phase; T-pile resting ants don't.
    def dominant_phase(values: np.ndarray, spacing: float) -> float:
        bins = np.arange(0, spacing + GRID_HIST_BIN_PX, GRID_HIST_BIN_PX)
        c, e = np.histogram(values % spacing, bins=bins)
        peak = int(np.argmax(c))
        return float((e[peak] + e[peak + 1]) / 2)

    x_offset = dominant_phase(pts[:, 0], grid_spacing)
    y_offset = dominant_phase(pts[:, 1], grid_spacing)
    print(f"lattice phase: x_offset={x_offset:.1f}, y_offset={y_offset:.1f}")

    def dist_to_lattice(values: np.ndarray, spacing: float, offset: float) -> np.ndarray:
        wrapped = (values - offset) % spacing
        return np.minimum(wrapped, spacing - wrapped)

    dx = dist_to_lattice(pts[:, 0], grid_spacing, x_offset)
    dy = dist_to_lattice(pts[:, 1], grid_spacing, y_offset)
    on_lattice = (dx < LATTICE_TOL_PX) & (dy < LATTICE_TOL_PX)

    is_grid = has_grid_neighbors & on_lattice
    grid_ids = stationary.index.values[is_grid]
    print(f"  {has_grid_neighbors.sum()} have >=2 NN at grid spacing")
    print(f"  {on_lattice.sum()} are on the lattice phase")
    print(f"grid-confirmed identities: {len(grid_ids):,} of {len(stationary):,} stationary")

    # Diagnostic: how many candidates were stationary but NOT lattice-aligned?
    # (Likely real ants resting briefly — we keep them.)
    n_kept_stationary = len(stationary) - len(grid_ids)
    if n_kept_stationary > 0:
        print(f"keeping {n_kept_stationary:,} stationary identities that lack lattice neighbors")

    if len(grid_ids) == 0:
        print("no grid-aligned identities found; nothing to remove")
        return

    if not BAK.exists():
        df.to_parquet(BAK)
        print(f"backed up original to {BAK.name}")
    else:
        print(f"backup {BAK.name} already exists; not overwriting")

    grid_set = set(int(x) for x in grid_ids)
    keep_mask = ~df.ant_id.isin(grid_set)
    df_filtered = df[keep_mask].reset_index(drop=True)
    df_filtered.to_parquet(SRC)
    print(f"removed {len(grid_ids):,} identities ({(~keep_mask).sum():,} rows)")
    print(f"output: {len(df_filtered):,} rows, {df_filtered.ant_id.nunique():,} identities")
    print(f"wrote {SRC.name}")


if __name__ == "__main__":
    main()
