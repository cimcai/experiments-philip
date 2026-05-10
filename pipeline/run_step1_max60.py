"""Thin wrapper: run pipeline.run at step=1 with MAX_LOST=60 source frames
(matching step=2's 1s grace window in real time)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import run as R

R.MAX_LOST = 60  # was 30 — at step=1 that's only 0.5s of grace, too short

OUT = Path("pipeline/tracks_step1_maxlost60.parquet")


def main() -> None:
    rows = R.run(start=0, end=None, frame_step=1, write_overlay=None)
    df = pd.DataFrame(rows, columns=["frame", "t_ms", "ant_id", "x", "y", "theta", "area"])
    df.to_parquet(OUT)
    print(f"\nwrote {OUT}: {len(df)} rows, {df['ant_id'].nunique()} unique IDs")
    if not df.empty:
        lens = df.groupby("ant_id").size()
        print(f"track length: median={int(lens.median())}, "
              f"p90={int(lens.quantile(0.9))}, max={int(lens.max())}")


if __name__ == "__main__":
    main()
