"""Visualize a single cluster event. Shows the source frame with the
involved ant_ids highlighted as colored trails (last K frames for merge,
next K frames for split). Convergence/divergence pattern should be obvious.

Usage: uv run python pipeline/visualize_event.py [event_id]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EVENTS = Path(__file__).resolve().parent / "cluster_events.parquet"
TRACKS = Path(
    "/Users/theosechopoulos/Development/cimc/ants-philip/pipeline/tracks_step1_maxlost60_clean.parquet"
)
VIDEO = Path("/Users/theosechopoulos/Development/cimc/ants-philip-sam3/ants_full.mp4")
OUT_DIR = Path(__file__).resolve().parent / "event_overlays"
TRAIL_FRAMES = 60   # 1 second @ 60fps source
ZOOM_PAD = 120      # px padding around event center


def color_for_id(oid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(oid) * 9176 + 17)
    h = float(rng.uniform(0, 180))
    hsv = np.uint8([[[int(h), 240, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_event(event_id: int) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    events = pd.read_parquet(EVENTS)
    ev = events[events.event_id == event_id].iloc[0]
    member_ids = list(ev.member_ids)
    print(f"event {event_id}: {ev.kind} at frame {ev.frame}  "
          f"({ev.n_members}-way) at ({ev.x:.0f}, {ev.y:.0f})")

    # Pull only the relevant member tracks for the surrounding window
    tracks = pd.read_parquet(TRACKS)
    if ev.kind == "merge":
        f0, f1 = int(ev.frame) - TRAIL_FRAMES, int(ev.frame)
    else:
        f0, f1 = int(ev.frame), int(ev.frame) + TRAIL_FRAMES
    sub = tracks[(tracks.ant_id.isin(member_ids)) &
                 (tracks.frame >= f0) & (tracks.frame <= f1)].copy()
    print(f"  trail rows: {len(sub):,}  for {sub.ant_id.nunique()}/{len(member_ids)} members")

    # Read frame
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ev.frame))
    ok, raw = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {ev.frame}")
    img = raw[: raw.shape[0] // 2, :].copy()  # top crop

    # Draw trails
    for oid, grp in sub.sort_values("frame").groupby("ant_id"):
        col = color_for_id(int(oid))
        pts = grp[["x", "y"]].to_numpy(dtype=np.int32)
        for i in range(1, len(pts)):
            cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), col, 1, cv2.LINE_AA)
        # Endpoint dot (final position for merge, initial for split)
        end_pt = tuple(pts[-1] if ev.kind == "merge" else pts[0])
        cv2.circle(img, end_pt, 5, col, 1, cv2.LINE_AA)
        cv2.putText(img, str(int(oid)), (end_pt[0] + 4, end_pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    # Mark event center
    cx, cy = int(ev.x), int(ev.y)
    cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.circle(img, (cx, cy), 50, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{ev.kind} @f{int(ev.frame)} N={int(ev.n_members)}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    out = OUT_DIR / f"event_{event_id:04d}_{ev.kind}_f{int(ev.frame)}_n{int(ev.n_members)}.png"
    cv2.imwrite(str(out), img)
    print(f"  wrote {out.name}")

    # Also write a zoomed crop
    x0, y0 = max(0, cx - ZOOM_PAD), max(0, cy - ZOOM_PAD)
    x1, y1 = min(img.shape[1], cx + ZOOM_PAD), min(img.shape[0], cy + ZOOM_PAD)
    crop = img[y0:y1, x0:x1]
    crop_big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                          interpolation=cv2.INTER_NEAREST)
    out_zoom = OUT_DIR / f"event_{event_id:04d}_{ev.kind}_f{int(ev.frame)}_n{int(ev.n_members)}_zoom.png"
    cv2.imwrite(str(out_zoom), crop_big)
    print(f"  wrote {out_zoom.name}")
    return out


def main() -> None:
    if len(sys.argv) > 1:
        ids = [int(a) for a in sys.argv[1:]]
    else:
        # Default: render the biggest merge and biggest split + a small one of each
        events = pd.read_parquet(EVENTS)
        ids = [
            int(events[events.kind == "merge"].nlargest(1, "n_members").event_id.iloc[0]),
            int(events[events.kind == "split"].nlargest(1, "n_members").event_id.iloc[0]),
            int(events[events.kind == "merge"]
                 .query("n_members == 2").event_id.iloc[10]),
            int(events[events.kind == "split"]
                 .query("n_members == 2").event_id.iloc[10]),
        ]
    for i in ids:
        render_event(i)


if __name__ == "__main__":
    main()
