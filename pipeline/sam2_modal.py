"""Modal app for SAM 2.1 video segmentation on ant footage.

Usage:
    uv run modal run pipeline/sam2_modal.py::smoke_test
    uv run modal run pipeline/sam2_modal.py::run_full

Pipeline:
1. Locally: classical detector (`seed_detect.py`) finds bounding boxes for
   every ant + the red T-shape on the seed frame(s).
2. Modal: a CUDA container loads SAM 2.1 Hiera-Large, opens the video chunk,
   adds box prompts at the seed frames, propagates masklets through the chunk,
   and returns per-frame centroid + area + theta as parquet bytes.
3. Locally: paste together chunk parquets, separate the load track from the
   ant tracks, run `export_viewer.py`.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import modal

APP_NAME = "ants-sam21"
WORKTREE = Path(__file__).resolve().parent.parent
VIDEO_LOCAL = WORKTREE / "ants_full.mp4"

# CUDA 12.4 + cuDNN — torch 2.5.1 is the SAM 2.1 minimum.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2.0",
        "opencv-python-headless",
        "Pillow",
        "huggingface_hub",
        "hydra-core",
        "iopath",
        "pandas",
        "pyarrow",
    )
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/sam2.git'",
    )
    .env({"HF_HOME": "/cache/hf", "TORCH_HOME": "/cache/torch"})
)

cache_vol = modal.Volume.from_name("sam2-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)

CKPT_NAME = "sam2.1_hiera_large.pt"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@app.function(
    gpu="H100",
    timeout=60 * 30,
    volumes={"/cache": cache_vol},
)
def run_chunk(
    video_bytes: bytes,
    seeds: list[dict],
    frame_offset: int = 0,
    debug: bool = False,
) -> bytes:
    """Run SAM 2.1 on a video chunk; return per-frame stats as parquet bytes."""
    import subprocess
    import tempfile

    import cv2
    import numpy as np
    import pandas as pd
    import torch

    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}  "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    ckpt_path = Path("/cache/sam21") / CKPT_NAME
    if not ckpt_path.exists():
        import urllib.request
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {CKPT_URL} → {ckpt_path}")
        urllib.request.urlretrieve(CKPT_URL, str(ckpt_path))
        cache_vol.commit()
    print(f"checkpoint: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)")

    from sam2.build_sam import build_sam2_video_predictor

    t0 = time.time()
    predictor = build_sam2_video_predictor(MODEL_CFG, str(ckpt_path), device="cuda")
    print(f"SAM 2.1 loaded in {time.time()-t0:.1f}s")

    # SAM 2 historically wanted JPEG folder; newer accepts mp4. Be safe — extract.
    workdir = Path(tempfile.mkdtemp())
    chunk_path = workdir / "chunk.mp4"
    chunk_path.write_bytes(video_bytes)
    frames_dir = workdir / "frames"
    frames_dir.mkdir()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(chunk_path), "-q:v", "2", "-start_number", "0",
         str(frames_dir / "%05d.jpg")],
        check=True,
    )
    n_frames = len(list(frames_dir.glob("*.jpg")))
    print(f"extracted {n_frames} frames")

    img0 = cv2.imread(str(frames_dir / "00000.jpg"))
    vid_h, vid_w = img0.shape[:2]
    print(f"video dims: {vid_w}×{vid_h}")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        t1 = time.time()
        state = predictor.init_state(video_path=str(frames_dir), async_loading_frames=True)
        print(f"init_state in {time.time()-t1:.1f}s")

        next_obj_id = 0
        load_obj_id: int | None = None
        added = 0
        t2 = time.time()
        for seed in seeds:
            f = seed["frame"]
            sx = vid_w / seed["frame_w"]
            sy = vid_h / seed["frame_h"]
            for (bx, by, bw, bh) in seed["ants"]:
                x1, y1, x2, y2 = bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=f, obj_id=next_obj_id, box=box,
                )
                next_obj_id += 1
                added += 1
            if seed.get("load"):
                bx, by, bw, bh = seed["load"]
                box = np.array([bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy], dtype=np.float32)
                if load_obj_id is None:
                    load_obj_id = next_obj_id
                    next_obj_id += 1
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=f, obj_id=load_obj_id, box=box,
                )
        print(f"seeded {added} ants and {1 if load_obj_id is not None else 0} load in {time.time()-t2:.1f}s")

        rows: list[tuple] = []
        t3 = time.time()
        n_prop = 0
        for prop_frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks = (mask_logits > 0).squeeze(1).cpu().numpy()
            for k, oid in enumerate(obj_ids):
                m = masks[k]
                if not m.any():
                    continue
                ys, xs = np.where(m)
                cx = float(xs.mean())
                cy = float(ys.mean())
                area = int(m.sum())
                if xs.size >= 5:
                    mx, my = xs.mean(), ys.mean()
                    dx = xs - mx; dy = ys - my
                    cxx = float((dx * dx).mean())
                    cyy = float((dy * dy).mean())
                    cxy = float((dx * dy).mean())
                    theta = 0.5 * float(np.arctan2(2 * cxy, cxx - cyy))
                else:
                    theta = 0.0
                kind = "load" if oid == load_obj_id else "ant"
                rows.append((frame_offset + prop_frame_idx, int(oid), kind, cx, cy, theta, area))
            n_prop += 1
            if debug and n_prop == 1:
                print(f"first propagate frame: {len(obj_ids)} objects, mask shape {masks.shape}")
        elapsed = time.time() - t3
        fps = n_prop / max(elapsed, 1e-3)
        print(f"propagated {n_prop} frames in {elapsed:.1f}s ({fps:.2f} fps)")

    df = pd.DataFrame(rows, columns=["frame", "obj_id", "kind", "x", "y", "theta", "area"])
    print(f"records: {len(df):,}  unique obj_ids: {df.obj_id.nunique()}")
    if len(df):
        per_f = df[df.kind == "ant"].groupby("frame").size()
        if len(per_f):
            print(f"ants/frame median={int(per_f.median())} min={int(per_f.min())} "
                  f"max={int(per_f.max())}")

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


@app.local_entrypoint()
def smoke_test(start_s: float = 80.0, dur_s: float = 5.0, fps: float = 15.0):
    """30 s slice through SAM 2.1 on 1× H100. Prints budget extrapolation."""
    import subprocess
    import sys
    import tempfile

    import numpy as np

    if not VIDEO_LOCAL.exists():
        raise RuntimeError(f"{VIDEO_LOCAL} missing; symlink ../ants-philip/ants_full.mp4")

    tmp = Path(tempfile.mkdtemp()) / "chunk.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}",
        "-i", str(VIDEO_LOCAL),
        "-vf", f"crop=iw:ih/2:0:0,fps={fps:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-an", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    chunk_bytes = tmp.read_bytes()
    print(f"chunk: {tmp.name}  {len(chunk_bytes)/1e6:.1f} MB  ({dur_s:.1f}s @ {fps:.0f}fps)")

    # Local: classical detector for seed boxes.
    sys.path.insert(0, str(WORKTREE))
    from pipeline.seed_detect import compute_background, seed_at_frame

    bg_path = WORKTREE / "pipeline" / "background.npy"
    if bg_path.exists():
        bg_gray = np.load(bg_path)
    else:
        bg_gray = compute_background(VIDEO_LOCAL)
        np.save(bg_path, bg_gray)
        print(f"wrote {bg_path}")

    src_fps = 60
    seed_frame_global = int(round(start_s * src_fps))
    seed = seed_at_frame(VIDEO_LOCAL, seed_frame_global, bg_gray)
    seed["frame"] = 0  # chunk-local frame 0
    print(f"seed: {len(seed['ants'])} ants  load={'yes' if seed['load'] else 'no'}  "
          f"({seed['frame_w']}×{seed['frame_h']})")
    if len(seed["ants"]) == 0:
        raise RuntimeError("no ants detected in seed frame; aborting")

    seeds = [seed]
    frame_offset = seed_frame_global

    t0 = time.time()
    parquet_bytes = run_chunk.remote(chunk_bytes, seeds, frame_offset, debug=True)
    elapsed = time.time() - t0
    print(f"\n=== remote done in {elapsed:.1f}s wall ({len(parquet_bytes)/1e6:.2f} MB)")

    out = WORKTREE / "pipeline" / "smoke_tracks.parquet"
    out.write_bytes(parquet_bytes)
    print(f"wrote {out}")

    import pandas as pd
    df = pd.read_parquet(out)
    print(f"\nrows: {len(df):,}  unique ids: {df.obj_id.nunique():,}")
    if len(df):
        ants = df[df.kind == "ant"]
        per_f = ants.groupby("frame").size()
        if len(per_f):
            print(f"ants/frame median={int(per_f.median())} "
                  f"min={int(per_f.min())} max={int(per_f.max())}")

    full_video_s = 211.0
    extrap = elapsed * full_video_s / dur_s / 60.0
    print(f"\n1× H100 extrapolation to full {full_video_s:.0f}s @ {fps:.0f}fps: {extrap:.1f} min")
    print(f"4× H100 parallel extrapolation: ~{extrap/4:.1f} min")
    if extrap / 4 > 30:
        print("WARNING: even 4× H100 parallel exceeds 30-min budget")
    else:
        print("budget: OK")
