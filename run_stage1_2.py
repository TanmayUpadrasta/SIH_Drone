#!/usr/bin/env python3
"""
Integrated Stage 1 -> Stage 2 runner.

Chains Stage 1 (Preprocessing, stage1_preprocess.py) into Stage 2
(SfM / Sparse Reconstruction, "role 2.py") for a single run, per
PIPELINE_CONTRACT.txt:

    raw/video.mp4 + raw/camera_intrinsics.json
        -> [Stage 1] -> keyframes/*.jpg
        -> [Stage 2] -> sparse/{cameras,images,points3D}.txt

Each stage is still runnable standalone (stage1_preprocess.py, "role 2.py")
-- this script just runs them back-to-back as subprocesses and enforces
the handoff: Stage 2 only starts once manifest.json actually confirms
stage_1_complete, so a Stage 1 failure never silently proceeds into a
COLMAP run against incomplete or missing keyframes.

Usage:
    python run_stage1_2.py --run_id run_001
    python run_stage1_2.py --run_id run_001 --data_root /data \
        --target_fps 2.0 --blur_threshold 100.0 --matcher sequential
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE1_SCRIPT = SCRIPT_DIR / "stage1_preprocess.py"
STAGE2_SCRIPT = SCRIPT_DIR / "role 2.py"


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def run_stage(label: str, cmd: list) -> int:
    log(f"{'=' * 60}")
    log(f"Starting {label}")
    log(f"$ {' '.join(str(c) for c in cmd)}")
    log(f"{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"{label} FAILED (exit code {result.returncode})")
    else:
        log(f"{label} finished successfully")
    return result.returncode


def manifest_flag(manifest_path: Path, key: str) -> bool:
    if not manifest_path.exists():
        return False
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return bool(manifest.get(key, False))


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 (Preprocessing) then Stage 2 (SfM) for one run."
    )
    parser.add_argument("--run_id", required=True, help="Run identifier, e.g. run_001")
    parser.add_argument("--data_root", default="/data", help="Root data directory (default: /data)")
    parser.add_argument("--target_fps", type=float, default=2.0, help="Stage 1: keyframe extraction rate")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Stage 1: blur rejection threshold")
    parser.add_argument(
        "--matcher", default="sequential", choices=["sequential", "exhaustive"],
        help="Stage 2: COLMAP matching strategy (default: sequential)",
    )
    parser.add_argument(
        "--skip_stage1", action="store_true",
        help="Skip Stage 1 and run Stage 2 only (assumes keyframes/ already exists)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.data_root) / args.run_id / "manifest.json"

    if not args.skip_stage1:
        stage1_cmd = [
            sys.executable, str(STAGE1_SCRIPT),
            "--run_id", args.run_id,
            "--data_root", args.data_root,
            "--target_fps", str(args.target_fps),
            "--blur_threshold", str(args.blur_threshold),
        ]
        rc = run_stage("Stage 1 (Preprocessing)", stage1_cmd)
        if rc != 0:
            sys.exit(rc)

        if not manifest_flag(manifest_path, "stage_1_complete"):
            log(
                "ERROR: Stage 1 exited successfully but manifest.json does not report "
                "stage_1_complete=true. Refusing to start Stage 2 against a possibly "
                "incomplete keyframe set."
            )
            sys.exit(1)
    else:
        log("Skipping Stage 1 (--skip_stage1); assuming keyframes/ is already populated.")

    stage2_cmd = [
        sys.executable, str(STAGE2_SCRIPT),
        "--run_id", args.run_id,
        "--data_root", args.data_root,
        "--matcher", args.matcher,
    ]
    rc = run_stage("Stage 2 (SfM / Sparse Reconstruction)", stage2_cmd)
    if rc != 0:
        sys.exit(rc)

    log("Stage 1 + Stage 2 complete. Sparse model ready for Role 3.")


if __name__ == "__main__":
    main()
