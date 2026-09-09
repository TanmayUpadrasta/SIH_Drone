#!/usr/bin/env python3
"""
Integrated Stage 1 -> 2 -> 3 -> 4 runner.

Chains Stage 1 (Preprocessing, stage1_preprocess.py), Stage 2 (SfM,
"role 2.py"), Stage 3 (Dense Reconstruction & Mesh, stage3_mesh.py), and
Stage 4 (Georeferencing, "role 4 code.py") for one run, per
PIPELINE_CONTRACT.txt:

    raw/video.mp4 + raw/camera_intrinsics.json
        -> [Stage 1] -> keyframes/*.jpg
        -> [Stage 2] -> sparse/{cameras,images,points3D}.txt
        -> [Stage 3] -> mesh/{model.obj,model.mtl,texture.jpg}
        -> [Stage 4] -> georeferenced/model_geo.{obj,mtl} + texture

Each stage is still runnable standalone -- this script just runs them in
order and enforces each handoff via manifest.json, EXCEPT the one place
the contract itself says not to: Stage 4 failing does not halt anything.

Stage-by-stage failure semantics (these differ on purpose, per each
stage's own contract, not by accident):
    Stage 1/2/3 failure -> STOP. Nothing later can run without their
        output, and Stage 3 in particular hard-fails with no fallback.
    Stage 4 failure -> does NOT stop this script. Per its own docstring,
        "role 4 code.py" is designed to write georeferenced/warning.txt
        and pass the mesh through unchanged as model_geo.* rather than
        halt the pipeline, so Stage 5 always has a consistent filename
        to load, georeferenced or not.

Usage:
    python run_pipeline.py --run_id run_001
    python run_pipeline.py --run_id run_001 --data_root /data \
        --target_fps 2.0 --blur_threshold 100.0 --matcher sequential \
        --nb_neighbors 20 --std_ratio 2.0 --radius_nb_points 16 \
        --radius 0.05 --min_component_fraction 0.05 --min_gps 5

    # Resume from Stage 3 (Stages 1-2 already done for this run_id):
    python run_pipeline.py --run_id run_001 --start_stage 3
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE1_SCRIPT = SCRIPT_DIR / "stage1_preprocess.py"
STAGE2_SCRIPT = SCRIPT_DIR / "role 2.py"
STAGE3_SCRIPT = SCRIPT_DIR / "stage3_mesh.py"
STAGE4_SCRIPT = SCRIPT_DIR / "role 4 code.py"


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def run_stage(label: str, cmd: list) -> int:
    log("=" * 60)
    log(f"Starting {label}")
    log(f"$ {' '.join(str(c) for c in cmd)}")
    log("=" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"{label} exited non-zero (code {result.returncode})")
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
        description="Run Stage 1 through Stage 4 for one run."
    )
    parser.add_argument("--run_id", required=True, help="Run identifier, e.g. run_001")
    parser.add_argument("--data_root", default="/data", help="Root data directory (default: /data)")
    parser.add_argument(
        "--start_stage", type=int, default=1, choices=[1, 2, 3, 4],
        help="Resume from this stage, assuming earlier stages already completed for this run_id "
             "(generalizes run_stage1_2.py's --skip_stage1 to the full 4-stage chain)",
    )
    # Stage 1
    parser.add_argument("--target_fps", type=float, default=2.0, help="Stage 1: keyframe extraction rate")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Stage 1: blur rejection threshold")
    # Stage 2
    parser.add_argument(
        "--matcher", default="sequential", choices=["sequential", "exhaustive"],
        help="Stage 2: COLMAP matching strategy (default: sequential)",
    )
    # Stage 3
    parser.add_argument("--nb_neighbors", type=int, default=20, help="Stage 3: statistical outlier removal neighbors")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Stage 3: statistical outlier removal std ratio")
    parser.add_argument("--radius_nb_points", type=int, default=16, help="Stage 3: radius outlier removal min points")
    parser.add_argument("--radius", type=float, default=0.05, help="Stage 3: radius outlier removal radius")
    parser.add_argument("--min_component_fraction", type=float, default=0.05, help="Stage 3: min mesh component size")
    # Stage 4
    parser.add_argument("--min_gps", type=int, default=5, help="Stage 4: min GPS-tagged keyframe matches required")
    args = parser.parse_args()

    manifest_path = Path(args.data_root) / args.run_id / "manifest.json"

    if args.start_stage <= 1:
        rc = run_stage(
            "Stage 1 (Preprocessing)",
            [
                sys.executable, str(STAGE1_SCRIPT),
                "--run_id", args.run_id,
                "--data_root", args.data_root,
                "--target_fps", str(args.target_fps),
                "--blur_threshold", str(args.blur_threshold),
            ],
        )
        if rc != 0 or not manifest_flag(manifest_path, "stage_1_complete"):
            log("Stage 1 did not complete successfully -- stopping.")
            sys.exit(1)

    if args.start_stage <= 2:
        rc = run_stage(
            "Stage 2 (SfM / Sparse Reconstruction)",
            [
                sys.executable, str(STAGE2_SCRIPT),
                "--run_id", args.run_id,
                "--data_root", args.data_root,
                "--matcher", args.matcher,
            ],
        )
        if rc != 0 or not manifest_flag(manifest_path, "stage_2_complete"):
            log("Stage 2 did not complete successfully -- stopping.")
            sys.exit(1)

    if args.start_stage <= 3:
        rc = run_stage(
            "Stage 3 (Dense Reconstruction & Mesh)",
            [
                sys.executable, str(STAGE3_SCRIPT),
                "--run_id", args.run_id,
                "--data_root", args.data_root,
                "--nb_neighbors", str(args.nb_neighbors),
                "--std_ratio", str(args.std_ratio),
                "--radius_nb_points", str(args.radius_nb_points),
                "--radius", str(args.radius),
                "--min_component_fraction", str(args.min_component_fraction),
            ],
        )
        if rc != 0 or not manifest_flag(manifest_path, "stage_3_complete"):
            log(
                "Stage 3 hard-failed -- this is Stage 3's designed behavior on any failure "
                "(no fallback output). Stopping. Check mesh/failure_log.txt."
            )
            sys.exit(1)

    # Stage 4 is intentionally NOT gated the same way as Stages 1-3: per its own
    # contract/docstring, a Stage 4 failure writes georeferenced/warning.txt and
    # passes the mesh through unchanged as model_geo.* rather than halting the
    # pipeline, so a downstream Stage 5 always has a consistent filename to load.
    # "role 4 code.py" itself takes run_id as a positional arg and hyphenated
    # flags (--data-root, --min-gps), unlike every other stage's --run_id/
    # --data_root -- that's its own existing CLI, not something to "fix" here.
    rc = run_stage(
        "Stage 4 (Georeferencing)",
        [
            sys.executable, str(STAGE4_SCRIPT),
            args.run_id,
            "--data-root", args.data_root,
            "--min-gps", str(args.min_gps),
        ],
    )
    if rc == 0 and manifest_flag(manifest_path, "stage_4_complete"):
        log("Stage 4 georeferenced the mesh successfully.")
    else:
        log(
            "Stage 4 could not georeference the mesh (see georeferenced/warning.txt) -- "
            "per its own contract this does NOT halt the pipeline. georeferenced/model_geo.obj "
            "is the Stage 3 mesh, ungeoreferenced, passed through unchanged for Stage 5."
        )

    log("Pipeline run finished (Stages 1-4).")


if __name__ == "__main__":
    main()
