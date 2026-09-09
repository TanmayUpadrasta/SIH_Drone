#!/usr/bin/env python3
"""
Role 2 — SfM Pipeline Integration (Structure-from-Motion)
===========================================================
Wraps COLMAP's CLI to turn a folder of keyframes + camera intrinsics
into a sparse reconstruction (camera poses + sparse point cloud),
per PIPELINE_CONTRACT.txt.

Contract in:
    /data/{run_id}/keyframes/*.jpg
    /data/{run_id}/raw/camera_intrinsics.json

Contract out:
    /data/{run_id}/sparse/cameras.txt
    /data/{run_id}/sparse/images.txt
    /data/{run_id}/sparse/points3D.txt
    /data/{run_id}/sparse/failed_images.txt   (on partial failure)
    /data/{run_id}/manifest.json  -> stage_2_complete updated

Requires the `colmap` CLI to be installed and on PATH.
    Ubuntu:  sudo apt install colmap
    Or build from source: https://colmap.github.io/install.html

Usage:
    python run_sfm.py --run_id run_001 [--data_root /data] [--matcher sequential]
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[role2-sfm] {msg}", flush=True)


def run(cmd: list, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output, raising on failure."""
    log("$ " + " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}"
        )
    return result


def check_colmap_installed() -> None:
    if shutil.which("colmap") is None:
        sys.exit(
            "ERROR: `colmap` CLI not found on PATH.\n"
            "Install it first (e.g. `sudo apt install colmap`, or build from source:\n"
            "https://colmap.github.io/install.html) and re-run this script."
        )


def load_intrinsics(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    required = [
        "focal_length_mm", "sensor_width_mm", "sensor_height_mm",
        "image_width_px", "image_height_px", "camera_model",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"camera_intrinsics.json missing keys: {missing}")
    return data


def intrinsics_to_colmap_params(intr: dict) -> tuple:
    """
    Convert mm-based intrinsics into COLMAP's PINHOLE pixel params:
    fx, fy, cx, cy.

    fx = focal_length_mm * image_width_px  / sensor_width_mm
    fy = focal_length_mm * image_height_px / sensor_height_mm
    cx, cy = image center
    """
    if intr["camera_model"] != "PINHOLE":
        log(
            f"WARNING: camera_model in intrinsics is '{intr['camera_model']}', "
            f"but this script assumes PINHOLE per the contract's example. "
            f"Proceeding with PINHOLE anyway."
        )

    fx = intr["focal_length_mm"] * intr["image_width_px"] / intr["sensor_width_mm"]
    fy = intr["focal_length_mm"] * intr["image_height_px"] / intr["sensor_height_mm"]
    cx = intr["image_width_px"] / 2.0
    cy = intr["image_height_px"] / 2.0
    return fx, fy, cx, cy


def update_manifest(manifest_path: Path, updates: dict) -> None:
    """Read-modify-write manifest.json, touching only the given keys."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "run_id": manifest_path.parent.name,
            "stage_1_complete": False,
            "stage_2_complete": False,
            "stage_3_complete": False,
            "stage_4_complete": False,
            "stage_5_complete": False,
        }
    manifest.update(updates)
    manifest["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"manifest.json updated: {updates}")


# ----------------------------------------------------------------------
# Core stages
# ----------------------------------------------------------------------

def feature_extraction(database_path: Path, image_dir: Path, camera_params: tuple) -> None:
    fx, fy, cx, cy = camera_params
    run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
        "--FeatureExtraction.use_gpu", "0",  # flip to 1 if a GPU is available
    ])


def feature_matching(database_path: Path, matcher: str) -> None:
    """
    Sequential matching is the right choice here: keyframes are named
    in capture order from a single continuous drone pass (per the
    contract), so consecutive-frame matching (+ a small overlap window)
    is far cheaper than exhaustive matching and less prone to spurious
    matches between visually-similar-but-distant frames. Exhaustive is
    offered as a fallback for footage that loops back on itself.
    """
    if matcher == "sequential":
        run([
            "colmap", "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", "10",
            "--FeatureMatching.use_gpu", "0",
        ])
    elif matcher == "exhaustive":
        run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "0",
        ])
    else:
        raise ValueError(f"Unknown matcher: {matcher}")


def sparse_reconstruction(database_path: Path, image_dir: Path, sparse_root: Path) -> Path:
    """
    Runs COLMAP's incremental mapper. Because footage can be tricky,
    the mapper may split the scene into multiple disconnected models
    (sparse/0, sparse/1, ...). We select the largest by registered
    image count as the "real" reconstruction to hand to Role 3.
    """
    sparse_root.mkdir(parents=True, exist_ok=True)
    run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_root),
    ])

    submodels = sorted([p for p in sparse_root.iterdir() if p.is_dir()])
    if not submodels:
        raise RuntimeError("COLMAP mapper produced no reconstruction models.")

    if len(submodels) > 1:
        log(
            f"WARNING: mapper produced {len(submodels)} disconnected sub-models "
            f"({[p.name for p in submodels]}). Picking the largest by image count. "
            f"This usually means the footage has a registration gap (motion blur, "
            f"low overlap, or a lighting jump) — worth flagging to Role 1 / Role 6."
        )

    def n_registered_images(model_dir: Path) -> int:
        # Cheap proxy before conversion: count non-comment, non-empty
        # lines in the binary images.bin isn't readable as text, so we
        # convert each candidate to TEXT temporarily to compare.
        tmp_txt = model_dir / "_tmp_txt"
        tmp_txt.mkdir(parents=True, exist_ok=True)
        run([
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(tmp_txt),
            "--output_type", "TXT",
        ])
        images_txt = tmp_txt / "images.txt"
        count = 0
        with open(images_txt) as f:
            lines = [l for l in f if not l.startswith("#")]
        # images.txt alternates 2 lines per registered image
        count = len(lines) // 2
        shutil.rmtree(tmp_txt)
        return count

    best_model, best_count = None, -1
    for m in submodels:
        c = n_registered_images(m)
        log(f"  sub-model {m.name}: {c} registered images")
        if c > best_count:
            best_model, best_count = m, c

    return best_model


def export_text_model(model_dir: Path, sparse_out: Path) -> None:
    sparse_out.mkdir(parents=True, exist_ok=True)
    run([
        "colmap", "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(sparse_out),
        "--output_type", "TXT",
    ])


def write_failed_images_log(sparse_out: Path, keyframe_dir: Path) -> tuple:
    """
    Cross-reference all input keyframes against what actually got
    registered in images.txt. Anything missing failed registration
    (bad matches, insufficient overlap, blur that slipped through
    Role 1's filter, etc.) and gets logged for Role 6 to inspect.
    Returns (num_input, num_registered).
    """
    images_txt = sparse_out / "images.txt"
    registered = set()
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        # format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = lines[i].split()
        if len(parts) >= 10:
            registered.add(parts[-1])

    all_keyframes = {p.name for p in keyframe_dir.glob("*.jpg")}
    failed = sorted(all_keyframes - registered)

    failed_log_path = sparse_out / "failed_images.txt"
    if failed:
        with open(failed_log_path, "w") as f:
            f.write(f"# {len(failed)} of {len(all_keyframes)} keyframes failed registration\n")
            for name in failed:
                f.write(name + "\n")
        log(f"WARNING: {len(failed)}/{len(all_keyframes)} keyframes failed registration "
            f"-> {failed_log_path}")
    else:
        log("All keyframes registered successfully — no failed_images.txt written.")

    return len(all_keyframes), len(registered)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Role 2 — SfM sparse reconstruction (COLMAP wrapper)")
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. run_001")
    parser.add_argument("--data_root", default="/data", help="Root data directory (default: /data)")
    parser.add_argument(
        "--matcher", default="sequential", choices=["sequential", "exhaustive"],
        help="Matching strategy (default: sequential — recommended for single continuous drone footage)",
    )
    args = parser.parse_args()

    check_colmap_installed()

    run_dir = Path(args.data_root) / args.run_id
    keyframe_dir = run_dir / "keyframes"
    intrinsics_path = run_dir / "raw" / "camera_intrinsics.json"
    sparse_out = run_dir / "sparse"
    manifest_path = run_dir / "manifest.json"
    work_dir = run_dir / "_sfm_work"  # scratch space for db + raw colmap output

    if not keyframe_dir.exists() or not any(keyframe_dir.glob("*.jpg")):
        sys.exit(f"ERROR: no keyframes found at {keyframe_dir} (expected frame_XXXX.jpg files).")
    if not intrinsics_path.exists():
        sys.exit(f"ERROR: camera_intrinsics.json not found at {intrinsics_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    database_path = work_dir / "database.db"
    raw_sparse_dir = work_dir / "sparse_raw"

    try:
        intr = load_intrinsics(intrinsics_path)
        camera_params = intrinsics_to_colmap_params(intr)
        log(f"Camera params (fx, fy, cx, cy) = {camera_params}")

        log("Stage 2.1 — feature extraction")
        feature_extraction(database_path, keyframe_dir, camera_params)

        log(f"Stage 2.2 — feature matching ({args.matcher})")
        feature_matching(database_path, args.matcher)

        log("Stage 2.3 — sparse reconstruction (incremental mapping)")
        best_model = sparse_reconstruction(database_path, keyframe_dir, raw_sparse_dir)

        log(f"Stage 2.4 — exporting TEXT model to {sparse_out}")
        export_text_model(best_model, sparse_out)

        num_input, num_registered = write_failed_images_log(sparse_out, keyframe_dir)
        registration_rate = num_registered / num_input if num_input else 0.0
        log(f"Registration rate: {num_registered}/{num_input} ({registration_rate:.1%})")

        if registration_rate < 0.8:
            log(
                "WARNING: registration rate is below 80% — Stage 3's assumption "
                "requires >80% of keyframes registered, or it will HARD FAIL with "
                "no fallback output. Flag this to Role 6 now, before handing off."
            )

        update_manifest(manifest_path, {"stage_2_complete": True})
        log("Stage 2 complete. Sparse model + camera poses ready for Role 3.")

    except Exception as e:
        log(f"ERROR: {e}")
        update_manifest(manifest_path, {"stage_2_complete": False})
        # Best-effort failure log even if we died before sparse_out existed
        sparse_out.mkdir(parents=True, exist_ok=True)
        with open(sparse_out / "failed_images.txt", "a") as f:
            f.write(f"# STAGE FAILED: {e}\n")
        sys.exit(1)

    finally:
        # keep raw colmap binary output + db around for debugging;
        # comment this out during a hackathon if disk space is tight
        log(f"(scratch/debug files kept at {work_dir})")


if __name__ == "__main__":
    main()
