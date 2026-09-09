#!/usr/bin/env python3
"""
Stage 3 (Dense Reconstruction & Mesh) — drone video reconstruction pipeline.

Reads /data/{run_id}/sparse/{cameras,images,points3D}.txt (Stage 2's COLMAP
TEXT output) and /data/{run_id}/keyframes/ (Stage 1's output), runs COLMAP
dense reconstruction (image undistortion + PatchMatch stereo + stereo
fusion), cleans and meshes the resulting dense point cloud with Open3D, and
writes /data/{run_id}/mesh/model.obj + model.mtl + texture.jpg per the
pipeline I/O contract.

This stage HARD FAILS by design: if dense reconstruction or meshing fails
for any reason, no partial output is written. /data/{run_id}/mesh/ will
contain failure_log.txt ONLY — check that file first when debugging a
failed run.

Usage:
    python stage3_mesh.py --run_id run_001
    python stage3_mesh.py --run_id run_001 --data_root /data \
        --nb_neighbors 20 --std_ratio 2.0 \
        --radius_nb_points 16 --radius 0.05 \
        --min_component_fraction 0.05
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# Poisson octree depth. Not exposed as a CLI flag (the contract's flag
# list is fixed) -- 9 is a safe middle ground for hackathon-scale scenes.
POISSON_DEPTH = 9

JPEG_QUALITY = 95


class StageFailure(Exception):
    """Carries which pipeline step failed + why, for write_failure_log."""

    def __init__(self, step, message, details=""):
        super().__init__(message)
        self.step = step
        self.message = message
        self.details = details


def log(msg):
    print(f"[stage3-mesh] {msg}", flush=True)


def run_streamed(cmd):
    """
    Run a subprocess, printing each line of stdout/stderr as it arrives
    (so a live COLMAP run is visible in real time) while also capturing
    the full text, so a failure can still be written verbatim to
    failure_log.txt afterward. Plain subprocess.run(capture_output=True)
    would hide output until the process ends; plain inherited stdio
    would lose it for the log. This gets both.
    """
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    captured = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    proc.wait()
    return proc.returncode, "".join(captured)


def check_colmap_installed():
    if shutil.which("colmap") is None:
        raise StageFailure(
            "environment",
            "`colmap` CLI not found on PATH. Install it and re-run "
            "(e.g. `brew install colmap` on macOS, `sudo apt install colmap` on Ubuntu).",
        )


def sanity_check_sparse(sparse_dir):
    """
    Stage 2 guarantees >80% registration, but per the contract we still
    verify the sparse folder isn't empty/corrupt before spending time on
    dense reconstruction against it.
    """
    required = ["cameras.txt", "images.txt", "points3D.txt"]
    for name in required:
        path = sparse_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise StageFailure("sanity_check", f"Missing or empty sparse file: {path}")

    def count_data_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip() and not line.startswith("#"))

    n_cameras = count_data_lines(sparse_dir / "cameras.txt")
    n_image_lines = count_data_lines(sparse_dir / "images.txt")  # 2 lines per registered image
    n_points = count_data_lines(sparse_dir / "points3D.txt")

    if n_cameras < 1:
        raise StageFailure("sanity_check", f"{sparse_dir / 'cameras.txt'} has no camera entries.")
    if n_image_lines < 2:
        raise StageFailure("sanity_check", f"{sparse_dir / 'images.txt'} has no registered images.")
    if n_points < 1:
        raise StageFailure("sanity_check", f"{sparse_dir / 'points3D.txt'} has no 3D points.")

    log(
        f"Sparse sanity check OK: {n_cameras} camera(s), "
        f"{n_image_lines // 2} registered image(s), {n_points} 3D point(s)"
    )


def run_dense_reconstruction(keyframes_dir, sparse_dir, work_dir):
    """
    COLMAP dense reconstruction: image undistortion -> PatchMatch stereo
    -> stereo fusion. Each step streams live and is checked individually
    so a failure is attributed to the specific step, not just "COLMAP
    failed". work_dir is scratch space OUTSIDE mesh/, so a failure here
    never leaves partial dense-reconstruction files inside mesh/.

    Note: COLMAP's PatchMatchStereo requires CUDA with no CPU fallback.
    On a machine without CUDA it aborts immediately (SIGABRT) regardless
    of input quality -- that failure is caught here like any other and
    reported with an explicit note, not left as a raw crash.
    """
    check_colmap_installed()
    work_dir.mkdir(parents=True, exist_ok=True)

    log("Dense reconstruction 1/3 — image undistortion")
    rc, out = run_streamed(
        [
            "colmap", "image_undistorter",
            "--image_path", str(keyframes_dir),
            "--input_path", str(sparse_dir),
            "--output_path", str(work_dir),
            "--output_type", "COLMAP",
        ]
    )
    if rc != 0:
        raise StageFailure("dense_reconstruction", f"image_undistorter failed (exit {rc})", out)

    log("Dense reconstruction 2/3 — PatchMatch stereo")
    rc, out = run_streamed(
        [
            "colmap", "patch_match_stereo",
            "--workspace_path", str(work_dir),
            "--workspace_format", "COLMAP",
        ]
    )
    if rc != 0:
        raise StageFailure(
            "dense_reconstruction",
            f"patch_match_stereo failed (exit {rc}). Note: COLMAP's PatchMatchStereo requires "
            f"CUDA and has no CPU fallback -- if this machine/build lacks CUDA, this step "
            f"cannot succeed regardless of input quality. Check for "
            f"'requires CUDA' in the captured output below.",
            out,
        )

    fused_path = work_dir / "fused.ply"
    log("Dense reconstruction 3/3 — stereo fusion")
    rc, out = run_streamed(
        [
            "colmap", "stereo_fusion",
            "--workspace_path", str(work_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(fused_path),
        ]
    )
    if rc != 0:
        raise StageFailure("dense_reconstruction", f"stereo_fusion failed (exit {rc})", out)

    if not fused_path.exists() or fused_path.stat().st_size < 200:
        raise StageFailure(
            "dense_reconstruction",
            f"stereo_fusion exited successfully but produced no usable dense point cloud "
            f"at {fused_path}",
        )

    with open(fused_path, "rb") as f:
        header = f.read(2000).decode("ascii", errors="ignore")
    match = re.search(r"element vertex (\d+)", header)
    n_dense_points = int(match.group(1)) if match else 0
    if n_dense_points < 1:
        raise StageFailure(
            "dense_reconstruction",
            f"stereo_fusion produced an empty dense point cloud (0 points) at {fused_path}",
        )

    log(f"Dense reconstruction produced {n_dense_points} points -> {fused_path}")
    return fused_path


def load_and_clean_point_cloud(fused_path, nb_neighbors, std_ratio, radius_nb_points, radius):
    """
    Load the dense point cloud, then two complementary outlier-removal
    passes before meshing:
      1. Statistical outlier removal -- drops points far from EVERYTHING
         (bad triangulation, isolated stray points).
      2. Radius outlier removal -- drops points in small, sparse local
         neighborhoods. This catches small coherent clusters (e.g.
         fragments from a moving object) that pass statistical removal,
         since a tight cluster's own internal spacing can still look
         locally "normal" to the first check even though the cluster as
         a whole is disconnected from the real surface.
    """
    pcd = o3d.io.read_point_cloud(str(fused_path))
    n_initial = len(pcd.points)
    if n_initial < 1:
        raise StageFailure("cleanup", f"Loaded dense point cloud has 0 points: {fused_path}")

    log(
        f"Loaded dense point cloud: {n_initial} points, "
        f"has_colors={pcd.has_colors()}, has_normals={pcd.has_normals()}"
    )

    pcd_stat, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    n_after_stat = len(pcd_stat.points)
    n_removed_stat = n_initial - n_after_stat
    log(
        f"Statistical outlier removal (nb_neighbors={nb_neighbors}, std_ratio={std_ratio}): "
        f"removed {n_removed_stat} points ({n_after_stat} remaining)"
    )

    pcd_radius, _ = pcd_stat.remove_radius_outlier(nb_points=radius_nb_points, radius=radius)
    n_after_radius = len(pcd_radius.points)
    n_removed_radius = n_after_stat - n_after_radius
    log(
        f"Radius outlier removal (nb_points={radius_nb_points}, radius={radius}): "
        f"removed {n_removed_radius} points ({n_after_radius} remaining)"
    )

    if n_after_radius < 4:
        raise StageFailure(
            "cleanup",
            f"Only {n_after_radius} points survived cleanup (nb_neighbors={nb_neighbors}, "
            f"std_ratio={std_ratio}, radius_nb_points={radius_nb_points}, radius={radius}) -- "
            f"not enough to mesh. Cleanup parameters are likely too aggressive for this "
            f"point cloud's density/scale.",
        )

    stats = {
        "n_initial": n_initial,
        "n_after_statistical": n_after_stat,
        "n_removed_statistical": n_removed_stat,
        "n_after_radius": n_after_radius,
        "n_removed_radius": n_removed_radius,
    }
    return pcd_radius, stats


def generate_mesh(pcd, depth=POISSON_DEPTH):
    """Poisson surface reconstruction. Normals are required; estimate + orient if missing."""
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    if len(mesh.triangles) < 1:
        raise StageFailure(
            "meshing", "Poisson surface reconstruction produced an empty mesh (0 triangles)."
        )

    log(
        f"Poisson reconstruction (depth={depth}): "
        f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
    )
    return mesh


def filter_small_components(mesh, min_component_fraction):
    """
    Standard mesh hygiene, then connected-component filtering: cluster
    triangles into connected groups and drop any group smaller than
    min_component_fraction of the LARGEST component's triangle count --
    a second, complementary defense (after point-cloud cleanup) against
    small leftover artifacts that still made it into the mesh.
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        raise StageFailure("meshing", "Mesh has no connected components after Poisson reconstruction.")

    largest = int(cluster_n_triangles.max())
    min_size = max(int(largest * min_component_fraction), 1)
    valid_clusters = np.where(cluster_n_triangles >= min_size)[0]
    n_removed_components = len(cluster_n_triangles) - len(valid_clusters)

    triangles_to_remove = ~np.isin(triangle_clusters, valid_clusters)
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    log(
        f"Component filter: kept {len(valid_clusters)}/{len(cluster_n_triangles)} components "
        f"(removed {n_removed_components} below {min_component_fraction * 100:.1f}% of the "
        f"largest component's {largest} triangles)"
    )

    if len(mesh.triangles) < 1:
        raise StageFailure(
            "meshing",
            f"Component filtering removed all mesh triangles -- min_component_fraction="
            f"{min_component_fraction} is likely too aggressive for this mesh.",
        )

    return mesh, n_removed_components


def apply_texture_and_export(mesh, mesh_dir):
    """
    Writes model.obj + model.mtl + texture.jpg.

    Texturing approach: COLMAP's stereo_fusion already assigns each dense
    point a color sampled from the source keyframe images during
    photo-consistent fusion, and Open3D's Poisson reconstruction
    interpolates those point colors onto the output mesh's vertices
    (verified: mesh.has_vertex_colors() is True after
    create_from_point_cloud_poisson on a colored input cloud). So each
    mesh vertex's color already traces back to keyframe pixels.

    Rather than depend on COLMAP's mesh_texturer (which expects a mesh
    produced by COLMAP's own delaunay/poisson mesher with matching
    internal visibility data -- not a good fit for a mesh we've since
    cleaned/filtered externally in Open3D), this bakes those per-vertex
    colors into a small square texture atlas image, with each vertex
    getting its own unique texel and a matching UV coordinate. This
    guarantees the exact required filenames/format and avoids a fragile
    COLMAP<->Open3D mesh handoff, at the cost of being a per-vertex
    texture rather than a true multi-view UV-projected texture atlas
    (see README's "known limitation" note for the upgrade path if a
    sharper texture is needed later, e.g. OpenMVS TextureMesh).
    """
    if not mesh.has_vertex_colors():
        log("WARNING: mesh has no vertex colors (input point cloud had none) -- using flat gray texture.")
        vertex_colors = np.full((len(mesh.vertices), 3), 0.6)
    else:
        vertex_colors = np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_vertices = len(vertices)

    if n_vertices < 3 or len(triangles) < 1:
        raise StageFailure("meshing", f"Final mesh is too small to export ({n_vertices} vertices, {len(triangles)} triangles).")

    atlas_size = max(2, int(np.ceil(np.sqrt(n_vertices))))
    atlas_rgb = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)

    rows = np.arange(n_vertices) // atlas_size
    cols = np.arange(n_vertices) % atlas_size
    atlas_rgb[rows, cols] = (vertex_colors * 255).astype(np.uint8)

    # OBJ uv origin is bottom-left (v increases upward); the atlas array's
    # row 0 is the image's top row, so v must be flipped from row index.
    u = (cols + 0.5) / atlas_size
    v = 1.0 - (rows + 0.5) / atlas_size

    obj_path = mesh_dir / "model.obj"
    mtl_path = mesh_dir / "model.mtl"
    texture_path = mesh_dir / "texture.jpg"

    atlas_bgr = cv2.cvtColor(atlas_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(texture_path), atlas_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl material0\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 1\n")
        f.write("map_Kd texture.jpg\n")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# Stage 3 mesh -- per-vertex color atlas texture\n")
        f.write("mtllib model.mtl\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for uu, vv in zip(u, v):
            f.write(f"vt {uu:.6f} {vv:.6f}\n")
        f.write("usemtl material0\n")
        for a, b, c in triangles:
            f.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")

    log(
        f"Exported {obj_path.name}, {mtl_path.name}, {texture_path.name} "
        f"({n_vertices} vertices, {len(triangles)} triangles, "
        f"{atlas_size}x{atlas_size} texture atlas)"
    )
    return obj_path, mtl_path, texture_path


def clear_dir(path):
    """Empty a directory's contents (keeping the directory itself), creating it if needed."""
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_failure_log(mesh_dir, step, message, details=""):
    """
    Per contract: on failure, /data/{run_id}/mesh/ must contain
    failure_log.txt and NOTHING else -- no partial mesh, no point cloud.
    Clears the directory first so this holds regardless of what any
    earlier step already wrote there.
    """
    clear_dir(mesh_dir)
    log_path = mesh_dir / "failure_log.txt"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Stage 3 (Dense Reconstruction & Mesh) FAILED at {timestamp}\n")
        f.write(f"Failed step: {step}\n")
        f.write(f"Reason: {message}\n")
        if details:
            f.write("\n--- captured output ---\n")
            f.write(details)
            if not details.endswith("\n"):
                f.write("\n")

    log(f"Failure log written to {log_path}")
    return log_path


def update_manifest(manifest_path, run_id, success):
    """
    Read-modify-write: only stage_3_complete + last_updated ever change
    here. If manifest.json doesn't exist yet (Stage 3 run standalone
    without Stage 1/2 having run first), create it with every other
    stage defaulted to false rather than assuming it's already there.
    """
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "run_id": run_id,
            "stage_1_complete": False,
            "stage_2_complete": False,
            "stage_3_complete": False,
            "stage_4_complete": False,
            "stage_5_complete": False,
        }

    manifest["stage_3_complete"] = bool(success)
    manifest["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"manifest.json updated: stage_3_complete={success}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 (Dense Reconstruction & Mesh): sparse model -> textured mesh."
    )
    parser.add_argument("--run_id", required=True, help="Run identifier, e.g. run_001")
    parser.add_argument("--data_root", default="/data", help="Root data directory (default: /data)")
    parser.add_argument("--nb_neighbors", type=int, default=20,
                         help="Statistical outlier removal: neighbors examined per point (default: 20)")
    parser.add_argument("--std_ratio", type=float, default=2.0,
                         help="Statistical outlier removal: std-dev multiplier threshold (default: 2.0)")
    parser.add_argument("--radius_nb_points", type=int, default=16,
                         help="Radius outlier removal: min neighbors required within --radius (default: 16)")
    parser.add_argument("--radius", type=float, default=0.05,
                         help="Radius outlier removal: neighborhood radius, scene units (default: 0.05)")
    parser.add_argument("--min_component_fraction", type=float, default=0.05,
                         help="Drop mesh components smaller than this fraction of the largest "
                              "component's triangle count (default: 0.05)")
    args = parser.parse_args()

    run_dir = Path(args.data_root) / args.run_id
    sparse_dir = run_dir / "sparse"
    keyframes_dir = run_dir / "keyframes"
    mesh_dir = run_dir / "mesh"
    manifest_path = run_dir / "manifest.json"
    work_dir = run_dir / "_dense_work"

    try:
        if not sparse_dir.exists():
            raise StageFailure("sanity_check", f"Sparse folder not found: {sparse_dir}")
        sanity_check_sparse(sparse_dir)

        fused_path = run_dense_reconstruction(keyframes_dir, sparse_dir, work_dir)

        pcd, cleanup_stats = load_and_clean_point_cloud(
            fused_path, args.nb_neighbors, args.std_ratio, args.radius_nb_points, args.radius
        )

        mesh = generate_mesh(pcd)
        mesh, n_removed_components = filter_small_components(mesh, args.min_component_fraction)

        clear_dir(mesh_dir)
        obj_path, mtl_path, texture_path = apply_texture_and_export(mesh, mesh_dir)

        update_manifest(manifest_path, args.run_id, success=True)

        print()
        print("=" * 60)
        print("STAGE 3 SUMMARY")
        print("=" * 60)
        print(f"Dense points (raw)      : {cleanup_stats['n_initial']}")
        print(f"Removed (statistical)   : {cleanup_stats['n_removed_statistical']}")
        print(f"Removed (radius)        : {cleanup_stats['n_removed_radius']}")
        print(f"Dense points (cleaned)  : {cleanup_stats['n_after_radius']}")
        print(f"Mesh components removed : {n_removed_components}")
        print(f"Final mesh vertices     : {len(mesh.vertices)}")
        print(f"Final mesh triangles    : {len(mesh.triangles)}")
        print(f"Output                  : {mesh_dir}")

    except StageFailure as e:
        log(f"STAGE 3 FAILED at step '{e.step}': {e.message}")
        write_failure_log(mesh_dir, e.step, e.message, e.details)
        update_manifest(manifest_path, args.run_id, success=False)
        sys.exit(1)

    except Exception as e:  # noqa: BLE001 -- must still hard-fail cleanly, not crash raw
        log(f"STAGE 3 FAILED with an unexpected error: {e}")
        write_failure_log(mesh_dir, "unexpected", str(e), traceback.format_exc())
        update_manifest(manifest_path, args.run_id, success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
