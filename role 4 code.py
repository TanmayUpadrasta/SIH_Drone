"""
Aligns the Stage 3 mesh to real-world GPS coordinates using a similarity
transform (scale + rotation + translation) fit from COLMAP camera centers
<-> GPS-tagged camera positions (Umeyama similarity transform).

Input-
    {data_root}/{run_id}/mesh/model.obj (+ .mtl + texture)
    {data_root}/{run_id}/raw/flight_metadata.json
    {data_root}/{run_id}/sparse/images.txt   (role 2 extra input)
    {data_root}/{run_id}/manifest.json

Output:
    {data_root}/{run_id}/georeferenced/model_geo.obj (+ .mtl + texture)
    {data_root}/{run_id}/georeferenced/transform.json   (on success)
    {data_root}/{run_id}/georeferenced/warning.txt       (on failure)
    updates manifest.json -> stage_4_complete

Failure handling: writing warning.txt,
        rather than crashing or halting the pipeline.
"""

import argparse
import json
import math
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

TEXTURE_KEYS = {"map_kd", "map_ka", "map_bump", "bump", "map_d"}

class GeoreferencingError(Exception):
    """Raised for known, expected failure conditions (as opposed to bugs)."""

# -- geodesy --
def geodetic_to_ecef(lat, lon, alt):
    lat, lon = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    x = (n + alt) * math.cos(lat) * math.cos(lon)
    y = (n + alt) * math.cos(lat) * math.sin(lon)
    z = (n * (1 - WGS84_E2) + alt) * math.sin(lat)
    return np.array([x, y, z])

def ecef_to_enu(ecef, ref_lat, ref_lon, ref_ecef):
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    d = ecef - ref_ecef
    t = np.array(
        [
            [-math.sin(lon), math.cos(lon), 0],
            [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        ]
    )
    return t @ d

def geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """WGS84 ellipsoid-accurate lat/lon/alt -> local East-North-Up meters."""
    ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    return ecef_to_enu(geodetic_to_ecef(lat, lon, alt), ref_lat, ref_lon, ref_ecef)


# ---------------------------------------------------------------- parsing --
def parse_images_txt(path):
    """COLMAP images.txt -> {image_name: camera_center (3,) in recon coords}. """
    centers = {}
    lines = [l for l in path.read_text().splitlines() if not l.lstrip().startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        pose_line = lines[i].strip()
        if not pose_line:
            continue
        parts = pose_line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = (float(v) for v in parts[1:5])
        tx, ty, tz = (float(v) for v in parts[5:8])
        name = parts[9]
        r = np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]
        )
        t = np.array([tx, ty, tz])
        centers[name] = -r.T @ t  # camera center = -R^T * t
    return centers

def parse_flight_metadata(path):
    data = json.loads(path.read_text())
    gps = {}
    for name, v in data.items():
        try:
            gps[name] = (float(v["lat"]), float(v["lon"]), float(v.get("alt", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed entries rather than failing the whole run
    return gps


# -- similarity fit --
def umeyama(src, dst):
    """Least-squares similarity transform src -> dst: dst ~= scale * R @ src + t.
    Returns R, scale, t, rmse (fit residual in dst units - meters here).   """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise GeoreferencingError("src/dst correspondence arrays must both be Nx3")

    n, m = src.shape
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    var_src = (src_c**2).sum() / n
    if var_src <= 1e-12:
        raise GeoreferencingError("degenerate camera positions (all centers coincide)")

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    r = u @ s @ vt
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    t = mu_dst - scale * r @ mu_src

    predicted = (scale * (r @ src.T)).T + t
    rmse = float(np.sqrt(np.mean(np.sum((predicted - dst) ** 2, axis=1))))
    return r, scale, t, rmse


# --- obj i/o --
def read_obj_vertices(obj_path):
    """Parse only the 'v ' lines; everything else in the file is passed
    through untouched later. Preserves an optional 4th (w) component."""
    vertices, w_values = [], []
    for line in obj_path.read_text().splitlines():
        if line.strip().startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            w_values.append(parts[4] if len(parts) >= 5 else None)
    if not vertices:
        raise GeoreferencingError(f"no 'v' vertex lines found in {obj_path}")
    return np.asarray(vertices, dtype=float), w_values


def write_transformed_obj(in_obj, out_obj, transformed, w_values):
    out_lines, vi = [], 0
    for line in in_obj.read_text().splitlines():
        if line.strip().startswith("v "):
            x, y, z = transformed[vi]
            w = w_values[vi]
            vi += 1
            if w is not None:
                out_lines.append(f"v {x:.9f} {y:.9f} {z:.9f} {w}")
            else:
                out_lines.append(f"v {x:.9f} {y:.9f} {z:.9f}")
        else:
            out_lines.append(line)
    out_obj.write_text("\n".join(out_lines) + "\n")


def find_mtllib(obj_path):
    for line in obj_path.read_text().splitlines():
        if line.strip().lower().startswith("mtllib "):
            return line.strip().split(maxsplit=1)[1]
    return None


def copy_mtl_and_textures(mesh_dir, out_dir, mtl_name):
    """Copy the .mtl  and every texture map it references (map_Kd/Ka/bump/d, case-insensitive)."""
    if not mtl_name:
        return
    src_mtl = mesh_dir / mtl_name
    if not src_mtl.exists():
        return
    dst_mtl = out_dir / "model_geo.mtl"
    shutil.copy2(src_mtl, dst_mtl)
    for line in src_mtl.read_text().splitlines():
        tokens = line.strip().split(maxsplit=1)
        if len(tokens) == 2 and tokens[0].lower() in TEXTURE_KEYS:
            tex_name = Path(tokens[1].strip())
            src_tex = src_mtl.parent / tex_name
            if src_tex.exists():
                shutil.copy2(src_tex, out_dir / tex_name.name)


def rewrite_mtllib_reference(obj_path):
    lines = obj_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("mtllib "):
            lines[i] = "mtllib model_geo.mtl"
            break
    obj_path.write_text("\n".join(lines) + "\n")


# -- manifest --
def load_manifest(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(path, manifest, stage4_ok):
    manifest["stage_4_complete"] = stage4_ok
    manifest["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(manifest, indent=2))


# -- fail path --
def safe_failure(mesh_dir, out_dir, reason):
    """Contract failure behavior: copy the mesh through unchanged as
    model_geo.* (so Role 5's viewer always has a consistent filename to
    load) and record why, without halting or crashing the pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_src = mesh_dir / "model.obj"
    if obj_src.exists():
        shutil.copy2(obj_src, out_dir / "model_geo.obj")
        mtl_name = find_mtllib(obj_src)
        copy_mtl_and_textures(mesh_dir, out_dir, mtl_name)
        if mtl_name:
            rewrite_mtllib_reference(out_dir / "model_geo.obj")

    (out_dir / "warning.txt").write_text(f"Role 4 georeferencing warning:\n{reason}\n")
    (out_dir / "transform.json").unlink(missing_ok=True)  # clear any stale transform from a prior successful run
    print(f"Georeferencing failed: {reason}")


# -- main --
def main():
    ap = argparse.ArgumentParser(description="Stage 4 - Georeferencing")
    ap.add_argument("run_id")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--min-gps", type=int, default=5, help="min GPS-tagged keyframe matches required")
    args = ap.parse_args()

    run_dir = Path(args.data_root) / args.run_id
    mesh_dir = run_dir / "mesh"
    geo_dir = run_dir / "georeferenced"
    manifest_path = run_dir / "manifest.json"

    manifest = load_manifest(manifest_path)
    if not manifest.get("stage_3_complete", False):
        print("Stage 3 not complete - stage 4 does not run.")
        return 1

    images_txt = run_dir / "sparse" / "images.txt"
    flight_json = run_dir / "raw" / "flight_metadata.json"

    try:
        if not (mesh_dir / "model.obj").exists():
            raise GeoreferencingError("missing mesh/model.obj")
        if not images_txt.exists():
            raise GeoreferencingError("missing sparse/images.txt")
        if not flight_json.exists():
            raise GeoreferencingError("missing raw/flight_metadata.json")

        centers = parse_images_txt(images_txt)
        gps = parse_flight_metadata(flight_json)
        matched = [name for name in centers if name in gps]

        if len(matched) < args.min_gps:
            raise GeoreferencingError(
                f"only {len(matched)} keyframes have matching GPS tags, need >= {args.min_gps}"
            )

        # reference point = first matched frame's GPS, so ENU coords stay small/well-conditioned
        ref_lat, ref_lon, ref_alt = gps[matched[0]]
        src = np.array([centers[n] for n in matched])
        dst = np.array([geodetic_to_enu(*gps[n], ref_lat, ref_lon, ref_alt) for n in matched])

        r, scale, t, rmse = umeyama(src, dst)

        vertices, w_values = read_obj_vertices(mesh_dir / "model.obj")
        transformed = (scale * (r @ vertices.T)).T + t

        geo_dir.mkdir(parents=True, exist_ok=True)
        write_transformed_obj(mesh_dir / "model.obj", geo_dir / "model_geo.obj", transformed, w_values)

        mtl_name = find_mtllib(mesh_dir / "model.obj")
        copy_mtl_and_textures(mesh_dir, geo_dir, mtl_name)
        if mtl_name:
            rewrite_mtllib_reference(geo_dir / "model_geo.obj")

        transform = {
            "method": "Umeyama 3D similarity transform",
            "coordinate_system": "local ENU meters (WGS84 ellipsoid); origin = first matched GPS tag",
            "scale": scale,
            "rotation": r.tolist(),
            "translation": t.tolist(),
            "reference_point": {"lat": ref_lat, "lon": ref_lon, "alt": ref_alt},
            "num_correspondences": len(matched),
            "matched_frames": matched,
            "fit_rmse_meters": rmse,
            "source": "sparse/images.txt (COLMAP camera centers) + raw/flight_metadata.json",
        }
        (geo_dir / "transform.json").write_text(json.dumps(transform, indent=2))
        (geo_dir / "warning.txt").unlink(missing_ok=True)  # clear any stale warning from a prior failed run

        save_manifest(manifest_path, manifest, stage4_ok=True)
        print(f"Georeferenced with {len(matched)} GPS correspondences, scale={scale:.4f}, rmse={rmse:.3f} m")
        return 0

    except GeoreferencingError as exc:
        safe_failure(mesh_dir, geo_dir, str(exc))
        save_manifest(manifest_path, manifest, stage4_ok=False)
        return 1
    except Exception as exc:  # noqa: BLE001 - contract requires we never hard-crash Stage 4
        safe_failure(mesh_dir, geo_dir, f"unexpected error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        save_manifest(manifest_path, manifest, stage4_ok=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
