"""
Role 3 — Dense Reconstruction & Mesh Generation
SIH 2026 — Video-to-3D-Model pipeline

Input : COLMAP MVS dense point cloud (fused.ply) from Role 2/your own
        colmap patch_match_stereo + stereo_fusion run.
Output: cleaned, colored, watertight-ish triangle mesh (.obj + .ply)
        ready for Role 4 (georeferencing/scaling) and Role 5 (web viewer).

DEPLOYMENT CONSTRAINT: this whole pipeline must run on a fully local
machine, no internet/cloud dependency. This is already true here --
Open3D, NumPy, SciPy, and scikit-learn are all local compute libraries
with no network calls. Nothing in this file phones out anywhere. Worth
confirming the same is true for Role 2 (COLMAP is a local binary, no
cloud API) and Role 5 (the web viewer should load the .obj/.ply from
local disk/localhost, not fetch it from a remote server) before the demo
-- a jury asking "does this need internet to work" should get a clean
"no" for the whole pipeline, not just this stage.

Usage:
    python mesh_pipeline.py --input fused.ply --output model.obj
"""
import argparse
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# STAGE 0: Parameter estimation from the point cloud itself
# ---------------------------------------------------------------------------
def estimate_parameters(pcd):
    """
    Real footage varies wildly in scale (a car vs. a building vs. a whole
    site) and density (short clip vs. long orbit). Hardcoded parameters
    tuned on one dataset silently break on another. Instead, derive
    sensible defaults from the cloud's own bounding box and point density.
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent())  # overall scene scale
    n_points = len(pcd.points)

    # Actual measured spacing between points, NOT a volume-based estimate.
    # Real point clouds lie on thin surfaces (walls, ground, roofs) inside
    # their bounding box, not filling it -- (bbox_volume / n_points)^(1/3)
    # badly overestimates spacing for anything surface-like, which is
    # every real photogrammetry scene. Instead, directly measure each
    # point's distance to its nearest neighbor (same technique used for
    # Ball Pivoting radius selection) and take the median, which is
    # robust to the outliers/clutter that are still in the cloud at this
    # point (we haven't cleaned it yet -- this runs before Stage 1).
    sample_n = min(n_points, 5000)
    sample_idx = np.random.default_rng(0).choice(n_points, sample_n, replace=False)
    sample_pts = np.asarray(pcd.points)[sample_idx]
    tree = cKDTree(np.asarray(pcd.points))
    nn_dist, _ = tree.query(sample_pts, k=2)  # k=1 is the point itself (dist 0)
    avg_spacing = float(np.median(nn_dist[:, 1]))

    voxel_size = max(avg_spacing * 1.5, diag * 0.0005)
    normal_radius = voxel_size * 4
    # Poisson depth: bigger/denser scenes can support more octree depth
    # without just fitting noise. Clamp to a sane demo-safe range.
    depth = int(np.clip(8 + np.log2(max(n_points, 1) / 20000), 8, 11))

    return {
        "voxel_size": round(voxel_size, 5),
        "normal_radius": round(normal_radius, 5),
        "poisson_depth": depth,
        "scene_diag": round(diag, 3),
        "avg_spacing": round(avg_spacing, 5),
    }


# ---------------------------------------------------------------------------
# STAGE 1: Cleanup the raw dense point cloud
# ---------------------------------------------------------------------------
def clean_point_cloud(pcd, voxel_size=None, nb_neighbors=20, std_ratio=2.0):
    """
    Remove floating noise / outlier points that COLMAP's MVS stage
    inevitably produces (bad matches, sky, reflective surfaces, etc).
    """
    if voxel_size:
        # Optional: downsample if the cloud is huge (millions of points).
        # Skip this if you want max detail and your machine can handle it.
        pcd = pcd.voxel_down_sample(voxel_size)

    # Statistical outlier removal: for each point, look at its nb_neighbors
    # nearest neighbors; if its average distance to them is > std_ratio
    # standard deviations from the mean, it's considered noise and dropped.
    #
    # IMPORTANT LIMITATION: this only catches points that are far from
    # EVERYTHING. A tight cluster of points (e.g. a small bush/vegetation
    # blob) has close neighbors *within itself*, so each point's local
    # neighbor distance looks normal -- statistical outlier removal alone
    # will NOT remove small, internally-dense clutter clusters. That's
    # what remove_sparse_clusters() below is for.
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return pcd_clean


def remove_sparse_clusters(pcd, eps, min_points=10, min_cluster_fraction=0.01):
    """
    Removes small, spatially-isolated clusters (vegetation blobs,
    reflective-surface fragments, small debris) BEFORE meshing, using
    pure geometric (3D position) DBSCAN clustering.

    Why this has to happen here and not after Poisson: Poisson
    reconstruction fits a *smooth global surface*, so if a small cluster
    sits close enough to the main structure, Poisson can bridge the gap
    and fuse it into the same connected mesh component -- at which point
    "keep the largest connected component" can no longer separate them.
    Filtering at the point-cloud level, by actual proximity, catches most
    of this before that fusion can happen.

    `min_cluster_fraction`: any cluster with fewer than this fraction of
    total points is treated as clutter and dropped. Lower = keep more
    small objects. Higher = more aggressive removal.

    LIMITATION: if an object genuinely, physically touches the main
    structure (e.g. a bush whose base touches the ground), pure
    position-based clustering CANNOT separate them -- they really are one
    connected blob in 3D space. See remove_by_color() for a targeted
    second pass that catches this specific case using color instead of
    position.
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    n_points = len(labels)
    if labels.max() < 0:
        return pcd  # nothing formed a cluster -- skip rather than empty the cloud

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    min_size = max(int(n_points * min_cluster_fraction), min_points)
    valid_clusters = unique[counts >= min_size]
    keep_mask = np.isin(labels, valid_clusters)

    n_removed_clusters = len(unique) - len(valid_clusters)
    n_removed_points = n_points - keep_mask.sum()
    print(f"    DBSCAN (position): found {len(unique)} clusters, removed "
          f"{n_removed_clusters} small ones ({n_removed_points} points, incl. noise)")

    return pcd.select_by_index(np.where(keep_mask)[0])


def remove_by_color(pcd, target_rgb, tolerance=0.12):
    """
    Removes points whose color is close to a known "unwanted" color
    signature (e.g. foliage green). This is a SEPARATE, second pass from
    geometric clustering, deliberately kept independent rather than
    merged into one distance metric -- mixing position and color into a
    single DBSCAN eps proved unstable (had to push color weight high
    enough to catch touching vegetation, which also started fragmenting
    and deleting real structure with any natural color variation, like
    the ground). Keeping the two signals as separate, independently
    tunable passes is more controllable and easier to reason about.

    This specifically solves the case remove_sparse_clusters() cannot:
    an object that is genuinely touching the main structure in 3D space,
    where only color (not position) can distinguish it.

    HONEST LIMITATION: this needs a known target color and only works
    when that color is genuinely distinct from the real structure's
    palette -- true for a synthetic scene with a fixed foliage color, not
    guaranteed for arbitrary real footage. A production version of this
    would use a trained semantic segmentation model (e.g. Open3D-ML)
    instead of a hardcoded color target -- this is a fast, explainable
    stand-in appropriate for a hackathon-scale MVP and demo scene.
    """
    colors = np.asarray(pcd.colors)
    dist = np.linalg.norm(colors - np.array(target_rgb), axis=1)
    keep_mask = dist > tolerance
    print(f"    Color filter (target={target_rgb}): removed "
          f"{(~keep_mask).sum()} points matching that color signature")
    return pcd.select_by_index(np.where(keep_mask)[0])


# ---------------------------------------------------------------------------
# STAGE 2: Surface reconstruction (point cloud -> mesh)
# ---------------------------------------------------------------------------
def reconstruct_mesh(pcd, depth=9, density_quantile=0.05, normal_radius=0.1):
    """
    Poisson surface reconstruction: fits a watertight implicit surface
    to the oriented point cloud. `depth` controls the octree resolution
    (mesh detail) -- 8-10 is the usual sweet spot for drone/aerial scenes.
    Higher depth = more detail but slower and more prone to noise bumps.

    Poisson always produces a *closed* surface, even in areas with no
    data (it "hallucinates" geometry to close holes) — those regions
    have low `density` values, which is why we trim them below.
    """
    # Normals are required for Poisson. COLMAP's fused.ply usually
    # already has normals; if not, estimate + orient them. Radius is
    # scale-aware (passed in from estimate_parameters), not a fixed
    # number that only makes sense at one scene scale.
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Trim low-density vertices -> these are the "hallucinated" / ghost
    # parts of the surface where Poisson had little real data to work with.
    densities = np.asarray(densities)
    low_density_mask = densities < np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(low_density_mask)

    return mesh


# ---------------------------------------------------------------------------
# STAGE 3: Mesh cleanup
# ---------------------------------------------------------------------------
def clean_mesh(mesh, keep_largest_component=True, min_component_fraction=0.02):
    """
    Standard mesh hygiene + removal of small floating fragments
    (disconnected islands of geometry — very common artifact of MVS
    reconstructions of scenes with clutter/vegetation/reflections).

    IMPORTANT: keeps ALL components with at least `min_component_fraction`
    of the total triangle count, NOT just the single largest one. This
    matters a lot for Ball Pivoting output specifically: unlike Poisson
    (which fits one continuous watertight surface), Ball Pivoting only
    builds triangles where balls of the given radii can physically reach,
    so a real structure can legitimately end up as multiple disconnected
    pieces (e.g. a sparsely-covered wall failing to bridge to the roof).
    Keeping only the single largest piece in that case silently deletes
    real geometry, not noise -- verified this by inspecting actual
    component sizes on Ball Pivoting output before settling on this
    fraction-based approach instead of a strict "keep #1 only" rule.
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if keep_largest_component:
        # Connected-component analysis: group triangles into clusters
        # that are geometrically connected, then keep every cluster
        # large enough to plausibly be real structure, dropping only the
        # small floating fragments (true noise/clutter remnants).
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        min_size = max(int(cluster_n_triangles.sum() * min_component_fraction), 1)
        valid_clusters = np.where(cluster_n_triangles >= min_size)[0]
        n_dropped = len(cluster_n_triangles) - len(valid_clusters)
        if n_dropped:
            print(f"    Component filter: kept {len(valid_clusters)}/"
                  f"{len(cluster_n_triangles)} components "
                  f"(dropped {n_dropped} below {min_component_fraction*100:.1f}% size)")

        triangles_to_remove = ~np.isin(triangle_clusters, valid_clusters)
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

    return mesh


# ---------------------------------------------------------------------------
# STAGE 4: "Texture" via vertex color transfer (fast MVP approach)
# ---------------------------------------------------------------------------
def transfer_colors(mesh, pcd, k=4):
    """
    Poisson reconstruction does NOT carry color from the point cloud to
    the new mesh vertices automatically. This transfers color from the
    nearest point-cloud points to each mesh vertex.

    Real dense clouds/meshes have 10^5-10^6+ vertices, so a plain Python
    per-vertex loop over Open3D's KDTreeFlann (fine for a demo of a few
    thousand points) becomes a real bottleneck. scipy's cKDTree supports
    batched, vectorized nearest-neighbor queries for the whole vertex
    array at once, which is what makes this practical at realistic scale.

    Using k>1 neighbors (averaged, inverse-distance weighted) instead of
    just the single nearest point also reduces blotchy/speckled color
    artifacts where the point cloud is sparse relative to mesh resolution.
    This is still vertex coloring, not true UV texture mapping.
    """
    pcd_colors = np.asarray(pcd.colors)
    pcd_points = np.asarray(pcd.points)
    vertices = np.asarray(mesh.vertices)

    tree = cKDTree(pcd_points)
    dists, idx = tree.query(vertices, k=k)
    dists = np.maximum(dists, 1e-8)  # avoid divide-by-zero for exact matches

    if k == 1:
        mesh_colors = pcd_colors[idx]
    else:
        weights = 1.0 / dists
        weights /= weights.sum(axis=1, keepdims=True)
        mesh_colors = np.einsum("vk,vkc->vc", weights, pcd_colors[idx])

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(mesh_colors, 0, 1))
    return mesh


# ---------------------------------------------------------------------------
# Alternative reconstruction method (fallback / comparison)
# ---------------------------------------------------------------------------
def reconstruct_mesh_ball_pivoting(pcd, avg_spacing=None, radii_multipliers=(1, 1.5, 2, 3, 4, 6)):
    """
    Ball Pivoting Algorithm (BPA): imagine rolling a ball of a given
    radius across the point cloud's surface -- wherever it can rest
    touching exactly three points without falling through, it forms a
    triangle there. Unlike Poisson, it does NOT invent/hallucinate
    geometry to close gaps -- it only builds triangles where the data
    actually supports them, so coverage gaps stay as real holes instead
    of being smoothly (and misleadingly) patched over. That's a genuine
    trade-off, not a strict downgrade: for a project where geometric
    honesty matters (you don't want the demo silently fabricating
    structure a sonar/measurement system might later trust), leaving a
    visible gap is arguably the more defensible behavior than Poisson's
    smooth guess.

    Multiple radii are used because a single ball size can't bridge both
    dense areas (small radius needed to avoid skipping detail) and
    sparser areas (needs a bigger radius to reach across gaps) in the
    same real, unevenly-sampled cloud. Open3D tries each radius in
    increasing order, filling in what smaller balls couldn't reach.
    Default multipliers (1, 1.5, 2, 3, 4, 6) were widened from an
    earlier narrower set after testing showed the narrower set left
    real structure (e.g. a sparsely-covered wall) disconnected from the
    rest of the mesh as a separate component -- the wider top radius
    (6x spacing) successfully bridges that gap, producing one single
    connected mesh instead of two, with no real speed cost.

    `avg_spacing` should come from estimate_parameters() so the radii
    are scale-aware -- reusing the already-computed value instead of
    recomputing nearest-neighbor distances again here.
    """
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=(avg_spacing or 0.05) * 4, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)

    if avg_spacing is None:
        # Fallback if called standalone without estimate_parameters()
        distances = pcd.compute_nearest_neighbor_distance()
        avg_spacing = float(np.mean(distances))

    radii = o3d.utility.DoubleVector([avg_spacing * m for m in radii_multipliers])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    return mesh


# ---------------------------------------------------------------------------
# STAGE 2.5: Hole filling (interpolation) -- OPTIONAL, off by default
# ---------------------------------------------------------------------------
def fill_mesh_holes(mesh, hole_size):
    """
    Triangulates across small boundary gaps left by Ball Pivoting (or any
    non-watertight mesh), interpolating a patch surface over each hole up
    to `hole_size` in radius. Uses Open3D's tensor-based
    TriangleMesh.fill_holes -- the legacy (non-tensor) TriangleMesh class
    has no equivalent method in this Open3D version (0.19), so this
    round-trips legacy <-> tensor for this one step.

    WHY THIS IS OPT-IN, NOT AUTOMATIC AFTER BALL PIVOTING:
    reconstruct_mesh_ball_pivoting() deliberately leaves real coverage
    gaps as open holes instead of Poisson's smooth hallucination -- that
    was an explicit, argued design choice (see that function's docstring:
    "you don't want the demo silently fabricating structure a
    sonar/measurement system might later trust"). Calling this
    unconditionally after Ball Pivoting would quietly undo that choice.
    Use it when small, genuinely-real gaps (a few missed triangles from
    uneven sampling) are hurting the mesh's visual/structural quality
    more than the honesty trade-off is worth -- e.g. Role 5's viewer
    showing distracting pinholes on an otherwise-solid wall. Do NOT crank
    `hole_size` up to patch large real openings (windows, doorways, the
    scene's own open boundary) -- that reintroduces exactly the
    fabrication problem Ball Pivoting was chosen to avoid.

    `hole_size` is in the same units as the point cloud/mesh coordinates
    (scene scale), not a triangle count -- pass something derived from
    avg_spacing (see estimate_parameters()), the same way Ball Pivoting's
    own radii are scale-aware, rather than a fixed number tuned on one
    dataset.

    New vertices created by filling get no color (fill_holes only knows
    geometry) -- call transfer_colors() AFTER this, not before, so the
    patched-in vertices get colored from the point cloud too instead of
    staying black/default.
    """
    n_verts_before = len(mesh.vertices)
    n_tris_before = len(mesh.triangles)

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tensor_mesh.fill_holes(hole_size=hole_size)
    filled_legacy = filled.to_legacy()

    n_added_verts = len(filled_legacy.vertices) - n_verts_before
    n_added_tris = len(filled_legacy.triangles) - n_tris_before
    print(f"    Hole filling (radius<={hole_size}): added {n_added_verts} vertices, "
          f"{n_added_tris} triangles patching boundary gaps")

    return filled_legacy


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(input_path, output_path, depth=None, voxel_size=None,
                  method="ball_pivoting", auto_params=True, min_cluster_fraction=0.01,
                  fill_holes=False, max_hole_size=None):
    t0 = time.time()
    print(f"Loading point cloud: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"  {len(pcd.points)} points, has_colors={pcd.has_colors()}, "
          f"has_normals={pcd.has_normals()}")

    params = estimate_parameters(pcd)
    if auto_params:
        voxel_size = voxel_size or params["voxel_size"]
        depth = depth or params["poisson_depth"]
        normal_radius = params["normal_radius"]
        print(f"  Auto-estimated params from scene scale "
              f"(diag={params['scene_diag']}): voxel_size={voxel_size}, "
              f"poisson_depth={depth}, normal_radius={normal_radius}")
    else:
        depth = depth or 9
        normal_radius = 0.1

    t1 = time.time()
    print("Cleaning point cloud (outlier removal)...")
    pcd = clean_point_cloud(pcd, voxel_size=voxel_size)
    print(f"  {len(pcd.points)} points remaining ({time.time()-t1:.1f}s)")

    t1b = time.time()
    print("Removing small isolated clusters (vegetation/clutter)...")
    declutter_eps = params["avg_spacing"] * 8  # scale-aware: tuned so real
    # connected surfaces (ground/walls/roof) stay one component despite
    # uneven sampling density, while small isolated clusters (vegetation,
    # clutter) stay separate rather than chain-connecting into the main
    # structure. See mesh_pipeline eps sweep notes if retuning for very
    # different footage -- this value is scene-density dependent.
    pcd = remove_sparse_clusters(pcd, eps=declutter_eps,
                                  min_cluster_fraction=min_cluster_fraction)
    print(f"  {len(pcd.points)} points remaining ({time.time()-t1b:.1f}s)")

    t2 = time.time()
    print(f"Reconstructing surface ({method})...")
    if method == "poisson":
        print(f"  (Poisson depth={depth})")
        mesh = reconstruct_mesh(pcd, depth=depth, normal_radius=normal_radius)
    else:
        print(f"  (Ball Pivoting, avg_spacing={params['avg_spacing']})")
        mesh = reconstruct_mesh_ball_pivoting(pcd, avg_spacing=params["avg_spacing"])
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles "
          f"({time.time()-t2:.1f}s)")

    t3 = time.time()
    print("Cleaning mesh (largest component, degenerate removal)...")
    mesh = clean_mesh(mesh)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles "
          f"({time.time()-t3:.1f}s)")

    if fill_holes and method == "ball_pivoting":
        t3b = time.time()
        hole_size = max_hole_size if max_hole_size is not None else params["avg_spacing"] * 20
        print(f"Filling small boundary holes (interpolation, hole_size={hole_size})...")
        mesh = fill_mesh_holes(mesh, hole_size=hole_size)
        print(f"  {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles "
              f"({time.time()-t3b:.1f}s)")
    elif fill_holes and method != "ball_pivoting":
        print("  (--fill_holes ignored: Poisson output is already closed/watertight, "
              "nothing to interpolate)")

    if pcd.has_colors():
        t4 = time.time()
        print("Transferring color from point cloud to mesh...")
        mesh = transfer_colors(mesh, pcd)
        print(f"  done ({time.time()-t4:.1f}s)")

    print(f"Writing output: {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh)
    # Also write a .ply alongside for compatibility with viewers/tools
    ply_path = output_path.rsplit(".", 1)[0] + ".ply"
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"Done. Total time: {time.time()-t0:.1f}s")
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense point cloud -> textured mesh")
    parser.add_argument("--input", required=True, help="Path to fused.ply (COLMAP MVS output)")
    parser.add_argument("--output", default="model.obj", help="Output mesh path (.obj)")
    parser.add_argument("--depth", type=int, default=9, help="Poisson octree depth (8-10 typical)")
    parser.add_argument("--voxel_size", type=float, default=None, help="Downsample voxel size (optional)")
    parser.add_argument("--method", choices=["poisson", "ball_pivoting"], default="ball_pivoting")
    parser.add_argument("--min_cluster_fraction", type=float, default=0.01,
                         help="Min fraction of total points a cluster needs to survive "
                              "declutter (0.01 = clusters under 1%% of the cloud are removed "
                              "as clutter). Lower = keep more small objects. Higher = more "
                              "aggressive removal of small clutter like vegetation.")
    parser.add_argument("--fill_holes", action="store_true",
                         help="After Ball Pivoting, interpolate a patch surface across small "
                              "boundary gaps (Open3D fill_holes). Off by default -- Ball "
                              "Pivoting leaves real gaps open on purpose (see "
                              "reconstruct_mesh_ball_pivoting docstring); only enable this if "
                              "leftover pinholes are hurting the mesh more than that honesty "
                              "trade-off is worth. No effect with --method poisson (already "
                              "watertight).")
    parser.add_argument("--max_hole_size", type=float, default=None,
                         help="Max hole radius to fill, in scene units, when --fill_holes is "
                              "set. Default: auto (avg_spacing * 20). Do not set this large "
                              "enough to patch genuinely open areas (windows, doorways) -- "
                              "that reintroduces the fabrication problem Ball Pivoting avoids.")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, depth=args.depth,
                 voxel_size=args.voxel_size, method=args.method,
                 min_cluster_fraction=args.min_cluster_fraction,
                 fill_holes=args.fill_holes, max_hole_size=args.max_hole_size)


# ---------------------------------------------------------------------------
# NOTE: Upgrade path to REAL UV texture mapping (stretch goal)
# ---------------------------------------------------------------------------
# Vertex coloring (above) is fast and fine for an MVP demo, but on a
# coarse mesh it can look blotchy compared to true photographic texture.
# If you have time after the MVP works end-to-end, the standard upgrade
# is OpenMVS, which plugs directly into COLMAP's output:
#
#   1. colmap image_undistorter --> OpenMVS InterfaceCOLMAP
#   2. OpenMVS DensifyPointCloud      (dense point cloud, like COLMAP MVS)
#   3. OpenMVS ReconstructMesh        (Delaunay-based meshing, often
#                                       cleaner than Poisson for outdoor/
#                                       aerial scenes with sharp edges)
#   4. OpenMVS RefineMesh             (optional, refines geometry)
#   5. OpenMVS TextureMesh            (projects the original video frames
#                                       onto the mesh as a proper UV-mapped
#                                       texture atlas -> .obj + .png)
#
# This gives Role 5's web viewer a mesh with an actual image texture
# instead of per-vertex color, which reads much better on a screen from
# a distance (i.e., in front of judges).
