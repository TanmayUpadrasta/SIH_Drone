# Stage 1 — Preprocessing

Extracts sharp, sequentially-numbered keyframes from a single continuous
drone take, per the pipeline I/O contract in [PIPELINE_CONTRACT.txt](PIPELINE_CONTRACT.txt).

## Running Stage 1 + Stage 2 together

[run_stage1_2.py](run_stage1_2.py) chains this script into `role 2.py`
(Stage 2 — SfM / Sparse Reconstruction) for one run, enforcing the
handoff: Stage 2 only starts once `manifest.json` actually confirms
`stage_1_complete`, so a Stage 1 failure never silently proceeds into
a COLMAP run against incomplete keyframes.

```bash
python run_stage1_2.py --run_id run_001
python run_stage1_2.py --run_id run_001 --data_root /data \
    --target_fps 2.0 --blur_threshold 100.0 --matcher sequential
```

Pass `--skip_stage1` to run Stage 2 only, against an already-populated
`keyframes/` directory. Each stage also still runs standalone exactly
as documented in its own section below / in `role 2.py`'s docstring.

Requires the `colmap` CLI on `PATH` for Stage 2 (`brew install colmap`
on macOS, `sudo apt install colmap` on Ubuntu). Note: `role 2.py` was
updated to use COLMAP's current flag names (`--FeatureExtraction.use_gpu`,
`--FeatureMatching.use_gpu`) and to pre-create `model_converter`'s output
directory — older COLMAP releases used the `--Sift*.use_gpu` names and
auto-created that directory; this repo has been tested against COLMAP
4.1.1. On COLMAP 4.x, `model_converter --output_type TXT` also emits
`rigs.txt` and `frames.txt` alongside the three files the contract
names — harmless additions, but flag it to Role 3 if their loader
expects only `cameras.txt` / `images.txt` / `points3D.txt`.

## Prerequisites

- Python 3.11 (or compatible 3.x)
- `opencv-python`, `numpy`, `tqdm` installed in your active virtual environment

```bash
pip install opencv-python numpy tqdm
```

No other setup is required — the script only reads from
`/data/{run_id}/raw/` and writes to `/data/{run_id}/keyframes/` and
`/data/{run_id}/manifest.json`.

## Usage

```bash
python stage1_preprocess.py --run_id run_001
```

Optional flags:

```bash
python stage1_preprocess.py \
    --run_id run_001 \
    --target_fps 2.0 \
    --blur_threshold 100.0 \
    --data_root /data
```

| Flag               | Default | Meaning                                                                 |
|--------------------|---------|--------------------------------------------------------------------------|
| `--run_id`         | —       | Required. Folder name under `--data_root`, e.g. `run_001`.               |
| `--target_fps`     | `2.0`   | Target keyframe rate for 1080p-class source video.                       |
| `--blur_threshold` | `100.0` | Minimum resolution-normalized sharpness score to accept a frame.         |
| `--data_root`      | `/data` | Root directory containing `{run_id}/` folders.                           |

Expects `/data/{run_id}/raw/video.mp4` and
`/data/{run_id}/raw/camera_intrinsics.json` to already exist.

## Resolution-aware behavior: 1080p vs 4K

**Extraction interval** (`compute_extraction_interval`): the script
detects whether the source is 4K-class (width ≥ 3840px) or
1080p-class. 4K frames carry more usable detail per frame for feature
matching, so 4K footage is extracted at 75% of `--target_fps` (a fixed
multiplier, not an adaptive scheme) while 1080p-class footage is
extracted at the full `--target_fps`. In both cases this is converted
to a source-frame interval using the video's actual fps.

**Blur scoring** (`compute_blur_score`): raw Laplacian variance scales
with pixel count, so a 4K frame and a 1080p frame of equally sharp
content produce different raw scores. Before scoring, each candidate
frame is resized (in memory, on a copy — the saved frame is untouched)
to a fixed reference width of 960px, so `--blur_threshold` means the
same thing regardless of source resolution.

**Output resolution**: keyframes are always saved at native source
resolution. Per the contract, output is never downscaled below 1080p
— 1080p source stays 1080p, 4K source stays native 4K (this script
does not offer a 4K→1080p downscale option, since native 4K is always
contract-compliant and preserves more detail for Stage 2).

## Output

- `/data/{run_id}/keyframes/frame_0001.jpg, frame_0002.jpg, ...` — accepted keyframes, sequential, 4-digit zero-padded, in capture order.
- `/data/{run_id}/keyframes/rejected_log.txt` — one line per rejected frame: original frame number, its blur score, and the rejection reason.
- `/data/{run_id}/manifest.json` — `stage_1_complete: true` and `last_updated` refreshed. Other stages' fields are read and preserved untouched if the file already exists, or initialized to `false` if the file is new.

## Tuning the blur threshold (for Role 6 / Stage 2 debugging)

If Stage 2 (SfM) reports poor feature registration, the keyframes it
received may be too blurry. `rejected_log.txt` logs the **actual
computed score** for every rejected frame (not just pass/fail), so:

1. Open `rejected_log.txt` and look at how close rejected scores are to `--blur_threshold`.
2. If many borderline-sharp frames were rejected, lower `--blur_threshold` and re-run.
3. If Stage 2 struggles despite frames passing Stage 1, raise `--blur_threshold` to be stricter.

The threshold default (`100.0`) is tuned against the normalized score
computed in `compute_blur_score` (960px-wide reference), not raw
Laplacian variance — don't compare it directly to blur scores computed
by other tools at full resolution.

# Stage 3 — Dense Reconstruction & Mesh

[stage3_mesh.py](stage3_mesh.py) turns Stage 2's sparse COLMAP model into
a textured mesh, per the pipeline I/O contract in [PIPELINE_CONTRACT.txt](PIPELINE_CONTRACT.txt).

**This stage hard-fails by design.** If dense reconstruction or meshing
fails for any reason, `/data/{run_id}/mesh/` will contain **only**
`failure_log.txt` — no partial mesh, no point cloud, nothing else. There
is no fallback output. **If a run failed, check `failure_log.txt` first**
— it names the exact step that failed (`sanity_check`, `dense_reconstruction`,
`cleanup`, `meshing`, or `unexpected`) and includes the raw COLMAP/Open3D
error output, not just "failed".

## Prerequisites

- Python 3.11 (or compatible 3.x)
- `opencv-python`, `numpy`, `open3d`, `tqdm` in your active virtual environment
- `colmap` CLI on `PATH` (verify with `colmap -h`)
- **COLMAP must be built with CUDA support.** `colmap patch_match_stereo`
  is GPU-only — it has no CPU fallback and will abort immediately with
  `Dense stereo reconstruction requires CUDA, which is not available on
  your system` on a CUDA-less build/machine (verified during development
  on this repo's own dev machine: it aborts with exit code -6/134
  regardless of input quality). Check `colmap --version` — if it prints
  `without CUDA`, this stage cannot complete dense reconstruction on that
  machine no matter how good the keyframes/sparse model are. That's a
  hard environment requirement, not a bug to work around here.

## Usage

```bash
python stage3_mesh.py --run_id run_001
```

```bash
python stage3_mesh.py --run_id run_001 --data_root /data \
    --nb_neighbors 20 --std_ratio 2.0 \
    --radius_nb_points 16 --radius 0.05 \
    --min_component_fraction 0.05
```

| Flag                      | Default | Meaning                                                                 |
|---------------------------|---------|--------------------------------------------------------------------------|
| `--run_id`                | —       | Required. Folder name under `--data_root`, e.g. `run_001`.               |
| `--data_root`             | `/data` | Root directory containing `{run_id}/` folders.                          |
| `--nb_neighbors`          | `20`    | Statistical outlier removal: neighbors examined per point.              |
| `--std_ratio`             | `2.0`   | Statistical outlier removal: std-dev multiplier threshold.              |
| `--radius_nb_points`      | `16`    | Radius outlier removal: min neighbors required within `--radius`.       |
| `--radius`                | `0.05`  | Radius outlier removal: neighborhood radius, in scene units.            |
| `--min_component_fraction`| `0.05`  | Drop mesh components smaller than this fraction of the largest one.     |

Expects `/data/{run_id}/sparse/{cameras,images,points3D}.txt` (Stage 2)
and `/data/{run_id}/keyframes/*.jpg` (Stage 1) to already exist.

## Pipeline steps

1. **Sanity-check** the sparse folder isn't empty/corrupt (per-file
   existence + non-comment line counts) before spending time on dense
   reconstruction against it.
2. **`run_dense_reconstruction`** — `colmap image_undistorter` →
   `patch_match_stereo` → `stereo_fusion`, each streamed live to the
   console and checked individually so a failure is attributed to the
   specific COLMAP step, not a generic "COLMAP failed". Scratch files
   live in `/data/{run_id}/_dense_work/`, outside `mesh/`, so a failure
   here never leaves partial dense-reconstruction files where the
   contract says only `failure_log.txt` may exist.
3. **`load_and_clean_point_cloud`** — two outlier-removal passes, in order:
   - **Statistical** (`--nb_neighbors`, `--std_ratio`): drops points far
     from *everything* — isolated stray points from bad triangulation.
   - **Radius** (`--radius_nb_points`, `--radius`): drops points in
     sparse local neighborhoods. This catches small, internally-coherent
     clusters (e.g. a fragment from a moving object in the scene) that
     statistical removal alone misses — a tight cluster's own points
     look locally "normal" to that first check even though the cluster
     as a whole sits apart from the real surface.
4. **`generate_mesh`** — Open3D Poisson surface reconstruction (fixed
   octree depth of 9, not exposed as a flag — the contract's CLI is fixed).
   On a ~40k-point cloud, normal estimation/orientation + Poisson took
   ~45s in testing; expect proportionally longer on denser real footage.
5. **`filter_small_components`** — standard mesh hygiene (degenerate/
   duplicate triangle and vertex removal, non-manifold edge removal),
   then connected-component filtering: drops any component smaller than
   `--min_component_fraction` of the **largest** component's triangle
   count. This is the second, complementary defense (after point-cloud
   cleanup) against small leftover artifacts that still made it into the
   mesh.
6. **`apply_texture_and_export`** — writes `model.obj` + `model.mtl` +
   `texture.jpg`.

## How texturing actually works here (read before debugging texture issues)

`colmap stereo_fusion` samples each dense point's color from the source
keyframe images during photo-consistent fusion, and Open3D's Poisson
reconstruction interpolates those point colors onto the output mesh's
vertices automatically (verified during development: `mesh.has_vertex_colors()`
is `True` after `create_from_point_cloud_poisson` on a colored input
cloud). So every mesh vertex's color already traces back to keyframe
pixels — just not through a per-triangle UV projection.

This stage bakes those per-vertex colors into a small square texture
**atlas**: every vertex gets its own unique texel and a matching UV
coordinate, and `model.obj`/`model.mtl`/`texture.jpg` are written by hand
(not via Open3D's mesh writer) to guarantee the exact filenames the
contract requires. Verified during development: the exported `.obj`
reloads cleanly with `has_triangle_uvs() == True` and a valid texture.

**Known limitation:** this is a per-vertex atlas texture, not a true
multi-view UV-projected texture. On a coarse mesh it can look blotchy up
close compared to real photographic texture mapping. If Stage 4/5 or the
final viewer need sharper texture, the standard upgrade path is COLMAP's
own `mesh_texturer` or OpenMVS's `TextureMesh` — both project the actual
keyframe images onto a proper UV atlas. That wasn't used here because
both expect a mesh produced by their own meshing step with matching
internal visibility data; wiring either to a mesh we've since cleaned/
filtered externally in Open3D is a separate, nontrivial integration, not
a drop-in swap.

## Tuning guide (if Stage 4/5 shows visible artifacts)

- **Floating specks / noisy mesh surface** → lower `--std_ratio` (more
  aggressive statistical removal) or raise `--nb_neighbors`.
- **Small disconnected blobs survive into the final mesh** (e.g. a
  moving object's ghost, vegetation) → raise `--radius_nb_points` or
  lower `--radius` (stricter radius removal), and/or raise
  `--min_component_fraction` (drops bigger leftover mesh islands).
- **Real thin structure (a railing, a sparse edge) is disappearing** →
  cleanup or component filtering is too aggressive for that structure's
  density — raise `--radius`, lower `--radius_nb_points`, or lower
  `--min_component_fraction`.
- **Cleanup step removes almost everything** (`cleanup` failure in
  `failure_log.txt`, "not enough points to mesh") → your `--radius` is
  probably too small relative to the dense cloud's actual point spacing;
  scale it up.
- **Blotchy/low-res-looking texture** → expected given the per-vertex
  atlas approach above; only a true UV-projected re-texturing (see
  Known limitation) fixes this — not a flag to tune here.

## Debugging a failed run

1. Read `/data/{run_id}/mesh/failure_log.txt` — it names the exact
   failed step and includes raw captured COLMAP/Open3D output.
2. If the step is `dense_reconstruction` and the captured output mentions
   `requires CUDA`, this is an environment limitation (see Prerequisites),
   not a data/parameter problem — no flag here can fix it.
3. `/data/{run_id}/_dense_work/` (COLMAP's undistorted images, depth
   maps, and `fused.ply`) is left on disk after both success and failure
   for inspection — it is not cleaned up automatically.
4. `manifest.json`'s `stage_3_complete` will be `false` and
   `stage_1_complete`/`stage_2_complete` are never touched by this stage.

# Stage 4 — Georeferencing

`role 4 code.py` aligns the Stage 3 mesh to real-world GPS coordinates
(Umeyama similarity transform fit from COLMAP camera centers ↔
GPS-tagged camera positions in `raw/flight_metadata.json`). Its own
docstring documents its I/O contract in full.

**Its failure semantics are deliberately different from Stages 1–3: a
Stage 4 failure never halts the pipeline.** If there aren't enough
GPS-tagged keyframe matches (`--min-gps`, default 5) or a required input
is missing, it writes `georeferenced/warning.txt` with the reason and
copies the Stage 3 mesh through **unchanged** as `model_geo.obj`/`.mtl`
+ texture — so Stage 5 always has a consistent filename to load,
georeferenced or not. `manifest.json`'s `stage_4_complete` is set to
`false` either way, but the run itself is not treated as failed.

Note its CLI differs from every other stage's: `run_id` is a
**positional** argument, and its flags are hyphenated (`--data-root`,
`--min-gps`), not `--run_id`/`--data_root` like Stages 1–3. That's
`role 4 code.py`'s own existing CLI, not something changed here.

## Running the full pipeline (Stages 1–4)

[run_pipeline.py](run_pipeline.py) chains all four stages in order,
gating each handoff through `manifest.json`:

```bash
python run_pipeline.py --run_id run_001
python run_pipeline.py --run_id run_001 --data_root /data \
    --target_fps 2.0 --blur_threshold 100.0 --matcher sequential \
    --nb_neighbors 20 --std_ratio 2.0 --radius_nb_points 16 \
    --radius 0.05 --min_component_fraction 0.05 --min_gps 5

# Resume from Stage 3 (Stages 1-2 already done for this run_id):
python run_pipeline.py --run_id run_001 --start_stage 3
```

Stages 1–3 are hard gates: if any of them fails, the script stops
immediately (matching each stage's own hard-fail contract). **Stage 4
is not a hard gate** — its exit code is logged, but the script always
finishes and exits 0 regardless of whether Stage 4 georeferenced
successfully, per Stage 4's own "never halt the pipeline" contract.

Verified end-to-end during development (real Stage 1→2 run on synthetic
footage, real `role 4 code.py` execution against a stand-in Stage 3
mesh — dense reconstruction itself can't run on this dev machine, see
Stage 3's CUDA note above): the orchestrator correctly stopped at the
Stage 3 hard-fail gate without invoking Stage 4, and — separately, with
a stand-in mesh — correctly produced a georeferenced `model_geo.obj`
when enough GPS matches were available, and correctly fell back to an
unchanged pass-through mesh + `warning.txt` (with the pipeline still
exiting 0) when they weren't. [run_stage1_2.py](run_stage1_2.py) still
exists separately for a Stage 1+2-only run.
