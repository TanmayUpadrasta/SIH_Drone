#!/usr/bin/env python3
"""
Stage 1 (Preprocessing) — drone video reconstruction pipeline.

Reads /data/{run_id}/raw/video.mp4 + camera_intrinsics.json and writes
sharp, evenly-spaced keyframes to /data/{run_id}/keyframes/.

Keyframe selection strategy — unchanged core, targeted gap-fill added:
    The interval-based sampling and the Laplacian-variance blur test are
    EXACTLY as before: one frame is tested at each fixed offset (0, 15,
    30, 45, 60, ...), and it's accepted if its blur score clears the
    threshold.

    What's new is what happens when that tested frame FAILS. Previously
    it was just dropped, with no substitute — if frames 30 and 45 both
    failed, the gap between accepted keyframes silently jumped from the
    intended ~15 frames to 45. That starves SfM feature matching of
    overlap right where it happens.

    Now, on a rejection, we search ONLY the window between the previous
    offset and the failed one (e.g. frames strictly between 15 and 30) for
    the sharpest available frame, and use that as the substitute. We are
    NOT rescanning the whole video and NOT scoring every frame by default
    — normal (non-rejected) offsets cost exactly one blur-score computation,
    same as the original script. The extra work only happens inside a
    window that actually needs it.

    This is possible cheaply because video decoding is already sequential
    (cap.read() has to walk through every frame regardless), so a small
    rolling buffer of the last `interval` decoded frames is kept as the
    pass moves forward. When a rejection happens, the candidates for that
    window are already sitting in memory from frames just decoded a moment
    ago — no reseek, no re-reading the file, no scoring outside that window.

Usage:
    python stage1_preprocess.py --run_id run_001
    python stage1_preprocess.py --run_id run_001 --target_fps 2.0 \
        --blur_threshold 100.0 --data_root /data
"""

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Frames at or above this width are treated as 4K-class.
FOUR_K_WIDTH_THRESHOLD = 3840

# 4K frames carry more usable detail per frame, so we extract at a
# lower effective fps than 1080p-class footage for the same duration.
FOUR_K_FPS_MULTIPLIER = 0.75

# Blur score is Laplacian variance computed after resizing the frame
# to this reference width, so the threshold is comparable across
# 1080p and 4K source footage (raw variance scales with pixel count).
BLUR_REFERENCE_WIDTH = 960

# Never write frames smaller than this, regardless of source/target.
MIN_OUTPUT_WIDTH = 1920
MIN_OUTPUT_HEIGHT = 1080

JPEG_QUALITY = 95


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def detect_video_properties(video_path):
    """Open the video and return its core properties, or fail cleanly."""
    if not video_path.exists():
        fail(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"Could not open video (unsupported codec or corrupt file): {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()

    cap.release()

    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        fail(
            f"Video reports invalid properties (fps={fps}, "
            f"{width}x{height}, frames={frame_count}): {video_path}"
        )

    is_4k = width >= FOUR_K_WIDTH_THRESHOLD

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": codec or "unknown",
        "is_4k": is_4k,
    }


def compute_extraction_interval(video_props, target_fps):
    """
    Frame interval (in source frames) between extracted candidates.
    UNCHANGED from the original script.

    4K footage extracts at a slightly lower effective fps than
    1080p-class footage, since each 4K frame carries more usable
    detail for feature matching.
    """
    effective_target_fps = target_fps
    if video_props["is_4k"]:
        effective_target_fps = target_fps * FOUR_K_FPS_MULTIPLIER

    interval = max(1, round(video_props["fps"] / effective_target_fps))
    return interval, effective_target_fps


def compute_blur_score(frame):
    """
    Resolution-normalized sharpness score: Laplacian variance computed
    on a copy of the frame resized to a fixed reference width, so the
    same threshold is meaningfully comparable across 1080p and 4K input.
    UNCHANGED from the original script.
    """
    height, width = frame.shape[:2]
    scale = BLUR_REFERENCE_WIDTH / float(width)
    ref_height = max(1, round(height * scale))

    resized = cv2.resize(
        frame, (BLUR_REFERENCE_WIDTH, ref_height), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_and_filter(video_path, video_props, interval, blur_threshold):
    """
    Same fixed-offset sampling and same blur threshold as the original
    script. The only addition: when the frame AT an offset fails the
    threshold, search the window strictly between the previous offset and
    this one for a substitute, instead of just dropping it.

    A small deque (maxlen=interval) buffers recently decoded frames as the
    single sequential pass moves forward, so that window's candidates are
    already in memory when a rejection happens — no rescanning.
    """
    if video_props["height"] < MIN_OUTPUT_HEIGHT or video_props["width"] < MIN_OUTPUT_WIDTH:
        fail(
            f"Source video ({video_props['width']}x{video_props['height']}) is "
            f"below the 1080p floor required by the pipeline contract."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"Could not reopen video for extraction: {video_path}")

    accepted = []
    selection_log = []
    recent_frames = deque(maxlen=interval)  # rolling buffer: (frame_number, image)
    previous_offset = None  # frame_number of the last evaluated offset (grid position)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        recent_frames.append((frame_index, frame))

        if frame_index % interval == 0:
            blur_score = compute_blur_score(frame)

            if blur_score >= blur_threshold:
                # Same as the original script's accept path — no extra work.
                accepted.append({"frame_number": frame_index, "blur_score": blur_score, "image": frame})
                selection_log.append(
                    {
                        "offset": frame_index,
                        "offset_blur_score": blur_score,
                        "status": "accepted_at_offset",
                        "selected_frame": frame_index,
                        "selected_blur_score": blur_score,
                        "candidates_scored": 0,
                    }
                )

            elif previous_offset is None:
                # First offset (frame 0) failed — nothing exists before it to
                # search, so it's kept as an unavoidable edge-case fallback.
                accepted.append({"frame_number": frame_index, "blur_score": blur_score, "image": frame})
                selection_log.append(
                    {
                        "offset": frame_index,
                        "offset_blur_score": blur_score,
                        "status": "first_frame_fallback",
                        "selected_frame": frame_index,
                        "selected_blur_score": blur_score,
                        "candidates_scored": 0,
                    }
                )

            else:
                # Rejected: search ONLY the window strictly between the
                # previous grid offset and this one — exactly what's
                # already sitting in `recent_frames`.
                candidates = [
                    (idx, img) for idx, img in recent_frames
                    if previous_offset < idx < frame_index
                ]

                if not candidates:
                    # interval == 1, or buffer too small to have history —
                    # nothing to substitute with, keep the tested frame.
                    accepted.append({"frame_number": frame_index, "blur_score": blur_score, "image": frame})
                    selection_log.append(
                        {
                            "offset": frame_index,
                            "offset_blur_score": blur_score,
                            "status": "fallback_no_candidates",
                            "selected_frame": frame_index,
                            "selected_blur_score": blur_score,
                            "candidates_scored": 0,
                        }
                    )
                else:
                    # Score ONLY these candidates — this is the entire extra
                    # cost of gap-filling, bounded to one window's worth of
                    # frames, never the whole video.
                    scored = [(idx, img, compute_blur_score(img)) for idx, img in candidates]
                    best_idx, best_img, best_score = max(scored, key=lambda t: t[2])

                    accepted.append({"frame_number": best_idx, "blur_score": best_score, "image": best_img})
                    status = "gap_fill_accepted" if best_score >= blur_threshold else "gap_fill_fallback_below_threshold"
                    selection_log.append(
                        {
                            "offset": frame_index,
                            "offset_blur_score": blur_score,
                            "status": status,
                            "selected_frame": best_idx,
                            "selected_blur_score": best_score,
                            "candidates_scored": len(scored),
                        }
                    )

            # Advance the grid marker to this offset regardless of outcome —
            # window boundaries stay locked to the fixed interval grid, not
            # to whichever frame ended up selected.
            previous_offset = frame_index

        frame_index += 1

    cap.release()
    return accepted, selection_log


def save_keyframes(accepted, keyframes_dir):
    """Save accepted frames as sequential 4-digit-zero-padded JPEGs."""
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for output_index, item in enumerate(accepted, start=1):
        filename = f"frame_{output_index:04d}.jpg"
        filepath = keyframes_dir / filename
        cv2.imwrite(str(filepath), item["image"], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved_paths.append(filepath)

    return saved_paths


def write_selection_log(selection_log, keyframes_dir):
    """
    One line per offset on the original sampling grid, showing whether it
    was a clean accept or a gap-filled substitute (and from how large a
    search that substitute came). Replaces the old rejected_log.txt with
    something that shows the fix actually working.
    """
    log_path = keyframes_dir / "selection_log.txt"
    gap_filled = sum(1 for s in selection_log if s["status"].startswith("gap_fill"))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Stage 1 selection log — {len(selection_log)} offsets, {gap_filled} gap-filled\n")
        f.write("# offset\toffset_blur\tstatus\tselected_frame\tselected_blur\tcandidates_scored\n")
        for s in selection_log:
            f.write(
                f"{s['offset']}\t{s['offset_blur_score']:.2f}\t{s['status']}\t"
                f"{s['selected_frame']}\t{s['selected_blur_score']:.2f}\t{s['candidates_scored']}\n"
            )

    return log_path, gap_filled


def update_manifest(data_root, run_id, manifest_path):
    """
    Update manifest.json, setting stage_1_complete=true and refreshing
    last_updated. Creates the file (with all other stages false) if it
    doesn't exist yet. Never touches other stages' fields.
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

    manifest["stage_1_complete"] = True
    manifest["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_camera_intrinsics(path):
    if not path.exists():
        fail(f"camera_intrinsics.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 (Preprocessing): extract sharp keyframes from drone video, "
        "with targeted gap-filling when a sampled frame fails the blur test."
    )
    parser.add_argument("--run_id", required=True, help="Run identifier, e.g. run_001")
    parser.add_argument(
        "--target_fps",
        type=float,
        default=2.0,
        help="Target keyframe extraction rate in fps for 1080p-class source video (default: 2.0). "
        "4K source video is extracted at a reduced effective rate automatically.",
    )
    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=100.0,
        help="Minimum resolution-normalized Laplacian variance for a frame to be accepted "
        "(default: 100.0). Raise to reject more borderline-sharp frames, lower to keep more.",
    )
    parser.add_argument(
        "--data_root",
        default="/data",
        help="Root data directory containing {run_id}/ folders (default: /data)",
    )
    args = parser.parse_args()

    run_dir = Path(args.data_root) / args.run_id
    video_path = run_dir / "raw" / "video.mp4"
    intrinsics_path = run_dir / "raw" / "camera_intrinsics.json"
    keyframes_dir = run_dir / "keyframes"
    manifest_path = run_dir / "manifest.json"

    intrinsics = load_camera_intrinsics(intrinsics_path)

    video_props = detect_video_properties(video_path)
    interval, effective_target_fps = compute_extraction_interval(video_props, args.target_fps)

    source_class = "4K" if video_props["is_4k"] else "1080p-class"
    print(f"Source video      : {video_path}")
    print(f"Resolution         : {video_props['width']}x{video_props['height']} ({source_class})")
    print(f"FPS / frame count  : {video_props['fps']:.2f} / {video_props['frame_count']}")
    print(f"Codec              : {video_props['codec']}")
    print(f"Camera model       : {intrinsics.get('camera_model', 'unknown')}")
    print(f"Extraction interval: every {interval} frames (~{effective_target_fps:.2f} fps effective)")
    print(f"Blur threshold     : {args.blur_threshold}")
    print("Gap-fill           : ON — rejected offsets trigger a search of ONLY their local window")
    print()

    accepted, selection_log = extract_and_filter(video_path, video_props, interval, args.blur_threshold)

    saved_paths = save_keyframes(accepted, keyframes_dir)
    log_path, gap_filled = write_selection_log(selection_log, keyframes_dir)
    update_manifest(args.data_root, args.run_id, manifest_path)

    print("=" * 60)
    print("STAGE 1 SUMMARY")
    print("=" * 60)
    print(f"Source resolution   : {video_props['width']}x{video_props['height']} ({source_class})")
    print(f"Offsets on grid     : {len(selection_log)}")
    print(f"Gap-filled offsets  : {gap_filled} (sampled frame failed blur test, substitute found locally)")
    print(f"Final keyframe count: {len(saved_paths)}")
    print(f"Keyframes written to: {keyframes_dir}")
    print(f"Selection log       : {log_path}")
    print(f"Manifest updated    : {manifest_path}")

    if not saved_paths:
        fail("No frames were accepted — check that the video actually contains readable frames.")


if __name__ == "__main__":
    main()
