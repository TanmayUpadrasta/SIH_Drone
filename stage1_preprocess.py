#!/usr/bin/env python3
"""
Stage 1 (Preprocessing) — drone video reconstruction pipeline.

Reads /data/{run_id}/raw/video.mp4 + camera_intrinsics.json, extracts
sharp keyframes at a target extraction rate (resolution-aware), and
writes them to /data/{run_id}/keyframes/ per the pipeline I/O contract.

Usage:
    python stage1_preprocess.py --run_id run_001
    python stage1_preprocess.py --run_id run_001 --target_fps 2.0 \
        --blur_threshold 100.0 --data_root /data
"""

import argparse
import json
import sys
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
    Walk the video at the computed interval, score each candidate frame
    for blur, and split into accepted / rejected lists.

    Accepted frames are never downscaled below 1080p: if the source is
    already below 1080p this would violate the contract, so that case
    is treated as a hard failure before extraction begins.
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
    rejected = []

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % interval == 0:
            blur_score = compute_blur_score(frame)
            if blur_score < blur_threshold:
                rejected.append(
                    {
                        "frame_number": frame_index,
                        "blur_score": blur_score,
                        "reason": f"blur_score {blur_score:.2f} < threshold {blur_threshold:.2f}",
                    }
                )
            else:
                accepted.append(
                    {
                        "frame_number": frame_index,
                        "blur_score": blur_score,
                        "image": frame,
                    }
                )

        frame_index += 1

    cap.release()
    return accepted, rejected


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


def write_rejected_log(rejected, keyframes_dir):
    """Log every rejected frame's number, blur score, and reason."""
    log_path = keyframes_dir / "rejected_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Stage 1 rejected frames — {len(rejected)} total\n")
        f.write("# frame_number\tblur_score\treason\n")
        for item in rejected:
            f.write(f"{item['frame_number']}\t{item['blur_score']:.2f}\t{item['reason']}\n")

    return log_path


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
        description="Stage 1 (Preprocessing): extract sharp keyframes from drone video."
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
    print()

    accepted, rejected = extract_and_filter(video_path, video_props, interval, args.blur_threshold)

    saved_paths = save_keyframes(accepted, keyframes_dir)
    log_path = write_rejected_log(rejected, keyframes_dir)
    update_manifest(args.data_root, args.run_id, manifest_path)

    print("=" * 60)
    print("STAGE 1 SUMMARY")
    print("=" * 60)
    print(f"Source resolution   : {video_props['width']}x{video_props['height']} ({source_class})")
    print(f"Candidate frames    : {len(accepted) + len(rejected)}")
    print(f"Frames rejected     : {len(rejected)}")
    print(f"Final keyframe count: {len(saved_paths)}")
    print(f"Keyframes written to: {keyframes_dir}")
    print(f"Rejected log        : {log_path}")
    print(f"Manifest updated    : {manifest_path}")

    if not saved_paths:
        fail("No frames were accepted — all candidates failed the blur threshold. See rejected_log.txt.")


if __name__ == "__main__":
    main()
