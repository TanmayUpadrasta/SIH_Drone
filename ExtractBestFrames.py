#!/usr/bin/env python3
"""
extract_best_frames.py
-----------------------
Role: Video Preprocessing & Keyframe Selection

Takes drone video(s) from an input folder, breaks each into frames, keeps
only the SHARP ones, and from those keeps only the NON-REPEATING ones
(i.e. each saved frame shows meaningfully new content vs. the last one
saved) — then stores the result to an output folder.

Usage:
    python3 extract_best_frames.py --input videos/ --output keyframes/

    # tune how aggressively frames are sampled / filtered
    python3 extract_best_frames.py --input videos/ --output keyframes/ \
        --sample-fps 2 --blur-percentile 20 --overlap-max 0.7
"""

import argparse
import json
import os
import glob
import cv2
import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


# --------------------------------------------------------------------- #
# STEP 1: how sharp is a frame?
# --------------------------------------------------------------------- #
def sharpness_score(frame_bgr):
    """
    Laplacian variance: a well-known, simple blur-detection trick.
    A sharp image has strong edges -> high variance in the 2nd derivative.
    A blurry image has soft edges -> low variance.
    Higher score = sharper frame.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# --------------------------------------------------------------------- #
# STEP 2: is this frame "new" compared to the last one we kept?
# --------------------------------------------------------------------- #
def compute_overlap(orb, matcher, gray_prev, gray_curr, min_matches=15):
    """
    Estimates how much of the CURRENT frame is still showing the same
    content as the PREVIOUS saved frame.

    How: find matching features (ORB) between the two frames, fit a
    homography (the geometric transform that maps one view to the other),
    then warp the previous frame's rectangle into the current frame and
    measure what fraction of the current frame it covers.

      overlap ~ 1.0  -> almost the same view (repeating, skip it)
      overlap ~ 0.0  -> completely different view (matching would fail,
                        take it now before it's too late)

    Returns None if there isn't enough reliable matching info (e.g. drone
    turned sharply) -- treat that as "definitely not repeating".
    """
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    # ratio test (Lowe's) to keep only confident matches
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_matches:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None or int(mask.sum()) < min_matches:
        return None

    h, w = gray_curr.shape
    corners_prev = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners_prev, H)
    except cv2.error:
        return None

    footprint = np.zeros((h, w), dtype=np.uint8)
    poly = np.clip(projected.reshape(-1, 2), -1e5, 1e5).astype(np.int32)
    cv2.fillConvexPoly(footprint, poly, 255)

    overlap_ratio = cv2.countNonZero(footprint) / (h * w)
    return min(overlap_ratio, 1.0)


# --------------------------------------------------------------------- #
# STEP 3: full per-video pipeline
# --------------------------------------------------------------------- #
def process_video(video_path, out_dir, sample_fps, blur_percentile, overlap_max):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Could not open {video_path}, skipping.")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(src_fps / sample_fps))
    print(f"  source fps={src_fps:.1f}, sampling every {stride} frames "
          f"(~{src_fps/stride:.2f} fps)")

    # --- Pass 1: pull candidate frames at a fixed sample rate, score sharpness ---
    candidates = []  # (frame_index, timestamp_sec, frame_bgr, sharpness)
    idx = 0
    while True:
        if idx % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            score = sharpness_score(frame)
            candidates.append((idx, idx / src_fps, frame, score))
        else:
            ok = cap.grab()  # skip decode for frames we don't need
            if not ok:
                break
        idx += 1
    cap.release()

    if not candidates:
        print("  [!] No frames read from video.")
        return []

    # --- Pass 2: drop the blurriest frames (adaptive threshold, not a fixed number) ---
    scores = np.array([c[3] for c in candidates])
    threshold = np.percentile(scores, blur_percentile)
    sharp_frames = [c for c in candidates if c[3] >= threshold]
    print(f"  {len(candidates)} candidates -> {len(sharp_frames)} after "
          f"dropping blurriest {blur_percentile:.0f}% (threshold={threshold:.1f})")

    # --- Pass 3: walk through in time order, only keep frames that add new content ---
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    kept = []
    prev_gray = None
    for frame_idx, ts, frame, blur in sharp_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            kept.append((frame_idx, ts, frame, blur, None))
            prev_gray = gray
            continue

        overlap = compute_overlap(orb, matcher, prev_gray, gray)

        # overlap is None -> content changed too much to even match reliably;
        # overlap <= overlap_max -> enough new content to be worth keeping
        if overlap is None or overlap <= overlap_max:
            kept.append((frame_idx, ts, frame, blur, overlap))
            prev_gray = gray
        # else: too similar to the last kept frame, skip (it's a repeat)

    print(f"  {len(sharp_frames)} sharp frames -> {len(kept)} non-repeating best frames")

    # --- save results ---
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    meta = []
    for i, (frame_idx, ts, frame, blur, overlap) in enumerate(kept):
        fname = f"{video_name}_frame_{i:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta.append({
            "file": fname,
            "source_video": os.path.basename(video_path),
            "source_frame_index": frame_idx,
            "timestamp_sec": round(ts, 3),
            "sharpness_score": round(blur, 2),
            "overlap_to_prev": None if overlap is None else round(overlap, 3),
        })
    return meta


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Extract best non-repeating frames from drone videos in a folder.")
    p.add_argument("--input", required=True, help="Folder containing video files")
    p.add_argument("--output", required=True, help="Folder to save selected frames + metadata.json")
    p.add_argument("--sample-fps", type=float, default=2.0, help="Raw frame sampling rate before filtering")
    p.add_argument("--blur-percentile", type=float, default=20.0, help="Drop the blurriest N%% of sampled frames")
    p.add_argument("--overlap-max", type=float, default=0.7,
                   help="Max allowed similarity to the last kept frame (lower = fewer, more distinct frames)")
    args = p.parse_args()

    videos = sorted(
        f for f in glob.glob(os.path.join(args.input, "*"))
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    if not videos:
        print(f"No video files found in {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    all_meta = []
    for video_path in videos:
        print(f"\nProcessing {os.path.basename(video_path)} ...")
        meta = process_video(video_path, args.output, args.sample_fps,
                              args.blur_percentile, args.overlap_max)
        all_meta.extend(meta)

    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nDone. {len(all_meta)} best frames saved to {args.output}/")
    print(f"Metadata (source video, timestamp, scores) saved to {args.output}/metadata.json")


if __name__ == "__main__":
    main()