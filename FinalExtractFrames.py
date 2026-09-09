"""
SfM / Photogrammetry Video Frame Selector
=========================================

Purpose:
    Convert a video into a high-quality sequence of frames suitable for
    SfM / photogrammetry / 3D reconstruction.

Main features:
    1. Two-pass processing
    2. Adaptive temporal sampling
    3. Sparse optical-flow motion estimation
    4. SIFT feature quality
    5. SIFT feature spatial coverage
    6. Perceptual-hash duplicate detection
    7. Homography / affine geometry
    8. Approximate visual overlap estimation
    9. Parallax proxy
   10. Pure-rotation detection
   11. Moving-object detection
   12. Blur detection
   13. Exposure-change detection
   14. Temporal coverage
   15. Maximum temporal gap
   16. Candidate scoring
   17. Diagnostic metadata
   18. Contact-sheet generation

Requirements:
    pip install opencv-python numpy

Optional:
    pip install opencv-contrib-python

OpenCV SIFT is included in modern OpenCV builds.

Example:
    python extract_sfm_frames.py --input VIDEO.mp4

Example with custom settings:
    python extract_sfm_frames.py \
        --input VIDEO.mp4 \
        --output sfm_frames \
        --sample-min 3 \
        --sample-max 10 \
        --target-overlap 0.70
"""

import os
import cv2
import json
import math
import argparse
import hashlib
import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_OUTPUT = "sfm_selected_frames"

# Image processing resolution used for analysis.
# Original full-resolution frames are saved.
ANALYSIS_WIDTH = 960

# Number of SIFT features to detect.
SIFT_FEATURES = 2500

# Optical flow
FLOW_POINTS = 500
FLOW_MAX_LEVEL = 3

# pHash size
PHASH_SIZE = 32

# Spatial feature grid
GRID_ROWS = 4
GRID_COLS = 4

# Temporal coverage
TIME_BINS = 20

# Desired overlap
TARGET_OVERLAP_LOW = 0.60
TARGET_OVERLAP_HIGH = 0.80

# Very high overlap = probably duplicate
DUPLICATE_OVERLAP = 0.92

# Minimum acceptable overlap before trying to force a selection
MIN_OVERLAP = 0.35

# Maximum gap fallback.
# This is automatically adapted to FPS.
MAX_GAP_SECONDS = 1.0

# Minimum frame separation.
MIN_FRAME_GAP = 2

# Optical flow / geometry
FLOW_RANSAC_THRESHOLD = 3.0
HOMOGRAPHY_RANSAC_THRESHOLD = 4.0

# SIFT Lowe ratio
SIFT_RATIO = 0.72

# Minimum useful SIFT matches
MIN_SIFT_MATCHES = 12

# Safety limits
MIN_SELECTED_FRAMES = 8


# ============================================================
# BASIC HELPERS
# ============================================================

def resize_for_analysis(image, width=ANALYSIS_WIDTH):
    """
    Resize image while preserving aspect ratio.
    """
    h, w = image.shape[:2]

    if w <= width:
        return image.copy()

    scale = width / float(w)

    new_h = max(1, int(h * scale))

    return cv2.resize(
        image,
        (width, new_h),
        interpolation=cv2.INTER_AREA
    )


def grayscale(image):
    """
    Convert BGR image to grayscale.
    """
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def robust_percentile(values, p, default=0.0):
    """
    Safe percentile.
    """
    values = np.asarray(values, dtype=np.float32)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return default

    return float(np.percentile(values, p))


def normalize_robust(value, low, high):
    """
    Normalize using robust bounds.
    """
    if high <= low:
        return 0.5

    return float(np.clip(
        (value - low) / (high - low),
        0.0,
        1.0
    ))


def clamp(value, low=0.0, high=1.0):
    return float(np.clip(value, low, high))


def safe_mean(values, default=0.0):
    if values is None or len(values) == 0:
        return default

    return float(np.mean(values))


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def perceptual_hash(image, size=PHASH_SIZE):
    """
    DCT-based perceptual hash.

    Returns a binary vector.
    """

    gray = grayscale(image)

    small = cv2.resize(
        gray,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    small = np.float32(small)

    dct = cv2.dct(small)

    # Ignore DC component.
    dct_low = dct[:8, :8]

    median = np.median(dct_low[1:, :])

    return (dct_low > median).flatten()


def phash_distance(hash1, hash2):
    """
    Normalized Hamming distance.
    """
    if hash1 is None or hash2 is None:
        return 1.0

    return float(np.mean(hash1 != hash2))


# ============================================================
# BLUR / SHARPNESS
# ============================================================

def laplacian_variance(gray):
    """
    Sharpness proxy.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ============================================================
# EXPOSURE
# ============================================================

def brightness_statistics(gray):
    """
    Robust brightness statistics.
    """
    pixels = gray.reshape(-1).astype(np.float32)

    return {
        "mean": float(np.mean(pixels)),
        "median": float(np.median(pixels)),
        "p10": float(np.percentile(pixels, 10)),
        "p90": float(np.percentile(pixels, 90)),
    }


def exposure_change(gray1, gray2):
    """
    Estimate global exposure/brightness change.

    Returns:
        0 = almost no change
        1 = very large change
    """

    s1 = brightness_statistics(gray1)
    s2 = brightness_statistics(gray2)

    mean_change = abs(
        s1["mean"] - s2["mean"]
    ) / max(1.0, s1["mean"])

    median_change = abs(
        s1["median"] - s2["median"]
    ) / max(1.0, s1["median"])

    p90_change = abs(
        s1["p90"] - s2["p90"]
    ) / max(1.0, s1["p90"])

    value = (
        0.45 * mean_change +
        0.35 * median_change +
        0.20 * p90_change
    )

    return clamp(value / 0.30)


# ============================================================
# SIFT
# ============================================================

def create_sift(nfeatures=SIFT_FEATURES):

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "SIFT is unavailable. Install a modern OpenCV build using:\n"
            "pip install opencv-contrib-python"
        )

    return cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=0.02,
        edgeThreshold=10,
        sigma=1.6
    )


def detect_sift(image, sift):

    gray = grayscale(image)

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


def feature_spatial_coverage(keypoints, width, height):
    """
    Measure how well features are distributed over the image.

    Prevents selecting frames where all features are concentrated
    in one small area.
    """

    if keypoints is None or len(keypoints) == 0:
        return 0.0

    grid = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=np.uint8
    )

    for kp in keypoints:

        x, y = kp.pt

        col = int(
            np.clip(
                x / width * GRID_COLS,
                0,
                GRID_COLS - 1
            )
        )

        row = int(
            np.clip(
                y / height * GRID_ROWS,
                0,
                GRID_ROWS - 1
            )
        )

        grid[row, col] = 1

    occupied = np.sum(grid)

    return float(
        occupied / (GRID_ROWS * GRID_COLS)
    )


# ============================================================
# OPTICAL FLOW
# ============================================================

def get_flow_points(gray):
    """
    Detect Shi-Tomasi points for sparse optical flow.
    """

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=FLOW_POINTS,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )

    return points


def optical_flow_motion(gray1, gray2):
    """
    Estimate motion between two frames using sparse Lucas-Kanade flow.

    Returns detailed motion information.
    """

    p0 = get_flow_points(gray1)

    if p0 is None or len(p0) < 10:

        return {
            "valid": False,
            "motion": 0.0,
            "median_flow": 0.0,
            "flow_std": 0.0,
            "inlier_ratio": 0.0,
            "translation": 0.0,
            "rotation": 0.0,
            "residual": 0.0,
            "moving_object_score": 0.0,
            "parallax_proxy": 0.0,
        }

    p1, status, error = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray2,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=FLOW_MAX_LEVEL,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if p1 is None or status is None:

        return {
            "valid": False,
            "motion": 0.0,
            "median_flow": 0.0,
            "flow_std": 0.0,
            "inlier_ratio": 0.0,
            "translation": 0.0,
            "rotation": 0.0,
            "residual": 0.0,
            "moving_object_score": 0.0,
            "parallax_proxy": 0.0,
        }

    status = status.reshape(-1)

    valid0 = p0.reshape(-1, 2)[status == 1]
    valid1 = p1.reshape(-1, 2)[status == 1]

    if len(valid0) < 10:

        return {
            "valid": False,
            "motion": 0.0,
            "median_flow": 0.0,
            "flow_std": 0.0,
            "inlier_ratio": 0.0,
            "translation": 0.0,
            "rotation": 0.0,
            "residual": 0.0,
            "moving_object_score": 0.0,
            "parallax_proxy": 0.0,
        }

    flow = valid1 - valid0

    magnitudes = np.linalg.norm(
        flow,
        axis=1
    )

    median_flow = float(
        np.median(magnitudes)
    )

    flow_std = float(
        np.std(magnitudes)
    )

    # --------------------------------------------------------
    # Estimate dominant affine motion
    # --------------------------------------------------------

    affine, inliers = cv2.estimateAffinePartial2D(
        valid0,
        valid1,
        method=cv2.RANSAC,
        ransacReprojThreshold=FLOW_RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.99
    )

    if affine is None:

        return {
            "valid": True,
            "motion": clamp(median_flow / 30.0),
            "median_flow": median_flow,
            "flow_std": flow_std,
            "inlier_ratio": 0.0,
            "translation": median_flow,
            "rotation": 0.0,
            "residual": 0.0,
            "moving_object_score": 0.0,
            "parallax_proxy": clamp(flow_std / 20.0),
        }

    inliers = inliers.reshape(-1).astype(bool)

    inlier_ratio = float(
        np.mean(inliers)
    )

    # --------------------------------------------------------
    # Affine translation
    # --------------------------------------------------------

    tx = float(affine[0, 2])
    ty = float(affine[1, 2])

    translation = math.sqrt(
        tx * tx + ty * ty
    )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    rotation = math.degrees(
        math.atan2(
            affine[1, 0],
            affine[0, 0]
        )
    )

    # --------------------------------------------------------
    # Residuals
    # --------------------------------------------------------

    predicted = cv2.transform(
        valid0.reshape(-1, 1, 2),
        affine
    ).reshape(-1, 2)

    residuals = np.linalg.norm(
        valid1 - predicted,
        axis=1
    )

    residual_median = float(
        np.median(residuals)
    )

    # --------------------------------------------------------
    # Parallax proxy
    # --------------------------------------------------------
    #
    # Important:
    # This is NOT metric parallax.
    #
    # Without camera calibration and depth estimation we cannot
    # calculate true metric translation/depth.
    #
    # Flow variation + residual variation provides a useful
    # proxy for depth-dependent motion.
    # --------------------------------------------------------

    if len(residuals) > 10:

        p75 = np.percentile(
            residuals,
            75
        )

        p25 = np.percentile(
            residuals,
            25
        )

        residual_spread = p75 - p25

    else:

        residual_spread = 0.0

    parallax_proxy = clamp(
        (
            0.45 * flow_std / 15.0 +
            0.55 * residual_spread / 8.0
        )
    )

    # --------------------------------------------------------
    # Moving-object proxy
    # --------------------------------------------------------
    #
    # A moving object usually produces a cluster of flow
    # residuals rather than residuals spread throughout the image.
    #
    # We therefore check:
    #   - high residual
    #   - spatial concentration
    #

    high_residual = residuals > max(
        4.0,
        np.percentile(residuals, 80)
    )

    moving_object_score = 0.0

    if np.sum(high_residual) >= 8:

        pts = valid1[high_residual]

        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        y_min = np.min(pts[:, 1])
        y_max = np.max(pts[:, 1])

        width = gray1.shape[1]
        height = gray1.shape[0]

        area_fraction = (
            (x_max - x_min) *
            (y_max - y_min)
        ) / max(
            1.0,
            width * height
        )

        concentration = (
            len(pts) /
            max(1, len(valid1))
        )

        # High residual points concentrated in a small region.
        if area_fraction < 0.25:

            moving_object_score = clamp(
                concentration * 3.0
            )

    # --------------------------------------------------------
    # Overall motion
    # --------------------------------------------------------

    motion = clamp(
        median_flow / 25.0
    )

    return {
        "valid": True,
        "motion": motion,
        "median_flow": median_flow,
        "flow_std": flow_std,
        "inlier_ratio": inlier_ratio,
        "translation": translation,
        "rotation": rotation,
        "residual": residual_median,
        "moving_object_score": moving_object_score,
        "parallax_proxy": parallax_proxy,
    }


# ============================================================
# HOMOGRAPHY / SIFT GEOMETRY
# ============================================================

def match_sift(
    kp1,
    des1,
    kp2,
    des2
):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    knn_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in knn_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < SIFT_RATIO * n.distance:
            good.append(m)

    return good


def estimate_geometry(
    kp1,
    kp2,
    matches,
    width,
    height
):

    result = {
        "matches": len(matches),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "overlap": 0.0,
        "homography_residual": 0.0,
        "affine_translation": 0.0,
        "affine_rotation": 0.0,
        "affine_scale": 1.0,
        "parallax_proxy": 0.0,
        "pure_rotation_score": 0.0,
    }

    if len(matches) < MIN_SIFT_MATCHES:
        return result

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        HOMOGRAPHY_RANSAC_THRESHOLD
    )

    if H is not None and mask is not None:

        mask = mask.reshape(-1).astype(bool)

        inlier_count = int(
            np.sum(mask)
        )

        result["inliers"] = inlier_count

        result["inlier_ratio"] = (
            inlier_count /
            max(1, len(matches))
        )

        if inlier_count >= 4:

            in1 = pts1[mask]
            in2 = pts2[mask]

            projected = cv2.perspectiveTransform(
                in1.reshape(-1, 1, 2),
                H
            ).reshape(-1, 2)

            residuals = np.linalg.norm(
                in2 - projected,
                axis=1
            )

            result["homography_residual"] = float(
                np.median(residuals)
            )

            # ------------------------------------------------
            # Approximate overlap
            # ------------------------------------------------

            corners = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ]).reshape(-1, 1, 2)

            try:

                warped_corners = cv2.perspectiveTransform(
                    corners,
                    H
                ).reshape(-1, 2)

                x_min = np.min(
                    warped_corners[:, 0]
                )

                x_max = np.max(
                    warped_corners[:, 0]
                )

                y_min = np.min(
                    warped_corners[:, 1]
                )

                y_max = np.max(
                    warped_corners[:, 1]
                )

                warped_width = (
                    x_max - x_min
                )

                warped_height = (
                    y_max - y_min
                )

                if (
                    warped_width > 1 and
                    warped_height > 1
                ):

                    # Estimate how much of the original image
                    # remains within the common frame.
                    common_width = min(
                        width,
                        warped_width
                    )

                    common_height = min(
                        height,
                        warped_height
                    )

                    overlap_area = (
                        common_width *
                        common_height
                    )

                    original_area = (
                        width * height
                    )

                    result["overlap"] = clamp(
                        overlap_area /
                        max(1.0, original_area)
                    )

            except cv2.error:
                pass

    # --------------------------------------------------------
    # Affine geometry
    # --------------------------------------------------------

    affine, affine_mask = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=FLOW_RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.99
    )

    if affine is not None:

        tx = float(
            affine[0, 2]
        )

        ty = float(
            affine[1, 2]
        )

        result["affine_translation"] = math.sqrt(
            tx * tx + ty * ty
        )

        result["affine_rotation"] = math.degrees(
            math.atan2(
                affine[1, 0],
                affine[0, 0]
            )
        )

        scale_x = math.sqrt(
            affine[0, 0] ** 2 +
            affine[1, 0] ** 2
        )

        scale_y = math.sqrt(
            affine[0, 1] ** 2 +
            affine[1, 1] ** 2
        )

        result["affine_scale"] = (
            scale_x + scale_y
        ) / 2.0

    # --------------------------------------------------------
    # Parallax proxy
    # --------------------------------------------------------
    #
    # Compare affine model and homography residual behavior.
    #
    # If homography explains the scene much better than affine,
    # there may be perspective/depth effects.
    #

    affine_residual = 0.0

    if affine is not None:

        predicted = cv2.transform(
            pts1.reshape(-1, 1, 2),
            affine
        ).reshape(-1, 2)

        affine_residuals = np.linalg.norm(
            pts2 - predicted,
            axis=1
        )

        affine_residual = float(
            np.median(affine_residuals)
        )

    homography_residual = result[
        "homography_residual"
    ]

    if affine_residual > 0:

        difference = (
            affine_residual -
            homography_residual
        )

        result["parallax_proxy"] = clamp(
            difference / 10.0
        )

    # --------------------------------------------------------
    # Pure rotation
    # --------------------------------------------------------

    translation = result[
        "affine_translation"
    ]

    rotation = abs(
        result["affine_rotation"]
    )

    # Strong rotation + very small translation.
    if rotation > 1.0:

        rotation_strength = clamp(
            rotation / 10.0
        )

        translation_strength = clamp(
            translation / max(20.0, width * 0.03)
        )

        result["pure_rotation_score"] = (
            rotation_strength *
            (1.0 - translation_strength)
        )

    return result


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


# ============================================================
# PASS 1 — MOTION SCAN
# ============================================================

def motion_scan(
    video_path,
    fps,
    frame_count,
    scan_step=2
):

    print()
    print("=" * 70)
    print("PASS 1 — MOTION / VIDEO ANALYSIS")
    print("=" * 70)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    samples = []

    previous_gray = None
    previous_frame_number = None

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % scan_step != 0:

            frame_number += 1
            continue

        small = resize_for_analysis(
            frame
        )

        gray = grayscale(
            small
        )

        blur = laplacian_variance(
            gray
        )

        brightness = brightness_statistics(
            gray
        )

        motion_data = {
            "motion": 0.0,
            "median_flow": 0.0,
            "flow_std": 0.0,
            "translation": 0.0,
            "rotation": 0.0,
            "parallax_proxy": 0.0,
            "moving_object_score": 0.0,
            "inlier_ratio": 0.0,
        }

        if previous_gray is not None:

            motion_data = optical_flow_motion(
                previous_gray,
                gray
            )

        samples.append({
            "frame": frame_number,
            "time": frame_number / fps,
            "blur": blur,
            "brightness": brightness["mean"],
            "motion": motion_data["motion"],
            "median_flow": motion_data["median_flow"],
            "flow_std": motion_data["flow_std"],
            "translation": motion_data["translation"],
            "rotation": motion_data["rotation"],
            "parallax_proxy": motion_data["parallax_proxy"],
            "moving_object_score": motion_data[
                "moving_object_score"
            ],
            "inlier_ratio": motion_data[
                "inlier_ratio"
            ]
        })

        previous_gray = gray
        previous_frame_number = frame_number

        frame_number += 1

        if len(samples) % 50 == 0:

            progress = (
                frame_number /
                max(1, frame_count)
            ) * 100

            print(
                f"  Scanned: {progress:5.1f}%"
            )

    cap.release()

    print(
        f"\nMotion samples collected: {len(samples)}"
    )

    return samples


# ============================================================
# MOTION STATISTICS
# ============================================================

def compute_motion_statistics(
    motion_samples
):

    motions = [
        x["median_flow"]
        for x in motion_samples
    ]

    translations = [
        x["translation"]
        for x in motion_samples
    ]

    parallax = [
        x["parallax_proxy"]
        for x in motion_samples
    ]

    blur = [
        x["blur"]
        for x in motion_samples
    ]

    stats = {

        "motion_p20":
            robust_percentile(
                motions,
                20,
                1.0
            ),

        "motion_p50":
            robust_percentile(
                motions,
                50,
                2.0
            ),

        "motion_p80":
            robust_percentile(
                motions,
                80,
                5.0
            ),

        "translation_p50":
            robust_percentile(
                translations,
                50,
                1.0
            ),

        "translation_p80":
            robust_percentile(
                translations,
                80,
                5.0
            ),

        "parallax_p50":
            robust_percentile(
                parallax,
                50,
                0.1
            ),

        "blur_p20":
            robust_percentile(
                blur,
                20,
                50.0
            ),

        "blur_p50":
            robust_percentile(
                blur,
                50,
                100.0
            ),

        "blur_p80":
            robust_percentile(
                blur,
                80,
                200.0
            ),
    }

    return stats


# ============================================================
# ADAPTIVE SAMPLING
# ============================================================

def choose_adaptive_stride(
    median_flow,
    stats,
    fps,
    sample_min,
    sample_max
):

    p20 = stats["motion_p20"]
    p80 = stats["motion_p80"]

    if p80 <= p20:

        normalized_motion = 0.5

    else:

        normalized_motion = normalize_robust(
            median_flow,
            p20,
            p80
        )

    # Fast camera movement -> small stride.
    # Slow camera movement -> large stride.
    #
    # Example at 30 FPS:
    #   slow -> around 10 frames
    #   medium -> around 6 frames
    #   fast -> around 3 frames

    stride = round(
        sample_max -
        normalized_motion *
        (sample_max - sample_min)
    )

    return int(
        np.clip(
            stride,
            sample_min,
            sample_max
        )
    )


def build_adaptive_candidates(
    motion_samples,
    frame_count,
    fps,
    sample_min,
    sample_max
):

    stats = compute_motion_statistics(
        motion_samples
    )

    candidates = []

    frame_to_sample = {
        x["frame"]: x
        for x in motion_samples
    }

    current_frame = 0

    while current_frame < frame_count:

        # Find nearest motion sample.
        nearest = min(
            motion_samples,
            key=lambda x:
                abs(x["frame"] - current_frame)
        )

        flow = nearest[
            "median_flow"
        ]

        stride = choose_adaptive_stride(
            flow,
            stats,
            fps,
            sample_min,
            sample_max
        )

        candidates.append(
            current_frame
        )

        current_frame += stride

    return candidates, stats


# ============================================================
# VIDEO FRAME READER
# ============================================================

def read_frame(
    cap,
    frame_number
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ret, frame = cap.read()

    if not ret:
        return None

    return frame


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_candidate(
    frame,
    previous_frame,
    sift,
    video_width,
    video_height
):

    analysis = resize_for_analysis(
        frame
    )

    gray = grayscale(
        analysis
    )

    kp, des = detect_sift(
        analysis,
        sift
    )

    feature_count = (
        len(kp)
        if kp is not None
        else 0
    )

    coverage = feature_spatial_coverage(
        kp,
        analysis.shape[1],
        analysis.shape[0]
    )

    blur = laplacian_variance(
        gray
    )

    result = {

        "feature_count":
            feature_count,

        "feature_coverage":
            coverage,

        "sharpness":
            blur,

        "keypoints":
            kp,

        "descriptors":
            des,

        "gray":
            gray,

        "analysis_image":
            analysis,

        "phash":
            perceptual_hash(
                analysis
            ),

        "matches":
            0,

        "inliers":
            0,

        "inlier_ratio":
            0.0,

        "overlap":
            1.0,

        "translation":
            0.0,

        "rotation":
            0.0,

        "parallax":
            0.0,

        "moving_object":
            0.0,

        "exposure_change":
            0.0,

        "flow_motion":
            0.0,

        "flow_std":
            0.0,

        "pure_rotation":
            0.0,
    }

    if previous_frame is None:
        return result

    previous_analysis = resize_for_analysis(
        previous_frame
    )

    previous_gray = grayscale(
        previous_analysis
    )

    # --------------------------------------------------------
    # Optical flow
    # --------------------------------------------------------

    flow = optical_flow_motion(
        previous_gray,
        gray
    )

    result["flow_motion"] = flow[
        "median_flow"
    ]

    result["flow_std"] = flow[
        "flow_std"
    ]

    result["translation"] = flow[
        "translation"
    ]

    result["rotation"] = flow[
        "rotation"
    ]

    result["parallax"] = flow[
        "parallax_proxy"
    ]

    result["moving_object"] = flow[
        "moving_object_score"
    ]

    result["pure_rotation"] = (
        1.0
        if (
            abs(result["rotation"]) > 1.5 and
            result["translation"] < 5
        )
        else 0.0
    )

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    result["exposure_change"] = exposure_change(
        previous_gray,
        gray
    )

    # --------------------------------------------------------
    # SIFT matching
    # --------------------------------------------------------

    kp_prev, des_prev = detect_sift(
        previous_analysis,
        sift
    )

    matches = match_sift(
        kp_prev,
        des_prev,
        kp,
        des
    )

    geometry = estimate_geometry(
        kp_prev,
        kp,
        matches,
        gray.shape[1],
        gray.shape[0]
    )

    result["matches"] = geometry[
        "matches"
    ]

    result["inliers"] = geometry[
        "inliers"
    ]

    result["inlier_ratio"] = geometry[
        "inlier_ratio"
    ]

    result["overlap"] = geometry[
        "overlap"
    ]

    result["parallax"] = max(
        result["parallax"],
        geometry["parallax_proxy"]
    )

    result["pure_rotation"] = max(
        result["pure_rotation"],
        geometry["pure_rotation_score"]
    )

    return result


# ============================================================
# QUALITY NORMALIZATION
# ============================================================

def feature_score(
    count,
    stats
):

    p20 = stats["feature_p20"]
    p50 = stats["feature_p50"]
    p80 = stats["feature_p80"]

    if p80 <= p20:
        return 0.5

    return normalize_robust(
        count,
        p20,
        p80
    )


def blur_score(
    sharpness,
    stats
):

    p20 = stats["blur_p20"]
    p80 = stats["blur_p80"]

    return normalize_robust(
        sharpness,
        p20,
        p80
    )


# ============================================================
# CANDIDATE SCORE
# ============================================================

def score_candidate(
    analysis,
    feature_stats,
    motion_stats,
    overlap_target
):

    # --------------------------------------------------------
    # Feature quality
    # --------------------------------------------------------

    fscore = feature_score(
        analysis["feature_count"],
        feature_stats
    )

    coverage = analysis[
        "feature_coverage"
    ]

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    sharpness = blur_score(
        analysis["sharpness"],
        feature_stats
    )

    # --------------------------------------------------------
    # SIFT geometry
    # --------------------------------------------------------

    inlier_score = clamp(
        analysis["inlier_ratio"] /
        0.70
    )

    match_score = clamp(
        analysis["matches"] /
        80.0
    )

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    overlap = analysis["overlap"]

    if (
        TARGET_OVERLAP_LOW <=
        overlap <=
        TARGET_OVERLAP_HIGH
    ):

        overlap_score = 1.0

    elif overlap > TARGET_OVERLAP_HIGH:

        overlap_score = clamp(
            1.0 -
            (overlap -
             TARGET_OVERLAP_HIGH)
            / 0.20
        )

    else:

        overlap_score = clamp(
            overlap /
            TARGET_OVERLAP_LOW
        )

    # --------------------------------------------------------
    # Motion / translation
    # --------------------------------------------------------

    translation = analysis[
        "translation"
    ]

    motion_score = normalize_robust(
        translation,
        motion_stats["translation_p20"],
        motion_stats["translation_p80"]
    )

    # --------------------------------------------------------
    # Parallax
    # --------------------------------------------------------

    parallax_score = analysis[
        "parallax"
    ]

    # --------------------------------------------------------
    # Penalties
    # --------------------------------------------------------

    duplicate_penalty = 0.0

    if overlap >= DUPLICATE_OVERLAP:
        duplicate_penalty = 1.0

    moving_penalty = analysis[
        "moving_object"
    ]

    rotation_penalty = analysis[
        "pure_rotation"
    ]

    exposure_penalty = analysis[
        "exposure_change"
    ]

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = (

        0.22 * fscore +

        0.12 * coverage +

        0.10 * sharpness +

        0.13 * inlier_score +

        0.05 * match_score +

        0.13 * overlap_score +

        0.10 * motion_score +

        0.10 * parallax_score

        -

        0.08 * duplicate_penalty -

        0.10 * moving_penalty -

        0.10 * rotation_penalty -

        0.06 * exposure_penalty
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# ADAPTIVE FEATURE THRESHOLDS
# ============================================================

def compute_feature_statistics(
    feature_counts
):

    if not feature_counts:

        return {
            "feature_p20": 100,
            "feature_p50": 200,
            "feature_p80": 400,
            "blur_p20": 50,
            "blur_p50": 100,
            "blur_p80": 200
        }

    counts = np.asarray(
        feature_counts,
        dtype=np.float32
    )

    # Do NOT use an unrestricted percentile as an absolute
    # threshold. That was the source of the earlier problem
    # where the threshold became too high.
    #
    # We use relative statistics instead.

    p20 = robust_percentile(
        counts,
        20,
        100
    )

    p50 = robust_percentile(
        counts,
        50,
        200
    )

    p80 = robust_percentile(
        counts,
        80,
        400
    )

    # Safety floor / ceiling.
    p20 = max(
        50,
        min(p20, 600)
    )

    p50 = max(
        p20,
        min(p50, 1500)
    )

    p80 = max(
        p50,
        min(p80, 2500)
    )

    return {
        "feature_p20": p20,
        "feature_p50": p50,
        "feature_p80": p80,

        "blur_p20": 50,
        "blur_p50": 100,
        "blur_p80": 200
    }


# ============================================================
# TEMPORAL COVERAGE
# ============================================================

def time_bin(
    frame_number,
    frame_count,
    bins=TIME_BINS
):

    if frame_count <= 1:
        return 0

    value = (
        frame_number /
        (frame_count - 1)
    )

    return int(
        np.clip(
            value * bins,
            0,
            bins - 1
        )
    )


# ============================================================
# SELECTION
# ============================================================

def select_frames(
    video_path,
    candidates,
    fps,
    frame_count,
    video_width,
    video_height,
    feature_stats,
    motion_stats,
    max_gap_frames,
    target_overlap
):

    print()
    print("=" * 70)
    print("PASS 2 — SIFT / GEOMETRY / FRAME SELECTION")
    print("=" * 70)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    sift = create_sift()

    selected = []

    selected_bins = set()

    last_selected_frame = None
    last_selected_image = None
    last_selected_hash = None

    candidate_cache = {}

    def get_candidate(frame_number):

        if frame_number not in candidate_cache:

            candidate_cache[
                frame_number
            ] = read_frame(
                cap,
                frame_number
            )

        return candidate_cache[
            frame_number
        ]

    for index, frame_number in enumerate(
        candidates
    ):

        frame = get_candidate(
            frame_number
        )

        if frame is None:
            continue

        if (
            last_selected_frame is not None and
            frame_number -
            last_selected_frame
            < MIN_FRAME_GAP
        ):
            continue

        analysis = analyze_candidate(
            frame,
            last_selected_image,
            sift,
            video_width,
            video_height
        )

        # ----------------------------------------------------
        # First frame
        # ----------------------------------------------------

        if last_selected_frame is None:

            selected.append({
                "frame": frame_number,
                "time": frame_number / fps,
                "score": 1.0,
                "reason": "initial_frame",
                "analysis": analysis,
                "image": frame
            })

            last_selected_frame = (
                frame_number
            )

            last_selected_image = (
                frame
            )

            last_selected_hash = (
                analysis["phash"]
            )

            selected_bins.add(
                time_bin(
                    frame_number,
                    frame_count
                )
            )

            print(
                f"  Selected frame "
                f"{frame_number}"
            )

            continue

        # ----------------------------------------------------
        # Duplicate check
        # ----------------------------------------------------

        phash_dist = phash_distance(
            last_selected_hash,
            analysis["phash"]
        )

        # Very small hash distance means visually almost
        # identical.
        duplicate = (
            phash_dist < 0.025
        )

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = score_candidate(
            analysis,
            feature_stats,
            motion_stats,
            target_overlap
        )

        analysis["phash_distance"] = (
            phash_dist
        )

        analysis["duplicate"] = (
            duplicate
        )

        # ----------------------------------------------------
        # Temporal coverage
        # ----------------------------------------------------

        current_bin = time_bin(
            frame_number,
            frame_count
        )

        new_time_bin = (
            current_bin not in selected_bins
        )

        # ----------------------------------------------------
        # Gap enforcement
        # ----------------------------------------------------

        gap = (
            frame_number -
            last_selected_frame
        )

        force_selection = (
            gap >= max_gap_frames
        )

        # ----------------------------------------------------
        # Overlap
        # ----------------------------------------------------

        overlap = analysis[
            "overlap"
        ]

        acceptable_geometry = (
            analysis["matches"]
            >= MIN_SIFT_MATCHES and
            analysis["inlier_ratio"]
            >= 0.20
        )

        # ----------------------------------------------------
        # Main selection decision
        # ----------------------------------------------------

        should_select = False
        reason = ""

        # Never select obvious duplicate.
        if duplicate:

            should_select = False

        # If this frame provides new temporal coverage,
        # allow slightly lower score.
        elif new_time_bin and score >= 0.38:

            should_select = True
            reason = "temporal_coverage"

        # Desired overlap + reasonable geometry.
        elif (
            TARGET_OVERLAP_LOW <= overlap <=
            TARGET_OVERLAP_HIGH and
            score >= 0.42
        ):

            should_select = True
            reason = "good_overlap"

        # Lower overlap but meaningful motion/parallax.
        elif (
            overlap >= MIN_OVERLAP and
            (
                analysis["parallax"] > 0.15 or
                analysis["translation"] > 4.0
            ) and
            score >= 0.45
        ):

            should_select = True
            reason = "motion_parallax"

        # Good frame even if geometry wasn't perfect.
        elif (
            score >= 0.62 and
            overlap >= MIN_OVERLAP
        ):

            should_select = True
            reason = "high_quality"

        # If there is a very large gap, force a useful frame.
        elif force_selection:

            if (
                analysis["feature_count"]
                >= feature_stats["feature_p20"]
                and
                analysis["feature_coverage"]
                >= 0.20
                and
                not duplicate
            ):

                should_select = True
                reason = "max_temporal_gap"

        # ----------------------------------------------------
        # Reject poor frames
        # ----------------------------------------------------

        if should_select:

            # Avoid pure rotation when possible.
            if (
                analysis["pure_rotation"]
                > 0.75 and
                analysis["parallax"]
                < 0.10 and
                not force_selection
            ):

                should_select = False
                reason = ""

        if should_select:

            selected.append({
                "frame": frame_number,
                "time": frame_number / fps,
                "score": score,
                "reason": reason,
                "analysis": analysis,
                "image": frame
            })

            last_selected_frame = (
                frame_number
            )

            last_selected_image = (
                frame
            )

            last_selected_hash = (
                analysis["phash"]
            )

            selected_bins.add(
                current_bin
            )

            print(
                f"  [{len(selected):03d}] "
                f"frame={frame_number:5d} "
                f"time={frame_number/fps:6.2f}s "
                f"score={score:.3f} "
                f"overlap={overlap:.2f} "
                f"features={analysis['feature_count']:4d} "
                f"reason={reason}"
            )

    cap.release()

    # --------------------------------------------------------
    # Safety pass
    # --------------------------------------------------------
    #
    # If selection is extremely small, add frames at regular
    # intervals while still avoiding duplicates.
    #

    if len(selected) < MIN_SELECTED_FRAMES:

        print()
        print(
            "WARNING: Very few frames selected."
        )

        print(
            "Running safety coverage pass..."
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        desired = max(
            MIN_SELECTED_FRAMES,
            min(
                30,
                int(frame_count / max(1, fps))
            )
        )

        safety_positions = np.linspace(
            0,
            frame_count - 1,
            desired,
            dtype=int
        )

        existing_frames = {
            x["frame"]
            for x in selected
        }

        for frame_number in safety_positions:

            if frame_number in existing_frames:
                continue

            frame = read_frame(
                cap,
                int(frame_number)
            )

            if frame is None:
                continue

            analysis = analyze_candidate(
                frame,
                last_selected_image,
                sift,
                video_width,
                video_height
            )

            selected.append({
                "frame": int(frame_number),
                "time": frame_number / fps,
                "score": 0.35,
                "reason": "safety_temporal_coverage",
                "analysis": analysis,
                "image": frame
            })

        cap.release()

        selected.sort(
            key=lambda x:
                x["frame"]
        )

    return selected


# ============================================================
# SAVE SELECTED FRAMES
# ============================================================

def save_selected_frames(
    selected,
    output_dir
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("SAVING SELECTED FRAMES")
    print("=" * 70)

    metadata = []

    for index, item in enumerate(
        selected,
        start=1
    ):

        frame = item["image"]

        filename = (
            f"frame_{index:04d}_"
            f"src_{item['frame']:06d}.jpg"
        )

        filepath = (
            output_dir /
            filename
        )

        # High JPEG quality.
        cv2.imwrite(
            str(filepath),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                97
            ]
        )

        analysis = item[
            "analysis"
        ]

        metadata.append({

            "output_index":
                index,

            "filename":
                filename,

            "source_frame":
                int(item["frame"]),

            "time_seconds":
                float(item["time"]),

            "selection_score":
                float(item["score"]),

            "selection_reason":
                item["reason"],

            "feature_count":
                int(
                    analysis["feature_count"]
                ),

            "feature_coverage":
                float(
                    analysis["feature_coverage"]
                ),

            "sharpness":
                float(
                    analysis["sharpness"]
                ),

            "sift_matches":
                int(
                    analysis["matches"]
                ),

            "sift_inliers":
                int(
                    analysis["inliers"]
                ),

            "inlier_ratio":
                float(
                    analysis["inlier_ratio"]
                ),

            "estimated_overlap":
                float(
                    analysis["overlap"]
                ),

            "optical_flow":
                float(
                    analysis["flow_motion"]
                ),

            "flow_std":
                float(
                    analysis["flow_std"]
                ),

            "translation_proxy":
                float(
                    analysis["translation"]
                ),

            "rotation_degrees":
                float(
                    analysis["rotation"]
                ),

            "parallax_proxy":
                float(
                    analysis["parallax"]
                ),

            "moving_object_score":
                float(
                    analysis["moving_object"]
                ),

            "pure_rotation_score":
                float(
                    analysis["pure_rotation"]
                ),

            "exposure_change":
                float(
                    analysis["exposure_change"]
                ),

            "phash_distance_from_previous":
                float(
                    analysis.get(
                        "phash_distance",
                        0.0
                    )
                )
        })

        print(
            f"  Saved {filename}"
        )

    return metadata


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    selected,
    output_path,
    thumb_width=320,
    columns=4
):

    if not selected:
        return

    thumbnails = []

    for item in selected:

        image = item["image"]

        h, w = image.shape[:2]

        scale = (
            thumb_width /
            float(w)
        )

        thumb_height = int(
            h * scale
        )

        thumb = cv2.resize(
            image,
            (
                thumb_width,
                thumb_height
            ),
            interpolation=cv2.INTER_AREA
        )

        # Add text information.
        canvas = thumb.copy()

        text1 = (
            f"#{item.get('frame', 0)}"
        )

        text2 = (
            f"{item['time']:.2f}s "
            f"S:{item['score']:.2f}"
        )

        cv2.putText(
            canvas,
            text1,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            text2,
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        thumbnails.append(
            canvas
        )

    rows = math.ceil(
        len(thumbnails) /
        columns
    )

    thumb_h = max(
        x.shape[0]
        for x in thumbnails
    )

    sheet = np.zeros(
        (
            rows * thumb_h,
            columns * thumb_width,
            3
        ),
        dtype=np.uint8
    )

    for i, thumb in enumerate(
        thumbnails
    ):

        row = i // columns
        col = i % columns

        y = row * thumb_h
        x = col * thumb_width

        sheet[
            y:y + thumb.shape[0],
            x:x + thumb.shape[1]
        ] = thumb

    cv2.imwrite(
        str(output_path),
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )


# ============================================================
# JSON SERIALIZATION
# ============================================================

def save_metadata(
    output_dir,
    video_info,
    motion_stats,
    feature_stats,
    metadata
):

    output_path = (
        Path(output_dir) /
        "metadata.json"
    )

    data = {

        "video": video_info,

        "analysis": {

            "motion_statistics":
                motion_stats,

            "feature_statistics":
                feature_stats,

            "target_overlap":
                [
                    TARGET_OVERLAP_LOW,
                    TARGET_OVERLAP_HIGH
                ],

            "analysis_resolution":
                ANALYSIS_WIDTH,

            "selection_method":
                "adaptive_sfm_selection"
        },

        "selected_frames":
            metadata
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive SfM / photogrammetry "
            "video frame selector"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output folder"
    )

    parser.add_argument(
        "--sample-min",
        type=int,
        default=3,
        help=(
            "Minimum source-frame sampling "
            "stride during fast movement"
        )
    )

    parser.add_argument(
        "--sample-max",
        type=int,
        default=10,
        help=(
            "Maximum source-frame sampling "
            "stride during slow movement"
        )
    )

    parser.add_argument(
        "--target-overlap",
        type=float,
        default=0.70,
        help=(
            "Desired overlap center. "
            "Default=0.70"
        )
    )

    parser.add_argument(
        "--scan-step",
        type=int,
        default=2,
        help=(
            "Pass-1 motion scan step. "
            "Default=2"
        )
    )

    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=MAX_GAP_SECONDS,
        help=(
            "Maximum allowed time between "
            "selected frames"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    input_path = Path(
        args.input
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input video does not exist:\n"
            f"{input_path}"
        )

    if args.sample_min < 1:
        args.sample_min = 1

    if args.sample_max < args.sample_min:
        args.sample_max = args.sample_min

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    video_info = get_video_info(
        input_path
    )

    fps = video_info[
        "fps"
    ]

    frame_count = video_info[
        "frame_count"
    ]

    width = video_info[
        "width"
    ]

    height = video_info[
        "height"
    ]

    print()
    print("=" * 70)
    print("ADAPTIVE SfM FRAME SELECTOR")
    print("=" * 70)

    print(
        f"Input       : {input_path}"
    )

    print(
        f"Resolution  : {width} x {height}"
    )

    print(
        f"FPS         : {fps:.2f}"
    )

    print(
        f"Frames      : {frame_count}"
    )

    print(
        f"Duration    : "
        f"{video_info['duration']:.2f} sec"
    )

    print(
        f"Sampling    : "
        f"{args.sample_min} - "
        f"{args.sample_max} frames"
    )

    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------

    motion_samples = motion_scan(
        input_path,
        fps,
        frame_count,
        args.scan_step
    )

    motion_stats = compute_motion_statistics(
        motion_samples
    )

    # --------------------------------------------------------
    # Adaptive candidate generation
    # --------------------------------------------------------

    candidates, adaptive_stats = (
        build_adaptive_candidates(
            motion_samples,
            frame_count,
            fps,
            args.sample_min,
            args.sample_max
        )
    )

    print()
    print(
        f"Adaptive candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Motion P20: "
        f"{motion_stats['motion_p20']:.2f} px"
    )

    print(
        f"Motion P50: "
        f"{motion_stats['motion_p50']:.2f} px"
    )

    print(
        f"Motion P80: "
        f"{motion_stats['motion_p80']:.2f} px"
    )

    # --------------------------------------------------------
    # Pre-scan SIFT features on candidates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PRE-SCAN — SIFT FEATURE STATISTICS")
    print("=" * 70)

    cap = cv2.VideoCapture(
        str(input_path)
    )

    sift = create_sift()

    feature_counts = []
    blur_values = []

    # Limit expensive statistics to candidates.
    for i, frame_number in enumerate(
        candidates
    ):

        frame = read_frame(
            cap,
            frame_number
        )

        if frame is None:
            continue

        small = resize_for_analysis(
            frame
        )

        gray = grayscale(
            small
        )

        kp, des = detect_sift(
            small,
            sift
        )

        count = (
            len(kp)
            if kp is not None
            else 0
        )

        feature_counts.append(
            count
        )

        blur_values.append(
            laplacian_variance(
                gray
            )
        )

        if (
            i + 1
        ) % 25 == 0:

            print(
                f"  Feature scan: "
                f"{i + 1}/{len(candidates)}"
            )

    cap.release()

    feature_stats = compute_feature_statistics(
        feature_counts
    )

    feature_stats["blur_p20"] = (
        robust_percentile(
            blur_values,
            20,
            50
        )
    )

    feature_stats["blur_p50"] = (
        robust_percentile(
            blur_values,
            50,
            100
        )
    )

    feature_stats["blur_p80"] = (
        robust_percentile(
            blur_values,
            80,
            200
        )
    )

    print()
    print(
        "Feature statistics:"
    )

    print(
        f"  P20 = "
        f"{feature_stats['feature_p20']:.0f}"
    )

    print(
        f"  P50 = "
        f"{feature_stats['feature_p50']:.0f}"
    )

    print(
        f"  P80 = "
        f"{feature_stats['feature_p80']:.0f}"
    )

    print(
        "Blur statistics:"
    )

    print(
        f"  P20 = "
        f"{feature_stats['blur_p20']:.1f}"
    )

    print(
        f"  P50 = "
        f"{feature_stats['blur_p50']:.1f}"
    )

    print(
        f"  P80 = "
        f"{feature_stats['blur_p80']:.1f}"
    )

    # --------------------------------------------------------
    # PASS 2
    # --------------------------------------------------------

    max_gap_frames = max(
        1,
        int(
            args.max_gap_seconds *
            fps
        )
    )

    selected = select_frames(
        input_path,
        candidates,
        fps,
        frame_count,
        width,
        height,
        feature_stats,
        motion_stats,
        max_gap_frames,
        args.target_overlap
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = save_selected_frames(
        selected,
        output_dir
    )

    metadata_path = save_metadata(
        output_dir,
        video_info,
        motion_stats,
        feature_stats,
        metadata
    )

    contact_sheet_path = (
        output_dir /
        "selected_contact_sheet.jpg"
    )

    create_contact_sheet(
        selected,
        contact_sheet_path
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Input frames       : "
        f"{frame_count}"
    )

    print(
        f"Initial candidates : "
        f"{len(candidates)}"
    )

    print(
        f"Selected frames    : "
        f"{len(selected)}"
    )

    reduction = (
        100 *
        (
            1 -
            len(selected) /
            max(1, frame_count)
        )
    )

    print(
        f"Frame reduction    : "
        f"{reduction:.1f}%"
    )

    print()
    print(
        f"Frames saved to    : "
        f"{output_dir}"
    )

    print(
        f"Metadata            : "
        f"{metadata_path}"
    )

    print(
        f"Contact sheet       : "
        f"{contact_sheet_path}"
    )

    print()
    print(
        "Frame selection complete."
    )


if __name__ == "__main__":
    main()