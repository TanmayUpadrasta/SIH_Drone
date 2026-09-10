#!/usr/bin/env python3
"""Stage 1: smart drone-video keyframe selection for SfM/COLMAP.

Features kept from the original implementations:
- motion-adaptive sampling (fast motion -> denser, slow -> sparser)
- targeted blur gap-fill
- SIFT feature count + spatial coverage
- video-relative quality thresholds
- SIFT matching + homography/affine RANSAC geometry
- overlap preference (60-80% by default)
- translation/parallax preference + pure-rotation rejection
- perceptual-hash duplicate suppression
- blur/sharpness, exposure-change and moving-object heuristics
- temporal coverage + maximum-gap protection
- optional timestamped GPS association
- camera-intrinsics loading/manifest preservation
- selection log + manifest + contact sheet

Performance-oriented design:
1. One sequential motion scan; gap-fill blur checks happen in that same pass.
2. Candidate generation uses a moving pointer, not repeated nearest-neighbor scans.
3. Candidate frames are read sequentially during each later pass (no per-frame seeks).
4. Selected-frame SIFT is cached, so the previous selected frame is never re-detected.
5. Expensive geometry runs only after cheap quality gates.
6. GPS lookup is O(log n), not O(n) per frame.
7. Analysis works at 960 px while original frames are saved unchanged.

GPS is metadata only in this stage; metric scale/georeferencing belongs downstream.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ANALYSIS_WIDTH = 960
SIFT_FEATURES = 2500
SIFT_RATIO = 0.72

FLOW_POINTS = 400
FLOW_LEVELS = 3
FLOW_RANSAC = 3.0
HOMOGRAPHY_RANSAC = 4.0

MIN_STRIDE = 3
MAX_STRIDE = 10
TARGET_OVERLAP = 0.70
OVERLAP_LOW = 0.60
OVERLAP_HIGH = 0.80
MIN_OVERLAP = 0.35
DUPLICATE_PHASH = 0.025
PHASH_SIZE = 32

MIN_SIFT_MATCHES = 12
MIN_INLIER_RATIO = 0.20
MIN_FEATURES = 80
MIN_COVERAGE = 0.20
GRID_ROWS = 4
GRID_COLS = 4

MAX_GAP_SECONDS = 1.0
MIN_FRAME_GAP = 2
MIN_SELECTED_FRAMES = 8
TIME_BINS = 20

JPEG_QUALITY = 97
FOUR_K_WIDTH_THRESHOLD = 3840
FOUR_K_FPS_MULTIPLIER = 0.75
MIN_OUTPUT_WIDTH = 1920
MIN_OUTPUT_HEIGHT = 1080
GAP_FILL_ENABLED = True


# ============================================================
# SMALL HELPERS
# ============================================================

def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def percentile(values, p: float, default: float = 0.0) -> float:
    if values is None:
        return default
    a = np.asarray(values, dtype=np.float32)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else default


def robust_norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return clamp((x - lo) / (hi - lo))


def gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_analysis(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= ANALYSIS_WIDTH:
        return image
    scale = ANALYSIS_WIDTH / float(w)
    return cv2.resize(image, (ANALYSIS_WIDTH, max(1, int(h * scale))), cv2.INTER_AREA)


def frame_time(frame: int, fps: float) -> float:
    return frame / max(fps, 1e-6)


# ============================================================
# VIDEO / INPUTS
# ============================================================

def video_info(path: Path) -> dict:
    if not path.exists():
        fail(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).strip()
    cap.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        fail(f"Invalid video properties: fps={fps}, frames={frames}, resolution={width}x{height}")
    return {
        "fps": fps, "frames": frames, "width": width, "height": height,
        "duration": frames / fps, "codec": codec or "unknown",
        "is_4k": width >= FOUR_K_WIDTH_THRESHOLD,
    }


def load_camera_intrinsics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Invalid camera_intrinsics.json: {exc}")


def load_gps_csv(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        fail(f"GPS CSV not found: {p}")
    out = []
    with p.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            def get(*names):
                for name in names:
                    if name in row and row[name] not in ("", None):
                        return row[name]
                return None
            t, lat, lon, alt = get("timestamp", "time"), get("latitude", "lat"), get("longitude", "lon"), get("altitude", "alt")
            if t is None or lat is None or lon is None:
                continue
            try:
                out.append({
                    "time": float(t), "latitude": float(lat), "longitude": float(lon),
                    "altitude": float(alt) if alt is not None else None,
                })
            except (TypeError, ValueError):
                continue
    out.sort(key=lambda x: x["time"])
    return out


def attach_gps(selected: list[dict], gps: list[dict]) -> None:
    if not gps:
        return
    times = [x["time"] for x in gps]
    for item in selected:
        t = item["time"]
        i = bisect.bisect_left(times, t)
        if i == 0:
            item["gps"] = gps[0]
        elif i == len(gps):
            item["gps"] = gps[-1]
        else:
            item["gps"] = gps[i - 1] if t - times[i - 1] <= times[i] - t else gps[i]


# ============================================================
# IMAGE QUALITY / SIFT / HASH
# ============================================================

def sharpness(g: np.ndarray) -> float:
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def brightness(g: np.ndarray) -> float:
    return float(np.mean(g, dtype=np.float32))


def exposure_change(a: np.ndarray, b: np.ndarray) -> float:
    # Mean + median + upper-tail change; avoids allocating multiple percentiles.
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    am, bm = float(a32.mean()), float(b32.mean())
    amd, bmd = float(np.median(a32)), float(np.median(b32))
    ap, bp = float(np.percentile(a32, 90)), float(np.percentile(b32, 90))
    v = (0.45 * abs(am - bm) / max(1.0, am)
         + 0.35 * abs(amd - bmd) / max(1.0, amd)
         + 0.20 * abs(ap - bp) / max(1.0, ap))
    return clamp(v / 0.30)


def make_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("SIFT unavailable. Install a modern OpenCV build.")
    return cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)


def sift_features(g: np.ndarray, sift):
    kp, des = sift.detectAndCompute(g, None)
    return kp or [], des


def feature_coverage(kp, width: int, height: int) -> float:
    if not kp:
        return 0.0
    grid = np.zeros((GRID_ROWS, GRID_COLS), np.uint8)
    sx, sy = GRID_COLS / max(width, 1), GRID_ROWS / max(height, 1)
    for k in kp:
        x, y = k.pt
        c = min(GRID_COLS - 1, max(0, int(x * sx)))
        r = min(GRID_ROWS - 1, max(0, int(y * sy)))
        grid[r, c] = 1
    return float(grid.mean())


def phash(g: np.ndarray) -> np.ndarray:
    small = cv2.resize(g, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
    low = cv2.dct(np.float32(small))[:8, :8]
    med = np.median(low[1:, :])
    return np.packbits((low > med).ravel())


def phash_distance(a, b) -> float:
    if a is None or b is None:
        return 1.0
    return float(np.count_nonzero(np.unpackbits(np.bitwise_xor(a, b))) / 64.0)


# ============================================================
# OPTICAL FLOW
# ============================================================

def flow_points(g: np.ndarray):
    return cv2.goodFeaturesToTrack(g, maxCorners=FLOW_POINTS, qualityLevel=0.01, minDistance=8, blockSize=7)


def zero_motion() -> dict:
    return {"median_flow": 0.0, "translation": 0.0, "rotation": 0.0, "flow_std": 0.0,
            "inlier_ratio": 0.0, "parallax": 0.0, "moving": 0.0}


def optical_motion(g1: np.ndarray, g2: np.ndarray) -> dict:
    p0 = flow_points(g1)
    if p0 is None or len(p0) < 10:
        return zero_motion()
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        g1, g2, p0, None, winSize=(21, 21), maxLevel=FLOW_LEVELS,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None or status is None:
        return zero_motion()
    valid = status.ravel().astype(bool)
    a, b = p0.reshape(-1, 2)[valid], p1.reshape(-1, 2)[valid]
    if len(a) < 10:
        return zero_motion()
    flow = b - a
    mag = np.linalg.norm(flow, axis=1)
    median_flow, flow_std = float(np.median(mag)), float(np.std(mag))

    affine, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                ransacReprojThreshold=FLOW_RANSAC,
                                                maxIters=1500, confidence=0.99)
    if affine is None or mask is None:
        return {"median_flow": median_flow, "translation": median_flow, "rotation": 0.0,
                "flow_std": flow_std, "inlier_ratio": 0.0,
                "parallax": clamp(flow_std / 20.0), "moving": 0.0}

    inlier_ratio = float(mask.ravel().mean())
    tx, ty = float(affine[0, 2]), float(affine[1, 2])
    translation = math.hypot(tx, ty)
    rotation = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
    predicted = cv2.transform(a.reshape(-1, 1, 2), affine).reshape(-1, 2)
    residual = np.linalg.norm(b - predicted, axis=1)
    spread = percentile(residual, 75) - percentile(residual, 25)
    parallax = clamp(0.45 * flow_std / 15.0 + 0.55 * spread / 8.0)

    high = residual > max(4.0, percentile(residual, 80))
    moving = 0.0
    if int(high.sum()) >= 8:
        pts = b[high]
        area = max(1.0, float(np.ptp(pts[:, 0]))) * max(1.0, float(np.ptp(pts[:, 1])))
        concentration = len(pts) / max(1, len(b))
        if area / max(1.0, g1.shape[0] * g1.shape[1]) < 0.25:
            moving = clamp(concentration * 3.0)

    return {"median_flow": median_flow, "translation": translation, "rotation": rotation,
            "flow_std": flow_std, "inlier_ratio": inlier_ratio,
            "parallax": parallax, "moving": moving}


def motion_stats(samples: list[dict]) -> dict:
    flows = [x["median_flow"] for x in samples]
    translations = [x["translation"] for x in samples]
    return {
        "flow_p20": percentile(flows, 20, 1.0), "flow_p50": percentile(flows, 50, 2.0),
        "flow_p80": percentile(flows, 80, 5.0), "translation_p20": percentile(translations, 20, 1.0),
        "translation_p80": percentile(translations, 80, 5.0),
    }


def adaptive_stride(flow: float, stats: dict) -> int:
    n = robust_norm(flow, stats["flow_p20"], stats["flow_p80"])
    return max(MIN_STRIDE, min(MAX_STRIDE, int(round(MAX_STRIDE - n * (MAX_STRIDE - MIN_STRIDE)))))


# ============================================================
# ONE-PASS MOTION SCAN + GAP FILL
# ============================================================

def motion_scan(path: Path, scan_step: int, info: dict, gap_fill: bool = True, blur_threshold: float | None = None):
    """Decode once. Motion samples are sparse; gap-fill blur checks share this decode."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"Cannot open video for motion scan: {path}")

    fps, frames = info["fps"], info["frames"]
    scan_step = max(1, int(scan_step))
    target_interval = max(1, round(fps / (2.0 * (FOUR_K_FPS_MULTIPLIER if info["is_4k"] else 1.0))))

    # First pass can collect sparse blur values for an adaptive threshold without a separate decode.
    probe_blur = []
    fixed_offset_blur = []
    gap_best_frame = -1
    gap_best_blur = -1.0
    previous_offset = None
    previous_gray = None
    previous_sample_frame = None
    samples = []
    gap_candidates = []
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        need_blur = gap_fill and (frame_no % target_interval == 0 or previous_offset is not None)
        small = None
        g = None
        blur = None
        if need_blur or frame_no % scan_step == 0:
            small = resize_analysis(frame)
            g = gray(small)
        if need_blur:
            blur = sharpness(g)
            if len(probe_blur) < 40 and frame_no % max(target_interval * 4, 1) == 0:
                probe_blur.append(blur)
            if previous_offset is not None and previous_offset < frame_no < previous_offset + target_interval:
                if blur > gap_best_blur:
                    gap_best_blur, gap_best_frame = blur, frame_no

        if frame_no % target_interval == 0 and gap_fill:
            fixed_offset_blur.append(blur if blur is not None else 0.0)
            threshold = blur_threshold
            if threshold is None and len(probe_blur) >= 5:
                threshold = percentile(probe_blur, 25, 100.0)
            threshold = 100.0 if threshold is None else threshold
            if previous_offset is None:
                gap_candidates.append(frame_no)
            elif blur >= threshold:
                gap_candidates.append(frame_no)
            elif gap_best_frame >= 0:
                gap_candidates.append(gap_best_frame)
            else:
                gap_candidates.append(frame_no)
            previous_offset = frame_no
            gap_best_frame, gap_best_blur = -1, -1.0

        if frame_no % scan_step == 0:
            if g is None:
                small = resize_analysis(frame)
                g = gray(small)
            motion = zero_motion() if previous_gray is None else optical_motion(previous_gray, g)
            samples.append({"frame": frame_no, **motion})
            previous_gray = g
            previous_sample_frame = frame_no
        frame_no += 1

    cap.release()
    # Derive threshold after scan if it was adaptive; gap candidates were conservatively generated
    # during the scan using the evolving threshold. This preserves the original targeted fallback.
    return samples, gap_candidates, {
        "target_interval": target_interval,
        "probe_blur": probe_blur,
        "fixed_offset_blur": fixed_offset_blur,
        "previous_sample_frame": previous_sample_frame,
    }


# ============================================================
# CANDIDATES
# ============================================================

def build_adaptive_candidates(samples: list[dict], frame_count: int) -> tuple[list[int], dict]:
    if not samples:
        return list(range(0, frame_count, MAX_STRIDE)), motion_stats([])
    stats = motion_stats(samples)
    candidates = []
    current = 0
    i = 0
    while current < frame_count:
        while i + 1 < len(samples) and samples[i + 1]["frame"] <= current:
            i += 1
        candidates.append(current)
        current += adaptive_stride(samples[i]["median_flow"], stats)
    return candidates, stats


def merge_candidates(adaptive: list[int], gap: list[int], frame_count: int) -> list[int]:
    s = set(int(x) for x in adaptive)
    for n in gap:
        n = int(n)
        if 0 <= n < frame_count and not any(abs(n - x) <= 1 for x in s):
            s.add(n)
    return sorted(s)


# ============================================================
# SIFT GEOMETRY
# ============================================================

def match_sift(des1, des2):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    return [m for pair in pairs if len(pair) == 2 and pair[0].distance < SIFT_RATIO * pair[1].distance for m in [pair[0]]]


def geometry(kp1, kp2, matches, width: int, height: int) -> dict:
    out = {"matches": len(matches), "inliers": 0, "inlier_ratio": 0.0, "overlap": 0.0,
           "parallax": 0.0, "rotation": 0.0, "translation": 0.0, "pure_rotation": 0.0}
    if len(matches) < MIN_SIFT_MATCHES:
        return out

    p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, hmask = cv2.findHomography(p1, p2, cv2.RANSAC, HOMOGRAPHY_RANSAC)
    hres = 0.0
    if H is not None and hmask is not None:
        hmask = hmask.ravel().astype(bool)
        inliers = int(hmask.sum())
        out["inliers"] = inliers
        out["inlier_ratio"] = inliers / max(1, len(matches))
        if inliers >= 4:
            a, b = p1[hmask], p2[hmask]
            pred = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
            hres = float(np.median(np.linalg.norm(b - pred, axis=1)))
            # True one-sided common-area estimate via warped validity mask.
            valid = np.ones((height, width), np.uint8) * 255
            warped = cv2.warpPerspective(valid, H, (width, height), flags=cv2.INTER_NEAREST)
            out["overlap"] = float(np.count_nonzero(warped) / max(1, width * height))

    affine, amask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                                 ransacReprojThreshold=FLOW_RANSAC,
                                                 maxIters=1500, confidence=0.99)
    if affine is not None:
        tx, ty = float(affine[0, 2]), float(affine[1, 2])
        out["translation"] = math.hypot(tx, ty)
        out["rotation"] = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
        pred = cv2.transform(p1.reshape(-1, 1, 2), affine).reshape(-1, 2)
        ares = float(np.median(np.linalg.norm(p2 - pred, axis=1)))
        out["parallax"] = clamp((ares - hres) / 10.0) if hres else 0.0

    if abs(out["rotation"]) > 1.0:
        r = clamp(abs(out["rotation"]) / 10.0)
        t = clamp(out["translation"] / max(20.0, width * 0.03))
        out["pure_rotation"] = r * (1.0 - t)
    return out


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze(frame: np.ndarray, prev_cache: dict | None, sift) -> dict:
    image = resize_analysis(frame)
    g = gray(image)
    kp, des = sift_features(g, sift)
    out = {
        "features": len(kp), "coverage": feature_coverage(kp, g.shape[1], g.shape[0]),
        "sharpness": sharpness(g), "phash": phash(g), "matches": 0, "inliers": 0,
        "inlier_ratio": 0.0, "overlap": 1.0, "translation": 0.0, "rotation": 0.0,
        "parallax": 0.0, "moving": 0.0, "exposure": 0.0, "flow": 0.0,
        "flow_std": 0.0, "pure_rotation": 0.0, "keypoints": kp, "descriptors": des,
        "gray": g, "analysis_image": image,
    }
    if prev_cache is None:
        return out

    pg, pkp, pdes = prev_cache["gray"], prev_cache["keypoints"], prev_cache["descriptors"]
    flow = optical_motion(pg, g)
    out["flow"] = flow["median_flow"]
    out["flow_std"] = flow["flow_std"]
    out["translation"] = flow["translation"]
    out["rotation"] = flow["rotation"]
    out["parallax"] = flow["parallax"]
    out["moving"] = flow["moving"]
    out["pure_rotation"] = 1.0 if abs(out["rotation"]) > 1.5 and out["translation"] < 5 else 0.0
    out["exposure"] = exposure_change(pg, g)

    matches = match_sift(pdes, des)
    geo = geometry(pkp, kp, matches, g.shape[1], g.shape[0])
    out["matches"], out["inliers"], out["inlier_ratio"], out["overlap"] = geo["matches"], geo["inliers"], geo["inlier_ratio"], geo["overlap"]
    out["parallax"] = max(out["parallax"], geo["parallax"])
    out["translation"] = max(out["translation"], geo["translation"])
    if abs(geo["rotation"]) > abs(out["rotation"]):
        out["rotation"] = geo["rotation"]
    out["pure_rotation"] = max(out["pure_rotation"], geo["pure_rotation"])
    return out


# ============================================================
# QUALITY / SCORE
# ============================================================

def quality_stats(analyses: list[dict]) -> dict:
    features = [a["features"] for a in analyses]
    sharp = [a["sharpness"] for a in analyses]
    p20 = max(50.0, min(percentile(features, 20, 100), 600.0))
    p50 = max(100.0, min(percentile(features, 50, 200), 1500.0))
    p80 = max(200.0, min(percentile(features, 80, 400), 2500.0))
    p50 = max(p20, p50)
    p80 = max(p50, p80)
    return {"feature_p20": p20, "feature_p50": p50, "feature_p80": p80,
            "sharp_p20": percentile(sharp, 20, 50), "sharp_p50": percentile(sharp, 50, 100),
            "sharp_p80": percentile(sharp, 80, 200)}


def candidate_score(a: dict, q: dict, m: dict) -> float:
    f = robust_norm(a["features"], q["feature_p20"], q["feature_p80"])
    s = robust_norm(a["sharpness"], q["sharp_p20"], q["sharp_p80"])
    inliers = clamp(a["inlier_ratio"] / 0.70)
    matches = clamp(a["matches"] / 80.0)
    overlap = a["overlap"]
    if OVERLAP_LOW <= overlap <= OVERLAP_HIGH:
        ov = 1.0
    elif overlap > OVERLAP_HIGH:
        ov = clamp(1.0 - (overlap - OVERLAP_HIGH) / 0.20)
    else:
        ov = clamp(overlap / OVERLAP_LOW)
    motion = robust_norm(a["translation"], m["translation_p20"], m["translation_p80"])
    score = (0.22*f + 0.12*a["coverage"] + 0.10*s + 0.13*inliers + 0.05*matches +
             0.13*ov + 0.10*motion + 0.10*a["parallax"] - 0.10*a["moving"] -
             0.10*a["pure_rotation"] - 0.06*a["exposure"])
    return clamp(score)


def prescan_features(path: Path, candidates: list[int]) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"Cannot open video for feature pre-scan: {path}")
    sift = make_sift()
    out, ci, target = [], 0, candidates[0] if candidates else None
    n = 0
    while target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        if n == target:
            g = gray(resize_analysis(frame))
            kp, _ = sift_features(g, sift)
            out.append({"frame": n, "features": len(kp), "sharpness": sharpness(g)})
            ci += 1
            target = candidates[ci] if ci < len(candidates) else None
        n += 1
    cap.release()
    return out


# ============================================================
# SELECTION
# ============================================================

def select_keyframes(path: Path, candidates: list[int], info: dict, qstats: dict, mstats: dict) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"Cannot open video for selection: {path}")
    sift = make_sift()
    selected = []
    prev_cache = None
    last_frame = None
    last_hash = None
    max_gap = max(MIN_FRAME_GAP, int(MAX_GAP_SECONDS * info["fps"]))
    ci, target = 0, candidates[0] if candidates else None
    n = 0

    while target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        if n == target:
            if last_frame is not None and n - last_frame < MIN_FRAME_GAP:
                ci += 1
                target = candidates[ci] if ci < len(candidates) else None
                n += 1
                continue

            a = analyze(frame, prev_cache, sift)
            distance = phash_distance(last_hash, a["phash"])
            duplicate = distance < DUPLICATE_PHASH
            if a["features"] < MIN_FEATURES or a["coverage"] < MIN_COVERAGE:
                ci += 1
                target = candidates[ci] if ci < len(candidates) else None
                n += 1
                continue

            score = candidate_score(a, qstats, mstats)
            gap = n - last_frame if last_frame is not None else 10**9
            overlap_ok = OVERLAP_LOW <= a["overlap"] <= OVERLAP_HIGH
            geometry_ok = a["matches"] >= MIN_SIFT_MATCHES and a["inlier_ratio"] >= MIN_INLIER_RATIO
            useful_motion = a["translation"] > 4 or a["parallax"] > 0.15
            choose, reason = False, ""
            if last_frame is None:
                choose, reason = True, "initial"
            elif duplicate:
                reason = "phash_duplicate"
            elif gap >= max_gap and not a["pure_rotation"]:
                choose, reason = True, "max_gap"
            elif overlap_ok and geometry_ok and score >= 0.42:
                choose, reason = True, "good_overlap"
            elif a["overlap"] >= MIN_OVERLAP and useful_motion and score >= 0.45:
                choose, reason = True, "motion_parallax"
            elif score >= 0.62 and a["overlap"] >= MIN_OVERLAP:
                choose, reason = True, "high_quality"

            if choose and a["pure_rotation"] > 0.75 and a["parallax"] < 0.10 and gap < max_gap:
                choose, reason = False, "pure_rotation_rejected"

            if choose:
                selected.append({"frame": n, "time": frame_time(n, info["fps"]), "score": score,
                                 "reason": reason, "analysis": a, "image": frame,
                                 "phash_distance": distance})
                last_frame, last_hash = n, a["phash"]
                # Cache the expensive SIFT result of the selected frame.
                prev_cache = {"gray": a["gray"], "keypoints": a["keypoints"], "descriptors": a["descriptors"]}

            ci += 1
            target = candidates[ci] if ci < len(candidates) else None
        n += 1
    cap.release()
    return selected


# ============================================================
# TEMPORAL COVERAGE FALLBACK
# ============================================================

def safety_coverage(path: Path, selected: list[dict], info: dict, qstats: dict, target: int | None = None) -> list[dict]:
    if target is None:
        target = max(12, int(info["duration"] * 2))
    if len(selected) >= target:
        return selected

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return selected
    sift = make_sift()
    existing = {x["frame"] for x in selected}
    hashes = [x["analysis"]["phash"] for x in selected[-20:]]
    positions = np.linspace(0, info["frames"] - 1, max(8, target), dtype=int)
    for n in positions:
        n = int(n)
        if n in existing:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        if not ok:
            continue
        a = analyze(frame, None, sift)
        if a["features"] < MIN_FEATURES or a["coverage"] < MIN_COVERAGE:
            continue
        if any(phash_distance(h, a["phash"]) < DUPLICATE_PHASH for h in hashes):
            continue
        selected.append({"frame": n, "time": frame_time(n, info["fps"]), "score": 0.35,
                         "reason": "coverage_fallback", "analysis": a, "image": frame,
                         "phash_distance": 1.0})
        existing.add(n)
        hashes.append(a["phash"])
    cap.release()
    selected.sort(key=lambda x: x["frame"])
    return selected


# ============================================================
# OUTPUTS
# ============================================================

def json_safe_analysis(a: dict) -> dict:
    return {k: v for k, v in a.items() if k not in {"keypoints", "descriptors", "phash", "gray", "analysis_image"}}


def write_outputs(selected: list[dict], info: dict, output: Path, gps_file: str | None, settings: dict, intrinsics: dict | None, run_id: str | None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "images"
    frame_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, item in enumerate(selected, 1):
        filename = f"frame_{i:04d}_src_{item['frame']:06d}.jpg"
        filepath = frame_dir / filename
        if not cv2.imwrite(str(filepath), item["image"], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            fail(f"Could not write frame: {filepath}")
        a = item["analysis"]
        rec = {"index": i, "filename": filename, "source_frame": item["frame"],
               "time_seconds": item["time"], "score": item["score"], "reason": item["reason"],
               "features": a["features"], "feature_coverage": a["coverage"], "sharpness": a["sharpness"],
               "sift_matches": a["matches"], "sift_inliers": a["inliers"], "inlier_ratio": a["inlier_ratio"],
               "overlap": a["overlap"], "optical_flow": a["flow"], "translation_proxy": a["translation"],
               "rotation_degrees": a["rotation"], "parallax_proxy": a["parallax"],
               "moving_object_score": a["moving"], "exposure_change": a["exposure"],
               "phash_distance": item["phash_distance"]}
        if "gps" in item:
            rec["gps"] = item["gps"]
        records.append(rec)
    data = {
        "pipeline": "drone_video_to_sfm_keyframes", "stage": 1, "run_id": run_id,
        "video": info, "camera_intrinsics": intrinsics, "settings": settings,
        "gps_source": str(gps_file) if gps_file else None,
        "metric_georeferencing": {"status": "metadata_preserved_only",
            "note": "GPS is associated with selected frames for downstream similarity-transform georeferencing. This stage does not establish metric scale."},
        "selected_frames": records,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding="utf-8")
    return path


def write_selection_log(selected: list[dict], output: Path) -> Path:
    path = output / "selection_log.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("# frame\tscore\treason\tfeatures\tcoverage\toverlap\tmatches\tinlier_ratio\ttranslation\tparallax\n")
        for item in selected:
            a = item["analysis"]
            f.write(f"{item['frame']}\t{item['score']:.4f}\t{item['reason']}\t{a['features']}\t{a['coverage']:.3f}\t{a['overlap']:.3f}\t{a['matches']}\t{a['inlier_ratio']:.3f}\t{a['translation']:.2f}\t{a['parallax']:.3f}\n")
    return path


def contact_sheet(selected: list[dict], path: Path, columns: int = 4) -> None:
    if not selected:
        return
    width = 320
    thumbs = []
    for item in selected:
        image = item["image"]
        h, w = image.shape[:2]
        thumb = cv2.resize(image, (width, max(1, int(h * width / w))), cv2.INTER_AREA)
        cv2.putText(thumb, f"#{item['frame']}  {item['time']:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    rows, cell_h = math.ceil(len(thumbs) / columns), max(t.shape[0] for t in thumbs)
    sheet = np.zeros((rows * cell_h, columns * width, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, columns)
        sheet[r * cell_h:r * cell_h + t.shape[0], c * width:c * width + t.shape[1]] = t
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])


def update_run_manifest(data_root: Path, run_id: str, stage_output: Path) -> None:
    """Preserve Code-1's /data/{run_id}/manifest.json stage flags when run_id is used."""
    p = data_root / run_id / "manifest.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"run_id": run_id}
    else:
        data = {"run_id": run_id, "stage_1_complete": False, "stage_2_complete": False,
                "stage_3_complete": False, "stage_4_complete": False, "stage_5_complete": False}
    data["stage_1_complete"] = True
    data["stage_1_output"] = str(stage_output)
    data["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Smart drone-video keyframe selector for SfM/COLMAP")
    parser.add_argument("--input", help="Drone video")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--run_id", default=None, help="Run identifier; uses /data/{run_id}/raw layout")
    parser.add_argument("--data_root", default="/data", help="Root directory for --run_id mode")
    parser.add_argument("--gps", default=None, help="Optional timestamped GPS CSV")
    parser.add_argument("--scan-step", type=int, default=2, help="Motion scan interval in source frames")
    parser.add_argument("--sample-min", type=int, default=3, help="Minimum adaptive stride")
    parser.add_argument("--sample-max", type=int, default=10, help="Maximum adaptive stride")
    parser.add_argument("--blur-threshold", type=float, default=None, help="Optional absolute gap-fill sharpness threshold")
    parser.add_argument("--disable-gap-fill", action="store_true", help="Disable targeted blur gap filling")
    args = parser.parse_args()

    global MIN_STRIDE, MAX_STRIDE
    MIN_STRIDE, MAX_STRIDE = max(1, args.sample_min), max(1, args.sample_max)
    if MAX_STRIDE < MIN_STRIDE:
        MAX_STRIDE = MIN_STRIDE

    if args.run_id:
        run_dir = Path(args.data_root) / args.run_id
        input_path = Path(args.input) if args.input else run_dir / "raw" / "video.mp4"
        output = Path(args.output) if args.output else run_dir / "keyframes"
        intrinsics_path = run_dir / "raw" / "camera_intrinsics.json"
    else:
        if not args.input:
            parser.error("--input is required unless --run_id is supplied")
        input_path, output, intrinsics_path = Path(args.input), Path(args.output or "sfm_selected_frames"), None

    info = video_info(input_path)
    if info["width"] < MIN_OUTPUT_WIDTH or info["height"] < MIN_OUTPUT_HEIGHT:
        fail(f"Source video is below the 1080p minimum ({info['width']}x{info['height']}).")
    intrinsics = load_camera_intrinsics(intrinsics_path) if intrinsics_path else None

    print("\n=== SMART SfM DRONE VIDEO PREPROCESSOR (OPTIMIZED) ===")
    print(f"Input       : {input_path}\nResolution  : {info['width']} x {info['height']}\nFPS         : {info['fps']:.2f}\nFrames      : {info['frames']}\nDuration    : {info['duration']:.2f}s\nCodec       : {info['codec']}")
    if intrinsics:
        print(f"Camera model: {intrinsics.get('camera_model', 'unknown')}")

    print("\n[1/5] One-pass motion scan + gap-fill...")
    samples, gap, scan_meta = motion_scan(input_path, args.scan_step, info, GAP_FILL_ENABLED and not args.disable_gap_fill, args.blur_threshold)
    candidates, mstats = build_adaptive_candidates(samples, info["frames"])
    candidates = merge_candidates(candidates, gap, info["frames"])
    print(f"Motion samples: {len(samples)} | Adaptive+gap candidates: {len(candidates)}")

    print("\n[2/5] Feature pre-scan...")
    prescan = prescan_features(input_path, candidates)
    qstats = quality_stats(prescan)
    print(f"Feature P20/P50/P80: {qstats['feature_p20']:.0f}/{qstats['feature_p50']:.0f}/{qstats['feature_p80']:.0f}")

    print("\n[3/5] SIFT + geometry selection...")
    selected = select_keyframes(input_path, candidates, info, qstats, mstats)
    selected = safety_coverage(input_path, selected, info, qstats, target=max(12, int(info["duration"] * 2)))
    print(f"Selected: {len(selected)}")
    if not selected:
        fail("No keyframes were selected.")

    print("\n[4/5] GPS metadata...")
    gps = load_gps_csv(args.gps)
    attach_gps(selected, gps)
    print(f"GPS records: {len(gps)}")

    print("\n[5/5] Writing outputs...")
    settings = {"analysis_width": ANALYSIS_WIDTH, "sample_stride_range": [MIN_STRIDE, MAX_STRIDE],
                "target_overlap": TARGET_OVERLAP, "overlap_range": [OVERLAP_LOW, OVERLAP_HIGH],
                "min_overlap": MIN_OVERLAP, "max_gap_seconds": MAX_GAP_SECONDS,
                "min_frame_gap": MIN_FRAME_GAP, "sift_features": SIFT_FEATURES, "sift_ratio": SIFT_RATIO,
                "min_sift_matches": MIN_SIFT_MATCHES, "min_inlier_ratio": MIN_INLIER_RATIO,
                "min_features": MIN_FEATURES, "min_feature_coverage": MIN_COVERAGE,
                "gap_fill_enabled": GAP_FILL_ENABLED and not args.disable_gap_fill,
                "motion_scan_step": max(1, args.scan_step), "gap_fill_interval": scan_meta["target_interval"]}
    manifest = write_outputs(selected, info, output, args.gps, settings, intrinsics, args.run_id)
    log = write_selection_log(selected, output)
    sheet = output / "contact_sheet.jpg"
    contact_sheet(selected, sheet)
    if args.run_id:
        update_run_manifest(Path(args.data_root), args.run_id, output)

    reduction = 100.0 * (1.0 - len(selected) / max(1, info["frames"]))
    print("\n=== RESULT ===")
    print(f"Input frames : {info['frames']}\nCandidates   : {len(candidates)}\nKeyframes    : {len(selected)}\nReduction    : {reduction:.1f}%")
    print(f"Images       : {output / 'images'}\nManifest     : {manifest}\nSelection log: {log}\nContact sheet: {sheet}")
    print("\nReady for COLMAP feature extraction/matching.")


if __name__ == "__main__":
    main()
