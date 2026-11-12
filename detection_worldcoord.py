"""
detection_worldcoord.py
========================================================================
GREEN WEED DETECTION  +  PIXEL -> WORLD COORDINATE CALCULATION

This script is meant to run as its own process on the Jetson Orin. It:

    1. Receives frames + already-detected CROP plant boxes from the
       DeepStream pipeline (deepstream_pipeline.py), which runs the YOLO
       crop-detector as a TensorRT engine via nvinfer. This module does
       NOT run YOLO itself anymore -- DeepStream is the only inference
       backend, on-device and in development alike.
    2. Receives the binary weed mask produced by the DeepStream CUDA ExG
       custom library. No PyTorch, kornia, or Python GPU tensor runtime is
       attached to this process.
    6. Converts every weed bbox centre from PIXEL coordinates to REAL
       WORLD coordinates (relative to the camera) using the camera
       intrinsic matrix K and extrinsic matrix Rt.
    7. Publishes each frame's detections (id, x, y, z, timestamp) to a
       shared JSON-lines file so the sprayer script can pick them up.
    8. Shows a live preview with trackbar sliders for every threshold
       (ExG threshold, minimum weed area, cluster-splitting area, and
       separation aggressiveness) so you can visually tune them while
       the video/camera is running. Press 'p' to freeze the current
       frame while you drag sliders, 'q' to quit.

--------------------------------------------------------------------------
ASSUMPTIONS / PLACEHOLDERS  (all clearly marked, edit to match your setup)
--------------------------------------------------------------------------
  * CAMERA INTRINSICS (K) and EXTRINSICS (Rt) below are the values taken
    from your calibration notebook. Replace with your actual calibration
    for the camera you mount on the rover.
  * VIDEO_SOURCE is a placeholder -- point it at your real file, or set
    it to 0 (or a sensor id) to use a live Jetson CSI camera. It's
    consumed by deepstream_pipeline.DeepStreamCamera, not cv2.VideoCapture.
  * Bounding-box extraction uses OpenCV connected-components on the binary
    weed mask returned by the DeepStream CUDA template. No PyTorch or kornia
    package is imported or required.
  * Inter-process communication with the sprayer script is done through
    a simple JSON-lines file (DETECTIONS_FILE). This is the easiest thing
    to get working first; swap it for a socket, ROS topic, or shared
    memory queue later without touching the detection logic.
  * The world-coordinate frame is CAMERA-CENTRIC (X right, Y up, Z
    forward on the ground plane), exactly as in your notebook. The
    sprayer script is responsible for turning this into rover/global
    coordinates using odometry.
========================================================================
"""

import os
import cv2
import json
import time
import numpy as np

# ------------------------------------------------------------------
# 1. CONFIG / PLACEHOLDERS -- edit these for your setup
# ------------------------------------------------------------------
VIDEO_SOURCE = 0                            # Default live CSI camera sensor 0
DETECTIONS_FILE = "weed_detections.jsonl"   # shared file the sprayer script reads
EXG_THRESHOLD = 20                          # vegetation threshold for ExG mask
MIN_CONTOUR_AREA = 3000                     # minimum weed contour area (pixels^2)
MAX_WEED_BOX_AREA = 100000                  # ignore final boxes larger than this (pixels^2)

# --- Cluster-splitting thresholds -------------------------------------
CLUSTER_AREA_THRESHOLD = 30000              # split blobs larger than this (pixels^2)
SEPARATION_KERNEL_PX = 9
MAX_WEED_DETECTIONS = 200            # bounds downstream work on dense vegetation frames
MAX_WATERSHED_SEEDS = 80             # avoids pathological segmentation on dense grass
MAX_OVERLAP_PASSES = 20              # safety bound for clipping-based overlap handling

SHOW_PREVIEW = False                  # GUI owns preview; set True only for legacy OpenCV preview

# Physical mounting measurement: optical centre to ground. The tractor lift
# raises the complete rig, so it is part of the ground projection.
BASE_CAMERA_HEIGHT_M = 0.42
TRACTOR_LIFT_HEIGHT_M = 0.22
CAMERA_HEIGHT_M = BASE_CAMERA_HEIGHT_M + TRACTOR_LIFT_HEIGHT_M

# ---------------------------------------------------------------------------
# STATIC_CAMERA_MODE  (set False when mounting on a moving tractor)
# ---------------------------------------------------------------------------
# When True (bench-test / fixed mount):
#   • Gaussian blur applied before ExG to kill per-pixel sensor noise
#   • Morphological open+close stabilises mask edges
#   • EMA temporal accumulator requires a pixel to be green across several
#     frames -- makes bounding-boxes rock-solid on a still image
# When False (tractor moving):
#   • Raw per-frame ExG, no blur, no morph, no EMA
#   • Every frame is detected independently so rapidly-changing scenes are
#     tracked without lag or ghost boxes from the previous position
STATIC_CAMERA_MODE = True

# ---------------------------------------------------------------------------
# TOP-DOWN (NADIR) CAMERA GEOMETRY
# ---------------------------------------------------------------------------
# With the camera mounted facing straight down at a fixed height (CAMERA_HEIGHT_M),
# extrinsic calibration matrices (R, t) are no longer needed. The relationship
# between pixel coordinates (u, v) and real-world ground coordinates (X, Y)
# is calculated directly via top-down pinhole projection:
#
#   X (lateral, right-positive) = (u - cx) * CAMERA_HEIGHT_M / fx
#   Y (forward, forward-positive) = (v - cy) * CAMERA_HEIGHT_M / fy
#
# Camera coordinate mapping:
CAMERA_XY_SWAP = False   # Set True only if camera is mounted rotated 90 degrees
CAMERA_X_FLIP  = True    # True: u > cx maps to Left motor (X < 0), u < cx maps to Right motor (X > 0)
CAMERA_Y_FLIP  = False   # Set True if forward/back is reversed


# --- Camera intrinsic matrix K (fx, fy, cx, cy) & Distortion Coefficients -
# Dynamically loaded from camera_calibration.npz (or camera_calibration/ folder),
# created by calibrate.py. Falls back to hardcoded defaults if missing.
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calibration.npz")
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calibration")

_DEFAULT_K = np.array([
    [2318.75030, 0.0, 1148.28753],
    [0.0, 2318.66170, 605.611832],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

_DEFAULT_DIST = np.array([
    -1.34575055e-01, 1.56354933e+00, 6.60235367e-03, 2.75444677e-03, -4.59716866e+00
], dtype=np.float64)


def load_camera_intrinsics(npz_path=CALIBRATION_FILE, dir_path=CALIBRATION_DIR):
    """Load camera matrix K and distortion coefficients from npz or directory, falling back to defaults."""
    if os.path.isfile(npz_path):
        try:
            with np.load(npz_path) as data:
                k = data.get('K') if data.get('K') is not None else data.get('camera_matrix')
                d = data.get('dist') if data.get('dist') is not None else data.get('dist_coeffs')
                if k is not None and d is not None:
                    return np.array(k, dtype=np.float64), np.array(d, dtype=np.float64).flatten()
        except Exception as exc:
            print(f"[CALIB][WARN] Failed to load {npz_path}: {exc}")

    k_file = os.path.join(dir_path, "K.npy")
    dist_file = os.path.join(dir_path, "dist.npy")
    if os.path.isfile(k_file) and os.path.isfile(dist_file):
        try:
            k = np.load(k_file)
            d = np.load(dist_file)
            return np.array(k, dtype=np.float64), np.array(d, dtype=np.float64).flatten()
        except Exception as exc:
            print(f"[CALIB][WARN] Failed to load from {dir_path}: {exc}")

    return _DEFAULT_K.copy(), _DEFAULT_DIST.copy()


K, DIST_COEFFS = load_camera_intrinsics()


def compute_ground_homography(K_mat, height_m=CAMERA_HEIGHT_M):
    """Compute direct top-down ground projection homography:
      When CAMERA_X_FLIP=True:
        u > cx -> X < 0 (Left motor)
        u < cx -> X > 0 (Right motor)
      Y = (v - cy) * height / fy
    """
    fx = float(K_mat[0, 0])
    fy = float(K_mat[1, 1])
    cx = float(K_mat[0, 2])
    cy = float(K_mat[1, 2])
    h = float(height_m)
    sx = -1.0 if CAMERA_X_FLIP else 1.0
    sy = -1.0 if CAMERA_Y_FLIP else 1.0
    return np.array([
        [sx * h / fx, 0.0,         -sx * h * cx / fx],
        [0.0,         sy * h / fy, -sy * h * cy / fy],
        [0.0,         0.0,          1.0             ],
    ], dtype=np.float32)


# Precompute ground-plane homography matrix H_P2W from intrinsics + height
H_P2W = compute_ground_homography(K, CAMERA_HEIGHT_M)

# CuPy GPU array acceleration (no PyTorch required) with seamless vectorized NumPy fallback
try:
    import cupy as cp
    H_GPU = cp.asarray(H_P2W, dtype=cp.float32)
    USE_CUPY = True
except ImportError:
    import numpy as cp
    H_GPU = np.asarray(H_P2W, dtype=np.float32)
    USE_CUPY = False


def set_camera_height(height_m):
    """Update camera height and recompute the ground projection matrix."""
    global CAMERA_HEIGHT_M, H_P2W, H_GPU
    CAMERA_HEIGHT_M = float(height_m)
    H_P2W = compute_ground_homography(K, CAMERA_HEIGHT_M)
    H_GPU = cp.asarray(H_P2W, dtype=cp.float32) if USE_CUPY else np.asarray(H_P2W, dtype=np.float32)


def set_tractor_lift_height(height_m):
    """Set the common camera/nozzle lift in metres and rebuild projection."""
    global TRACTOR_LIFT_HEIGHT_M
    height_m = float(height_m)
    if height_m < 0:
        raise ValueError("Tractor lift height cannot be negative")
    TRACTOR_LIFT_HEIGHT_M = height_m
    set_camera_height(BASE_CAMERA_HEIGHT_M + TRACTOR_LIFT_HEIGHT_M)


# --- Principal point alignment -------------------------------------------
AUTO_CENTER_PRINCIPAL_POINT = False
_principal_point_aligned = False


def align_principal_point(frame_width, frame_height):
    """Re-center K's principal point on real frame size and recompute projection."""
    global K, H_P2W, H_GPU, _principal_point_aligned
    if not AUTO_CENTER_PRINCIPAL_POINT or _principal_point_aligned:
        return
    K[0, 2] = frame_width / 2.0
    K[1, 2] = frame_height / 2.0
    H_P2W = compute_ground_homography(K, CAMERA_HEIGHT_M)
    H_GPU = cp.asarray(H_P2W, dtype=cp.float32) if USE_CUPY else np.asarray(H_P2W, dtype=np.float32)
    _principal_point_aligned = True


def batch_world_coords(centroids_px):
    """Batch-compute ground plane world coordinates (meters) on GPU/NumPy without PyTorch.

    centroids_px: Nx2 array-like of (u, v) pixel coordinates.
    Returns: Nx2 numpy array of (X, Y) world coordinates (camera-centric, meters).
    """
    if len(centroids_px) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = cp.asarray(centroids_px, dtype=cp.float32)
    ones = cp.ones((pts.shape[0], 1), dtype=cp.float32)
    homog = cp.concatenate([pts, ones], axis=1)          # Nx3
    world = homog @ H_GPU.T                               # Nx3
    world = world[:, :2] / world[:, 2:3]                  # normalize homogeneous coordinates
    if USE_CUPY:
        return cp.asnumpy(world)
    return np.asarray(world)


# ------------------------------------------------------------------
# 2. PIXEL -> WORLD COORDINATE TRANSFORM (scalar backward compatibility)
# ------------------------------------------------------------------
def pixel_to_world(u, v, K=None, height_m=None, *args, **kwargs):
    """Back-project a pixel (u, v) onto the ground plane using top-down geometry.
    Returns [X, Y, 0.0] in the camera-centric world frame (meters).
    """
    if K is None:
        K = globals()['K']
    if height_m is None:
        height_m = CAMERA_HEIGHT_M
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    sx = -1.0 if CAMERA_X_FLIP else 1.0
    sy = -1.0 if CAMERA_Y_FLIP else 1.0
    X = sx * (u - cx) * (height_m / fx)
    Y = sy * (v - cy) * (height_m / fy)
    if CAMERA_XY_SWAP:
        X, Y = Y, X
    return np.array([X, Y, 0.0], dtype=np.float32)


# ------------------------------------------------------------------
# 3. GREEN (ExG) VEGETATION SEGMENTATION
# ------------------------------------------------------------------
def plant_bounding_box(plants, mask):
    """
    Build a filled mask + convex hull covering all detected CROP plant
    boxes, so it can be subtracted from the vegetation mask.
    `plants` is a list of [x1, y1, x2, y2].
    """
    if plants is None or len(plants) == 0:
        return None, None, np.zeros_like(mask)

    plant_mask = np.zeros_like(mask)
    for (x1, y1, x2, y2) in plants:
        cv2.rectangle(plant_mask, (x1, y1), (x2, y2), 255, -1)

    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, plant_mask

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)

    all_x1 = [p[0] for p in plants]
    all_y1 = [p[1] for p in plants]
    all_x2 = [p[2] for p in plants]
    all_y2 = [p[3] for p in plants]
    merged = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))
    return merged, hull, plant_mask


def find_weed_boxes(mask, min_area=MIN_CONTOUR_AREA,
                     cluster_area_threshold=CLUSTER_AREA_THRESHOLD,
                     separation_px=SEPARATION_KERNEL_PX):
    """Extract weed boxes with OpenCV from the CUDA-produced binary mask."""
    return _find_weed_boxes_cpu(mask, min_area, cluster_area_threshold, separation_px)


def _find_weed_boxes_cpu(mask, min_area, cluster_area_threshold, separation_px):
    """
    OpenCV connected-components implementation for the CUDA-produced mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []

    for label in range(1, num_labels):     # label 0 is the background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        if area <= cluster_area_threshold:
            boxes.append((x, y, x + w, y + h))
            continue

        # Big blob -> likely several weeds merged together. Crop just
        # this component and split it.
        component_mask = np.where(labels[y:y + h, x:x + w] == label, 255, 0).astype(np.uint8)
        sub_boxes = split_cluster(component_mask, separation_px, min_area)
        for (sx1, sy1, sx2, sy2) in sub_boxes:
            boxes.append((x + sx1, y + sy1, x + sx2, y + sy2))

    return boxes


def split_cluster(component_mask, separation_px, min_area):
    """
    Split ONE large connected green blob into several individual weed
    bounding boxes using distance-transform local maxima + watershed.

    This works even when weeds visually OVERLAP (not just touch at a
    thin neck): the distance transform peaks near the middle of each
    individual weed, and each peak becomes a separate "seed" that
    watershed then grows outward, drawing the split line where two
    weeds meet.

    `separation_px` controls how far apart two peaks must be to count
    as two different weeds (it's the size of the local-maxima search
    window): SMALLER -> splits more aggressively (more, smaller weeds
    found); LARGER -> merges nearby peaks together (less splitting,
    closer to the old "one big bbox" behaviour).
    """
    separation_px = max(1, int(separation_px))
    if separation_px % 2 == 0:
        separation_px += 1  # odd kernel sizes behave better for morphology

    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        x, y, w, h = cv2.boundingRect(component_mask)
        return [(x, y, x + w, y + h)]

    # Find local maxima of the distance transform: a pixel is a peak if
    # it equals the max of its neighbourhood (size = separation_px) and
    # isn't just background noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (separation_px, separation_px))
    dilated = cv2.dilate(dist, kernel)
    local_max = np.logical_and(dist == dilated, dist > 0.15 * dist.max())
    seed_mask = local_max.astype(np.uint8) * 255

    num_seeds, seed_labels = cv2.connectedComponents(seed_mask)
    if num_seeds <= 2:  # only background + one seed -> nothing to split
        x, y, w, h = cv2.boundingRect(component_mask)
        return [(x, y, x + w, y + h)]
    if num_seeds - 1 > MAX_WATERSHED_SEEDS:
        # Dense grass can produce thousands of local maxima. Processing every
        # marker would block the live feed. It is not a reliable individual
        # weed detection, so suppress it rather than drawing one huge box.
        return []

    # Watershed: grow each seed back out to fill the original blob, and
    # let OpenCV draw the dividing line between touching/overlapping weeds.
    unknown = cv2.subtract(component_mask, seed_mask)
    markers = seed_labels + 1
    markers[unknown == 255] = 0

    color_mask = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_mask, markers)

    boxes = []
    for lbl in range(2, markers.max() + 1):
        region = np.where(markers == lbl, 255, 0).astype(np.uint8)
        if cv2.countNonZero(region) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(region)
        boxes.append((x, y, x + w, y + h))

    if not boxes:   # splitting produced nothing usable -> fall back to one box
        x, y, w, h = cv2.boundingRect(component_mask)
        boxes = [(x, y, x + w, y + h)]

    return boxes


def remove_boxes_inside_hull(hull, boxes):
    """Keep only weed boxes that are completely outside the crop-plant hull."""
    if hull is None:
        return boxes

    hull = np.asarray(hull, dtype=np.float32).reshape(-1, 2)
    hull_contour = hull.reshape(-1, 1, 2)
    kept = []
    for (x1, y1, x2, y2) in boxes:
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # A box is valid only when it neither overlaps nor touches the chilli
        # plant polygon. Checking both polygon directions catches the case
        # where the hull lies entirely inside a large candidate box.
        box_corner_in_hull = any(
            cv2.pointPolygonTest(hull_contour, tuple(point), False) >= 0
            for point in box
        )
        hull_point_in_box = np.any(
            (hull[:, 0] >= x1) & (hull[:, 0] <= x2) &
            (hull[:, 1] >= y1) & (hull[:, 1] <= y2)
        )
        overlap_area, _ = cv2.intersectConvexConvex(hull_contour, box.reshape(-1, 1, 2))
        if not box_corner_in_hull and not hull_point_in_box and overlap_area <= 0:
            kept.append((x1, y1, x2, y2))
    return kept


def box_overlaps_hull(hull, box):
    """Return True when a candidate box touches or overlaps a plant hull."""
    return len(remove_boxes_inside_hull(hull, [box])) == 0


def remove_overlaps(boxes, min_area=MIN_CONTOUR_AREA,
                    max_passes=MAX_OVERLAP_PASSES):
    """Clip overlapping boxes using the original overlap-handler strategy.

    The original code is retained, with a pass limit and a no-change guard to
    prevent one problematic frame from looping indefinitely.
    """
    def intersection(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix1 < ix2 and iy1 < iy2:
            return ix1, iy1, ix2, iy2
        return None

    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def clip_bbox(small, large):
        ix1, iy1, ix2, iy2 = intersection(small, large)
        sx1, sy1, sx2, sy2 = small
        candidates = []
        if ix1 > sx1:
            candidates.append((sx1, sy1, ix1, sy2))
        if ix2 < sx2:
            candidates.append((ix2, sy1, sx2, sy2))
        if iy1 > sy1:
            candidates.append((sx1, sy1, sx2, iy1))
        if iy2 < sy2:
            candidates.append((sx1, iy2, sx2, sy2))
        return max(candidates, key=area) if candidates else None

    boxes = sorted(boxes, key=area, reverse=True)[:MAX_WEED_DETECTIONS]
    for _ in range(max_passes):
        changed = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            for j in range(len(boxes)):
                if i == j or used[j]:
                    continue
                if intersection(boxes[i], boxes[j]):
                    if area(boxes[i]) >= area(boxes[j]):
                        clipped = clip_bbox(boxes[j], boxes[i])
                        used[j] = True
                    else:
                        clipped = clip_bbox(boxes[i], boxes[j])
                        used[i] = True
                    if clipped and area(clipped) >= min_area:
                        new_boxes.append(clipped)
                    changed = True
                    break
            if not used[i]:
                new_boxes.append(boxes[i])
        if not changed or new_boxes == boxes:
            return new_boxes
        boxes = new_boxes
    return boxes


def draw_detections(image, boxes, color=(0, 0, 255)):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


# ------------------------------------------------------------------
# 4. FULL PIPELINE FOR ONE FRAME
# ------------------------------------------------------------------
_next_id = 0

# Temporal mask smoothing via Exponential Moving Average (EMA).
# On a static image, per-frame noise causes individual pixels to flip
# on/off each frame → contour edges jitter → detection boxes flicker.
# EMA accumulates a float (0..255) for each pixel across frames:
#   acc = alpha * new_mask + (1 - alpha) * acc
# Only pixels that are consistently green (acc > 127) survive the final
# threshold, effectively requiring a pixel to be "on" in the majority of
# recent frames.  alpha=0.35 gives ~3-4 frames of settling time.
_mask_accumulator: np.ndarray = None   # float32, same H×W as frame
_MASK_EMA_ALPHA = 0.35                 # blend weight for the current frame


def reset_detection_ids():
    """Reset the display/command id counter when a new GUI stream starts."""
    global _mask_accumulator, _next_id
    _mask_accumulator = None
    _next_id = 0


def exg_mask_from_bgr(frame, threshold):
    """Build the binary ExG mask from untouched BGR pixels.

    This is used by the raw-preview GUI: it keeps the camera image visible
    while still supplying the exact ``2G - R - B > threshold`` mask needed
    for red weed bounding boxes.

    In STATIC_CAMERA_MODE a 5×5 Gaussian blur is applied BEFORE thresholding
    to suppress per-pixel sensor noise that makes mask edges flicker on still
    images.  In tractor mode (STATIC_CAMERA_MODE=False) the blur is skipped
    so every frame is detected independently without temporal lag.
    """
    src = cv2.GaussianBlur(frame, (5, 5), 0) if STATIC_CAMERA_MODE else frame
    blue, green, red = cv2.split(src)
    exg = 2 * green.astype(np.int16) - red.astype(np.int16) - blue.astype(np.int16)
    return np.where(exg > int(threshold), 255, 0).astype(np.uint8)


def process_frame(frame, plants, weed_mask=None,
                   exg_threshold=EXG_THRESHOLD,
                   min_area=MIN_CONTOUR_AREA,
                   cluster_area_threshold=CLUSTER_AREA_THRESHOLD,
                   separation_px=SEPARATION_KERNEL_PX,
                   max_box_area=MAX_WEED_BOX_AREA, plant_track_ids=None,
                   tracked_weed_boxes=None, weed_track_ids=None,
                   pipeline_compatible=False, weed_mode="yolo"):
    """
    Run the full weed-detection + world-coordinate pipeline on a single
    BGR frame, given the CROP plant boxes [[x1, y1, x2, y2], ...] that
    DeepStream's nvinfer stage already found for this frame.

    This is the only detection entry point now -- DeepStream is the sole
    inference backend (see deepstream_pipeline.DeepStreamCamera), so
    there's no separate Python YOLO path to choose between here anymore.
    The ExG mask and bounding-box extraction inside this
    function receives NvSORT-tracked weed boxes from DeepStream in the live
    path. ExG thresholding, chilli-box suppression, and connected-component
    extraction have already run before nvtracker. When tracked_weed_boxes is
    omitted, the OpenCV fallback below detects boxes locally.

    Area and cluster limits apply in the OpenCV fallback. For the native live
    path, they are configured at camera construction and the resulting
    NvSORT-tracked boxes are passed in through tracked_weed_boxes.

    Returns (annotated_frame, list_of_world_detections, mask) where each
    detection is {"id", "x", "y", "z", "pixel": (u, v)}.
    """
    if weed_mode not in ("yolo", "gi"):
        raise ValueError("weed_mode must be 'yolo' or 'gi'")
    # Normal operation consumes boxes from the second DeepStream GIE.  GI
    # keeps the earlier green-index/ExG path for comparison and tuning.
    native_weed_tracking = weed_mode == "yolo" and tracked_weed_boxes is not None
    if native_weed_tracking:
        # In the live DeepStream path, ExG and box extraction already ran
        # upstream. Return a box-coverage mask for the existing UI statistic.
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in tracked_weed_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    elif pipeline_compatible:
        # Match the native CUDA path: raw ExG threshold, no temporal blur/EMA.
        blue, green, red = cv2.split(frame)
        exg = 2 * green.astype(np.int16) - red.astype(np.int16) - blue.astype(np.int16)
        mask = np.where(exg > int(exg_threshold), 255, 0).astype(np.uint8)
    else:
        # Offline/fallback path when no upstream NvSORT weed objects are given.
        mask = (exg_mask_from_bgr(frame, exg_threshold) if weed_mask is None
                else weed_mask.copy())

    if STATIC_CAMERA_MODE and not native_weed_tracking and not pipeline_compatible:
        # Morphological stabilisation: open removes tiny noise islands (salt);
        # close fills small holes inside real weed blobs.
        # TRACTOR MODE: skip both (raw mask, no lag).
        _morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _morph_k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _morph_k, iterations=2)

        # Temporal EMA smoothing: pixel must be consistently green across frames.
        # TRACTOR MODE: skip (every frame detected independently, no ghost boxes).
        global _mask_accumulator
        mask_f = mask.astype(np.float32)
        if _mask_accumulator is None or _mask_accumulator.shape != mask_f.shape:
            _mask_accumulator = mask_f.copy()
        else:
            _mask_accumulator = _MASK_EMA_ALPHA * mask_f + (1.0 - _MASK_EMA_ALPHA) * _mask_accumulator
        mask = (_mask_accumulator > 127).astype(np.uint8) * 255

    # DeepStream normally supplies Python lists, but callers such as tuning
    # tools may supply NumPy arrays.  Never use ``array or []`` / ``if array``:
    # NumPy intentionally rejects that ambiguous boolean conversion.
    plants = [] if plants is None else list(plants)
    plant_ids = [] if plant_track_ids is None else list(plant_track_ids)
    plant_hull = None
    if len(plants) > 0:
        _, plant_hull, _ = plant_bounding_box(plants, mask)
        if plant_hull is not None and not native_weed_tracking:
            cv2.fillPoly(mask, [plant_hull], 0)

    if native_weed_tracking:
        weed_boxes, weed_ids = [], []
        source_ids = [] if weed_track_ids is None else list(weed_track_ids)
        for index, source_box in enumerate(tracked_weed_boxes):
            box = list(map(int, source_box))
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area < min_area or (max_box_area and max_box_area > 0
                                        and box_area > max_box_area):
                continue
            # The models share a GPU frame, then reject detections that touch
            # a chilli plant's convex hull before geometry/actuation. This is
            # more reliable than CPU image masking and adds no extra copy.
            if not box_overlaps_hull(plant_hull, box):
                weed_boxes.append(box)
                weed_ids.append(source_ids[index] if index < len(source_ids) else -1)
    else:
        weed_boxes = find_weed_boxes(mask, min_area=min_area,
                                      cluster_area_threshold=cluster_area_threshold,
                                      separation_px=separation_px)
        weed_boxes = remove_boxes_inside_hull(plant_hull, weed_boxes)
        weed_boxes = remove_overlaps(weed_boxes, min_area=min_area)
        if max_box_area and max_box_area > 0:
            weed_boxes = [
                box for box in weed_boxes
                if (box[2] - box[0]) * (box[3] - box[1]) <= max_box_area
            ]
        weed_ids = []

    global _next_id
    world_detections = []
    if len(weed_boxes) > 0:
        centroids = np.array(
            [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in weed_boxes],
            dtype=np.float32,
        )
        coords = batch_world_coords(centroids)
        for box_index, ((u, v), (X, Y)) in enumerate(zip(centroids, coords)):
            # Step 1 – XY swap: the homography's X/Y may be transposed relative
            # to the physical tractor axes because of how the ChArUco board was
            # oriented during calibration.  CAMERA_XY_SWAP=True exchanges them
            # so that image columns drive lateral (X) and rows drive forward (Y).
            if CAMERA_XY_SWAP:
                X, Y = Y, X
            wx = float(X)
            wy = float(Y)
            track_id = weed_ids[box_index] if box_index < len(weed_ids) else -1
            world_detections.append({
                "id": track_id if track_id is not None and track_id >= 0 else _next_id,
                "tracker_id": track_id if native_weed_tracking and track_id is not None and track_id >= 0 else None,
                "x": wx,
                "y": wy,
                "z": 0.0,
                "pixel": (float(u), float(v)),
                "frame_center_x": frame.shape[1] / 2.0,
            })
            if not native_weed_tracking or track_id is None or track_id < 0:
                _next_id += 1

    annotated = draw_detections(frame.copy(), weed_boxes)
    for (x1, y1, x2, y2), det in zip(weed_boxes, world_detections):
        cv2.putText(annotated, str(det["id"]), (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if len(plants) > 0:
        annotated = draw_detections(annotated, plants, color=(0, 255, 0))
        for (x1, y1, _, _), track_id in zip(plants, plant_ids):
            label = f"chilli {track_id}" if track_id >= 0 else "chilli"
            cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated, world_detections, mask


# ------------------------------------------------------------------
# 6. PUBLISH DETECTIONS TO SHARED FILE (simple IPC placeholder)
# ------------------------------------------------------------------
def publish_detections(detections, path=DETECTIONS_FILE):
    """
    Append one JSON line per frame's batch of detections, tagged with a
    timestamp. The sprayer script tails this file.

    ASSUMPTION: replace this with a proper queue / socket / ROS topic for
    a production system -- a flat file is just the simplest thing that
    works for a first integration pass.
    """
    record = {"t": time.time(), "detections": detections}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _nothing(_x):
    """No-op callback required by cv2.createTrackbar."""
    pass


def _make_trackbars():
    """Create the live-tuning window with one slider per threshold."""
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("ExG Threshold", "Controls", EXG_THRESHOLD, 100, _nothing)
    cv2.createTrackbar("Min Weed Area", "Controls", MIN_CONTOUR_AREA, 3000, _nothing)
    cv2.createTrackbar("Cluster Area Thr", "Controls", CLUSTER_AREA_THRESHOLD, 30000, _nothing)
    cv2.createTrackbar("Separation px", "Controls", SEPARATION_KERNEL_PX, 41, _nothing)


def _read_trackbars():
    exg_thr = cv2.getTrackbarPos("ExG Threshold", "Controls")
    min_area = cv2.getTrackbarPos("Min Weed Area", "Controls")
    cluster_thr = cv2.getTrackbarPos("Cluster Area Thr", "Controls")
    sep_px = cv2.getTrackbarPos("Separation px", "Controls")
    sep_px = max(1, sep_px)
    return exg_thr, min_area, cluster_thr, sep_px


# ------------------------------------------------------------------
# 5. MAIN LOOP (standalone test harness -- the GUI drives its own loop)
# ------------------------------------------------------------------
def main():
    from deepstream_pipeline import DeepStreamCamera

    cam = DeepStreamCamera(source=VIDEO_SOURCE)
    cam.start()

    if SHOW_PREVIEW:
        _make_trackbars()

    print("Detection pipeline started (DeepStream backend).")
    print("[INFO] ExG/crop exclusion: DeepStream CUDA template; boxes: OpenCV CPU.")
    print("  'q' = quit   'p' = pause/resume (freeze the frame while you tune sliders)")

    paused = False
    frame, plants = None, []

    try:
        while True:
            if not paused or frame is None:
                (new_frame, new_plants, new_plant_ids, new_weed_boxes,
                 new_weed_ids, new_mask, _fps, _sequence) = cam.get_latest_tracking(timeout=1.0)
                if new_frame is None:
                    print("End of stream / camera read failed.")
                    break
                frame, plants, weed_mask = new_frame, new_plants, new_mask
                if not _principal_point_aligned:
                    align_principal_point(frame.shape[1], frame.shape[0])

            if SHOW_PREVIEW:
                exg_thr, min_area, cluster_thr, sep_px = _read_trackbars()
            else:
                exg_thr, min_area, cluster_thr, sep_px = (
                    EXG_THRESHOLD, MIN_CONTOUR_AREA,
                    CLUSTER_AREA_THRESHOLD, SEPARATION_KERNEL_PX,
                )

            annotated, world_detections, mask = process_frame(
                frame, plants, weed_mask=weed_mask,
                plant_track_ids=new_plant_ids,
                tracked_weed_boxes=new_weed_boxes,
                weed_track_ids=new_weed_ids,
                exg_threshold=exg_thr,
                min_area=min_area,
                cluster_area_threshold=cluster_thr,
                separation_px=sep_px,
            )

            if not paused and world_detections:
                publish_detections(world_detections)
                for d in world_detections:
                    print(f"[DETECTION] id={d['id']} "
                          f"X={d['x']:.3f} Y={d['y']:.3f} Z={d['z']:.3f}")

            if SHOW_PREVIEW:
                info = (f"ExG>{exg_thr}  minArea={min_area}  "
                        f"clusterThr={cluster_thr}px2  sep={sep_px}px  "
                        f"weeds={len(world_detections)}")
                if paused:
                    info += "  [PAUSED - tune sliders, press 'p' to resume]"
                cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 0), 2)

                cv2.imshow("Weed Detection", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
