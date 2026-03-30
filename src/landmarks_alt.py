"""
Alternative landmark detection approaches for the Shroud of Turin.

MediaPipe FaceLandmarker fails on the actual Enrie photograph because
the image is too different from normal photos. This module tries:
1. OpenCV Haar cascade (simpler detector, more tolerant of unusual images)
2. Dlib HOG face detector + 68-point shape predictor
3. Extremely aggressive preprocessing to make the image more photo-like
4. Depth-map-guided landmark estimation (using brightness peaks)

If automated detection fails entirely, we provide a semi-manual approach
where the user can refine AI-estimated positions.
"""

import cv2
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEASUREMENTS_DIR = PROJECT_ROOT / "data" / "measurements"

# Key landmark indices matching the main landmarks.py format
KEY_LANDMARK_NAMES = [
    "left_eye_inner", "left_eye_outer", "right_eye_inner", "right_eye_outer",
    "left_pupil", "right_pupil",
    "nose_tip", "nose_bridge_top", "nose_left_alar", "nose_right_alar",
    "upper_lip_center", "lower_lip_center", "mouth_left", "mouth_right",
    "chin", "jaw_left", "jaw_right",
    "left_eyebrow_inner", "left_eyebrow_outer",
    "right_eyebrow_inner", "right_eyebrow_outer",
    "left_cheek", "right_cheek",
]


def try_haar_cascade(positive: np.ndarray) -> dict | None:
    """Try OpenCV Haar cascade face detection.

    Haar cascades are simpler and sometimes more tolerant of unusual images.
    We just need a face bounding box to constrain further analysis.
    """
    print("\n--- Trying Haar Cascade Face Detection ---")

    # Try multiple preprocessing levels
    test_images = []

    # Raw positive
    test_images.append(("raw", positive))

    # Heavy blur to kill cloth texture
    for k in [21, 31, 41, 51]:
        blurred = cv2.GaussianBlur(positive, (k, k), 0)
        test_images.append((f"blur_{k}", blurred))

    # Blur + histogram equalization
    for k in [31, 41]:
        blurred = cv2.GaussianBlur(positive, (k, k), 0)
        equalized = cv2.equalizeHist(blurred)
        test_images.append((f"blur_{k}_histeq", equalized))

    # Blur + CLAHE
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
    for k in [31, 41]:
        blurred = cv2.GaussianBlur(positive, (k, k), 0)
        enhanced = clahe.apply(blurred)
        test_images.append((f"blur_{k}_clahe6", enhanced))

    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("  ERROR: Could not load Haar cascade classifier")
        return None

    for name, img in test_images:
        # Try multiple scale factors and min neighbors
        for scale_factor in [1.05, 1.1, 1.2, 1.3]:
            for min_neighbors in [1, 2, 3]:
                faces = face_cascade.detectMultiScale(
                    img,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(img.shape[0] // 6, img.shape[0] // 6),
                )

                if len(faces) > 0:
                    # Take the largest detection
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    print(f"  DETECTED with {name} (scale={scale_factor}, minN={min_neighbors})")
                    print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
                    return {
                        "x": int(x), "y": int(y),
                        "w": int(w), "h": int(h),
                        "preprocessing": name,
                        "method": "haar_cascade",
                    }

    print("  No face detected with Haar cascade")
    return None


def try_dlib_detection(positive: np.ndarray) -> dict | None:
    """Try Dlib HOG face detector + shape predictor.

    Dlib's HOG-based detector uses a different approach than MediaPipe
    and may be more robust to unusual image characteristics.
    """
    print("\n--- Trying Dlib HOG Face Detection ---")

    try:
        import dlib
    except ImportError:
        print("  Dlib not installed. Install with: pip install dlib")
        print("  (Note: dlib requires CMake and C++ build tools)")
        return None

    detector = dlib.get_frontal_face_detector()

    # Try multiple preprocessing levels
    test_images = []
    test_images.append(("raw", positive))

    for k in [21, 31, 41]:
        blurred = cv2.GaussianBlur(positive, (k, k), 0)
        test_images.append((f"blur_{k}", blurred))

    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
    for k in [31, 41]:
        blurred = cv2.GaussianBlur(positive, (k, k), 0)
        enhanced = clahe.apply(blurred)
        test_images.append((f"blur_{k}_clahe", enhanced))

    for name, img in test_images:
        # Dlib can upsample for better detection
        for upsample in [0, 1, 2]:
            faces = detector(img, upsample)
            if len(faces) > 0:
                face = faces[0]
                print(f"  DETECTED with {name} (upsample={upsample})")
                print(f"  Bounding box: left={face.left()}, top={face.top()}, "
                      f"right={face.right()}, bottom={face.bottom()}")
                return {
                    "x": face.left(), "y": face.top(),
                    "w": face.right() - face.left(),
                    "h": face.bottom() - face.top(),
                    "preprocessing": name,
                    "method": "dlib_hog",
                }

    print("  No face detected with Dlib HOG")
    return None


def estimate_landmarks_from_depth(depth_map: np.ndarray, haar_bbox: dict | None = None) -> dict | None:
    """Estimate facial landmark positions using the depth map.

    The Shroud's depth map has clear peaks at anatomical features:
    - Nose tip: highest/brightest point in center
    - Cheekbones: lateral peaks
    - Eye sockets: dark depressions between nose and brow
    - Brow ridge: horizontal ridge above eyes

    This is a geometric/heuristic approach that doesn't require
    a face detector trained on photographs.
    """
    print("\n--- Estimating Landmarks from Depth Map ---")

    h, w = depth_map.shape[:2]

    # Use the CLAHE-processed depth map (same one that generates the heatmap).
    # The raw Enrie negative has cloth-edge artifacts that are brighter than
    # the face. The CLAHE depth map enhances LOCAL contrast, making the nose
    # ridge clearly visible as Michael confirmed in the heatmap review.
    depth_npy = PROCESSED_DIR / "depth_map.npy"
    if depth_npy.exists():
        depth = np.load(str(depth_npy))
        h, w = depth.shape
        print(f"  Using CLAHE depth map: {depth.shape}")
    elif len(depth_map.shape) == 3:
        depth = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        h, w = depth.shape
    else:
        depth = depth_map.copy()
        h, w = depth.shape

    # Smooth heavily to find major anatomical features (kill cloth texture)
    smoothed = cv2.GaussianBlur(depth, (61, 61), 0)

    # NOSE TIP DETECTION — Two-step approach:
    #
    # Step 1: Find the nose RIDGE (vertically sustained bright stripe).
    # The nose ridge is the single most prominent feature in the depth map —
    # a vertical bright line running down the face midline. We find it by
    # looking for columns with the highest AVERAGE brightness in the face region.
    #
    # Step 2: Find the nose TIP along the ridge — the peak brightness point
    # in the y=25-40% range (between eyes and mouth).

    # Step 1: Establish face midline x-position.
    #
    # The Enrie hires image is a portrait crop of the face — the face is
    # roughly centered horizontally. CLAHE artifacts make pure brightness-
    # based ridge detection unreliable, so we use bilateral symmetry instead.
    #
    # For each candidate x in the center 30% of the image, compute the
    # correlation between the left and right halves at the face level.
    # The x with best symmetry is the face midline (nose ridge).
    face_y1 = int(h * 0.15)
    face_y2 = int(h * 0.50)
    face_region = smoothed[face_y1:face_y2, :]

    best_sym_x = w // 2
    best_sym_score = -1
    half_width = int(w * 0.20)  # check symmetry over 20% on each side

    for candidate_x in range(int(w * 0.40), int(w * 0.60)):
        left_start = max(0, candidate_x - half_width)
        right_end = min(w, candidate_x + half_width)
        actual_half = min(candidate_x - left_start, right_end - candidate_x)
        if actual_half < 50:
            continue
        left_strip = face_region[:, candidate_x - actual_half:candidate_x]
        right_strip = face_region[:, candidate_x:candidate_x + actual_half]
        # Flip right strip and compare
        right_flipped = right_strip[:, ::-1]
        # Normalized cross-correlation
        left_norm = left_strip - left_strip.mean()
        right_norm = right_flipped - right_flipped.mean()
        denom = np.sqrt((left_norm ** 2).sum() * (right_norm ** 2).sum())
        if denom > 0:
            score = (left_norm * right_norm).sum() / denom
        else:
            score = 0
        if score > best_sym_score:
            best_sym_score = score
            best_sym_x = candidate_x

    nose_ridge_x = best_sym_x
    print(f"  Face midline x (symmetry): {nose_ridge_x} ({nose_ridge_x/w*100:.1f}% across)")
    print(f"  Symmetry score: {best_sym_score:.4f}")

    # Step 2: Find nose TIP — peak brightness on the midline in y=25-40%.
    # Use a narrow strip around the midline (±30px) to be robust.
    tip_y1 = int(h * 0.25)
    tip_y2 = int(h * 0.40)
    strip_half = 30
    ridge_left = max(0, nose_ridge_x - strip_half)
    ridge_right = min(w, nose_ridge_x + strip_half)
    nose_strip = smoothed[tip_y1:tip_y2, ridge_left:ridge_right]
    max_loc = np.unravel_index(nose_strip.argmax(), nose_strip.shape)
    nose_tip_y = max_loc[0] + tip_y1
    nose_tip_x = max_loc[1] + ridge_left

    print(f"  Nose tip (depth coords): ({nose_tip_x}, {nose_tip_y}) = {nose_tip_y/h*100:.1f}% down")
    print(f"  Depth at nose tip: {smoothed[nose_tip_y, nose_tip_x]}")

    # COORDINATE MAPPING
    # The depth map may be the full Enrie (3000x2388) while the positive
    # image is the trimmed face region (2700x2150). Map coordinates.
    positive_path = PROCESSED_DIR / "enrie_positive_raw.png"
    if positive_path.exists():
        pos_img = cv2.imread(str(positive_path), cv2.IMREAD_GRAYSCALE)
        pos_h, pos_w = pos_img.shape
    else:
        pos_h, pos_w = h, w

    # If dimensions differ, compute the offset (5% margin trim)
    if (h, w) != (pos_h, pos_w):
        source_dir = PROJECT_ROOT / "data" / "source"
        enrie_path = source_dir / "enrie_1931_face_hires.jpg"
        if enrie_path.exists():
            src = cv2.imread(str(enrie_path))
            src_h, src_w = src.shape[:2]
            offset_y = int(src_h * 0.05)
            offset_x = int(src_w * 0.05)
        else:
            offset_y = (h - pos_h) // 2
            offset_x = (w - pos_w) // 2
        print(f"  Coordinate offset (depth->positive): x-{offset_x}, y-{offset_y}")
    else:
        offset_x, offset_y = 0, 0

    # Convert nose tip to positive image coordinates
    nose_tip_x_pos = nose_tip_x - offset_x
    nose_tip_y_pos = nose_tip_y - offset_y
    print(f"  Nose tip (positive coords): ({nose_tip_x_pos}, {nose_tip_y_pos})")

    # FACE CENTER X — use the nose ridge as the face midline
    face_center_x = nose_ridge_x
    face_center_x_pos = face_center_x - offset_x
    print(f"  Face center x (positive coords): {face_center_x_pos}")

    # FACE WIDTH DETECTION
    # Use multiple horizontal profiles across the face to estimate width.
    # The face width varies: widest at cheekbones (slightly below eyes),
    # narrower at forehead and jaw.
    widths = []
    for y_offset_pct in [-0.08, -0.04, 0.0, 0.04]:
        probe_y = nose_tip_y + int(h * y_offset_pct)
        probe_y = max(0, min(h - 1, probe_y))
        row = smoothed[probe_y, :]
        # Background level: median of the outer 20% on each side
        bg_left = np.median(row[:int(w * 0.15)])
        bg_right = np.median(row[int(w * 0.85):])
        bg = (bg_left + bg_right) / 2
        # Face pixels are significantly above background
        threshold = bg + (row.max() - bg) * 0.35
        face_pixels = np.where(row > threshold)[0]
        if len(face_pixels) > 20:
            # Find contiguous cluster around face_center_x
            diffs = np.diff(face_pixels)
            breaks = np.where(diffs > 30)[0]
            clusters = np.split(face_pixels, breaks + 1)
            best = min(clusters, key=lambda c: abs(c.mean() - face_center_x))
            if len(best) > 10:
                widths.append(best[-1] - best[0])

    if widths:
        face_width = int(np.median(widths))
    else:
        face_width = int(w * 0.30)
        print(f"  WARNING: Could not measure face width, using fallback")

    face_left = face_center_x - face_width // 2
    face_right = face_center_x + face_width // 2

    # Sanity check
    if face_width < w * 0.10 or face_width > w * 0.65:
        print(f"  WARNING: Face width ({face_width}px) out of range, using 30% of image")
        face_width = int(w * 0.30)

    print(f"  Estimated face width: {face_width}px")

    # Anthropometric proportions (relative to face width)
    # Based on forensic reconstruction standards for adult male faces
    ipd = face_width * 0.42  # Interpupillary distance ~42% of face width
    eye_y_offset = -face_width * 0.35  # Eyes above nose tip
    nose_length = face_width * 0.30  # Nose length
    nose_width = face_width * 0.22  # Nose alar width
    mouth_y_offset = face_width * 0.20  # Mouth below nose tip
    mouth_width = face_width * 0.25  # Mouth width
    chin_y_offset = face_width * 0.55  # Chin below nose tip
    brow_y_offset = -face_width * 0.45  # Eyebrows above nose
    jaw_width = face_width * 0.85  # Jaw width
    cheek_width = face_width * 0.90  # Cheekbone width

    # Build landmark dict — all coordinates in POSITIVE IMAGE space
    cx = face_center_x_pos
    ny = nose_tip_y_pos

    key_landmarks = {
        "nose_tip": {"x": float(nose_tip_x), "y": float(ny), "z": 0.0},
        "nose_bridge_top": {"x": float(cx), "y": float(ny - nose_length), "z": 0.0},
        "nose_left_alar": {"x": float(cx - nose_width/2), "y": float(ny + nose_width*0.3), "z": 0.0},
        "nose_right_alar": {"x": float(cx + nose_width/2), "y": float(ny + nose_width*0.3), "z": 0.0},
        "left_eye_inner": {"x": float(cx - ipd*0.25), "y": float(ny + eye_y_offset), "z": 0.0},
        "left_eye_outer": {"x": float(cx - ipd*0.55), "y": float(ny + eye_y_offset), "z": 0.0},
        "right_eye_inner": {"x": float(cx + ipd*0.25), "y": float(ny + eye_y_offset), "z": 0.0},
        "right_eye_outer": {"x": float(cx + ipd*0.55), "y": float(ny + eye_y_offset), "z": 0.0},
        "left_pupil": {"x": float(cx - ipd/2), "y": float(ny + eye_y_offset), "z": 0.0},
        "right_pupil": {"x": float(cx + ipd/2), "y": float(ny + eye_y_offset), "z": 0.0},
        "upper_lip_center": {"x": float(cx), "y": float(ny + mouth_y_offset * 0.6), "z": 0.0},
        "lower_lip_center": {"x": float(cx), "y": float(ny + mouth_y_offset * 1.2), "z": 0.0},
        "mouth_left": {"x": float(cx - mouth_width/2), "y": float(ny + mouth_y_offset), "z": 0.0},
        "mouth_right": {"x": float(cx + mouth_width/2), "y": float(ny + mouth_y_offset), "z": 0.0},
        "chin": {"x": float(cx), "y": float(ny + chin_y_offset), "z": 0.0},
        "jaw_left": {"x": float(cx - jaw_width/2), "y": float(ny + chin_y_offset * 0.6), "z": 0.0},
        "jaw_right": {"x": float(cx + jaw_width/2), "y": float(ny + chin_y_offset * 0.6), "z": 0.0},
        "left_eyebrow_inner": {"x": float(cx - ipd*0.2), "y": float(ny + brow_y_offset), "z": 0.0},
        "left_eyebrow_outer": {"x": float(cx - ipd*0.6), "y": float(ny + brow_y_offset + face_width*0.02), "z": 0.0},
        "right_eyebrow_inner": {"x": float(cx + ipd*0.2), "y": float(ny + brow_y_offset), "z": 0.0},
        "right_eyebrow_outer": {"x": float(cx + ipd*0.6), "y": float(ny + brow_y_offset + face_width*0.02), "z": 0.0},
        "left_cheek": {"x": float(cx - cheek_width/2), "y": float(ny - face_width*0.05), "z": 0.0},
        "right_cheek": {"x": float(cx + cheek_width/2), "y": float(ny - face_width*0.05), "z": 0.0},
    }

    # Refine landmarks using depth map features.
    # Landmarks are in positive-image coords; depth map may be offset.
    # Convert to depth coords for searching, then back.
    print("  Refining landmarks using depth map features...")

    peak_landmarks = ["nose_tip", "left_cheek", "right_cheek",
                      "left_eyebrow_inner", "right_eyebrow_inner",
                      "left_eyebrow_outer", "right_eyebrow_outer",
                      "chin"]
    valley_landmarks = ["left_eye_inner", "left_eye_outer",
                        "right_eye_inner", "right_eye_outer",
                        "left_pupil", "right_pupil"]

    search_radius = int(face_width * 0.08)

    for name in peak_landmarks:
        if name in key_landmarks:
            lm = key_landmarks[name]
            # Convert positive coords → depth coords for search
            dx = int(lm["x"]) + offset_x
            dy = int(lm["y"]) + offset_y
            y1 = max(0, dy - search_radius)
            y2 = min(h, dy + search_radius)
            x1 = max(0, dx - search_radius)
            x2 = min(w, dx + search_radius)
            region = smoothed[y1:y2, x1:x2]
            if region.size > 0:
                local_max = np.unravel_index(region.argmax(), region.shape)
                # Convert back to positive coords
                key_landmarks[name]["x"] = float(x1 + local_max[1] - offset_x)
                key_landmarks[name]["y"] = float(y1 + local_max[0] - offset_y)

    for name in valley_landmarks:
        if name in key_landmarks:
            lm = key_landmarks[name]
            dx = int(lm["x"]) + offset_x
            dy = int(lm["y"]) + offset_y
            y1 = max(0, dy - search_radius)
            y2 = min(h, dy + search_radius)
            x1 = max(0, dx - search_radius)
            x2 = min(w, dx + search_radius)
            region = smoothed[y1:y2, x1:x2]
            if region.size > 0:
                local_min = np.unravel_index(region.argmin(), region.shape)
                key_landmarks[name]["x"] = float(x1 + local_min[1] - offset_x)
                key_landmarks[name]["y"] = float(y1 + local_min[0] - offset_y)

    # Add index field for compatibility with measurements.py
    for i, name in enumerate(key_landmarks):
        key_landmarks[name]["index"] = i

    landmarks = {
        "key_landmarks": key_landmarks,
        "all_landmarks": {str(i): v for i, v in enumerate(key_landmarks.values())},
        "image_size": {"width": pos_w, "height": pos_h},
        "num_landmarks": len(key_landmarks),
        "detection_confidence": 0.0,
        "preprocessing": "depth_map_guided",
        "detection_scale": 1.0,
        "method": "depth_map_heuristic",
        "note": "Landmarks estimated from depth map peaks/valleys + anthropometric proportions. "
                "These are approximate and should be visually verified.",
    }

    print(f"  Estimated {len(key_landmarks)} key landmarks from depth map")
    return landmarks


def run_alternative_detection():
    """Try all alternative detection methods."""
    print("=" * 60)
    print("  ALTERNATIVE LANDMARK DETECTION")
    print("  (MediaPipe failed on Enrie data — trying alternatives)")
    print("=" * 60)

    # Load the Enrie positive
    positive_path = PROCESSED_DIR / "enrie_positive_raw.png"
    if not positive_path.exists():
        # Generate it from Enrie negative
        source_dir = PROJECT_ROOT / "data" / "source"
        enrie_path = source_dir / "enrie_1931_face_hires.jpg"
        if not enrie_path.exists():
            print("Enrie source image not found")
            return None
        img = cv2.imread(str(enrie_path))
        h, w = img.shape[:2]
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        face = img[margin_y:h - margin_y, margin_x:w - margin_x]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        positive = cv2.bitwise_not(gray)
        cv2.imwrite(str(positive_path), positive)
    else:
        positive = cv2.imread(str(positive_path), cv2.IMREAD_GRAYSCALE)

    print(f"Loaded Enrie positive: {positive.shape}")

    # Method 1: Haar Cascade
    bbox = try_haar_cascade(positive)

    # Method 2: Dlib HOG
    if bbox is None:
        bbox = try_dlib_detection(positive)

    if bbox is not None:
        print(f"\n  Face bounding box found via {bbox['method']}!")
        print(f"  Now attempting MediaPipe within the detected region...")
        # Crop to detected face and retry MediaPipe
        x, y, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        # Add padding
        pad = int(max(bw, bh) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(positive.shape[1], x + bw + pad)
        y2 = min(positive.shape[0], y + bh + pad)
        face_crop = positive[y1:y2, x1:x2]

        # Try MediaPipe on this cropped region with heavy preprocessing
        from src.landmarks import detect_landmarks
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(cv2.GaussianBlur(face_crop, (21, 21), 0))
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        landmarks = detect_landmarks(rgb)
        if landmarks is not None:
            # Offset landmarks back to full image coordinates
            for idx in landmarks["all_landmarks"]:
                landmarks["all_landmarks"][idx]["x"] += x1
                landmarks["all_landmarks"][idx]["y"] += y1
            for name in landmarks["key_landmarks"]:
                landmarks["key_landmarks"][name]["x"] += x1
                landmarks["key_landmarks"][name]["y"] += y1
            landmarks["image_size"] = {
                "width": positive.shape[1],
                "height": positive.shape[0]
            }
            print("  MediaPipe succeeded on cropped face region!")
            _save_and_visualize(positive, landmarks, "haar_then_mediapipe")
            return landmarks

    # Method 3: Depth-map-guided estimation, constrained by Haar bbox if available
    print("\n  Falling back to depth-map-guided landmark estimation...")

    depth_path = PROCESSED_DIR / "depth_map.npy"
    if not depth_path.exists():
        print("  Depth map not found. Run depth extraction first.")
        return None

    depth = np.load(str(depth_path))
    landmarks = estimate_landmarks_from_depth(depth, haar_bbox=bbox)

    if landmarks is not None:
        _save_and_visualize(positive, landmarks, "depth_guided")

    return landmarks


def _save_and_visualize(positive: np.ndarray, landmarks: dict, method: str):
    """Save landmarks and create visualization overlay."""
    from src.landmarks import draw_landmarks, save_landmarks

    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    save_landmarks(landmarks)

    # Draw on the positive image
    vis_img = cv2.cvtColor(positive, cv2.COLOR_GRAY2BGR)
    vis = draw_landmarks(vis_img, landmarks)
    vis_path = PROCESSED_DIR / "landmarks_overlay.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"\n  Saved landmark overlay: {vis_path}")
    print(f"  Detection method: {method}")

    # Also save a version on the depth map for comparison
    depth_vis_path = PROCESSED_DIR / "depth_map.png"
    if depth_vis_path.exists():
        depth_img = cv2.imread(str(depth_vis_path))
        depth_vis = draw_landmarks(depth_img, landmarks)
        out_path = PROCESSED_DIR / "landmarks_on_depth.png"
        cv2.imwrite(str(out_path), depth_vis)
        print(f"  Saved depth overlay: {out_path}")


if __name__ == "__main__":
    run_alternative_detection()
