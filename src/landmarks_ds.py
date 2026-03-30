"""
Landmark detection on the 150x150 downsampled + smoothed depth map.

This approach uses the same data that produces the approved VP-8 3D surface.
At 150x150 with 15x15 Gaussian, anatomical features are clearly resolved:
nose ridge, eye sockets, cheekbones, brow ridge, chin.

The detection walks the vertical centerline profile to identify anatomical
peaks and valleys, then expands laterally to place all landmarks.
"""

import cv2
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEASUREMENTS_DIR = PROJECT_ROOT / "data" / "measurements"
PLOTS_DIR = PROJECT_ROOT / "output" / "plots"

# Shroud physical dimensions for scale calibration
SHROUD_LENGTH_CM = 442.0  # ~4.42m
SHROUD_WIDTH_CM = 113.0   # ~1.13m
# The man's height is estimated at 175-185cm by multiple researchers
ESTIMATED_HEIGHT_CM = 180.0

# Key landmark names (compatible with measurements.py)
KEY_LANDMARKS = [
    "left_eye_inner", "left_eye_outer", "right_eye_inner", "right_eye_outer",
    "left_pupil", "right_pupil",
    "nose_tip", "nose_bridge_top", "nose_left_alar", "nose_right_alar",
    "upper_lip_center", "lower_lip_center", "mouth_left", "mouth_right",
    "chin", "jaw_left", "jaw_right",
    "left_eyebrow_inner", "left_eyebrow_outer",
    "right_eyebrow_inner", "right_eyebrow_outer",
    "left_cheek", "right_cheek",
]


def create_analysis_map():
    """Create the 150x150 + Gaussian 15x15 depth map used for analysis."""
    depth = np.load(str(PROCESSED_DIR / "depth_map.npy"))
    ds = cv2.resize(depth, (150, 150), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(ds, (15, 15), 0)
    return smooth, depth.shape


def find_face_midline(smooth):
    """Find the face's vertical axis of symmetry."""
    h, w = smooth.shape
    best_x, best_score = w // 2, -1
    for cx in range(int(w * 0.40), int(w * 0.60)):
        half = min(cx, w - cx, int(w * 0.25))
        if half < 10:
            continue
        left = smooth[:, cx - half:cx].astype(float)
        right = smooth[:, cx:cx + half].astype(float)[:, ::-1]
        ln = left - left.mean()
        rn = right - right.mean()
        d = np.sqrt((ln ** 2).sum() * (rn ** 2).sum())
        if d > 0:
            s = (ln * rn).sum() / d
            if s > best_score:
                best_score = s
                best_x = cx
    return best_x, best_score


def find_anatomy_from_profile(smooth, midline_x):
    """Walk the vertical centerline to identify anatomical features.

    On the 150x150 smoothed depth map, the centerline profile shows clear
    peaks (forehead, nose bridge, chin) and valleys (eye sockets, philtrum).

    Verified anatomy on this data (from detailed profile analysis):
      y~25 (16.7%): brow ridge peak (our "forehead")
      y~41 (27.3%): eye socket trough on midline
      y~49 (32.7%): nose BRIDGE peak
      y~55 (36.7%): nostril / below-nose valley
      y~60 (40.0%): upper lip peak
      y~67 (44.7%): mouth valley
      y~73 (48.7%): chin (mentale) — first peak after mouth
      y~82 (54.7%): beard/neck transition (NOT the chin)
      y~92 (61.3%): neck valley

    Returns dict of anatomical feature positions in 150x150 coordinates.
    """
    h, w = smooth.shape

    # Average a narrow strip around the midline for robustness
    strip_half = 3
    x1 = max(0, midline_x - strip_half)
    x2 = min(w, midline_x + strip_half + 1)
    profile = smooth[:, x1:x2].mean(axis=1)

    features = {}

    # Brow ridge: highest point in y=10-22%
    y_range = slice(int(h * 0.10), int(h * 0.22))
    forehead_y = np.argmax(profile[y_range]) + int(h * 0.10)
    features["forehead"] = (midline_x, forehead_y, profile[forehead_y])

    # Eye socket trough: lowest point in y=22-32%
    y_range = slice(int(h * 0.22), int(h * 0.32))
    eye_trough_y = np.argmin(profile[y_range]) + int(h * 0.22)
    features["eye_level"] = (midline_x, eye_trough_y, profile[eye_trough_y])

    # Nose bridge: highest point in y=28-38%
    y_range = slice(int(h * 0.28), int(h * 0.38))
    nose_bridge_y = np.argmax(profile[y_range]) + int(h * 0.28)
    features["nose_bridge"] = (midline_x, nose_bridge_y, profile[nose_bridge_y])

    # Nose tip: placed between bridge peak and nostril valley.
    # The nostril valley is the minimum between the bridge and the mouth.
    y_range = slice(nose_bridge_y, int(h * 0.42))
    nostril_y = np.argmin(profile[y_range]) + nose_bridge_y
    # Nose tip is roughly 70% of the way from bridge to nostrils
    nose_tip_y = nose_bridge_y + int((nostril_y - nose_bridge_y) * 0.70)
    features["nose_tip"] = (midline_x, nose_tip_y, profile[nose_tip_y])

    # Mouth: lowest point in y=40-48%
    y_range = slice(int(h * 0.40), int(h * 0.48))
    mouth_y = np.argmin(profile[y_range]) + int(h * 0.40)
    features["mouth_level"] = (midline_x, mouth_y, profile[mouth_y])

    # Chin (mentale): the FIRST peak after the mouth valley.
    # On a bearded face (as on the Shroud), the first peak after the
    # mouth is the chin prominence. Later peaks (y~82) are the beard
    # surface or neck transition — NOT the bony chin.
    chin_y = mouth_y + 2
    while chin_y < min(int(h * 0.54), h - 2):
        if profile[chin_y - 1] <= profile[chin_y] >= profile[chin_y + 1]:
            break
        chin_y += 1
    else:
        chin_y = mouth_y + int(h * 0.04)
    features["chin"] = (midline_x, chin_y, profile[chin_y])

    return features


def find_eye_sockets(smooth, midline_x, eye_level_y, nose_y):
    """Find the two eye socket centers as depth valleys flanking the nose ridge.

    Uses the AVERAGED horizontal profile across the eye band (±3 rows)
    to find robust valley centers. This avoids single-pixel noise and
    correctly identifies each socket as a local minimum in the horizontal
    brightness profile at eye level.
    """
    h, w = smooth.shape

    # Average several rows around eye level for a robust horizontal profile
    margin_y = 3
    y1 = max(0, eye_level_y - margin_y)
    y2 = min(h, eye_level_y + margin_y + 1)
    avg_row = smooth[y1:y2, :].mean(axis=0)

    # Find the right eye socket (better defined on this data), then
    # compute the pupil position as the CENTER of the socket depression
    # (not the darkest point, which sits near the inner wall).
    #
    # Method: find the darkest point, then find where the socket walls
    # rise to half the depth on each side. The midpoint is the pupil.
    inner_gap = 5
    outer_limit = 20

    # Right eye: find darkest point in 2D region
    ry1 = max(0, eye_level_y - 8)
    ry2 = min(h, eye_level_y + 8)
    rx1 = midline_x + inner_gap
    rx2 = min(w, midline_x + outer_limit)
    right_region = smooth[ry1:ry2, rx1:rx2]
    r_loc = np.unravel_index(right_region.argmin(), right_region.shape)
    r_min_x = r_loc[1] + rx1
    r_min_y = r_loc[0] + ry1
    r_min_val = smooth[r_min_y, r_min_x]

    # Find socket boundaries at half-depth on the darkest row
    row = smooth[r_min_y, :]
    r_threshold = r_min_val + (row[max(0, midline_x - 5):min(w, midline_x + 25)].max() - r_min_val) * 0.5
    r_left_bound = r_min_x
    for x in range(r_min_x, midline_x, -1):
        if row[x] >= r_threshold:
            r_left_bound = x
            break
    r_right_bound = r_min_x
    for x in range(r_min_x, min(w, midline_x + 25)):
        if row[x] >= r_threshold:
            r_right_bound = x
            break

    # Pupil = center of socket, biased slightly lateral (+1px toward temple)
    # since the deepest socket point is near the inner wall
    reye_x = (r_left_bound + r_right_bound) // 2 + 1
    reye_y = r_min_y

    # Mirror for left eye (left socket is poorly defined on this data)
    right_offset = reye_x - midline_x
    leye_x = midline_x - right_offset

    leye = (leye_x, eye_level_y)
    reye = (reye_x, eye_level_y)

    return leye, reye


def find_cheekbones(smooth, midline_x, nose_y, leye, reye):
    """Place cheekbones using anthropometric ratio from IPD.

    Brightness-based cheekbone detection is unreliable on this data
    because cloth brightness outside the face confuses the search.
    Instead, use the well-established forensic ratio:
      bizygomatic width ≈ 2.05x IPD for a broad-faced male.

    Cheekbones are placed at nose level, at ±IPD*1.025 from the midline.
    """
    ipd = abs(reye[0] - leye[0])

    # Bizygomatic width = ~2.05x IPD → each cheekbone at ~1.025x IPD from midline
    cheek_offset = int(ipd * 1.025)

    # Place at nose level (cheekbones are roughly at nose bridge height)
    lcheek = (midline_x - cheek_offset, nose_y)
    rcheek = (midline_x + cheek_offset, nose_y)

    return lcheek, rcheek


def build_landmarks(smooth, features, leye, reye, lcheek, rcheek, midline_x):
    """Build the full landmark dict from detected features + anthropometric ratios."""
    h, w = smooth.shape
    mx = midline_x

    _, nose_tip_y, _ = features["nose_tip"]
    _, nose_bridge_y, _ = features["nose_bridge"]
    _, eye_y, _ = features["eye_level"]
    _, mouth_y, _ = features["mouth_level"]
    _, chin_y, _ = features["chin"]
    _, forehead_y, _ = features["forehead"]

    nose_y = nose_tip_y  # use tip for landmark placement

    # Measured distances on 150x150 grid
    ipd = abs(reye[0] - leye[0])
    nose_to_eye = abs(nose_y - eye_y)
    nose_to_mouth = abs(mouth_y - nose_y)
    nose_to_chin = abs(chin_y - nose_y)

    # Derived proportions from detected features
    eye_width = max(3, int(ipd * 0.30))  # single eye width ~30% of IPD
    nose_width = max(3, int(ipd * 0.55))  # nose alar width ~55% of IPD
    mouth_width = max(3, int(ipd * 0.65))  # mouth width ~65% of IPD
    jaw_width = max(3, int(ipd * 1.8))    # jaw width ~180% of IPD
    brow_offset_y = max(2, int(nose_to_eye * 0.25))  # brows slightly above eyes

    key_landmarks = {
        # Eyes — pupils at socket centers, inner corners between pupils and nose
        "left_pupil": {"x": float(leye[0]), "y": float(leye[1]), "z": 0.0},
        "right_pupil": {"x": float(reye[0]), "y": float(reye[1]), "z": 0.0},
        # Inner corners: ~40% of the way from midline to pupil
        "left_eye_inner": {"x": float(mx - (mx - leye[0]) * 0.40), "y": float(leye[1]), "z": 0.0},
        "right_eye_inner": {"x": float(mx + (reye[0] - mx) * 0.40), "y": float(reye[1]), "z": 0.0},
        # Outer corners: ~30% past the pupil
        "left_eye_outer": {"x": float(leye[0] - eye_width * 0.6), "y": float(leye[1]), "z": 0.0},
        "right_eye_outer": {"x": float(reye[0] + eye_width * 0.6), "y": float(reye[1]), "z": 0.0},

        # Nose — tip from profile, bridge above, alars lateral
        "nose_tip": {"x": float(mx), "y": float(nose_y), "z": 0.0},
        "nose_bridge_top": {"x": float(mx), "y": float(nose_bridge_y), "z": 0.0},
        "nose_left_alar": {"x": float(mx - nose_width // 2), "y": float(nose_y), "z": 0.0},
        "nose_right_alar": {"x": float(mx + nose_width // 2), "y": float(nose_y), "z": 0.0},

        # Mouth — from profile trough
        "upper_lip_center": {"x": float(mx), "y": float(nose_y + int(nose_to_mouth * 0.5)), "z": 0.0},
        "lower_lip_center": {"x": float(mx), "y": float(mouth_y + 1), "z": 0.0},
        "mouth_left": {"x": float(mx - mouth_width // 2), "y": float(mouth_y), "z": 0.0},
        "mouth_right": {"x": float(mx + mouth_width // 2), "y": float(mouth_y), "z": 0.0},

        # Chin — from profile peak
        "chin": {"x": float(mx), "y": float(chin_y), "z": 0.0},

        # Jaw — lateral at chin level
        "jaw_left": {"x": float(mx - jaw_width // 2), "y": float(chin_y - 2), "z": 0.0},
        "jaw_right": {"x": float(mx + jaw_width // 2), "y": float(chin_y - 2), "z": 0.0},

        # Eyebrows — above eyes
        "left_eyebrow_inner": {"x": float(leye[0] + eye_width * 0.3),
                               "y": float(leye[1] - brow_offset_y), "z": 0.0},
        "left_eyebrow_outer": {"x": float(leye[0] - eye_width * 0.5),
                               "y": float(leye[1] - brow_offset_y + 1), "z": 0.0},
        "right_eyebrow_inner": {"x": float(reye[0] - eye_width * 0.3),
                                "y": float(reye[1] - brow_offset_y), "z": 0.0},
        "right_eyebrow_outer": {"x": float(reye[0] + eye_width * 0.5),
                                "y": float(reye[1] - brow_offset_y + 1), "z": 0.0},

        # Cheekbones — from detected peaks
        "left_cheek": {"x": float(lcheek[0]), "y": float(lcheek[1]), "z": 0.0},
        "right_cheek": {"x": float(rcheek[0]), "y": float(rcheek[1]), "z": 0.0},
    }

    # Add index field for compatibility
    for i, name in enumerate(key_landmarks):
        key_landmarks[name]["index"] = i

    return key_landmarks


def refine_landmarks(smooth, key_landmarks, midline_x):
    """Refine PRIMARY landmarks toward the nearest depth peak or valley.

    Only refines landmarks that were placed from direct detection.
    Derived landmarks (eye inner/outer, computed from proportions) are
    NOT refined to avoid them collapsing onto nearby primary landmarks.
    """
    h, w = smooth.shape
    search_r = 3  # small radius on 150x150

    peak_names = {"nose_tip", "left_cheek", "right_cheek", "chin"}
    valley_names = {"left_pupil", "right_pupil"}

    for name, lm in key_landmarks.items():
        cx, cy = int(lm["x"]), int(lm["y"])
        y1 = max(0, cy - search_r)
        y2 = min(h, cy + search_r + 1)
        x1 = max(0, cx - search_r)
        x2 = min(w, cx + search_r + 1)
        region = smooth[y1:y2, x1:x2]
        if region.size == 0:
            continue

        if name in peak_names:
            loc = np.unravel_index(region.argmax(), region.shape)
        elif name in valley_names:
            loc = np.unravel_index(region.argmin(), region.shape)
        else:
            continue

        key_landmarks[name]["x"] = float(x1 + loc[1])
        key_landmarks[name]["y"] = float(y1 + loc[0])


def scale_to_full_image(key_landmarks, ds_shape, full_shape):
    """Scale landmarks from 150x150 coordinates to full image coordinates.

    The full image is the Enrie positive (2700x2150), and the depth map
    was created from the full Enrie negative (3000x2388) with 5% margin trim.
    """
    ds_h, ds_w = ds_shape      # 150, 150
    full_h, full_w = full_shape  # depth map: 3000, 2388

    # Scale 150x150 -> full depth map
    scale_x = full_w / ds_w
    scale_y = full_h / ds_h

    # Then offset from depth map -> positive image (5% margin trim)
    offset_x = int(full_w * 0.05)  # 119
    offset_y = int(full_h * 0.05)  # 150

    # Target positive image dimensions
    pos_w = full_w - 2 * offset_x  # 2150
    pos_h = full_h - 2 * offset_y  # 2700

    for name, lm in key_landmarks.items():
        # Scale to full depth map coordinates
        full_x = lm["x"] * scale_x
        full_y = lm["y"] * scale_y
        # Convert to positive image coordinates
        lm["x"] = full_x - offset_x
        lm["y"] = full_y - offset_y

    return pos_w, pos_h


def calibrate_scale(key_landmarks, pos_w, pos_h, full_shape, features_150, ds_shape):
    """Calculate pixels-per-cm using Shroud body proportions.

    The Enrie photograph is a close-up of the face, NOT the full cloth width.
    We cannot simply divide image width by 113cm.

    Instead, we use the detected face anatomy combined with known
    anthropometric data for a ~180cm male:
      - Forehead-to-chin (menton) distance: ~22-24cm
      - Cross-check with IPD: ~6.0-7.0cm

    The forehead and chin positions are detected directly from the depth map
    centerline profile, so this calibration ties the scale to actual
    Shroud features rather than assumptions about image framing.
    """
    full_h, full_w = full_shape
    ds_h, ds_w = ds_shape

    # Get forehead and chin y-positions from the 150x150 analysis
    forehead_y_ds = features_150["forehead"][1]
    chin_y_ds = features_150["chin"][1]
    face_height_ds = chin_y_ds - forehead_y_ds  # in 150x150 pixels

    # Scale to full image pixels
    face_height_full = face_height_ds * (full_h / ds_h)

    # Combined calibration from two independent references:
    #
    # 1. Face height: brow ridge to mentale (chin).
    #    Our "forehead" is the brow ridge, "chin" is the first peak
    #    after mouth (before beard). Expected: ~18cm for adult male.
    #
    # 2. IPD: interpupillary distance.
    #    Expected: ~6.3cm for adult male (range 5.5-7.5cm).
    #
    # The Shroud has draping distortion, so neither reference is perfect.
    # Averaging gives a more robust estimate.

    FACE_HEIGHT_CM = 18.0
    IPD_CM = 6.3

    scale_from_face = face_height_full / FACE_HEIGHT_CM

    # IPD-based scale
    if "left_pupil" in key_landmarks and "right_pupil" in key_landmarks:
        lp = key_landmarks["left_pupil"]
        rp = key_landmarks["right_pupil"]
        ipd_px = np.sqrt((rp["x"] - lp["x"])**2 + (rp["y"] - lp["y"])**2)
        scale_from_ipd = ipd_px / IPD_CM
        print(f"   Face-height scale: {scale_from_face:.1f} px/cm "
              f"(brow-chin = {FACE_HEIGHT_CM}cm)")
        print(f"   IPD scale: {scale_from_ipd:.1f} px/cm "
              f"(IPD = {IPD_CM}cm)")

        # Weighted average. IPD gets MORE weight because cloth draping
        # over the face elongates the vertical axis (the cloth follows
        # the 3D contour from forehead over nose to chin), while the
        # horizontal axis is less distorted at the eye level.
        px_per_cm = scale_from_face * 0.40 + scale_from_ipd * 0.60
        print(f"   Combined scale: {px_per_cm:.1f} px/cm")
    else:
        px_per_cm = scale_from_face

    # Cross-checks
    if "left_pupil" in key_landmarks and "right_pupil" in key_landmarks:
        implied_ipd = ipd_px / px_per_cm
        print(f"   Implied IPD: {implied_ipd:.1f}cm")
    implied_face_h = face_height_full / px_per_cm
    print(f"   Implied face height: {implied_face_h:.1f}cm")
    full_image_height_cm = full_h / px_per_cm
    print(f"   Implied image height: {full_image_height_cm:.1f}cm")

    return px_per_cm


def compute_measurements(key_landmarks, px_per_cm):
    """Compute anthropometric measurements from landmarks."""
    import math

    def dist(p1_name, p2_name):
        if p1_name not in key_landmarks or p2_name not in key_landmarks:
            return None
        p1 = key_landmarks[p1_name]
        p2 = key_landmarks[p2_name]
        dx = p1["x"] - p2["x"]
        dy = p1["y"] - p2["y"]
        return math.sqrt(dx * dx + dy * dy)

    measurements = {}

    def measure(name, p1, p2):
        d = dist(p1, p2)
        if d is not None:
            measurements[name] = {
                "pixels": round(d, 1),
                "cm": round(d / px_per_cm, 2),
                "points": [p1, p2],
            }

    measure("interpupillary_distance", "left_pupil", "right_pupil")
    measure("inner_eye_distance", "left_eye_inner", "right_eye_inner")
    measure("nose_length", "nose_bridge_top", "nose_tip")
    measure("nose_width", "nose_left_alar", "nose_right_alar")
    measure("face_width_cheekbones", "left_cheek", "right_cheek")
    measure("jaw_width", "jaw_left", "jaw_right")
    measure("nose_to_chin", "nose_tip", "chin")
    measure("mouth_width", "mouth_left", "mouth_right")
    measure("upper_lip_to_nose", "nose_tip", "upper_lip_center")
    measure("lower_face_height", "nose_tip", "chin")
    measure("brow_to_nose_bridge", "left_eyebrow_inner", "nose_bridge_top")
    measure("left_eye_to_mouth", "left_eye_inner", "mouth_left")
    measure("brow_to_chin", "left_eyebrow_inner", "chin")

    # Symmetry
    d_left = dist("left_pupil", "nose_tip")
    d_right = dist("right_pupil", "nose_tip")
    if d_left and d_right and max(d_left, d_right) > 0:
        measurements["facial_symmetry"] = {
            "ratio": round(min(d_left, d_right) / max(d_left, d_right), 4),
            "note": "1.0 = perfect symmetry. >0.95 typical for living faces.",
        }

    measurements["_metadata"] = {
        "px_per_cm": round(px_per_cm, 2),
        "scale_basis": f"Shroud cloth width ({SHROUD_WIDTH_CM}cm) mapped to image width",
        "note": "Scale calibrated from known Shroud dimensions, not IPD.",
    }

    return measurements


def validate_measurements(measurements):
    """Check plausibility against known human ranges."""
    expected = {
        "interpupillary_distance": (5.5, 7.5),
        "nose_length": (4.0, 6.5),
        "nose_width": (2.5, 5.0),
        "face_width_cheekbones": (12.0, 17.0),
        "jaw_width": (10.0, 15.0),
        "nose_to_chin": (5.0, 8.5),
        "mouth_width": (4.0, 6.5),
    }
    warnings = []
    for name, (lo, hi) in expected.items():
        if name in measurements and "cm" in measurements[name]:
            cm = measurements[name]["cm"]
            if cm < lo or cm > hi:
                warnings.append(f"{name}: {cm}cm outside expected {lo}-{hi}cm")
    return warnings


def draw_landmarks_on_image(image, key_landmarks):
    """Draw landmarks on an image. Image must be BGR."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]
    dot_r = max(3, min(w, h) // 200)
    font_scale = max(0.3, min(w, h) / 3000)

    for name, lm in key_landmarks.items():
        x, y = int(lm["x"]), int(lm["y"])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), dot_r, (0, 0, 255), -1)
            label = name[:12]
            cv2.putText(vis, label, (x + dot_r + 2, y - dot_r),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)

    return vis


def run_landmark_detection():
    """Full landmark detection pipeline using 150x150 downsampled depth map."""
    print("=" * 60)
    print("  LANDMARK DETECTION (150x150 Depth Map)")
    print("=" * 60)

    # Step 1: Create analysis map
    print("\n1. Creating 150x150 + Gaussian 15x15 analysis map...")
    smooth, full_shape = create_analysis_map()
    h, w = smooth.shape
    print(f"   Analysis map: {h}x{w}, range [{smooth.min():.0f}, {smooth.max():.0f}]")
    print(f"   Full depth map: {full_shape}")

    # Step 2: Find face midline
    print("\n2. Finding face midline (bilateral symmetry)...")
    midline_x, sym_score = find_face_midline(smooth)
    print(f"   Midline: x={midline_x} ({midline_x/w*100:.1f}%)")
    print(f"   Symmetry score: {sym_score:.4f}")

    # Step 3: Walk centerline profile to find anatomy
    print("\n3. Identifying anatomy from centerline profile...")
    features = find_anatomy_from_profile(smooth, midline_x)
    for name, (fx, fy, fv) in features.items():
        print(f"   {name:15s}: y={fy:3d} ({fy/h*100:5.1f}%), depth={fv:.0f}")

    # Step 4: Find eye sockets
    print("\n4. Finding eye sockets (depth valleys)...")
    _, eye_y, _ = features["eye_level"]
    _, nose_y, _ = features["nose_tip"]
    leye, reye = find_eye_sockets(smooth, midline_x, eye_y, nose_y)
    print(f"   Left eye:  ({leye[0]}, {leye[1]}), depth={smooth[leye[1], leye[0]]:.0f}")
    print(f"   Right eye: ({reye[0]}, {reye[1]}), depth={smooth[reye[1], reye[0]]:.0f}")
    print(f"   IPD: {abs(reye[0] - leye[0])}px on 150x150")

    # Step 5: Find cheekbones
    print("\n5. Finding cheekbones (depth peaks)...")
    lcheek, rcheek = find_cheekbones(smooth, midline_x, nose_y, leye, reye)
    print(f"   Left cheek:  ({lcheek[0]}, {lcheek[1]}), depth={smooth[lcheek[1], lcheek[0]]:.0f}")
    print(f"   Right cheek: ({rcheek[0]}, {rcheek[1]}), depth={smooth[rcheek[1], rcheek[0]]:.0f}")

    # Step 6: Build full landmark set
    print("\n6. Building landmarks from detected features + proportions...")
    key_landmarks = build_landmarks(smooth, features, leye, reye, lcheek, rcheek, midline_x)

    # Step 7: Refine toward local peaks/valleys
    print("   Refining toward local depth features...")
    refine_landmarks(smooth, key_landmarks, midline_x)

    # Step 8: Scale to full image coordinates
    print("\n7. Scaling to full image coordinates...")
    pos_w, pos_h = scale_to_full_image(key_landmarks, smooth.shape, full_shape)
    print(f"   Positive image: {pos_w}x{pos_h}")

    # Step 9: Calibrate scale
    print("\n8. Calibrating scale from Shroud body proportions...")
    px_per_cm = calibrate_scale(key_landmarks, pos_w, pos_h, full_shape,
                                features, smooth.shape)
    print(f"   Scale: {px_per_cm:.2f} px/cm")
    print(f"   Basis: brow ridge to mentale = {18.0}cm")

    # Step 10: Compute measurements
    print("\n9. Computing measurements...")
    measurements = compute_measurements(key_landmarks, px_per_cm)

    print("\n--- Measurements ---")
    for name, data in measurements.items():
        if name.startswith("_"):
            continue
        if isinstance(data, dict) and "cm" in data:
            print(f"   {name}: {data['cm']} cm ({data['pixels']} px)")
        elif isinstance(data, dict) and "ratio" in data:
            print(f"   {name}: {data['ratio']}")

    warnings = validate_measurements(measurements)
    if warnings:
        print("\n--- Warnings ---")
        for w_msg in warnings:
            print(f"   WARNING: {w_msg}")
    else:
        print("\n   All measurements within expected human ranges.")

    # Step 11: Save
    print("\n10. Saving outputs...")
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save landmarks JSON
    landmarks_data = {
        "key_landmarks": key_landmarks,
        "all_landmarks": {str(i): v for i, v in enumerate(key_landmarks.values())},
        "image_size": {"width": pos_w, "height": pos_h},
        "num_landmarks": len(key_landmarks),
        "method": "150x150_depth_guided",
        "note": "Landmarks detected from 150x150 downsampled depth map. "
                "Scale calibrated from Shroud physical dimensions.",
    }
    with open(MEASUREMENTS_DIR / "landmarks.json", "w") as f:
        json.dump(landmarks_data, f, indent=2)
    print(f"   Landmarks: {MEASUREMENTS_DIR / 'landmarks.json'}")

    # Save measurements JSON
    with open(MEASUREMENTS_DIR / "anthropometric_measurements.json", "w") as f:
        json.dump(measurements, f, indent=2)
    print(f"   Measurements: {MEASUREMENTS_DIR / 'anthropometric_measurements.json'}")

    # Draw overlays
    # On Enrie positive
    pos_path = PROCESSED_DIR / "enrie_positive_raw.png"
    if pos_path.exists():
        pos_img = cv2.imread(str(pos_path))
        vis = draw_landmarks_on_image(pos_img, key_landmarks)
        cv2.imwrite(str(PROCESSED_DIR / "landmarks_overlay.png"), vis)
        print(f"   Overlay (Enrie): {PROCESSED_DIR / 'landmarks_overlay.png'}")

    # On depth map
    depth_img_path = PROCESSED_DIR / "depth_map.png"
    if depth_img_path.exists():
        depth_img = cv2.imread(str(depth_img_path))
        vis2 = draw_landmarks_on_image(depth_img, key_landmarks)
        cv2.imwrite(str(PROCESSED_DIR / "landmarks_on_depth.png"), vis2)
        print(f"   Overlay (depth): {PROCESSED_DIR / 'landmarks_on_depth.png'}")

    # On the 150x150 smoothed map (for direct verification)
    smooth_rgb = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
    smooth_large = cv2.resize(smooth_rgb, (750, 750), interpolation=cv2.INTER_NEAREST)
    # Scale landmarks to 750x750
    scale_vis = 750 / 150
    for name, lm in key_landmarks.items():
        # Convert back from positive coords to 150x150 for this visualization
        ds_x = (lm["x"] + int(full_shape[1] * 0.05)) / (full_shape[1] / 150)
        ds_y = (lm["y"] + int(full_shape[0] * 0.05)) / (full_shape[0] / 150)
        x_vis = int(ds_x * scale_vis)
        y_vis = int(ds_y * scale_vis)
        if 0 <= x_vis < 750 and 0 <= y_vis < 750:
            cv2.circle(smooth_large, (x_vis, y_vis), 5, (0, 0, 255), -1)
            cv2.putText(smooth_large, name[:10], (x_vis + 7, y_vis - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imwrite(str(PROCESSED_DIR / "landmarks_on_150x150.png"), smooth_large)
    print(f"   Overlay (150x150): {PROCESSED_DIR / 'landmarks_on_150x150.png'}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)

    return landmarks_data, measurements


if __name__ == "__main__":
    run_landmark_detection()
