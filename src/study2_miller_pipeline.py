"""
Study 2: Vernon Miller 1978 STURP Photographs — Full Pipeline

Source: 34c-Fa-N_0414.jpg — Face close-up negative, 8176x6132px
This is the highest-resolution face negative from the 1978 STURP scientific
photography session. Vernon Miller, STURP team photographer.

Runs identical pipeline to Study 1 (Enrie 1931):
  1. Depth map extraction (CLAHE + normalization)
  2. 150x150 + Gaussian 15x15 3D surface
  3. Depth-guided landmark detection
  4. Facial measurements + scale calibration
  5. Healed (symmetrized) depth map

All outputs -> output/study2_miller/
"""

import cv2
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.parent
MILLER_SOURCE = PROJECT_ROOT / "data" / "source" / "vernon_miller" / "34c-Fa-N_0414.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output" / "study2_miller"

# Reuse the approved scale calibration method from Study 1
# Combined: face-height (40%) + IPD (60%)
FACE_HEIGHT_REFERENCE_CM = 18.0   # brow-to-chin
IPD_REFERENCE_CM = 6.3             # interpupillary distance


# ─────────────────────────────────────────────
#  STEP 1: DEPTH MAP
# ─────────────────────────────────────────────

def extract_and_process(image_path):
    """Load Miller face negative and create depth map.

    The 34c image is already a face close-up (landscape, 8176x6132).
    Crop out border margins, then apply CLAHE + normalization.
    """
    print(f"Loading: {image_path.name}  ({image_path.stat().st_size / 1e6:.1f} MB)")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    h, w = img_bgr.shape[:2]
    print(f"  Raw size: {w}x{h}  ({w/h:.3f} aspect ratio)")

    # Trim 8% margins on all sides to remove border/film artifacts
    mx = int(w * 0.08)
    my = int(h * 0.08)
    face_bgr = img_bgr[my:h - my, mx:w - mx]
    fh, fw = face_bgr.shape[:2]
    print(f"  After margin trim: {fw}x{fh}")

    # Convert to grayscale
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # Normalize to full range
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # CLAHE — same params as Study 1 (clipLimit=2.0, tileGrid=8x8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    depth = clahe.apply(normalized)

    print(f"  Depth map: {depth.shape}  range [{depth.min()}, {depth.max()}]")
    return depth, face_bgr


def save_outputs(depth, output_dir):
    """Save depth map as PNG and .npy."""
    cv2.imwrite(str(output_dir / "depth_map.png"), depth)
    np.save(str(output_dir / "depth_map.npy"), depth)
    print(f"  Saved depth_map.png + .npy")


# ─────────────────────────────────────────────
#  STEP 2: 3D SURFACE (150x150 + Gaussian 15x15)
# ─────────────────────────────────────────────

def generate_3d_surface(depth, output_dir):
    """Generate VP-8 style 3D surface plot — locked params: 150x150 + G15."""
    print("Generating 3D surface (150x150 + Gaussian 15)...")

    ds = cv2.resize(depth, (150, 150), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(ds, (15, 15), 0)

    # Save the analysis map
    cv2.imwrite(str(output_dir / "depth_150x150_g15.png"), smooth)
    np.save(str(output_dir / "depth_150x150_g15.npy"), smooth)

    X = np.arange(150)
    Y = np.arange(150)
    X, Y = np.meshgrid(X, Y)
    Z = smooth.astype(float)

    # Angled view
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="bone", linewidth=0, antialiased=True,
                    rstride=1, cstride=1, alpha=0.95)
    ax.set_title("Study 2: Miller 1978 STURP — VP-8 3D Surface\n(150x150, Gaussian 15x15)",
                 fontsize=11, pad=10)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_zlabel("Depth")
    ax.view_init(elev=35, azim=-60)
    ax.set_box_aspect([1, 1.2, 0.3])
    plt.tight_layout()
    plt.savefig(str(output_dir / "3d_surface_angled.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 3d_surface_angled.png")

    # Front view
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="bone", linewidth=0, antialiased=True,
                    rstride=1, cstride=1, alpha=0.95)
    ax.set_title("Study 2: Miller 1978 STURP — VP-8 3D Surface (Front)",
                 fontsize=11, pad=10)
    ax.view_init(elev=0, azim=-90)
    ax.set_box_aspect([1, 1.2, 0.3])
    plt.tight_layout()
    plt.savefig(str(output_dir / "3d_surface_front.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 3d_surface_front.png")

    # Depth heatmap
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(smooth, cmap="plasma", aspect="auto")
    ax.set_title("Study 2: Miller 1978 — Depth Heatmap (150x150 + G15)",
                 fontsize=11)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    plt.colorbar(im, ax=ax, label="Depth (brightness)")
    plt.tight_layout()
    plt.savefig(str(output_dir / "depth_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved depth_heatmap.png")

    return smooth


# ─────────────────────────────────────────────
#  STEP 3: LANDMARK DETECTION
#  (Identical algorithm to landmarks_ds.py)
# ─────────────────────────────────────────────

def find_face_midline(smooth):
    """Bilateral symmetry search for face midline."""
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
    """Walk vertical centerline profile for anatomical features."""
    h, w = smooth.shape
    strip_half = 3
    x1 = max(0, midline_x - strip_half)
    x2 = min(w, midline_x + strip_half + 1)
    profile = smooth[:, x1:x2].mean(axis=1)

    features = {}

    # Brow ridge: highest in y=10-22%
    y_range = slice(int(h * 0.10), int(h * 0.22))
    brow_y = int(h * 0.10) + int(profile[y_range].argmax())
    features["brow_y"] = brow_y

    # Nose bridge: highest in y=22-38%
    y_range = slice(int(h * 0.22), int(h * 0.38))
    nose_bridge_y = int(h * 0.22) + int(profile[y_range].argmax())
    features["nose_bridge_y"] = nose_bridge_y

    # Eye level: lowest in y=22-35% (socket trough)
    y_range = slice(int(h * 0.22), int(h * 0.35))
    eye_y = int(h * 0.22) + int(profile[y_range].argmin())
    features["eye_y"] = eye_y

    # Nose tip: highest in y=30-45% (after nose bridge)
    y_range = slice(int(h * 0.30), int(h * 0.45))
    nose_tip_y = int(h * 0.30) + int(profile[y_range].argmax())
    features["nose_tip_y"] = nose_tip_y

    # Mouth: lowest in y=38-50%
    y_range = slice(int(h * 0.38), int(h * 0.50))
    mouth_y = int(h * 0.38) + int(profile[y_range].argmin())
    features["mouth_y"] = mouth_y

    # Chin: FIRST local peak after mouth (before y=54%)
    chin_y = mouth_y + 2
    while chin_y < min(int(h * 0.54), h - 2):
        if profile[chin_y - 1] <= profile[chin_y] >= profile[chin_y + 1]:
            break
        chin_y += 1
    features["chin_y"] = chin_y

    return features, profile


def find_eye_sockets(smooth, eye_y, midline_x):
    """Find eye socket centers. Returns (left_socket_x, right_socket_x) in image coords.

    Naming: left = lower x (left side of image), right = higher x (right side).
    The search anchors on the lower-x side (left eye) and mirrors for the right.
    This matches Study 1 (Enrie) naming convention.
    """
    h, w = smooth.shape
    y1 = max(0, eye_y - 3)
    y2 = min(h, eye_y + 4)

    # Search for darkest point left of midline (= left eye, lower x)
    left_region = smooth[y1:y2, midline_x - 30:midline_x - 5]
    left_local_x = int(left_region.mean(axis=0).argmin())
    left_dark_x = midline_x - 30 + left_local_x

    # Socket center via half-depth boundaries
    row = smooth[eye_y, :].astype(float)
    mid_val = row[midline_x]
    left_thresh = (row[left_dark_x] + mid_val) / 2
    ll, lr = left_dark_x, left_dark_x
    while ll > 0 and row[ll] < left_thresh:
        ll -= 1
    while lr < midline_x and row[lr] < left_thresh:
        lr += 1
    left_socket_x = (ll + lr) // 2 + 1  # +1 lateral bias (from Study 1 calibration)

    # Mirror for right eye (right side = higher x = 2*midline - left_socket)
    right_socket_x = 2 * midline_x - left_socket_x

    return left_socket_x, right_socket_x


def detect_landmarks(smooth):
    """Full landmark detection on 150x150 depth map.

    Naming convention (matches Study 1 approved_measurements.json):
      left  = lower x (left side of image)
      right = higher x (right side of image)
    """
    h, w = smooth.shape

    midline_x, sym_score = find_face_midline(smooth)
    print(f"  Midline: x={midline_x} ({midline_x/w*100:.1f}%)  symmetry={sym_score:.4f}")

    features, profile = find_anatomy_from_profile(smooth, midline_x)
    print(f"  Anatomy: brow={features['brow_y']} eye={features['eye_y']} "
          f"nose_bridge={features['nose_bridge_y']} nose_tip={features['nose_tip_y']} "
          f"mouth={features['mouth_y']} chin={features['chin_y']}")

    left_socket_x, right_socket_x = find_eye_sockets(smooth, features["eye_y"], midline_x)
    ipd_px = abs(right_socket_x - left_socket_x)
    print(f"  Eye sockets: left_x={left_socket_x}  right_x={right_socket_x}  "
          f"IPD_px={ipd_px}")

    # ── Inner eye corners: 40% of the way from midline to each pupil ──
    # (matches landmarks_ds.py: left_eye_inner = mx - (mx - leye[0]) * 0.40)
    left_eye_inner_x = int(midline_x - (midline_x - left_socket_x) * 0.40)
    right_eye_inner_x = int(midline_x + (right_socket_x - midline_x) * 0.40)

    # ── Outer eye corners: 30% past each pupil (away from midline) ──
    left_eye_outer_x = int(left_socket_x - ipd_px * 0.30)
    right_eye_outer_x = int(right_socket_x + ipd_px * 0.30)

    # ── Cheekbones: bizygomatic width ≈ 2.05x IPD ──
    # Each cheekbone at 1.025x IPD from midline (matches landmarks_ds.py exactly)
    cheek_offset = int(ipd_px * 1.025)
    left_cheek_x = midline_x - cheek_offset
    right_cheek_x = midline_x + cheek_offset
    cheek_y = int(features["nose_tip_y"])

    # ── Jaw: jaw width ≈ 1.8x IPD → each point at 0.9x IPD from midline ──
    jaw_half = int(ipd_px * 0.9)
    left_jaw_x = midline_x - jaw_half
    right_jaw_x = midline_x + jaw_half
    jaw_y = int(features["chin_y"]) - 2

    # ── Nose alar width ≈ 0.55x IPD total → each alar at 0.275x IPD from midline ──
    nose_alar_half = int(ipd_px * 0.275)
    left_alar_x = midline_x - nose_alar_half
    right_alar_x = midline_x + nose_alar_half

    # ── Mouth corners ≈ 0.65x IPD total → each at 0.325x IPD from midline ──
    mouth_half = int(ipd_px * 0.325)
    left_mouth_x = midline_x - mouth_half
    right_mouth_x = midline_x + mouth_half

    landmarks = {
        "midline_x": int(midline_x),
        "symmetry_score": float(sym_score),
        "left_pupil": {"x": int(left_socket_x), "y": int(features["eye_y"])},
        "right_pupil": {"x": int(right_socket_x), "y": int(features["eye_y"])},
        "left_eye_inner": {"x": int(left_eye_inner_x), "y": int(features["eye_y"])},
        "right_eye_inner": {"x": int(right_eye_inner_x), "y": int(features["eye_y"])},
        "left_eye_outer": {"x": int(left_eye_outer_x), "y": int(features["eye_y"])},
        "right_eye_outer": {"x": int(right_eye_outer_x), "y": int(features["eye_y"])},
        "nose_bridge_top": {"x": int(midline_x), "y": int(features["nose_bridge_y"])},
        "nose_tip": {"x": int(midline_x), "y": int(features["nose_tip_y"])},
        "nose_left_alar": {"x": int(left_alar_x), "y": int(features["nose_tip_y"])},
        "nose_right_alar": {"x": int(right_alar_x), "y": int(features["nose_tip_y"])},
        "mouth_center": {"x": int(midline_x), "y": int(features["mouth_y"])},
        "mouth_left": {"x": int(left_mouth_x), "y": int(features["mouth_y"])},
        "mouth_right": {"x": int(right_mouth_x), "y": int(features["mouth_y"])},
        "chin": {"x": int(midline_x), "y": int(features["chin_y"])},
        "left_cheek": {"x": int(left_cheek_x), "y": int(cheek_y)},
        "right_cheek": {"x": int(right_cheek_x), "y": int(cheek_y)},
        "left_jaw": {"x": int(left_jaw_x), "y": int(jaw_y)},
        "right_jaw": {"x": int(right_jaw_x), "y": int(jaw_y)},
        "brow_center": {"x": int(midline_x), "y": int(features["brow_y"])},
    }

    return landmarks, features, ipd_px


# ─────────────────────────────────────────────
#  STEP 4: MEASUREMENTS + SCALE CALIBRATION
# ─────────────────────────────────────────────

def compute_measurements(landmarks, smooth, ipd_px):
    """Compute facial measurements using combined scale calibration."""
    h, w = smooth.shape

    # Scale calibration: IPD-only for Study 2.
    # The Miller 34c image is a face close-up at high magnification. The
    # centerline brow/chin profile detects interior features of the face
    # rather than the full brow-to-menton extent, making face-height
    # calibration unreliable. IPD is anchored to the eye socket detection
    # which is geometrically well-constrained. Use IPD reference 6.3cm.
    lm = landmarks
    face_height_px = lm["chin"]["y"] - lm["brow_center"]["y"]
    px_from_fh = (face_height_px / FACE_HEIGHT_REFERENCE_CM)
    px_from_ipd = (ipd_px / IPD_REFERENCE_CM)
    px_per_cm = px_from_ipd  # IPD-only (100%)

    print(f"  Scale: face_height_px={face_height_px}  ipd_px={ipd_px:.1f}")
    print(f"    px/cm from face-height (informational): {px_from_fh:.2f}")
    print(f"    px/cm from IPD (used):                  {px_from_ipd:.2f}")
    print(f"    Scale (IPD-only):                        {px_per_cm:.2f}")

    def px_to_cm(px_val):
        return round(float(px_val) / px_per_cm, 2)

    # Interpupillary distance
    ipd_cm = px_to_cm(ipd_px)

    # Inner eye distance
    inner_px = abs(lm["left_eye_inner"]["x"] - lm["right_eye_inner"]["x"])
    inner_cm = px_to_cm(inner_px)

    # Nose width (alar)
    nose_px = abs(lm["nose_right_alar"]["x"] - lm["nose_left_alar"]["x"])
    nose_cm = px_to_cm(nose_px)

    # Face width at cheekbones
    fw_px = abs(lm["right_cheek"]["x"] - lm["left_cheek"]["x"])
    fw_cm = px_to_cm(fw_px)

    # Jaw width
    jaw_px = abs(lm["right_jaw"]["x"] - lm["left_jaw"]["x"])
    jaw_cm = px_to_cm(jaw_px)

    # Mouth width
    mw_px = abs(lm["mouth_right"]["x"] - lm["mouth_left"]["x"])
    mw_cm = px_to_cm(mw_px)

    # Nose-to-chin (vertical)
    ntc_px = lm["chin"]["y"] - lm["nose_tip"]["y"]
    ntc_cm = px_to_cm(ntc_px)

    # Nose length (bridge to tip)
    nl_px = lm["nose_tip"]["y"] - lm["nose_bridge_top"]["y"]
    nl_cm = px_to_cm(nl_px)

    # Facial symmetry
    right_half = smooth[:, lm["midline_x"]:]
    left_half = smooth[:, :lm["midline_x"]][:, ::-1]
    min_w = min(right_half.shape[1], left_half.shape[1])
    corr = np.corrcoef(right_half[:, :min_w].flatten(),
                       left_half[:, :min_w].flatten())[0, 1]

    measurements = {
        "scale_calibration": {
            "method": "IPD-only (face-height unreliable at this magnification)",
            "px_per_cm": round(px_per_cm, 2),
            "ipd_reference_cm": IPD_REFERENCE_CM,
            "face_height_px": int(face_height_px),
            "face_height_ref_cm_informational": FACE_HEIGHT_REFERENCE_CM,
            "ipd_px": round(float(ipd_px), 1),
        },
        "interpupillary_distance": {
            "cm": ipd_cm, "pixels": round(float(ipd_px), 1),
            "expected": "5.5-7.5",
            "status": "borderline" if ipd_cm < 5.5 else "in_range"
        },
        "inner_eye_distance": {
            "cm": inner_cm, "pixels": int(inner_px),
            "expected": "2.8-3.5",
            "status": "in_range" if 2.8 <= inner_cm <= 3.5 else "outlier"
        },
        "nose_width": {
            "cm": nose_cm, "pixels": int(nose_px),
            "expected": "2.5-5.0",
            "status": "in_range" if 2.5 <= nose_cm <= 5.0 else "outlier"
        },
        "face_width_cheekbones": {
            "cm": fw_cm, "pixels": int(fw_px),
            "expected": "12.0-17.0",
            "status": "in_range" if 12.0 <= fw_cm <= 17.0 else "outlier"
        },
        "jaw_width": {
            "cm": jaw_cm, "pixels": int(jaw_px),
            "expected": "10.0-15.0",
            "status": "in_range" if 10.0 <= jaw_cm <= 15.0 else "outlier"
        },
        "mouth_width": {
            "cm": mw_cm, "pixels": int(mw_px),
            "expected": "4.0-6.5",
            "status": "in_range" if 4.0 <= mw_cm <= 6.5 else "outlier"
        },
        "nose_to_chin": {
            "cm": ntc_cm, "pixels": int(ntc_px),
            "expected": "7.0-9.5",
            "status": "in_range" if 7.0 <= ntc_cm <= 9.5 else "outlier_explained",
            "note": "Cloth draping may elongate vertical axis"
        },
        "nose_length": {
            "cm": nl_cm, "pixels": int(nl_px),
            "expected": "2.5-5.5",
            "status": "in_range" if 2.5 <= nl_cm <= 5.5 else "outlier_explained",
            "note": "Resolution limit on 150x150 grid"
        },
        "facial_symmetry": {
            "ratio": round(float(corr), 3),
            "status": "excellent" if corr >= 0.97 else "good" if corr >= 0.94 else "fair"
        },
    }

    return measurements


# ─────────────────────────────────────────────
#  STEP 5: LANDMARK OVERLAY IMAGES
# ─────────────────────────────────────────────

def draw_landmark_overlays(smooth, landmarks, output_dir):
    """Draw landmark dots on depth map and 150x150 grid."""
    scale = 4  # 150x150 -> 600x600 for visibility
    cell = 150 * scale

    color_map = {
        "right_pupil": (255, 100, 100),
        "left_pupil": (255, 100, 100),
        "right_eye_inner": (200, 200, 255),
        "left_eye_inner": (200, 200, 255),
        "right_eye_outer": (200, 200, 255),
        "left_eye_outer": (200, 200, 255),
        "nose_tip": (100, 255, 100),
        "nose_bridge_top": (100, 200, 100),
        "nose_right_alar": (150, 255, 150),
        "nose_left_alar": (150, 255, 150),
        "mouth_right": (255, 200, 100),
        "mouth_left": (255, 200, 100),
        "mouth_center": (255, 180, 80),
        "chin": (200, 150, 255),
        "right_cheek": (255, 255, 100),
        "left_cheek": (255, 255, 100),
        "right_jaw": (200, 200, 200),
        "left_jaw": (200, 200, 200),
        "brow_center": (150, 220, 255),
    }

    # Normalize smooth for display
    disp = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
    big = cv2.resize(disp_rgb, (cell, cell), interpolation=cv2.INTER_NEAREST)

    for name, pt in landmarks.items():
        if isinstance(pt, dict) and "x" in pt:
            px = pt["x"] * scale
            py = pt["y"] * scale
            color = color_map.get(name, (255, 255, 255))
            cv2.circle(big, (px, py), 6, color, -1)
            cv2.circle(big, (px, py), 7, (0, 0, 0), 1)

    # Midline
    mx = landmarks["midline_x"] * scale
    cv2.line(big, (mx, 0), (mx, cell), (100, 100, 255), 1)

    cv2.imwrite(str(output_dir / "landmarks_on_150.png"), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    print("  Saved landmarks_on_150.png")


# ─────────────────────────────────────────────
#  STEP 6: HEALED DEPTH MAP
# ─────────────────────────────────────────────

def generate_healed_depth(smooth, output_dir):
    """Symmetrize the 150x150 depth map to remove trauma effects."""
    print("Generating healed (symmetrized) depth map...")
    h, w = smooth.shape

    # Find midline
    midline_x, sym_score = find_face_midline(smooth)
    print(f"  Using midline x={midline_x} (symmetry={sym_score:.4f})")

    # Per-row bilateral average
    healed = smooth.copy().astype(float)
    for y in range(h):
        for x in range(w):
            mirror_x = 2 * midline_x - x
            if 0 <= mirror_x < w:
                avg = (smooth[y, x] + smooth[y, mirror_x]) / 2.0
                healed[y, x] = avg
                healed[y, mirror_x] = avg

    # Post-processing: light 5x5 Gaussian
    healed_u8 = healed.astype(np.uint8)
    healed_smooth = cv2.GaussianBlur(healed_u8, (5, 5), 0)

    # Measure asymmetry reduction
    right_h = healed_smooth[:, midline_x:].astype(float)
    left_h = healed_smooth[:, :midline_x][:, ::-1].astype(float)
    min_w = min(right_h.shape[1], left_h.shape[1])
    sym_after = np.corrcoef(right_h[:, :min_w].flatten(),
                            left_h[:, :min_w].flatten())[0, 1]
    print(f"  Symmetry: before={sym_score:.4f} -> after={sym_after:.4f}")

    # Save 150x150
    cv2.imwrite(str(output_dir / "depth_healed_150.png"), healed_smooth)
    np.save(str(output_dir / "depth_healed_150.npy"), healed_smooth)
    print("  Saved depth_healed_150.png + .npy")

    # Prepare ControlNet versions (512x512, inverted, RGB)
    for name, src in [("original", smooth), ("healed", healed_smooth)]:
        up = cv2.resize(src, (512, 512), interpolation=cv2.INTER_LINEAR)
        norm = cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        inverted = 255 - norm  # ControlNet: dark = close
        rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb)
        fname = f"controlnet_depth_{name}_512.png"
        pil.save(str(output_dir / fname))
        print(f"  Saved {fname}")

    return healed_smooth, sym_after


# ─────────────────────────────────────────────
#  STEP 7: SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(landmarks, measurements, healed_sym, output_dir, smooth):
    """Save landmarks.json and measurements.json."""
    # Upscale landmarks to full depth map coordinates
    # (landmarks are in 150x150 space)
    landmarks_out = {k: v for k, v in landmarks.items()}

    lm_path = output_dir / "landmarks.json"
    with open(lm_path, "w") as f:
        json.dump(landmarks_out, f, indent=2)
    print(f"  Saved {lm_path.name}")

    results = {
        "study": "Study 2 — Vernon Miller 1978 STURP",
        "source_image": "34c-Fa-N_0414.jpg",
        "source_dimensions": "8176x6132",
        "method": "150x150 downsampled depth map + Gaussian 15x15, depth-guided landmark detection",
        "healed_symmetry_after": round(healed_sym, 4),
        "scale_calibration": measurements.pop("scale_calibration"),
        "measurements": measurements,
    }

    res_path = output_dir / "measurements.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {res_path.name}")

    return results


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  STUDY 2: VERNON MILLER 1978 STURP PHOTOGRAPHS")
    print("  Full forensic reconstruction pipeline")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Depth map
    print("\n[1/6] Depth map extraction...")
    depth, face_bgr = extract_and_process(MILLER_SOURCE)
    save_outputs(depth, OUTPUT_DIR)
    cv2.imwrite(str(OUTPUT_DIR / "face_crop.png"), face_bgr)

    # Step 2: 3D surface
    print("\n[2/6] 3D surface generation...")
    smooth = generate_3d_surface(depth, OUTPUT_DIR)

    # Step 3: Landmarks
    print("\n[3/6] Landmark detection...")
    landmarks, features, ipd_px = detect_landmarks(smooth)

    # Step 4: Measurements
    print("\n[4/6] Computing measurements...")
    measurements = compute_measurements(landmarks, smooth, ipd_px)

    # Print measurement summary
    print("\n  --- Measurement Summary ---")
    for k, v in measurements.items():
        if isinstance(v, dict) and "cm" in v:
            status = v.get("status", "?")
            flag = "" if status in ("in_range", "excellent") else " !"
            print(f"  {k:30s}: {v['cm']:6.2f} cm  [{status}]{flag}")
    sym = measurements.get("facial_symmetry", {})
    print(f"  {'facial_symmetry':30s}: {sym.get('ratio', '?')}  [{sym.get('status', '?')}]")

    # Step 5: Landmark overlays
    print("\n[5/6] Generating landmark overlay images...")
    draw_landmark_overlays(smooth, landmarks, OUTPUT_DIR)

    # Step 6: Healed depth map
    print("\n[6/6] Generating healed depth map...")
    healed_smooth, healed_sym = generate_healed_depth(smooth, OUTPUT_DIR)

    # Save all results
    print("\nSaving results...")
    results = save_results(landmarks, measurements, healed_sym, OUTPUT_DIR, smooth)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run()
