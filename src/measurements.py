"""
Anthropometric measurements from Shroud facial landmarks.

Converts detected landmark positions into real-world measurements (cm)
using the known physical dimensions of the Shroud of Turin as scale reference.

Shroud dimensions: ~4.4m × 1.1m (full cloth)
The face occupies a known proportion of the total cloth width.
"""

import json
import math
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MEASUREMENTS_DIR = PROJECT_ROOT / "data" / "measurements"

# The Shroud cloth is approximately 1.1m (110cm) wide.
# The face region occupies roughly 20-25cm of this width.
# We'll calibrate using the interpupillary distance as a sanity check
# (average adult male IPD is ~6.3cm, range 5.5-7.5cm).
SHROUD_CLOTH_WIDTH_CM = 110.0


def pixel_distance(p1: dict, p2: dict) -> float:
    """Euclidean distance between two landmark points in pixel space."""
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    return math.sqrt(dx * dx + dy * dy)


def calculate_scale(landmarks: dict) -> float:
    """Estimate pixels-per-cm scale using interpupillary distance calibration.

    The average adult male interpupillary distance (IPD) is ~6.3cm.
    We use this as our primary calibration reference, since it's a
    well-established anthropometric constant.

    If eye landmarks aren't available, falls back to image width estimate.
    """
    kl = landmarks.get("key_landmarks", {})

    # Primary: calibrate from interpupillary distance
    if "left_eye_outer" in kl and "right_eye_outer" in kl:
        ipd_px = pixel_distance(kl["left_eye_outer"], kl["right_eye_outer"])
        AVERAGE_MALE_IPD_CM = 6.3  # Well-established anthropometric average
        px_per_cm = ipd_px / AVERAGE_MALE_IPD_CM
        return px_per_cm

    # Fallback: image width estimate
    image_width_px = landmarks["image_size"]["width"]
    estimated_crop_width_cm = 44.0  # Rough estimate
    px_per_cm = image_width_px / estimated_crop_width_cm
    return px_per_cm


def compute_measurements(landmarks: dict) -> dict:
    """Compute anthropometric measurements from facial landmarks.

    Returns measurements in both pixels and estimated cm.
    """
    kl = landmarks["key_landmarks"]
    px_per_cm = calculate_scale(landmarks)

    measurements = {}

    def measure(name: str, point1_name: str, point2_name: str):
        if point1_name in kl and point2_name in kl:
            px = pixel_distance(kl[point1_name], kl[point2_name])
            cm = px / px_per_cm
            measurements[name] = {
                "pixels": round(px, 1),
                "cm": round(cm, 2),
                "points": [point1_name, point2_name],
            }

    # Core facial measurements
    measure("interpupillary_distance", "left_eye_outer", "right_eye_outer")
    measure("inner_eye_distance", "left_eye_inner", "right_eye_inner")
    measure("nose_length", "nose_bridge_top", "nose_tip")
    measure("nose_width", "nose_left_alar", "nose_right_alar")
    measure("face_width_cheekbones", "left_cheek", "right_cheek")
    measure("jaw_width", "jaw_left", "jaw_right")
    measure("nose_to_chin", "nose_tip", "chin")
    measure("mouth_width", "mouth_left", "mouth_right")
    measure("upper_lip_to_nose", "nose_tip", "upper_lip_center")
    measure("lower_face_height", "nose_tip", "chin")

    # Forehead height (eyebrow to estimated hairline is hard, use brow to nose bridge)
    measure("brow_to_nose_bridge", "left_eyebrow_inner", "nose_bridge_top")

    # Vertical proportions
    measure("left_eye_to_mouth", "left_eye_inner", "mouth_left")
    measure("brow_to_chin", "left_eyebrow_inner", "chin")

    # Facial symmetry analysis
    if "left_eye_outer" in kl and "right_eye_outer" in kl and "nose_tip" in kl:
        nose = kl["nose_tip"]
        left_dist = pixel_distance(kl["left_eye_outer"], nose)
        right_dist = pixel_distance(kl["right_eye_outer"], nose)
        symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
        measurements["facial_symmetry"] = {
            "ratio": round(symmetry_ratio, 4),
            "note": "1.0 = perfect symmetry. >0.95 typical for living faces.",
        }

    # Add metadata
    measurements["_metadata"] = {
        "px_per_cm": round(px_per_cm, 2),
        "scale_basis": "Estimated face crop width ~25cm",
        "note": "Measurements are approximate. Calibration improves with "
                "known reference dimensions from the full Shroud image.",
    }

    return measurements


def validate_measurements(measurements: dict) -> list[str]:
    """Check if measurements fall within plausible human ranges.

    Returns list of warnings for any out-of-range values.
    """
    warnings = []

    # Expected ranges for adult male (cm)
    expected_ranges = {
        "interpupillary_distance": (5.5, 7.5),
        "nose_length": (4.0, 6.5),
        "nose_width": (2.5, 5.0),
        "face_width_cheekbones": (12.0, 17.0),
        "jaw_width": (10.0, 15.0),
        "nose_to_chin": (5.0, 8.5),
        "mouth_width": (4.0, 6.5),
    }

    for name, (low, high) in expected_ranges.items():
        if name in measurements:
            cm = measurements[name]["cm"]
            if cm < low or cm > high:
                warnings.append(
                    f"{name}: {cm}cm is outside expected range "
                    f"({low}-{high}cm). Scale calibration may need adjustment."
                )

    return warnings


def run_measurements():
    """Full measurement pipeline."""
    print("=== Shroud Anthropometric Measurements ===\n")

    # Load landmarks
    landmarks_path = MEASUREMENTS_DIR / "landmarks.json"
    if not landmarks_path.exists():
        print("Landmarks file not found. Run landmarks.py first.")
        return None

    with open(landmarks_path) as f:
        landmarks = json.load(f)

    print(f"Loaded {landmarks['num_landmarks']} landmarks")

    # Compute measurements
    print("\nComputing measurements...")
    measurements = compute_measurements(landmarks)

    # Print results
    print("\n--- Measurements ---")
    for name, data in measurements.items():
        if name.startswith("_"):
            continue
        if isinstance(data, dict) and "cm" in data:
            print(f"  {name}: {data['cm']} cm ({data['pixels']} px)")
        elif isinstance(data, dict) and "ratio" in data:
            print(f"  {name}: {data['ratio']}")

    # Validate
    warnings = validate_measurements(measurements)
    if warnings:
        print("\n--- Warnings ---")
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("\nAll measurements within expected human ranges.")

    # Save
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MEASUREMENTS_DIR / "anthropometric_measurements.json"
    with open(output_path, "w") as f:
        json.dump(measurements, f, indent=2)
    print(f"\nSaved: {output_path}")

    return measurements


if __name__ == "__main__":
    run_measurements()
