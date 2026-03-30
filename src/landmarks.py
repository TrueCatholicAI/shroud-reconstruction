"""
Facial landmark detection on the Shroud of Turin.

Uses MediaPipe FaceLandmarker (tasks API, 478 landmarks) to detect facial geometry.
The Shroud image is unusual input -- we apply preprocessing to improve
detection reliability.
"""

import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEASUREMENTS_DIR = PROJECT_ROOT / "data" / "measurements"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "face_landmarker.task"

# Key landmark indices in MediaPipe Face Mesh (478 landmarks)
KEY_LANDMARKS = {
    # Eyes
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_pupil": 468,  # Iris landmarks
    "right_pupil": 473,
    # Nose
    "nose_tip": 1,
    "nose_bridge_top": 6,
    "nose_left_alar": 129,
    "nose_right_alar": 358,
    # Mouth
    "upper_lip_center": 0,
    "lower_lip_center": 17,
    "mouth_left": 61,
    "mouth_right": 291,
    # Jaw / Face outline
    "chin": 152,
    "jaw_left": 234,
    "jaw_right": 454,
    # Forehead / Brow
    "left_eyebrow_inner": 107,
    "left_eyebrow_outer": 70,
    "right_eyebrow_inner": 336,
    "right_eyebrow_outer": 300,
    # Cheekbones
    "left_cheek": 123,
    "right_cheek": 352,
}


def preprocess_for_detection(face_img: np.ndarray) -> np.ndarray:
    """Preprocess the Shroud face image to improve landmark detection.

    The Shroud image is very different from typical face photos:
    - Monochrome / sepia toned
    - Low contrast
    - Negative image characteristics
    """
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img.copy()

    # Histogram equalization to maximize contrast
    equalized = cv2.equalizeHist(gray)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(equalized)

    # Convert back to 3-channel RGB
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return rgb


def preprocess_inverted(face_img: np.ndarray) -> np.ndarray:
    """Invert the Shroud negative to create a positive-like image.

    The Shroud face on the Enrie negative appears as light features on dark
    background. Inverting makes it more like a conventional photograph,
    which face detectors are trained on.
    """
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img.copy()

    # Invert: negative -> positive-like
    inverted = cv2.bitwise_not(gray)

    # CLAHE on the inverted image
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)

    # Convert to RGB
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb


def preprocess_with_bilateral(face_img: np.ndarray) -> np.ndarray:
    """Smooth and enhance while preserving edges."""
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img.copy()

    # Bilateral filter preserves edges while smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Strong CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(filtered)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb


def detect_landmarks(face_img: np.ndarray) -> dict | None:
    """Detect facial landmarks using MediaPipe FaceLandmarker (tasks API).

    Tries multiple preprocessing strategies since the Shroud image is
    very different from typical face photos.

    Returns dict with landmark coordinates, or None if no face detected.
    """
    h_orig, w_orig = face_img.shape[:2]

    # Try multiple preprocessing approaches
    preprocessors = [
        ("standard", preprocess_for_detection),
        ("inverted", preprocess_inverted),
        ("bilateral", preprocess_with_bilateral),
    ]

    for preproc_name, preproc_fn in preprocessors:
        preprocessed = preproc_fn(face_img)
        h, w = preprocessed.shape[:2]

        # Also try at smaller scale (face detectors sometimes work
        # better on typical photo resolutions like 640x480 or 1280x960)
        scales = [1.0]
        if max(h, w) > 1500:
            scales.append(800 / max(h, w))

        for scale in scales:
            if scale != 1.0:
                sh, sw = int(h * scale), int(w * scale)
                img_scaled = cv2.resize(preprocessed, (sw, sh))
            else:
                img_scaled = preprocessed
                sh, sw = h, w

            result = _try_detect(img_scaled, sw, sh, scale, w, h, preproc_name)
            if result is not None:
                return result

    print("  WARNING: No face detected with any preprocessing strategy.")
    return None


def _try_detect(preprocessed, w, h, scale, orig_w, orig_h, preproc_name):
    """Try detection at a specific confidence level."""
    for min_confidence in [0.3, 0.2, 0.1, 0.05]:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=0.1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            # MediaPipe expects RGB format
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=preprocessed,
            )
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                face_lms = result.face_landmarks[0]
                scale_label = f" @ {scale:.2f}x" if scale != 1.0 else ""
                print(f"  Face detected! (preprocessing: {preproc_name}, "
                      f"confidence: {min_confidence}{scale_label})")

                # Extract all landmarks, scaled back to original image coords
                all_landmarks = {}
                for idx, lm in enumerate(face_lms):
                    all_landmarks[idx] = {
                        "x": lm.x * orig_w,
                        "y": lm.y * orig_h,
                        "z": lm.z * orig_w,
                    }

                # Extract key named landmarks
                key_landmarks = {}
                for name, idx in KEY_LANDMARKS.items():
                    if idx < len(face_lms):
                        lm = face_lms[idx]
                        key_landmarks[name] = {
                            "x": lm.x * orig_w,
                            "y": lm.y * orig_h,
                            "z": lm.z * orig_w,
                            "index": idx,
                        }

                return {
                    "all_landmarks": all_landmarks,
                    "key_landmarks": key_landmarks,
                    "image_size": {"width": orig_w, "height": orig_h},
                    "detection_confidence": min_confidence,
                    "num_landmarks": len(all_landmarks),
                    "preprocessing": preproc_name,
                    "detection_scale": scale,
                }

    return None


def draw_landmarks(face_img: np.ndarray, landmarks: dict) -> np.ndarray:
    """Draw detected landmarks on the face image for visualization."""
    vis = face_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Scale dot/text size relative to image dimensions
    h, w = vis.shape[:2]
    dot_radius = max(1, min(w, h) // 500)
    key_radius = max(3, min(w, h) // 200)
    font_scale = max(0.3, min(w, h) / 3000)

    # Draw all landmarks as small dots
    for idx_str, lm in landmarks["all_landmarks"].items():
        x, y = int(lm["x"]), int(lm["y"])
        cv2.circle(vis, (x, y), dot_radius, (0, 255, 0), -1)

    # Draw key landmarks as larger labeled circles
    for name, lm in landmarks["key_landmarks"].items():
        x, y = int(lm["x"]), int(lm["y"])
        cv2.circle(vis, (x, y), key_radius, (0, 0, 255), -1)
        cv2.putText(vis, name[:10], (x + key_radius + 2, y - key_radius),
                     cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)

    return vis


def save_landmarks(landmarks: dict, name: str = "landmarks"):
    """Save landmarks as JSON."""
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MEASUREMENTS_DIR / f"{name}.json"

    serializable = landmarks.copy()
    serializable["all_landmarks"] = {
        str(k): v for k, v in landmarks["all_landmarks"].items()
    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved landmarks: {path}")
    return path


def _load_enrie_face() -> np.ndarray | None:
    """Load the Enrie 1931 face image (the actual Shroud photograph).

    The Enrie negative is inverted to create a positive-like image
    since face detectors are trained on normal photographs.

    IMPORTANT: We deliberately do NOT use the Holy Face 1909 reproduction
    because it is a devotional artistic copy, not the actual Shroud data.
    Using it would introduce exactly the artistic bias this project
    exists to eliminate.
    """
    source_dir = PROJECT_ROOT / "data" / "source"

    # Primary source: Enrie 1931 high-res face (photographic negative)
    enrie_path = source_dir / "enrie_1931_face_hires.jpg"
    if not enrie_path.exists():
        print(f"  Enrie source image not found at: {enrie_path}")
        return None

    img = cv2.imread(str(enrie_path))
    if img is None:
        print(f"  Failed to load: {enrie_path}")
        return None

    print(f"  Loaded Enrie 1931 negative: {img.shape}")

    # Extract face region (same logic as depth_map.py)
    h, w = img.shape[:2]
    aspect = h / w if w > 0 else 1
    if aspect > 1.1:
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        face = img[margin_y:h - margin_y, margin_x:w - margin_x]
    else:
        face = img

    # Convert to grayscale and invert (negative -> positive)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    positive = cv2.bitwise_not(gray)

    # Save the raw positive for reference
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PROCESSED_DIR / "enrie_positive_raw.png"), positive)

    return positive


def _preprocess_enrie_for_detection(positive: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Create multiple preprocessed versions of the Enrie positive for detection.

    The Shroud image has heavy cloth texture noise that confuses face detectors.
    We try several preprocessing strategies to help the detector find the face
    while preserving the actual Shroud geometry.

    Returns list of (name, rgb_image) tuples to try.
    """
    versions = []

    # Version 1: CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(positive)
    versions.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)))

    # Version 2: Gaussian blur to reduce cloth texture + CLAHE
    blurred = cv2.GaussianBlur(positive, (15, 15), 0)
    enhanced2 = clahe.apply(blurred)
    versions.append(("blur15_clahe", cv2.cvtColor(enhanced2, cv2.COLOR_GRAY2RGB)))

    # Version 3: Stronger blur (21px) + CLAHE
    blurred21 = cv2.GaussianBlur(positive, (21, 21), 0)
    enhanced3 = clahe.apply(blurred21)
    versions.append(("blur21_clahe", cv2.cvtColor(enhanced3, cv2.COLOR_GRAY2RGB)))

    # Version 4: Bilateral filter (edge-preserving smoothing) + CLAHE
    bilateral = cv2.bilateralFilter(positive, 11, 75, 75)
    enhanced4 = clahe.apply(bilateral)
    versions.append(("bilateral_clahe", cv2.cvtColor(enhanced4, cv2.COLOR_GRAY2RGB)))

    # Version 5: Heavy blur (31px) + histogram equalization (last resort)
    blurred31 = cv2.GaussianBlur(positive, (31, 31), 0)
    equalized = cv2.equalizeHist(blurred31)
    versions.append(("blur31_histeq", cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)))

    return versions


def run_landmark_detection():
    """Full landmark detection pipeline.

    Uses the Enrie 1931 photograph (inverted negative -> positive) as the
    source image. Does NOT use the Holy Face 1909 devotional reproduction.
    """
    print("=== Shroud Facial Landmark Detection ===\n")
    print("Source: Enrie 1931 photograph (NOT Holy Face 1909 reproduction)")
    print("Reason: Holy Face is an artistic copy that introduces bias.\n")

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Run: python -c \"import requests; open('data/models/face_landmarker.task','wb').write(requests.get('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task').content)\"")
        return None

    # Load the actual Enrie Shroud photograph
    positive = _load_enrie_face()
    if positive is None:
        print("Cannot proceed without Enrie source image.")
        return None

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save the positive as the working face crop (replaces any old Holy Face crop)
    face_rgb = cv2.cvtColor(positive, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(PROCESSED_DIR / "face_crop.png"), face_rgb)
    print(f"  Saved Enrie positive as face_crop.png: {positive.shape}")

    # Try detection on multiple preprocessed versions
    print("\nRunning MediaPipe FaceLandmarker on Enrie positive...")
    print("(Trying multiple preprocessing strategies for cloth texture noise)\n")

    preprocessed_versions = _preprocess_enrie_for_detection(positive)

    landmarks = None
    successful_preproc = None

    for preproc_name, preproc_img in preprocessed_versions:
        print(f"  Trying preprocessing: {preproc_name}...")
        # Save each preprocessed version for inspection
        cv2.imwrite(
            str(PROCESSED_DIR / f"enrie_preproc_{preproc_name}.png"),
            preproc_img,
        )

        result = detect_landmarks(preproc_img)
        if result is not None:
            landmarks = result
            successful_preproc = preproc_name
            print(f"  SUCCESS with {preproc_name}!")
            break
        else:
            print(f"  No face detected with {preproc_name}")

    # Also try the depth map as a last resort (it's already smoothed)
    if landmarks is None:
        print("\n  Trying depth map image (already smoothed)...")
        depth_path = PROCESSED_DIR / "depth_map.png"
        if depth_path.exists():
            depth_img = cv2.imread(str(depth_path))
            landmarks = detect_landmarks(depth_img)
            if landmarks is not None:
                successful_preproc = "depth_map"
                print("  SUCCESS with depth map!")

    if landmarks is None:
        print("\n  RESULT: Landmark detection failed on all Enrie preprocessing variants.")
        print("  This is not unexpected -- the Shroud image is very different from typical photos.")
        print("  Next steps to try:")
        print("    - Dlib HOG face detector (different architecture)")
        print("    - Manual landmark annotation (guided by depth map features)")
        print("    - Hybrid: use depth map peaks to seed approximate landmark positions")
        return None

    print(f"\nDetected {landmarks['num_landmarks']} landmarks")
    print(f"Key landmarks found: {len(landmarks['key_landmarks'])}")
    print(f"Successful preprocessing: {successful_preproc}")

    # Save landmarks
    save_landmarks(landmarks)

    # Draw visualization on the Enrie positive
    vis = draw_landmarks(face_rgb, landmarks)
    vis_path = PROCESSED_DIR / "landmarks_overlay.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"Saved landmark overlay: {vis_path}")

    # Also save overlay on the preprocessed version that worked
    if successful_preproc and successful_preproc != "depth_map":
        for name, img in preprocessed_versions:
            if name == successful_preproc:
                vis2 = draw_landmarks(img, landmarks)
                vis2_path = PROCESSED_DIR / "landmarks_overlay_preprocessed.png"
                cv2.imwrite(str(vis2_path), vis2)
                print(f"Saved preprocessed overlay: {vis2_path}")
                break

    return landmarks


if __name__ == "__main__":
    run_landmark_detection()
