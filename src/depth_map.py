"""
Depth map extraction from the Shroud of Turin.

Scientific basis: The Shroud's image intensity is directly proportional to
cloth-to-body distance (proven by STURP/VP-8 Image Analyzer research).
Darker regions = closer to body, lighter = farther.

This module converts the Shroud image into a depth map where pixel brightness
represents estimated distance from cloth to body surface.
"""

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_shroud_face(image_path: Path = None) -> np.ndarray:
    """Load the Shroud face image.

    If the full-length image is provided, crop to the face region.
    Otherwise load the face comparison image directly.
    """
    if image_path is None:
        # Try high-res Enrie first, fall back to other available images
        candidates = [
            SOURCE_DIR / "enrie_1931_face_hires.jpg",
            SOURCE_DIR / "holy_face_1909.jpg",
            SOURCE_DIR / "enrie_face_comparison.jpg",
        ]
        for candidate in candidates:
            if candidate.exists():
                image_path = candidate
                break
        else:
            image_path = candidates[0]  # Will error with helpful message

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return img


def extract_face_region(img: np.ndarray, source_name: str = "") -> np.ndarray:
    """Extract just the face region from a Shroud image.

    The Enrie high-res image is already a face-only negative.
    The comparison image has positive on left, negative on right.
    Other images may need different cropping.
    """
    h, w = img.shape[:2]
    aspect = h / w if w > 0 else 1

    # If the image is taller than wide (portrait), it's likely already the face
    # (e.g., enrie_1931_face_hires.jpg is 2388x3000)
    if aspect > 1.1:
        # Already a face image — just trim edges slightly
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        face = img[margin_y:h - margin_y, margin_x:w - margin_x]
        return face

    # If roughly square or landscape, it may be a comparison image
    # (positive on left, negative on right)
    if aspect < 0.9 or (0.9 <= aspect <= 1.1 and w > 600):
        # Take the right half (negative) for comparison images
        negative = img[:, w // 2:, :]
        nh, nw = negative.shape[:2]
        face = negative[int(nh * 0.05):int(nh * 0.95), int(nw * 0.1):int(nw * 0.9)]
        return face

    # Default: use center crop
    cx, cy = w // 2, h // 2
    crop_w, crop_h = int(w * 0.7), int(h * 0.9)
    face = img[cy - crop_h // 2:cy + crop_h // 2, cx - crop_w // 2:cx + crop_w // 2]
    return face


def create_depth_map(face_img: np.ndarray, smooth_sigma: float = 3.0) -> np.ndarray:
    """Convert the Shroud face image to a depth map.

    The Shroud's image formation creates a direct brightness-to-distance
    relationship: brighter areas on the negative = closer to body.

    Steps:
    1. Convert to grayscale
    2. Normalize to full 0-255 range
    3. The negative image already has the correct polarity:
       brighter = closer (nose, cheekbones) — this IS the depth map
    4. Apply Gaussian smoothing to reduce noise
    5. Optionally apply CLAHE for local contrast enhancement

    Returns depth map where higher values = closer to body surface.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Normalize brightness to use full 0-255 range
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This enhances local contrast without blowing out global contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # Gaussian smoothing to reduce noise while preserving structure
    if smooth_sigma > 0:
        kernel_size = int(smooth_sigma * 6) | 1  # Ensure odd
        smoothed = cv2.GaussianBlur(enhanced, (kernel_size, kernel_size), smooth_sigma)
    else:
        smoothed = enhanced

    return smoothed


def save_depth_map(depth_map: np.ndarray, name: str = "depth_map") -> tuple[Path, Path]:
    """Save depth map as both image and numpy array."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    img_path = PROCESSED_DIR / f"{name}.png"
    npy_path = PROCESSED_DIR / f"{name}.npy"

    cv2.imwrite(str(img_path), depth_map)
    np.save(str(npy_path), depth_map)

    print(f"Saved depth map image: {img_path}")
    print(f"Saved depth map array: {npy_path} (shape: {depth_map.shape})")

    return img_path, npy_path


def run_depth_extraction():
    """Full depth map extraction pipeline."""
    print("=== Shroud Depth Map Extraction ===\n")

    # Load image
    print("Loading Shroud face image...")
    img = load_shroud_face()
    print(f"  Image loaded: {img.shape}")

    # Extract face region from the negative
    print("Extracting face region (negative)...")
    face = extract_face_region(img)
    print(f"  Face region: {face.shape}")

    # Save the cropped face for reference
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PROCESSED_DIR / "face_crop.png"), face)

    # Create depth map
    print("Creating depth map (brightness -> distance)...")
    depth = create_depth_map(face, smooth_sigma=3.0)
    print(f"  Depth map: {depth.shape}, range [{depth.min()}, {depth.max()}]")

    # Save
    save_depth_map(depth)

    # Also save an unsmoothed version for comparison
    depth_raw = create_depth_map(face, smooth_sigma=0)
    save_depth_map(depth_raw, "depth_map_raw")

    print("\nDepth extraction complete.")
    return depth


if __name__ == "__main__":
    run_depth_extraction()
