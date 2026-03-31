"""Extract bloodstain regions from Sudarium photograph using color thresholding."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SUDARIUM_PATH = PROJECT / "data" / "source" / "sudarium" / "sudarium.jpg"
OUT_DIR = PROJECT / "output" / "sudarium"
RESULTS_JSON = PROJECT / "output" / "task_results" / "sudarium_stains_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(SUDARIUM_PATH))
if img is None:
    raise FileNotFoundError(f"Cannot load {SUDARIUM_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(f"Loaded Sudarium: {img.shape}")

# Blood/brownish-red stain detection in HSV space
# Blood on aged linen appears as brownish-red to dark brown
# Hue: 0-20 (red-orange) and 160-180 (red wrap-around)
# Saturation: moderate (40-255) — not grayish background
# Value: moderate (30-200) — not too bright (linen) or too dark (shadow)

mask1 = cv2.inRange(hsv, (0, 40, 30), (20, 255, 200))      # red-orange
mask2 = cv2.inRange(hsv, (160, 40, 30), (180, 255, 200))    # red wrap
stain_mask = cv2.bitwise_or(mask1, mask2)

# Morphological cleanup: remove small noise, fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel)
stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, kernel)

# Coverage statistics
total_pixels = stain_mask.size
stain_pixels = int(np.count_nonzero(stain_mask))
coverage_pct = round(stain_pixels / total_pixels * 100, 2)
print(f"Stain coverage: {stain_pixels}/{total_pixels} pixels = {coverage_pct}%")

# Find contours for region analysis
contours, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
print(f"Found {len(contours)} stain regions")
if contour_areas:
    print(f"Largest region: {contour_areas[0]:.0f} px, top 5: {[f'{a:.0f}' for a in contour_areas[:5]]}")

BG = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = 'white'
image_paths = []

# Save binary mask
mask_path = OUT_DIR / 'sudarium_stain_mask.png'
cv2.imwrite(str(mask_path), stain_mask)
image_paths.append(str(mask_path.relative_to(PROJECT)))

# Figure 1: Side by side — original, mask, overlay
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_color(WHITE)

axes[0].imshow(img_rgb)
axes[0].set_title('Sudarium of Oviedo', color=GOLD, fontweight='bold')

axes[1].imshow(stain_mask, cmap='Reds')
axes[1].set_title(f'Stain Mask ({coverage_pct}% coverage)', color=GOLD, fontweight='bold')

# Overlay: original with red-tinted stain regions
overlay = img_rgb.copy()
stain_color = np.zeros_like(overlay)
stain_color[:, :, 0] = 255  # red
overlay[stain_mask > 0] = cv2.addWeighted(
    overlay[stain_mask > 0], 0.5,
    stain_color[stain_mask > 0], 0.5, 0
)
axes[2].imshow(overlay)
axes[2].set_title('Stain Overlay', color=GOLD, fontweight='bold')

fig.suptitle('Sudarium Bloodstain Extraction', color=GOLD, fontsize=14, fontweight='bold')
plt.tight_layout()
path = OUT_DIR / 'sudarium_stain_extraction.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

results = {
    "source_image": str(SUDARIUM_PATH.relative_to(PROJECT)),
    "source_resolution": f"{img.shape[1]}x{img.shape[0]}",
    "method": "HSV color thresholding for brownish-red regions (H:0-20/160-180, S:40-255, V:30-200) with morphological cleanup",
    "total_pixels": total_pixels,
    "stain_pixels": stain_pixels,
    "coverage_pct": coverage_pct,
    "num_regions": len(contours),
    "largest_region_area_px": int(contour_areas[0]) if contour_areas else 0,
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps(results, indent=2))
