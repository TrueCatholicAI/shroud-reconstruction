"""Register Sudarium stain mask with Shroud face depth map."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
STAIN_MASK_PATH = PROJECT / "output" / "sudarium" / "sudarium_stain_mask.png"
DEPTH_PATH = PROJECT / "data" / "final" / "depth_150x150_g15.npy"
LANDMARKS_PATH = PROJECT / "data" / "measurements" / "landmarks.json"
SUDARIUM_PATH = PROJECT / "data" / "source" / "sudarium" / "sudarium.jpg"
OUT_DIR = PROJECT / "output" / "sudarium"
RESULTS_JSON = PROJECT / "output" / "task_results" / "sudarium_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

# Load data
stain_mask = cv2.imread(str(STAIN_MASK_PATH), cv2.IMREAD_GRAYSCALE)
depth = np.load(DEPTH_PATH).astype(np.float64)
sudarium_img = cv2.imread(str(SUDARIUM_PATH))
sudarium_rgb = cv2.cvtColor(sudarium_img, cv2.COLOR_BGR2RGB)

with open(LANDMARKS_PATH) as f:
    landmarks_data = json.load(f)

print(f"Stain mask: {stain_mask.shape}")
print(f"Depth map: {depth.shape}")
print(f"Sudarium: {sudarium_img.shape}")

# The Sudarium image coordinate system: full image is 4160x3120
# The depth map is 150x150 face crop
# Landmarks are in original image coords (2150x2700 space)

# Scale landmarks to 150x150 depth map space
img_w = landmarks_data["image_size"]["width"]   # 2150
img_h = landmarks_data["image_size"]["height"]  # 2700
scale_x = 150.0 / img_w
scale_y = 150.0 / img_h

key_landmarks = landmarks_data["key_landmarks"]
landmarks_150 = {}
for name, lm in key_landmarks.items():
    landmarks_150[name] = {
        "x": round(lm["x"] * scale_x, 1),
        "y": round(lm["y"] * scale_y, 1),
    }

print(f"\nLandmarks in 150x150 space:")
for name in ["left_pupil", "right_pupil", "nose_tip", "chin", "upper_lip_center"]:
    if name in landmarks_150:
        lm = landmarks_150[name]
        print(f"  {name}: ({lm['x']}, {lm['y']})")

# Scale stain mask to match 150x150 depth map
# Assume both the Sudarium and Shroud covered the same face
# The Sudarium is roughly face-sized, so we scale it to match the depth map
stain_150 = cv2.resize(stain_mask, (150, 150), interpolation=cv2.INTER_AREA)
stain_binary = (stain_150 > 127).astype(np.uint8)

# Define facial landmark regions (circles of radius ~8px in 150x150 space)
REGION_RADIUS = 8
regions = {
    "left_eye": landmarks_150.get("left_pupil", {"x": 67, "y": 36}),
    "right_eye": landmarks_150.get("right_pupil", {"x": 84, "y": 34}),
    "nose": landmarks_150.get("nose_tip", {"x": 78, "y": 47}),
    "mouth": landmarks_150.get("upper_lip_center", {"x": 75, "y": 58}),
    "forehead": {"x": 75, "y": 20},  # approximate
    "chin": landmarks_150.get("chin", {"x": 78, "y": 72}),
    "left_cheek": landmarks_150.get("left_cheek", {"x": 49, "y": 47}),
    "right_cheek": landmarks_150.get("right_cheek", {"x": 101, "y": 47}),
}

# Compute overlap for each region
overlap_results = {}
for name, center in regions.items():
    cx, cy = int(round(center["x"])), int(round(center["y"]))
    # Create circular mask for this region
    region_mask = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(region_mask, (cx, cy), REGION_RADIUS, 1, -1)

    region_pixels = int(np.sum(region_mask))
    overlap_pixels = int(np.sum(region_mask & stain_binary))
    overlap_pct = round(overlap_pixels / region_pixels * 100, 1) if region_pixels > 0 else 0.0

    overlap_results[name] = {
        "center_x": cx,
        "center_y": cy,
        "region_pixels": region_pixels,
        "stain_pixels": overlap_pixels,
        "overlap_pct": overlap_pct,
    }
    print(f"  {name:15s}: {overlap_pct:5.1f}% stain overlap ({overlap_pixels}/{region_pixels} px)")

# Overall face overlap
face_mask = np.zeros((150, 150), dtype=np.uint8)
cv2.ellipse(face_mask, (75, 50), (45, 55), 0, 0, 360, 1, -1)  # rough face ellipse
face_pixels = int(np.sum(face_mask))
face_overlap = int(np.sum(face_mask & stain_binary))
face_overlap_pct = round(face_overlap / face_pixels * 100, 1) if face_pixels > 0 else 0.0
print(f"\n  Overall face: {face_overlap_pct}% stain overlap ({face_overlap}/{face_pixels} px)")

BG = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = 'white'
image_paths = []

# Figure 1: Registration overlay — depth map + stain mask
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_color(WHITE)

axes[0].imshow(depth, cmap='inferno')
axes[0].set_title('Shroud Face Depth Map', color=GOLD, fontweight='bold')

axes[1].imshow(stain_150, cmap='Reds')
axes[1].set_title('Sudarium Stains (scaled)', color=GOLD, fontweight='bold')

# Overlay
overlay = np.zeros((150, 150, 3))
depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
overlay[:, :, 0] = depth_norm * 0.7  # depth in red channel
overlay[:, :, 1] = depth_norm * 0.5  # depth in green
overlay[:, :, 2] = depth_norm * 0.3  # depth in blue
# Add stain as red overlay
stain_norm = stain_150.astype(np.float64) / 255.0
overlay[:, :, 0] = np.clip(overlay[:, :, 0] + stain_norm * 0.5, 0, 1)

# Mark landmarks
for name, center in regions.items():
    cx, cy = int(round(center["x"])), int(round(center["y"]))
    for ax in [axes[0], axes[2]]:
        ax.plot(cx, cy, 'o', color=GOLD, markersize=6, markeredgecolor=WHITE, markeredgewidth=1)

axes[2].imshow(overlay)
axes[2].set_title(f'Registration Overlay ({face_overlap_pct}% face coverage)', color=GOLD, fontweight='bold')

fig.suptitle('Sudarium of Oviedo — Shroud Face Registration', color=GOLD, fontsize=14, fontweight='bold')
plt.tight_layout()
path = OUT_DIR / 'sudarium_registration.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Figure 2: Region overlap bar chart
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(BG)
names = list(overlap_results.keys())
pcts = [overlap_results[n]["overlap_pct"] for n in names]
bars = ax.barh(names, pcts, color=GOLD, edgecolor=WHITE, linewidth=0.5)
ax.set_xlabel('Stain Overlap (%)', color=WHITE)
ax.set_title('Sudarium Stain Overlap by Facial Region', color=GOLD, fontsize=13, fontweight='bold')
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', va='center', color=WHITE, fontsize=10)
ax.set_xlim(0, max(pcts) * 1.2 if pcts else 100)
plt.tight_layout()
path = OUT_DIR / 'sudarium_overlap_chart.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Include stain extraction image from previous step
stain_extract_path = OUT_DIR / 'sudarium_stain_extraction.png'
if stain_extract_path.exists():
    image_paths.append(str(stain_extract_path.relative_to(PROJECT)))

results = {
    "method": "Scale Sudarium stain mask to 150x150, overlay with Shroud face depth map, compute stain-to-landmark overlap in circular ROIs",
    "registration": "Simple scaling (Sudarium resized to match 150x150 face crop). No geometric transformation or feature-based alignment.",
    "face_overlap_pct": face_overlap_pct,
    "region_overlap": overlap_results,
    "region_radius_px": REGION_RADIUS,
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps({k: v for k, v in results.items() if k != 'region_overlap'}, indent=2))
