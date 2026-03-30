"""Task H: Scourge mark pattern analysis on full-body depth map."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm

print("=== Scourge Mark Pattern Analysis ===")

# Load full-body frontal source and depth
img = cv2.imread('data/source/shroud_full_negatives.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
frontal = img[:, :w//2]
frontal = frontal[:, :int(frontal.shape[1] * 0.95)]
print(f"Frontal body: {frontal.shape}")

# CLAHE depth
norm_img = cv2.normalize(frontal, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_raw = clahe.apply(norm_img)

# Heavy smoothing for the "expected" body surface
depth_smooth = gaussian_filter(depth_raw.astype(np.float64), sigma=20)

# High-frequency residual = raw - smooth
residual = depth_raw.astype(np.float64) - depth_smooth
print(f"Body residual: range [{residual.min():.1f}, {residual.max():.1f}], std={residual.std():.1f}")

# Scourge marks would appear as dumbbell-shaped depressions (from Roman flagrum tips)
# Focus on torso region (roughly 15-55% of body height, excluding face and legs)
bh, bw = frontal.shape
torso_y1 = int(bh * 0.15)
torso_y2 = int(bh * 0.55)

torso_residual = residual[torso_y1:torso_y2, :]
torso_src = frontal[torso_y1:torso_y2, :]
torso_depth = depth_raw[torso_y1:torso_y2, :]
print(f"Torso region: {torso_residual.shape}")
print(f"Torso residual: range [{torso_residual.min():.1f}, {torso_residual.max():.1f}], std={torso_residual.std():.1f}")

# Threshold for candidate mark regions
mean_t = torso_residual.mean()
std_t = torso_residual.std()

# Scourge marks should appear as raised features (brighter in negative)
# or depressed features depending on interpretation
# Try both positive and negative thresholds
raised_mask = torso_residual > (mean_t + 1.5 * std_t)
depressed_mask = torso_residual < (mean_t - 1.5 * std_t)

print(f"Raised features (>{mean_t + 1.5*std_t:.1f}): {np.sum(raised_mask)} px ({100*np.sum(raised_mask)/raised_mask.size:.1f}%)")
print(f"Depressed features (<{mean_t - 1.5*std_t:.1f}): {np.sum(depressed_mask)} px ({100*np.sum(depressed_mask)/depressed_mask.size:.1f}%)")

# Try to detect dumbbell/elongated features using morphological analysis
# Roman flagrum tips create small (~1cm) paired marks
# At full-body resolution: 1cm ~ 2370/200cm body height * 1cm ~ 12px
# Apply connected component analysis to the threshold mask
raised_uint8 = (raised_mask * 255).astype(np.uint8)

# Morphological opening to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(raised_uint8, cv2.MORPH_OPEN, kernel)

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)
print(f"\nConnected components (raised, after opening): {num_labels - 1}")

# Filter by size — flagrum marks should be small (5-50 px area at this resolution)
small_marks = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    w_comp = stats[i, cv2.CC_STAT_WIDTH]
    h_comp = stats[i, cv2.CC_STAT_HEIGHT]
    if 5 <= area <= 80:
        aspect = max(w_comp, h_comp) / (min(w_comp, h_comp) + 1e-8)
        small_marks.append({
            'label': i, 'area': area, 'cx': centroids[i][0], 'cy': centroids[i][1],
            'aspect': aspect, 'w': w_comp, 'h': h_comp,
        })

print(f"Small mark candidates (5-80 px area): {len(small_marks)}")

# === Visualization 1: Full body residual ===
fig, axes = plt.subplots(1, 3, figsize=(18, 14))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(frontal, cmap='gray')
axes[0].set_title('Frontal Body (Negative)', color='white', fontsize=12)
axes[0].axhline(y=torso_y1, color='red', linestyle='--', alpha=0.5)
axes[0].axhline(y=torso_y2, color='red', linestyle='--', alpha=0.5)
axes[0].axis('off')

vmax = max(abs(residual.min()), abs(residual.max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
axes[1].imshow(residual, cmap='RdBu_r', norm=norm)
axes[1].set_title('High-Frequency Residual', color='white', fontsize=12)
axes[1].axhline(y=torso_y1, color='yellow', linestyle='--', alpha=0.5)
axes[1].axhline(y=torso_y2, color='yellow', linestyle='--', alpha=0.5)
axes[1].axis('off')

# Torso zoom with marks
torso_rgb = cv2.cvtColor(torso_src, cv2.COLOR_GRAY2RGB)
for mark in small_marks:
    cx, cy = int(mark['cx']), int(mark['cy'])
    cv2.circle(torso_rgb, (cx, cy), 5, (0, 255, 0), 1)

axes[2].imshow(torso_rgb)
axes[2].set_title(f'Torso Region — {len(small_marks)} Candidate Marks', color='white', fontsize=12)
axes[2].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Scourge Mark Pattern Analysis (Exploratory)', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/scourge_marks_overview.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: scourge_marks_overview.png")

# === Visualization 2: Torso residual detail ===
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(torso_src, cmap='gray')
axes[0].set_title('Torso Source', color='white', fontsize=12)
axes[0].axis('off')

vmax_t = max(abs(torso_residual.min()), abs(torso_residual.max()))
norm_t = TwoSlopeNorm(vmin=-vmax_t, vcenter=0, vmax=vmax_t)
axes[1].imshow(torso_residual, cmap='RdBu_r', norm=norm_t)
axes[1].set_title('Torso Residual (red=raised, blue=depressed)', color='white', fontsize=12)
axes[1].axis('off')

# Mark overlay on depth
torso_depth_rgb = cv2.cvtColor(torso_depth, cv2.COLOR_GRAY2RGB)
for mark in small_marks:
    cx, cy = int(mark['cx']), int(mark['cy'])
    cv2.circle(torso_depth_rgb, (cx, cy), 4, (0, 255, 0), 1)
axes[2].imshow(torso_depth_rgb)
axes[2].set_title(f'Candidate Marks on Depth ({len(small_marks)} found)', color='white', fontsize=12)
axes[2].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Torso Scourge Mark Analysis', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/scourge_marks_torso.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: scourge_marks_torso.png")

# === 3D surface with marks ===
from scipy.ndimage import zoom as scipy_zoom

# Downsample torso for 3D
ds_h, ds_w = 200, 100
torso_3d = scipy_zoom(torso_depth.astype(np.float32), (ds_h/torso_depth.shape[0], ds_w/torso_depth.shape[1]), order=1)
mark_3d = cv2.resize(opened, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)

from matplotlib.cm import inferno
depth_norm = (torso_3d - torso_3d.min()) / (torso_3d.max() - torso_3d.min() + 1e-8)
colors = inferno(depth_norm)
colors[mark_3d > 0, 0] = 0.0
colors[mark_3d > 0, 1] = 1.0
colors[mark_3d > 0, 2] = 0.0

X = np.arange(ds_w)
Y = np.arange(ds_h)
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(8, 12))
fig.patch.set_facecolor('#1a1a1a')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#1a1a1a')
ax.plot_surface(X, Y, torso_3d, facecolors=colors, linewidth=0, antialiased=True, rstride=1, cstride=1)
ax.set_zlim(0, 280)
ax.view_init(elev=30, azim=135)
ax.set_title('Torso 3D Surface with Candidate Marks (green)', color='white', fontsize=13)
ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.tight_layout()
plt.savefig('output/analysis/scourge_marks_3d.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: scourge_marks_3d.png")

print("\n--- Summary ---")
print(f"Total small mark candidates in torso: {len(small_marks)}")
print(f"Published literature suggests ~120 scourge marks on the full body")
print(f"Our detection found {len(small_marks)} candidates in the torso region")
print("\nLIMITATIONS:")
print("- Source resolution is only ~12 px/cm at body scale")
print("- Cannot distinguish scourge marks from cloth weave intersections")
print("- No morphological filtering for dumbbell shape (would need higher resolution)")
print("- Roman flagrum marks (~10mm) are only ~12px at this resolution")
print("- This is exploratory proof-of-concept only")
print("- Multi-spectral data (UV fluorescence) would be required for validation")

print("\n=== Scourge Mark Analysis Complete ===")
