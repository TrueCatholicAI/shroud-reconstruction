"""Phase 4.2: Blood stain spatial mapping — isolate high-frequency surface features."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

print("=== Blood Stain Spatial Mapping (Exploratory) ===")

# Load Enrie face source and depth
enrie_src = cv2.imread('data/source/enrie_1931_face_hires.jpg', cv2.IMREAD_GRAYSCALE)
depth_full = np.load('data/processed/depth_map_smooth_15.npy')  # 3000x2388
print(f"Enrie source: {enrie_src.shape}")
print(f"Depth (smooth 15): {depth_full.shape}")

# The idea: subtract the smoothed depth map from the original image
# The residual contains high-frequency features: bloodstains, cloth texture, etc.
# The depth map IS the smoothed version; we need the raw CLAHE depth too
depth_raw = np.load('data/processed/depth_map.npy')
print(f"Depth (raw CLAHE): {depth_raw.shape}, range [{depth_raw.min()}, {depth_raw.max()}]")

# === Method 1: Source minus smoothed depth ===
# Normalize both to same range first
src_norm = cv2.normalize(enrie_src, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
depth_norm = cv2.normalize(depth_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)

# Ensure same size
if src_norm.shape != depth_norm.shape:
    # Resize depth to match source
    depth_resized = cv2.resize(depth_norm, (src_norm.shape[1], src_norm.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
else:
    depth_resized = depth_norm

# High-frequency residual = source - smooth
residual = src_norm - depth_resized
print(f"Residual (source - smooth): range [{residual.min():.1f}, {residual.max():.1f}], std={residual.std():.1f}")

# === Method 2: Raw CLAHE minus smoothed CLAHE ===
raw_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
if raw_norm.shape != depth_norm.shape:
    raw_resized = cv2.resize(raw_norm, (depth_norm.shape[1], depth_norm.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
else:
    raw_resized = raw_norm

hf_signal = raw_resized - depth_resized
print(f"HF signal (raw CLAHE - smooth): range [{hf_signal.min():.1f}, {hf_signal.max():.1f}], std={hf_signal.std():.1f}")

# === Threshold for candidate stain regions ===
# Blood stains on the Shroud appear as darker patches (in the negative, they appear brighter)
# In the residual, they should appear as positive spikes (brighter than smooth depth)
# Use upper tail: residual > mean + 2*std
mean_r = residual.mean()
std_r = residual.std()
threshold = mean_r + 2.0 * std_r
stain_mask = residual > threshold
print(f"Candidate stain threshold: {threshold:.1f} (mean + 2*std)")
print(f"Candidate stain pixels: {np.sum(stain_mask)} ({100*np.sum(stain_mask)/stain_mask.size:.1f}%)")

# Also try the negative tail (depressed features)
threshold_neg = mean_r - 2.0 * std_r
depressed_mask = residual < threshold_neg
print(f"Depressed feature pixels: {np.sum(depressed_mask)} ({100*np.sum(depressed_mask)/depressed_mask.size:.1f}%)")

# === Visualization 1: Residual map ===
fig, axes = plt.subplots(1, 3, figsize=(21, 8))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(enrie_src, cmap='gray')
axes[0].set_title('Enrie Source (Negative)', color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(depth_full, cmap='inferno')
axes[1].set_title('Smoothed Depth Map', color='white', fontsize=12)
axes[1].axis('off')

from matplotlib.colors import TwoSlopeNorm
vmax = max(abs(residual.min()), abs(residual.max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = axes[2].imshow(residual, cmap='RdBu_r', norm=norm)
axes[2].set_title('High-Frequency Residual', color='white', fontsize=12)
axes[2].axis('off')
cbar = plt.colorbar(im, ax=axes[2], fraction=0.03, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Blood Stain Isolation: Source Minus Smoothed Depth', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/bloodstain/residual_map.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: residual_map.png")

# === Visualization 2: Stain candidate overlay on depth ===
# Create colored overlay: red for raised (candidate stains), blue for depressed
overlay = np.zeros((*depth_full.shape, 3), dtype=np.uint8)
# Base: depth in grayscale
depth_vis = cv2.normalize(depth_full, None, 0, 200, cv2.NORM_MINMAX).astype(np.uint8)
overlay[:, :, 0] = depth_vis
overlay[:, :, 1] = depth_vis
overlay[:, :, 2] = depth_vis

# Resize masks to match depth_full if needed
if stain_mask.shape != depth_full.shape:
    stain_mask_rs = cv2.resize(stain_mask.astype(np.uint8),
                               (depth_full.shape[1], depth_full.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    depressed_mask_rs = cv2.resize(depressed_mask.astype(np.uint8),
                                   (depth_full.shape[1], depth_full.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
else:
    stain_mask_rs = stain_mask
    depressed_mask_rs = depressed_mask

# Red channel for raised (candidate stains)
overlay[stain_mask_rs, 0] = 255
overlay[stain_mask_rs, 1] = 50
overlay[stain_mask_rs, 2] = 50

# Blue channel for depressed
overlay[depressed_mask_rs, 0] = 50
overlay[depressed_mask_rs, 1] = 50
overlay[depressed_mask_rs, 2] = 255

fig, ax = plt.subplots(figsize=(8, 10))
fig.patch.set_facecolor('#1a1a1a')
ax.imshow(overlay)
ax.set_title('Candidate Stain Regions on Depth Map\n(Red=raised, Blue=depressed)',
             color='white', fontsize=13)
ax.axis('off')
ax.set_facecolor('#1a1a1a')
plt.tight_layout()
plt.savefig('output/bloodstain/stain_overlay_depth.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: stain_overlay_depth.png")

# === Visualization 3: 3D surface with stain overlay ===
# Downsample for 3D rendering
from scipy.ndimage import zoom as scipy_zoom
ds = 150
h, w = depth_full.shape
depth_3d = scipy_zoom(depth_full.astype(np.float32), (ds/h, ds/w), order=1)
stain_3d = cv2.resize(stain_mask.astype(np.uint8),
                       (ds, ds), interpolation=cv2.INTER_NEAREST).astype(bool)
depressed_3d = cv2.resize(depressed_mask.astype(np.uint8),
                           (ds, ds), interpolation=cv2.INTER_NEAREST).astype(bool)

# Create color array for surface
colors = np.zeros((ds, ds, 4))
# Base: inferno colormap
from matplotlib.cm import inferno
depth_norm_3d = (depth_3d - depth_3d.min()) / (depth_3d.max() - depth_3d.min() + 1e-8)
base_colors = inferno(depth_norm_3d)
colors = base_colors.copy()

# Overlay stain regions
colors[stain_3d, 0] = 1.0  # Red
colors[stain_3d, 1] = 0.2
colors[stain_3d, 2] = 0.2
colors[stain_3d, 3] = 1.0

colors[depressed_3d, 0] = 0.2
colors[depressed_3d, 1] = 0.2
colors[depressed_3d, 2] = 1.0  # Blue
colors[depressed_3d, 3] = 1.0

fig = plt.figure(figsize=(10, 10))
fig.patch.set_facecolor('#1a1a1a')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#1a1a1a')

X = np.arange(ds)
Y = np.arange(ds)
X, Y = np.meshgrid(X, Y)

ax.plot_surface(X, Y, depth_3d,
                facecolors=colors,
                linewidth=0, antialiased=True,
                rstride=1, cstride=1)
ax.set_zlim(0, 280)
ax.view_init(elev=35, azim=135)
ax.set_title('3D Depth Surface with Stain Candidates\n(Red=raised, Blue=depressed)',
             color='white', fontsize=13)
ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.tight_layout()
plt.savefig('output/bloodstain/stain_3d_surface.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: stain_3d_surface.png")

# === Documentation ===
print("\n--- Findings ---")
print(f"Method: Source image minus Gaussian-smoothed VP-8 depth map")
print(f"Residual std: {std_r:.1f}")
print(f"Threshold: mean + 2*std = {threshold:.1f}")
print(f"Raised candidates (potential bloodstains): {np.sum(stain_mask)} px ({100*np.sum(stain_mask)/stain_mask.size:.1f}%)")
print(f"Depressed candidates: {np.sum(depressed_mask)} px ({100*np.sum(depressed_mask)/depressed_mask.size:.1f}%)")
print("\nLIMITATIONS:")
print("- Cannot distinguish bloodstains from cloth texture artifacts")
print("- 2-sigma threshold is arbitrary; no ground truth for calibration")
print("- Source image resolution (2388x3000) limits spatial precision")
print("- The smooth depth map is derived FROM the source, creating circular dependency")
print("- This is exploratory only — not a validated blood pattern analysis")

print("\n=== Blood Stain Mapping Complete ===")
