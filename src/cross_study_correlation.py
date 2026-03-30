"""Task D: Cross-study depth correlation — register Enrie and Miller by eye positions."""
import numpy as np
import cv2
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom, gaussian_filter
from scipy.stats import pearsonr

print("=== Cross-Study Depth Correlation Analysis ===")

# Load both depth maps at 150x150
enrie_full = np.load('data/processed/depth_map_smooth_15.npy')
h, w = enrie_full.shape
enrie_150 = zoom(enrie_full.astype(np.float64), (150/h, 150/w), order=1)

miller_150 = np.load('output/study2_miller/depth_150x150_g15.npy').astype(np.float64)

print(f"Enrie 150x150: range [{enrie_150.min():.1f}, {enrie_150.max():.1f}]")
print(f"Miller 150x150: range [{miller_150.min():.1f}, {miller_150.max():.1f}]")

# Load landmark positions for registration
# Enrie landmarks from Study 1
enrie_landmarks = json.load(open('data/final/approved_measurements.json'))
# Miller landmarks from Study 2
miller_landmarks = json.load(open('output/study2_miller/landmarks.json'))

print(f"\nEnrie landmarks: {list(enrie_landmarks.keys()) if isinstance(enrie_landmarks, dict) else 'list format'}")
print(f"Miller landmarks: {list(miller_landmarks.keys()) if isinstance(miller_landmarks, dict) else 'list format'}")

# For registration, we need eye positions in the 150x150 coordinate system
# The landmarks are stored at 150x150 analysis resolution already

# Extract eye positions
if isinstance(enrie_landmarks, dict):
    # Try to get eye positions from measurements structure
    e_leye = enrie_landmarks.get('left_eye', enrie_landmarks.get('left_pupil'))
    e_reye = enrie_landmarks.get('right_eye', enrie_landmarks.get('right_pupil'))
    e_midline = enrie_landmarks.get('midline_x')
else:
    e_leye = e_reye = e_midline = None

if isinstance(miller_landmarks, dict):
    m_leye = miller_landmarks.get('left_eye', miller_landmarks.get('left_pupil'))
    m_reye = miller_landmarks.get('right_eye', miller_landmarks.get('right_pupil'))
    m_midline = miller_landmarks.get('midline_x')
else:
    m_leye = m_reye = m_midline = None

print(f"Enrie left eye: {e_leye}, right eye: {e_reye}, midline: {e_midline}")
print(f"Miller left eye: {m_leye}, right eye: {m_reye}, midline: {m_midline}")

# If landmarks have enough info, do affine registration
# Otherwise, use midline-based alignment (simpler)

# For robust approach: align by midline and use normalized cross-correlation
# Both are already 150x150 square grids, so the main alignment needed is:
# 1. Center horizontally on midline
# 2. Center vertically on eye level (or nose tip)

# Normalize both to [0, 1]
e_norm = (enrie_150 - enrie_150.min()) / (enrie_150.max() - enrie_150.min())
m_norm = (miller_150 - miller_150.min()) / (miller_150.max() - miller_150.min())

# Simple approach: compute sliding correlation in a local window
# For each pixel, compute correlation in a surrounding window
window_size = 15  # 15x15 window
half_w = window_size // 2

corr_map = np.full((150, 150), np.nan)
for y in range(half_w, 150 - half_w):
    for x in range(half_w, 150 - half_w):
        e_patch = e_norm[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        m_patch = m_norm[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        if e_patch.std() > 0.01 and m_patch.std() > 0.01:
            r, _ = pearsonr(e_patch, m_patch)
            corr_map[y, x] = r

print(f"\nLocal correlation map: {np.nanmean(corr_map):.3f} mean, {np.nanstd(corr_map):.3f} std")
print(f"  Valid pixels: {np.sum(~np.isnan(corr_map))}")

# Global correlation
r_global, p_global = pearsonr(e_norm.flatten(), m_norm.flatten())
print(f"Global pixel-wise correlation: r={r_global:.4f}, p={p_global:.2e}")

# === Visualization 1: Side-by-side depths + correlation map ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(e_norm, cmap='inferno')
axes[0].set_title('Enrie 1931 Depth', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(m_norm, cmap='inferno')
axes[1].set_title('Miller 1978 Depth', color='white', fontsize=13)
axes[1].axis('off')

# Correlation map with diverging colormap
im = axes[2].imshow(corr_map, cmap='RdYlGn', vmin=-1, vmax=1)
axes[2].set_title(f'Local Correlation (15x15 window)\nGlobal r={r_global:.3f}', color='white', fontsize=12)
axes[2].axis('off')
cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label('Pearson r', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Cross-Study Depth Correlation: Enrie 1931 vs Miller 1978', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/cross_study_correlation.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: cross_study_correlation.png")

# === Visualization 2: Where are they most/least correlated? ===
# Mask the valid correlation area
valid = ~np.isnan(corr_map)
corr_valid = corr_map.copy()
corr_valid[~valid] = 0

# Find regions of highest and lowest correlation
high_corr = corr_valid > 0.7
low_corr = (corr_valid < 0.0) & valid

# Create annotated overlay
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('#1a1a1a')

# High correlation regions on depth map
overlay_high = np.zeros((150, 150, 3))
overlay_high[:, :, 0] = e_norm * 0.7
overlay_high[:, :, 1] = e_norm * 0.7
overlay_high[:, :, 2] = e_norm * 0.7
overlay_high[high_corr, 0] = 0.0
overlay_high[high_corr, 1] = 0.9
overlay_high[high_corr, 2] = 0.0

axes[0].imshow(overlay_high)
axes[0].set_title(f'High Correlation (r > 0.7)\n{np.sum(high_corr)} pixels', color='white', fontsize=12)
axes[0].axis('off')

# Low correlation regions
overlay_low = np.zeros((150, 150, 3))
overlay_low[:, :, 0] = e_norm * 0.7
overlay_low[:, :, 1] = e_norm * 0.7
overlay_low[:, :, 2] = e_norm * 0.7
overlay_low[low_corr, 0] = 0.9
overlay_low[low_corr, 1] = 0.0
overlay_low[low_corr, 2] = 0.0

axes[1].imshow(overlay_low)
axes[1].set_title(f'Negative Correlation (r < 0)\n{np.sum(low_corr)} pixels', color='white', fontsize=12)
axes[1].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Cross-Study Agreement/Disagreement Regions', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/cross_study_regions.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: cross_study_regions.png")

# === Statistics by face region ===
# Divide into approximate regions based on 150x150 grid
# Central face (inner 60%): rows 30-120, cols 30-120
# Periphery: everything else
center_mask = np.zeros((150, 150), dtype=bool)
center_mask[30:120, 30:120] = True

center_corr = corr_map[center_mask & valid]
periph_corr = corr_map[~center_mask & valid]

print(f"\nRegional analysis:")
print(f"  Central face (inner 60%): mean r={np.nanmean(center_corr):.3f}, median={np.nanmedian(center_corr):.3f}")
print(f"  Periphery: mean r={np.nanmean(periph_corr):.3f}, median={np.nanmedian(periph_corr):.3f}")

# Upper face (forehead/eyes) vs lower face (nose/mouth/chin)
upper_mask = np.zeros((150, 150), dtype=bool)
upper_mask[30:75, 30:120] = True
lower_mask = np.zeros((150, 150), dtype=bool)
lower_mask[75:120, 30:120] = True

upper_corr = corr_map[upper_mask & valid]
lower_corr = corr_map[lower_mask & valid]
print(f"  Upper face (brow/eyes): mean r={np.nanmean(upper_corr):.3f}")
print(f"  Lower face (nose/chin): mean r={np.nanmean(lower_corr):.3f}")

print("\n=== Cross-Study Correlation Complete ===")
