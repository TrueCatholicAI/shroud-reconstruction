"""Phase 2.1: FFT weave pattern separation on Miller full-resolution face crop."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

print("=== FFT Weave Pattern Separation ===")

# Load Miller source
img = cv2.imread('data/source/vernon_miller/34c-Fa-N_0414.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Miller source: {img.shape} ({img.dtype})")

# Crop face region (8% margins as in study2 pipeline)
h, w = img.shape
margin = 0.08
y1, y2 = int(h * margin), int(h * (1 - margin))
x1, x2 = int(w * margin), int(w * (1 - margin))
face = img[y1:y2, x1:x2].copy()
print(f"Face crop: {face.shape}")

# === Step 1: Compute 2D FFT ===
f = np.float32(face)
# Pad to optimal size for FFT
rows, cols = face.shape
optimal_rows = cv2.getOptimalDFTSize(rows)
optimal_cols = cv2.getOptimalDFTSize(cols)
padded = np.zeros((optimal_rows, optimal_cols), dtype=np.float32)
padded[:rows, :cols] = f
print(f"Padded to: {padded.shape}")

# FFT
dft = np.fft.fft2(padded)
dft_shift = np.fft.fftshift(dft)

# Magnitude spectrum (log scale for visualization)
magnitude = 20 * np.log(np.abs(dft_shift) + 1)

print(f"FFT magnitude range: [{magnitude.min():.1f}, {magnitude.max():.1f}]")

# === Step 2: Identify weave frequency peaks ===
# The weave pattern shows as periodic peaks in the frequency domain
# We look for bright spots away from the center (DC component)
cy, cx = optimal_rows // 2, optimal_cols // 2

# Create a mask that excludes the low-frequency center (radius ~20 pixels)
mag_abs = np.abs(dft_shift)
mag_search = mag_abs.copy()

# Zero out center region (low frequencies we want to keep)
low_freq_radius = min(optimal_rows, optimal_cols) // 30  # ~3% of image
Y, X = np.ogrid[:optimal_rows, :optimal_cols]
center_mask = (Y - cy)**2 + (X - cx)**2 < low_freq_radius**2
mag_search[center_mask] = 0

# Find peaks — top frequency components outside center
# We'll use a threshold approach: anything above 90th percentile of non-center magnitudes
non_center_mags = mag_search[mag_search > 0]
if len(non_center_mags) > 0:
    threshold = np.percentile(non_center_mags, 99.5)
    peak_mask = mag_search > threshold
    peak_coords = np.argwhere(peak_mask)
    print(f"Found {len(peak_coords)} peak pixels above 99.5th percentile threshold")

    # Compute distances from center for peaks
    peak_distances = np.sqrt((peak_coords[:, 0] - cy)**2 + (peak_coords[:, 1] - cx)**2)
    print(f"Peak distance range: [{peak_distances.min():.0f}, {peak_distances.max():.0f}] pixels")
else:
    print("No significant peaks found outside center")
    peak_mask = np.zeros_like(mag_search, dtype=bool)

# === Step 3: Create notch filter ===
# Suppress frequencies at the weave peak locations
# Use a Gaussian-shaped notch at each peak cluster
notch_filter = np.ones((optimal_rows, optimal_cols), dtype=np.float32)

# Dilate peak mask to create broader notch regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
peak_mask_uint8 = peak_mask.astype(np.uint8) * 255
dilated = cv2.dilate(peak_mask_uint8, kernel, iterations=2)
notch_regions = dilated > 0

# Apply Gaussian taper to notch (softer than hard cutoff)
notch_filter[notch_regions] = 0.0
notch_filter = gaussian_filter(notch_filter, sigma=3)

# Don't suppress the center (DC + low frequencies)
notch_filter[center_mask] = 1.0

print(f"Notch filter: {np.sum(notch_filter < 0.5)} pixels suppressed")

# === Step 4: Apply filter and inverse FFT ===
filtered_dft = dft_shift * notch_filter
filtered_dft_ishift = np.fft.ifftshift(filtered_dft)
filtered_img = np.real(np.fft.ifft2(filtered_dft_ishift))
filtered_face = filtered_img[:rows, :cols]  # Remove padding

# Normalize to uint8
filtered_face = np.clip(filtered_face, 0, 255).astype(np.uint8)
print(f"Filtered image: {filtered_face.shape}, range [{filtered_face.min()}, {filtered_face.max()}]")

# Save filtered image
cv2.imwrite('output/fft_weave/miller_filtered.png', filtered_face)

# === Step 5: Generate visualizations ===

# 5a. FFT magnitude spectrum
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a1a')
# Clip for better visibility
mag_vis = np.clip(magnitude, 0, np.percentile(magnitude, 99.9))
ax.imshow(mag_vis, cmap='inferno', aspect='auto')
ax.set_title('FFT Magnitude Spectrum (log scale)', color='white', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig('output/fft_weave/fft_magnitude_spectrum.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: fft_magnitude_spectrum.png")

# 5b. Notch filter mask
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.imshow(notch_filter, cmap='gray', aspect='auto')
ax.set_title('Notch Filter Mask (white=pass, black=suppress)', color='white', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig('output/fft_weave/notch_filter_mask.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: notch_filter_mask.png")

# 5c. Before/after comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor('#1a1a1a')

# Show a zoomed region to make weave visible
# Pick center region for zoom
zh, zw = 600, 600
zy = rows // 2 - zh // 2
zx = cols // 2 - zw // 2

axes[0].imshow(face[zy:zy+zh, zx:zx+zw], cmap='gray')
axes[0].set_title('Original (zoomed center)', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(filtered_face[zy:zy+zh, zx:zx+zw], cmap='gray')
axes[1].set_title('FFT Weave-Filtered (zoomed center)', color='white', fontsize=13)
axes[1].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Miller 1978 - FFT Weave Pattern Removal', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/fft_weave/before_after_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: before_after_comparison.png")

# 5d. Full-image before/after
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#1a1a1a')
axes[0].imshow(face, cmap='gray')
axes[0].set_title('Original Miller Face Crop', color='white', fontsize=13)
axes[0].axis('off')
axes[1].imshow(filtered_face, cmap='gray')
axes[1].set_title('FFT Weave-Filtered', color='white', fontsize=13)
axes[1].axis('off')
for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Miller 1978 - Full Face FFT Weave Separation', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/fft_weave/full_before_after.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: full_before_after.png")

# === Step 6: Run depth pipeline on filtered image ===
print("\n--- Running depth pipeline on FFT-filtered image ---")

# CLAHE + normalization (same as study2 pipeline)
norm_img = cv2.normalize(filtered_face, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_map = clahe.apply(norm_img)

print(f"Depth map from filtered: {depth_map.shape}, range [{depth_map.min()}, {depth_map.max()}]")

# Downsample to 150x150 + Gaussian 15
from scipy.ndimage import zoom as scipy_zoom
dh, dw = depth_map.shape
depth_150 = scipy_zoom(depth_map.astype(np.float32), (150/dh, 150/dw), order=1)
depth_150_g15 = gaussian_filter(depth_150, sigma=15/6.0)  # 15px kernel ~ sigma=2.5
depth_150_g15 = np.clip(depth_150_g15, 0, 255).astype(np.uint8)

print(f"Filtered depth 150x150+G15: range [{depth_150_g15.min()}, {depth_150_g15.max()}]")

# Save
np.save('output/fft_weave/depth_150x150_g15_filtered.npy', depth_150_g15)
cv2.imwrite('output/fft_weave/depth_150x150_g15_filtered.png', depth_150_g15)

# Load unfiltered for comparison
unfiltered_150 = np.load('output/study2_miller/depth_150x150_g15.npy')

# Comparison: filtered vs unfiltered depth
from scipy.stats import pearsonr
r, p = pearsonr(depth_150_g15.flatten().astype(float), unfiltered_150.flatten().astype(float))
print(f"Filtered vs unfiltered depth correlation: r={r:.4f}")

diff = depth_150_g15.astype(float) - unfiltered_150.astype(float)
print(f"Depth difference: mean={diff.mean():.2f}, std={diff.std():.2f}, max_abs={np.abs(diff).max():.1f}")

# Comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(unfiltered_150, cmap='inferno')
axes[0].set_title('Unfiltered Depth (Study 2)', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(depth_150_g15, cmap='inferno')
axes[1].set_title('FFT-Filtered Depth', color='white', fontsize=13)
axes[1].axis('off')

from matplotlib.colors import TwoSlopeNorm
vmax = max(abs(diff.min()), abs(diff.max()))
if vmax < 1e-8:
    vmax = 1.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = axes[2].imshow(diff, cmap='RdBu_r', norm=norm)
axes[2].set_title(f'Difference (r={r:.3f})', color='white', fontsize=13)
axes[2].axis('off')
cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('FFT Weave Filtering Effect on Depth Map', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/fft_weave/depth_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: depth_comparison.png")

print("\n=== FFT Weave Separation Complete ===")
