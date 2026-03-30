"""Task G: Dorsal image extraction and depth pipeline."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, zoom

print("=== Dorsal Image Extraction ===")

# Load full negatives image
img = cv2.imread('data/source/shroud_full_negatives.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(f"Full negatives: {img.shape}")

# The dorsal image is on the RIGHT half
midpoint = w // 2
dorsal = img[:, midpoint:]
# Trim left edge (gap between panels)
dorsal = dorsal[:, int(dorsal.shape[1] * 0.05):]
print(f"Dorsal crop: {dorsal.shape}")

# The dorsal image is MIRRORED relative to the frontal — the back of the body
# Head is at the TOP (same orientation as frontal in this image)

# === CLAHE depth extraction ===
norm_img = cv2.normalize(dorsal, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_dorsal = clahe.apply(norm_img)
print(f"Dorsal depth: {depth_dorsal.shape}, range [{depth_dorsal.min()}, {depth_dorsal.max()}]")

cv2.imwrite('output/full_body/dorsal_raw.png', dorsal)
cv2.imwrite('output/full_body/dorsal_depth_fullres.png', depth_dorsal)
np.save('output/full_body/dorsal_depth_fullres.npy', depth_dorsal)

# === Downsample to match frontal (618x300) ===
fh, fw = depth_dorsal.shape
target_w = 300
target_h = int(target_w * (fh / fw))
print(f"Dorsal downsample target: {target_h}x{target_w}")

depth_ds = zoom(depth_dorsal.astype(np.float32), (target_h/fh, target_w/fw), order=1)
sigma = 20 / 6.0
depth_smooth = gaussian_filter(depth_ds, sigma=sigma)
depth_smooth = np.clip(depth_smooth, 0, 255).astype(np.uint8)
print(f"Dorsal smooth: {depth_smooth.shape}, range [{depth_smooth.min()}, {depth_smooth.max()}]")

np.save('output/full_body/dorsal_depth_smooth.npy', depth_smooth)
cv2.imwrite('output/full_body/dorsal_depth_smooth.png', depth_smooth)

# === Load frontal for comparison ===
frontal_smooth = np.load('output/full_body/depth_body_smooth.npy')
print(f"Frontal smooth: {frontal_smooth.shape}")

# Resize dorsal to match frontal dimensions exactly
dorsal_resized = cv2.resize(depth_smooth, (frontal_smooth.shape[1], frontal_smooth.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

# === Heatmap comparison ===
fig, axes = plt.subplots(1, 3, figsize=(21, 8))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(frontal_smooth, cmap='inferno')
axes[0].set_title('Frontal Depth', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(dorsal_resized, cmap='inferno')
axes[1].set_title('Dorsal Depth', color='white', fontsize=13)
axes[1].axis('off')

# Combine: frontal + dorsal should give body thickness estimate
combined = frontal_smooth.astype(np.float64) + dorsal_resized.astype(np.float64)
combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

axes[2].imshow(combined_norm, cmap='inferno')
axes[2].set_title('Combined (Frontal + Dorsal)\nEstimated Body Thickness', color='white', fontsize=12)
axes[2].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Frontal vs Dorsal Depth Comparison', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/full_body/frontal_dorsal_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: frontal_dorsal_comparison.png")

# === Dorsal 3D surface ===
rows, cols = depth_smooth.shape
X = np.arange(cols)
Y = np.arange(rows)
X, Y = np.meshgrid(X, Y)
stride = max(1, rows // 200)

fig = plt.figure(figsize=(10, 16))
fig.patch.set_facecolor('#1a1a1a')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#1a1a1a')
ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride],
                depth_smooth[::stride, ::stride].astype(float),
                cmap='inferno', linewidth=0, antialiased=True,
                rstride=1, cstride=1)
ax.set_zlim(0, 280)
ax.view_init(elev=25, azim=135)
ax.set_title('Dorsal VP-8 3D Surface', color='white', fontsize=14)
ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('output/full_body/dorsal_3d_surface.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: dorsal_3d_surface.png")

# === Centerline profile comparison ===
mid_x_f = frontal_smooth.shape[1] // 2
mid_x_d = dorsal_resized.shape[1] // 2
strip = 2

frontal_profile = frontal_smooth[:, mid_x_f-strip:mid_x_f+strip+1].mean(axis=1)
dorsal_profile = dorsal_resized[:, mid_x_d-strip:mid_x_d+strip+1].mean(axis=1)

fig, ax = plt.subplots(figsize=(8, 12))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#222')

ax.plot(frontal_profile, np.arange(len(frontal_profile)), color='#c4a35a', linewidth=1.5, label='Frontal')
ax.plot(dorsal_profile, np.arange(len(dorsal_profile)), color='#3498db', linewidth=1.5, label='Dorsal')
ax.set_xlabel('Depth Intensity', color='white')
ax.set_ylabel('Row (top=head)', color='white')
ax.set_title('Frontal vs Dorsal Centerline Profiles', color='white', fontsize=13)
ax.invert_yaxis()
ax.tick_params(colors='white')
ax.legend(facecolor='#333', labelcolor='white')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate
body_h = len(frontal_profile)
for frac, label in [(0.05, 'Head'), (0.18, 'Chest'), (0.45, 'Hands'), (0.75, 'Legs')]:
    y_pos = int(body_h * frac)
    ax.axhline(y=y_pos, color='#555555', linestyle='--', alpha=0.5)
    ax.text(max(frontal_profile.max(), dorsal_profile.max()) * 0.85, y_pos, label,
            color='#888888', fontsize=9, va='center')

plt.tight_layout()
plt.savefig('output/full_body/frontal_dorsal_profiles.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: frontal_dorsal_profiles.png")

# === Back-of-head analysis (head region zoom) ===
head_end = int(depth_smooth.shape[0] * 0.15)
head_region = depth_smooth[:head_end, :]
print(f"\nDorsal head region: {head_region.shape}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a1a')

# Frontal head
f_head = frontal_smooth[:int(frontal_smooth.shape[0]*0.15), :]
axes[0].imshow(f_head, cmap='inferno')
axes[0].set_title('Frontal — Face Region', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(head_region, cmap='inferno')
axes[1].set_title('Dorsal — Back of Head', color='white', fontsize=13)
axes[1].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Head Region: Frontal vs Dorsal', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/full_body/head_frontal_vs_dorsal.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: head_frontal_vs_dorsal.png")

# Thickness consistency check
# If frontal+dorsal ~ constant, it means consistent body thickness
combined_profile = frontal_profile + dorsal_profile
thickness_std = combined_profile.std()
thickness_mean = combined_profile.mean()
cv = thickness_std / thickness_mean
print(f"\nBody thickness consistency:")
print(f"  Mean combined intensity: {thickness_mean:.1f}")
print(f"  Std: {thickness_std:.1f}")
print(f"  Coefficient of variation: {cv:.3f}")
print(f"  (Lower CV = more consistent body thickness)")

from scipy.stats import pearsonr
r, p = pearsonr(frontal_profile, dorsal_profile)
print(f"  Frontal-dorsal centerline correlation: r={r:.3f}, p={p:.2e}")

print("\n=== Dorsal Extraction Complete ===")
