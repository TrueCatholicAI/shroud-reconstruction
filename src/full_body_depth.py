"""Phase 4.1: Full body depth extraction from Enrie 1931 full-length negative."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, zoom

print("=== Full Body Depth Extraction ===")

# Load full negatives image
img = cv2.imread('data/source/shroud_full_negatives.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Full negatives image: {img.shape}")

# The image shows frontal (left) and dorsal (right) side by side
# Crop the frontal half (left side)
h, w = img.shape
midpoint = w // 2
frontal = img[:, :midpoint]
print(f"Frontal crop: {frontal.shape}")

# There's a dark border/gap at the center — trim a bit from the right edge
frontal = frontal[:, :int(midpoint * 0.95)]
print(f"Frontal trimmed: {frontal.shape}")

# Also try to get the higher-res version
# For now, work with what we have

# === Step 1: CLAHE depth extraction (same pipeline as face studies) ===
norm_img = cv2.normalize(frontal, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_full = clahe.apply(norm_img)
print(f"Full-body depth map: {depth_full.shape}, range [{depth_full.min()}, {depth_full.max()}]")

# Save full-res depth
cv2.imwrite('output/full_body/depth_fullres.png', depth_full)
np.save('output/full_body/depth_fullres.npy', depth_full)

# === Step 2: Downsample to ~300x400 (preserve aspect ratio) ===
fh, fw = depth_full.shape
aspect = fh / fw
# Target: width=300, height preserving aspect
target_w = 300
target_h = int(target_w * aspect)
print(f"Downsample target: {target_h}x{target_w}")

depth_ds = zoom(depth_full.astype(np.float32), (target_h/fh, target_w/fw), order=1)

# Gaussian smoothing — proportional to face study: 150px -> G15, so 300px -> ~G20
sigma = 20 / 6.0
depth_smooth = gaussian_filter(depth_ds, sigma=sigma)
depth_smooth = np.clip(depth_smooth, 0, 255).astype(np.uint8)
print(f"Smoothed depth: {depth_smooth.shape}, range [{depth_smooth.min()}, {depth_smooth.max()}]")

np.save('output/full_body/depth_body_smooth.npy', depth_smooth)
cv2.imwrite('output/full_body/depth_body_smooth.png', depth_smooth)

# === Step 3: Heatmap visualization ===
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(frontal, cmap='gray')
axes[0].set_title('Frontal Body (Negative)', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(depth_smooth, cmap='inferno')
axes[1].set_title('Full-Body Depth Map', color='white', fontsize=13)
axes[1].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Full Body VP-8 Depth Extraction', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/full_body/body_heatmap.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: body_heatmap.png")

# === Step 4: Full-body 3D surface ===
rows, cols = depth_smooth.shape
X = np.arange(cols)
Y = np.arange(rows)
X, Y = np.meshgrid(X, Y)

# Angled view
fig = plt.figure(figsize=(10, 16))
fig.patch.set_facecolor('#1a1a1a')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#1a1a1a')

# Subsample for rendering speed
stride = max(1, rows // 200)
ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride],
                depth_smooth[::stride, ::stride].astype(float),
                cmap='inferno', linewidth=0, antialiased=True,
                rstride=1, cstride=1)
ax.set_zlim(0, 280)
ax.view_init(elev=25, azim=135)
ax.set_title('Full-Body VP-8 3D Surface', color='white', fontsize=14)
ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
# Invert Y so head is at top
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('output/full_body/body_3d_surface_angled.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: body_3d_surface_angled.png")

# Front view
fig = plt.figure(figsize=(10, 16))
fig.patch.set_facecolor('#1a1a1a')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#1a1a1a')
ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride],
                depth_smooth[::stride, ::stride].astype(float),
                cmap='inferno', linewidth=0, antialiased=True,
                rstride=1, cstride=1)
ax.set_zlim(0, 280)
ax.view_init(elev=0, azim=0)
ax.set_title('Full-Body VP-8 3D Surface (front)', color='white', fontsize=14)
ax.tick_params(colors='white')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('output/full_body/body_3d_surface_front.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: body_3d_surface_front.png")

# === Step 5: Centerline profile (vertical intensity along midline) ===
mid_x = cols // 2
strip_width = 5
centerline = depth_smooth[:, mid_x - strip_width:mid_x + strip_width + 1].mean(axis=1)

fig, ax = plt.subplots(figsize=(6, 14))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.plot(centerline, np.arange(len(centerline)), color='#c4a35a', linewidth=1.5)
ax.set_xlabel('Depth Intensity', color='white')
ax.set_ylabel('Row (top=head)', color='white')
ax.set_title('Full-Body Centerline Profile', color='white', fontsize=13)
ax.invert_yaxis()
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate approximate body regions
body_height = len(centerline)
annotations = [
    (0.05, 'Head'),
    (0.18, 'Chest'),
    (0.35, 'Abdomen'),
    (0.45, 'Hands/Pelvis'),
    (0.55, 'Upper legs'),
    (0.75, 'Lower legs'),
    (0.90, 'Feet'),
]
for frac, label in annotations:
    y_pos = int(body_height * frac)
    ax.axhline(y=y_pos, color='#555555', linestyle='--', alpha=0.5)
    ax.text(centerline.max() * 0.85, y_pos, label, color='#888888', fontsize=9, va='center')

plt.tight_layout()
plt.savefig('output/full_body/centerline_profile.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: centerline_profile.png")

print("\n=== Full Body Depth Extraction Complete ===")
