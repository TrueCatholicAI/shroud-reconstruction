"""Task L: Hand geometry analysis — crossing angle, wrist position, thumb visibility."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import find_peaks

print("=== Hand Geometry Analysis (Task L) ===")

# --- Step 1: Load full-body frontal depth ---
img = cv2.imread('data/source/shroud_full_negatives.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Source image: {img.shape}")

h, w = img.shape
midpoint = w // 2
frontal = img[:, :midpoint]
frontal = frontal[:, :int(midpoint * 0.95)]  # trim 5% right edge
print(f"Frontal trimmed: {frontal.shape}")

# CLAHE enhancement
norm_img = cv2.normalize(frontal, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_full = clahe.apply(norm_img)

# Downsample to ~618x300
fh, fw = depth_full.shape
target_w = 300
target_h = int(target_w * (fh / fw))
depth_ds = zoom(depth_full.astype(np.float32), (target_h / fh, target_w / fw), order=1)
depth_ds = gaussian_filter(depth_ds, sigma=20)
print(f"Downsampled depth: {depth_ds.shape}")

# --- Step 2: Extract hand/forearm region (rows 45-55% of body height) ---
row_start = int(depth_ds.shape[0] * 0.45)
row_end = int(depth_ds.shape[0] * 0.55)
hand_region = depth_ds[row_start:row_end, :]
print(f"Hand region: rows {row_start}-{row_end}, shape {hand_region.shape}")

# --- Step 3: Analyze crossing geometry ---
# Horizontal profiles at multiple vertical positions through the hand zone
n_profiles = 5
profile_rows = np.linspace(0, hand_region.shape[0] - 1, n_profiles, dtype=int)
profiles = []
for r in profile_rows:
    prof = hand_region[r, :]
    profiles.append(prof)

# Average horizontal profile across the hand zone
avg_profile = np.mean(hand_region, axis=0)

# Find peaks in horizontal profile — these represent wrist/hand ridges
peaks, properties = find_peaks(avg_profile, height=np.percentile(avg_profile, 60),
                                distance=20, prominence=5)
print(f"Detected {len(peaks)} peaks in horizontal hand profile")

# Vertical centerline profile through the hand region
center_col = depth_ds.shape[1] // 2
vert_profile = depth_ds[row_start:row_end, center_col]

# Estimate crossing angle from the diagonal orientation of intensity
# Use gradient analysis to find dominant angle
gy, gx = np.gradient(hand_region)
magnitude = np.sqrt(gx**2 + gy**2)
angles = np.arctan2(gy, gx) * 180 / np.pi

# Weight angles by gradient magnitude to find dominant direction
mask = magnitude > np.percentile(magnitude, 70)
if np.sum(mask) > 0:
    weighted_angles = angles[mask]
    # The crossing angle is relative to horizontal
    median_angle = np.median(weighted_angles)
    crossing_angle = abs(median_angle)
else:
    crossing_angle = 0
    median_angle = 0

# Analyze left vs right hand elevation
mid_col = hand_region.shape[1] // 2
left_half_mean = np.mean(hand_region[:, :mid_col])
right_half_mean = np.mean(hand_region[:, mid_col:])

# Thumb visibility: look for small features (high-frequency content) in the hand zone
hand_hf = hand_region - gaussian_filter(hand_region, sigma=10)
thumb_energy = np.std(hand_hf)

print(f"\n--- FINDINGS ---")
print(f"Hand region: {row_start/depth_ds.shape[0]*100:.1f}% to {row_end/depth_ds.shape[0]*100:.1f}% of body height")
print(f"Dominant gradient angle: {median_angle:.1f} degrees")
print(f"Estimated crossing angle: {crossing_angle:.1f} degrees from horizontal")
print(f"Left half mean depth: {left_half_mean:.1f}")
print(f"Right half mean depth: {right_half_mean:.1f}")
if left_half_mean > right_half_mean:
    print(f"Left wrist appears ELEVATED (closer to cloth) — right hand crossed over left")
else:
    print(f"Right wrist appears ELEVATED (closer to cloth) — left hand crossed over right")
print(f"High-frequency detail energy (thumb/finger visibility): {thumb_energy:.2f}")
if thumb_energy < 3.0:
    print("Thumbs NOT clearly visible — consistent with post-mortem adduction or concealment")
else:
    print("Some fine structure detected — possible thumb/finger articulation visible")

# Peak analysis
if len(peaks) >= 2:
    peak_separation_px = abs(peaks[-1] - peaks[0])
    # The width of the image represents roughly the torso width
    print(f"Peak-to-peak wrist separation: {peak_separation_px} pixels ({peak_separation_px/depth_ds.shape[1]*100:.1f}% of image width)")
    print(f"Peak positions: {peaks}")

# --- Step 4: Generate visualization ---
fig = plt.figure(figsize=(16, 12), facecolor='#1a1a1a')
gold = '#c4a35a'

# Panel 1: Source hand region (top-left)
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(frontal[int(frontal.shape[0]*0.45):int(frontal.shape[0]*0.55), :],
           cmap='gray', aspect='auto')
ax1.set_title('Source: Hand/Forearm Region', color=gold, fontsize=13, fontweight='bold')
ax1.set_xlabel('Horizontal position', color='#aaa', fontsize=10)
ax1.set_ylabel('Vertical position', color='#aaa', fontsize=10)
ax1.tick_params(colors='#666')
ax1.set_facecolor('#1a1a1a')

# Panel 2: Depth heatmap of hand region (top-right)
ax2 = fig.add_subplot(2, 2, 2)
im = ax2.imshow(hand_region, cmap='inferno', aspect='auto')
ax2.set_title('Depth Heatmap: Crossed Hands', color=gold, fontsize=13, fontweight='bold')
ax2.set_xlabel('Horizontal position', color='#aaa', fontsize=10)
ax2.set_ylabel('Row within hand zone', color='#aaa', fontsize=10)
ax2.tick_params(colors='#666')
ax2.set_facecolor('#1a1a1a')
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors='#666')
cbar.set_label('Depth (relative)', color='#aaa', fontsize=10)

# Panel 3: Horizontal profile across hands (bottom-left)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#1a1a1a')
x_axis = np.arange(len(avg_profile))
ax3.plot(x_axis, avg_profile, color=gold, linewidth=2, label='Avg horizontal profile')
for i, prof in enumerate(profiles):
    alpha = 0.3 + 0.1 * i
    ax3.plot(x_axis, prof, color='#888', linewidth=0.8, alpha=alpha)
if len(peaks) > 0:
    ax3.scatter(peaks, avg_profile[peaks], color='#ff4444', s=80, zorder=5, label='Peaks (wrists/hands)')
ax3.axvline(x=mid_col, color='#555', linestyle='--', linewidth=1, label='Centerline')
ax3.set_title('Horizontal Depth Profile Across Hands', color=gold, fontsize=13, fontweight='bold')
ax3.set_xlabel('Horizontal position (px)', color='#aaa', fontsize=10)
ax3.set_ylabel('Depth intensity', color='#aaa', fontsize=10)
ax3.tick_params(colors='#666')
ax3.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='#555', labelcolor='#ccc')

# Panel 4: Gradient angle distribution (bottom-right)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#1a1a1a')
if np.sum(mask) > 0:
    ax4.hist(weighted_angles, bins=60, color=gold, alpha=0.7, edgecolor='#1a1a1a')
    ax4.axvline(x=median_angle, color='#ff4444', linewidth=2, linestyle='--',
                label=f'Median: {median_angle:.1f}°')
ax4.set_title('Gradient Angle Distribution (Crossing Direction)', color=gold, fontsize=13, fontweight='bold')
ax4.set_xlabel('Angle (degrees)', color='#aaa', fontsize=10)
ax4.set_ylabel('Count', color='#aaa', fontsize=10)
ax4.tick_params(colors='#666')
ax4.legend(fontsize=10, facecolor='#2a2a2a', edgecolor='#555', labelcolor='#ccc')

plt.tight_layout(pad=2.0)
import os
os.makedirs('output/analysis', exist_ok=True)
plt.savefig('output/analysis/hand_geometry.png', dpi=150, facecolor='#1a1a1a',
            bbox_inches='tight')
plt.close()
print(f"\nSaved: output/analysis/hand_geometry.png")
