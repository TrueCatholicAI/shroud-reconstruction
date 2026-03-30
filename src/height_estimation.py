"""Task M: Height estimation from full-body frontal depth profile."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import find_peaks
import os

print("=== Height Estimation (Task M) ===")

# --- Step 1: Load full-body frontal depth ---
img = cv2.imread('data/source/shroud_full_negatives.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Source image: {img.shape}")

h, w = img.shape
midpoint = w // 2
frontal = img[:, :midpoint]
frontal = frontal[:, :int(midpoint * 0.95)]  # trim 5% right edge
print(f"Frontal trimmed: {frontal.shape}")

# CLAHE enhancement (for depth visualization)
norm_img = cv2.normalize(frontal, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_full = clahe.apply(norm_img)

# Downsample to ~618x300
fh, fw = depth_full.shape
target_w = 300
target_h = int(target_w * (fh / fw))
depth_ds = zoom(depth_full.astype(np.float32), (target_h / fh, target_w / fw), order=1)
depth_ds = gaussian_filter(depth_ds, sigma=20)
print(f"Downsampled CLAHE depth: {depth_ds.shape}")

# Also downsample the RAW frontal for body-extent detection
# In this image, the body is DARKER than the surrounding cloth
raw_ds = zoom(frontal.astype(np.float32), (target_h / fh, target_w / fw), order=1)
raw_smooth = gaussian_filter(raw_ds, sigma=5)

# --- Step 2: Centerline intensity profile (raw, for body detection) ---
# Average over a central strip (middle 30% of width) for robustness
strip_left = int(raw_smooth.shape[1] * 0.35)
strip_right = int(raw_smooth.shape[1] * 0.65)
raw_centerline = np.mean(raw_smooth[:, strip_left:strip_right], axis=1)
raw_centerline_smooth = gaussian_filter(raw_centerline, sigma=8)

# Also compute CLAHE centerline for visualization
clahe_centerline = np.mean(depth_ds[:, strip_left:strip_right], axis=1)
clahe_centerline_smooth = gaussian_filter(clahe_centerline, sigma=5)

print(f"Centerline profile: {len(raw_centerline)} rows")

# --- Step 3: Detect body extent ---
# The body appears as a DEPRESSION (darker = lower values) in the raw negative image.
# The cloth border/background is brighter (higher values).
# Invert so body = high, background = low, making peak detection natural.
inverted = np.max(raw_centerline_smooth) - raw_centerline_smooth

# Background = low values in inverted (bright cloth)
# Body = high values in inverted (dark body region)
bg_level = np.percentile(inverted, 15)
body_peak_level = np.percentile(inverted, 85)
detection_threshold = bg_level + 0.35 * (body_peak_level - bg_level)
print(f"Inverted: bg={bg_level:.1f}, body peak={body_peak_level:.1f}, threshold={detection_threshold:.1f}")

# Head: search top 25% for where inverted signal first rises above threshold
top_search = int(len(inverted) * 0.25)
top_above = np.where(inverted[:top_search] > detection_threshold)[0]
if len(top_above) > 0:
    head_row = top_above[0]
else:
    head_row = int(len(inverted) * 0.08)  # fallback ~8%

# Find head peak (local max near head_row)
head_search_start = max(0, head_row - 5)
head_search_end = min(head_row + 50, top_search)
head_region = inverted[head_search_start:head_search_end]
head_peaks_local, _ = find_peaks(head_region, distance=5, prominence=2)
if len(head_peaks_local) > 0:
    head_peak = head_search_start + head_peaks_local[0]
else:
    head_peak = head_row

print(f"Head (crown of head): row {head_row} ({head_row/len(inverted)*100:.1f}% from top)")

# Feet: search bottom 20% for where inverted signal last exceeds threshold
bottom_search_start = int(len(inverted) * 0.80)
bottom_above = np.where(inverted[bottom_search_start:] > detection_threshold)[0]
if len(bottom_above) > 0:
    feet_row = bottom_search_start + bottom_above[-1]
else:
    feet_row = int(len(inverted) * 0.92)  # fallback ~92%

print(f"Feet position: row {feet_row} ({feet_row/len(inverted)*100:.1f}% from top)")

# --- Step 4: Pixel distance ---
pixel_distance = feet_row - head_row
total_image_rows = depth_ds.shape[0]
print(f"Head-to-feet pixel distance: {pixel_distance} px")
print(f"Total image rows: {total_image_rows}")
print(f"Body occupies {pixel_distance/total_image_rows*100:.1f}% of image height")

# --- Step 5: Calibrate using cloth dimensions ---
# The Shroud of Turin is 4.36m x 1.10m total.
# The frontal image occupies roughly the first half of the cloth (~2.18m).
# Our cropped frontal image represents this frontal cloth section.
# The image height (618 px) corresponds to the full frontal cloth length.
CLOTH_FRONTAL_LENGTH_M = 2.18  # meters — frontal half of the 4.36m cloth

# Scale: full image height = frontal cloth length
pixels_per_meter = total_image_rows / CLOTH_FRONTAL_LENGTH_M

# Body height from pixel measurement
body_height_m = pixel_distance / pixels_per_meter

# Uncertainty analysis:
# - Head detection: +/- ~10 pixels (the crown is ambiguous)
# - Feet detection: +/- ~15 pixels (feet signal fades gradually)
# - Cloth calibration: +/- 5 cm (the exact body-to-cloth mapping has margin)
head_unc_px = 10
feet_unc_px = 15
px_unc = np.sqrt(head_unc_px**2 + feet_unc_px**2)
cloth_unc_m = 0.05

# Propagate
height_low_px = (pixel_distance - px_unc) / pixels_per_meter
height_high_px = (pixel_distance + px_unc) / pixels_per_meter
height_low_cloth = pixel_distance / (total_image_rows / (CLOTH_FRONTAL_LENGTH_M - cloth_unc_m))
height_high_cloth = pixel_distance / (total_image_rows / (CLOTH_FRONTAL_LENGTH_M + cloth_unc_m))

uncertainty_low = min(height_low_px, height_low_cloth)
uncertainty_high = max(height_high_px, height_high_cloth)

# Convert to imperial
height_in_total = body_height_m * 39.3701
feet_imperial = int(height_in_total // 12)
inches_imperial = height_in_total % 12

low_in = uncertainty_low * 39.3701
high_in = uncertainty_high * 39.3701
low_ft = int(low_in // 12)
low_inch = low_in % 12
high_ft = int(high_in // 12)
high_inch = high_in % 12

print(f"\n--- HEIGHT ESTIMATION ---")
print(f"Calibration: {total_image_rows} px = {CLOTH_FRONTAL_LENGTH_M:.2f} m ({pixels_per_meter:.1f} px/m)")
print(f"Pixel distance head-to-feet: {pixel_distance} px")
print(f"Estimated height: {body_height_m:.2f} m ({feet_imperial}'{inches_imperial:.1f}\")")
print(f"Uncertainty range: {uncertainty_low:.2f} - {uncertainty_high:.2f} m")
print(f"  = {low_ft}'{low_inch:.1f}\" - {high_ft}'{high_inch:.1f}\"")
print(f"\nNote: Historical estimates place the Shroud man at ~175-181 cm (5'9\"-5'11\")")
print(f"This is consistent with a tall individual for 1st-century Palestine.")

# --- Step 6: Visualization ---
fig = plt.figure(figsize=(16, 10), facecolor='#1a1a1a')
gold = '#c4a35a'

# Panel 1: Full body CLAHE depth with head/feet markers (left)
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(depth_ds, cmap='inferno', aspect='auto')
# Head marker
ax1.axhline(y=head_row, color='#00ff88', linewidth=2, linestyle='--', label=f'Head: row {head_row}')
if head_peak != head_row:
    ax1.axhline(y=head_peak, color='#00ff88', linewidth=1, linestyle=':', alpha=0.7)
# Feet marker
ax1.axhline(y=feet_row, color='#ff4444', linewidth=2, linestyle='--', label=f'Feet: row {feet_row}')
# Central strip indicators
ax1.axvline(x=strip_left, color='#555', linewidth=1, linestyle=':')
ax1.axvline(x=strip_right, color='#555', linewidth=1, linestyle=':')
# Height annotation arrow
mid_y = (head_row + feet_row) // 2
ax1.annotate('', xy=(depth_ds.shape[1] - 15, head_row), xytext=(depth_ds.shape[1] - 15, feet_row),
             arrowprops=dict(arrowstyle='<->', color=gold, lw=2))
ax1.text(depth_ds.shape[1] - 10, mid_y,
         f'{body_height_m:.2f} m\n({feet_imperial}\'{inches_imperial:.0f}")',
         color=gold, fontsize=11, fontweight='bold', ha='left', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=gold, alpha=0.9))
ax1.set_title('Full Body Depth \u2014 Head & Feet Detection', color=gold, fontsize=13, fontweight='bold')
ax1.set_xlabel('Horizontal position', color='#aaa', fontsize=10)
ax1.set_ylabel('Vertical position (top to bottom)', color='#aaa', fontsize=10)
ax1.tick_params(colors='#666')
ax1.legend(loc='lower left', fontsize=9, facecolor='#2a2a2a', edgecolor='#555', labelcolor='#ccc')
ax1.set_facecolor('#1a1a1a')

# Panel 2: Inverted centerline profile with annotations (right)
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor('#1a1a1a')
rows = np.arange(len(inverted))
ax2.plot(inverted, rows, color=gold, linewidth=2, label='Body signal (inverted)')
ax2.plot(np.max(raw_centerline_smooth) - raw_centerline, rows, color='#666',
         linewidth=0.8, alpha=0.4, label='Raw (inverted)')

# Threshold
ax2.axvline(x=detection_threshold, color='#555', linewidth=1, linestyle=':',
            label=f'Detection threshold')

# Head marker
ax2.axhline(y=head_row, color='#00ff88', linewidth=2, linestyle='--')
ax2.scatter([inverted[head_row]], [head_row], color='#00ff88', s=100, zorder=5)
ax2.text(inverted[head_row] + 3, head_row, 'HEAD', color='#00ff88', fontsize=10,
         fontweight='bold', va='center')

# Feet marker
ax2.axhline(y=feet_row, color='#ff4444', linewidth=2, linestyle='--')
ax2.scatter([inverted[feet_row]], [feet_row], color='#ff4444', s=100, zorder=5)
ax2.text(inverted[feet_row] + 3, feet_row, 'FEET', color='#ff4444', fontsize=10,
         fontweight='bold', va='center')

# Height annotation box
ax2.annotate(f'Height: {body_height_m:.2f} m ({feet_imperial}\'{inches_imperial:.0f}")\n'
             f'Range: {uncertainty_low:.2f}\u2013{uncertainty_high:.2f} m\n'
             f'({low_ft}\'{low_inch:.0f}"\u2013{high_ft}\'{high_inch:.0f}")',
             xy=(np.max(inverted) * 0.55, mid_y),
             fontsize=11, fontweight='bold', color=gold,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', edgecolor=gold, alpha=0.9),
             ha='center', va='center')

# Mark anatomical peaks
body_peaks, _ = find_peaks(inverted, height=detection_threshold, distance=25, prominence=5)
for p in body_peaks:
    ax2.scatter([inverted[p]], [p], color='#888', s=25, alpha=0.5, zorder=4)

ax2.set_title('Centerline Intensity Profile with Annotations', color=gold, fontsize=13, fontweight='bold')
ax2.set_xlabel('Body signal intensity (inverted)', color='#aaa', fontsize=10)
ax2.set_ylabel('Row (top to bottom)', color='#aaa', fontsize=10)
ax2.tick_params(colors='#666')
ax2.invert_yaxis()
ax2.legend(loc='lower right', fontsize=9, facecolor='#2a2a2a', edgecolor='#555', labelcolor='#ccc')

plt.tight_layout(pad=2.0)
os.makedirs('output/analysis', exist_ok=True)
plt.savefig('output/analysis/height_estimation.png', dpi=150, facecolor='#1a1a1a',
            bbox_inches='tight')
plt.close()
print(f"\nSaved: output/analysis/height_estimation.png")
