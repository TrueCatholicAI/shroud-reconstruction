"""Body Proportions Analysis: Compute body segment ratios from the full-body
Shroud depth extraction and compare against Vitruvian proportions and modern
anthropometric data."""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter
import os
import shutil

print("=" * 60)
print("  Body Proportions Analysis")
print("  Shroud Depth vs Vitruvian vs Modern Anthropometric")
print("=" * 60)

# ── Output directories ──────────────────────────────────────────────────────
os.makedirs('output/analysis', exist_ok=True)
os.makedirs('docs/images', exist_ok=True)

# ── Style constants ─────────────────────────────────────────────────────────
BG_COLOR = '#1a1a1a'
GOLD = '#c4a35a'
LIGHT_GOLD = '#e8d5a3'
PALE_GOLD = '#f0e6cc'
WHITE = '#ffffff'
GRAY = '#888888'
DARK_GRAY = '#333333'

# ── Step 1: Load depth data ────────────────────────────────────────────────
depth_2d = None
if os.path.exists('output/full_body/depth_body_smooth.npy'):
    depth_2d = np.load('output/full_body/depth_body_smooth.npy').astype(np.float32)
    print(f"Loaded depth_body_smooth.npy: {depth_2d.shape}")
elif os.path.exists('output/full_body/depth_body_healed.npy'):
    depth_2d = np.load('output/full_body/depth_body_healed.npy').astype(np.float32)
    print(f"Loaded depth_body_healed.npy: {depth_2d.shape}")
else:
    print("No preprocessed depth found, extracting from source image...")
    img = cv2.imread('data/source/shroud_full_negatives.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    frontal = img[:, :w // 2]
    from scipy.ndimage import zoom as nd_zoom
    target_h, target_w = 618, 300
    depth_2d = nd_zoom(frontal.astype(np.float32),
                       (target_h / frontal.shape[0], target_w / frontal.shape[1]), order=1)
    depth_2d = gaussian_filter(depth_2d, sigma=3)
    print(f"Extracted from source: {depth_2d.shape}")

rows, cols = depth_2d.shape
print(f"Depth map: {rows} x {cols}")

# ── Step 2: Calibration ────────────────────────────────────────────────────
TOTAL_HEIGHT_PX = rows  # 618 pixels
TOTAL_HEIGHT_M = 1.76   # established calibration
PX_PER_M = TOTAL_HEIGHT_PX / TOTAL_HEIGHT_M
MM_PER_PX = 1000.0 / PX_PER_M
print(f"Calibration: {TOTAL_HEIGHT_PX} px = {TOTAL_HEIGHT_M} m")
print(f"  {PX_PER_M:.1f} px/m, {MM_PER_PX:.2f} mm/px")

# ── Step 3: Extract profiles ───────────────────────────────────────────────
# Vertical centerline depth profile (average of central 30%)
strip_l = int(cols * 0.35)
strip_r = int(cols * 0.65)
center_profile = np.mean(depth_2d[:, strip_l:strip_r], axis=1)
center_smooth = gaussian_filter(center_profile, sigma=5)

# Body width profile: for each row, count how many pixels are "body" (above threshold)
body_thresh = np.percentile(depth_2d, 30)
width_profile = np.sum(depth_2d > body_thresh, axis=1).astype(np.float32)
width_smooth = gaussian_filter(width_profile, sigma=8)

# Depth gradient (first derivative of centerline)
depth_grad = np.gradient(center_smooth)
depth_grad_smooth = gaussian_filter(depth_grad, sigma=5)

# Second derivative for inflection points
depth_grad2 = np.gradient(depth_grad_smooth)
depth_grad2_smooth = gaussian_filter(depth_grad2, sigma=5)

print(f"\nProfile range: {center_smooth.min():.1f} - {center_smooth.max():.1f}")
print(f"Width range: {width_smooth.min():.1f} - {width_smooth.max():.1f}")

# ── Step 4: Identify body landmarks ────────────────────────────────────────
# Strategy: combine depth profile analysis, body width profile, and anatomical
# constraints.  The Shroud image has the body oriented head-at-top, feet-at-bottom.
# The depth data is a CLAHE-enhanced image where the body surface is bright.
# Key challenge: the folded hands obscure the waist/navel area.

print("\n--- Landmark Detection ---")

# TOP OF HEAD: start of significant signal
noise_floor = np.percentile(center_smooth[:30], 50)
signal_thresh = noise_floor + 0.2 * (center_smooth.max() - noise_floor)
head_top_candidates = np.where(center_smooth[:60] > signal_thresh)[0]
head_top = head_top_candidates[0] if len(head_top_candidates) > 0 else 0
print(f"Head top: row {head_top}")

# FACE PEAK: the highest depth in the head region (nose/forehead)
face_region = center_smooth[head_top:head_top + 130]
face_peaks, _ = find_peaks(face_region, distance=15, prominence=5)
if len(face_peaks) > 0:
    face_peak = head_top + face_peaks[0]
else:
    face_peak = head_top + np.argmax(face_region)
print(f"Face peak: row {face_peak}")

# CHIN: the valley/drop below the face.  Search below the face peak.
chin_search_start = face_peak + 15
chin_search_end = min(chin_search_start + 70, rows)
chin_region = center_smooth[chin_search_start:chin_search_end]
chin_valleys, _ = find_peaks(-chin_region, distance=10, prominence=2)
if len(chin_valleys) > 0:
    chin = chin_search_start + chin_valleys[0]
else:
    chin = chin_search_start + np.argmin(chin_region)
print(f"Chin: row {chin}")

# SHOULDERS: use two complementary signals:
#   (a) first significant widening of body width after chin
#   (b) depth profile: region where the chest broadens
# The shoulder line is typically at about chin + 0.3*head_height anatomically.
# We search width profile for the first strong increase after the chin.
head_h_est = chin - head_top
shoulder_search_start = chin
shoulder_search_end = chin + int(head_h_est * 1.0)  # within ~1 head-length below chin
shoulder_search_end = min(shoulder_search_end, rows)

# Find where width first reaches near-maximum in that region
shoulder_width_region = width_smooth[shoulder_search_start:shoulder_search_end]
# Use a threshold: where width exceeds 70% of the max in that region
sw_max = np.max(shoulder_width_region)
sw_thresh = 0.70 * sw_max
sw_above = np.where(shoulder_width_region > sw_thresh)[0]
if len(sw_above) > 0:
    shoulders = shoulder_search_start + sw_above[0]
else:
    shoulders = chin + int(head_h_est * 0.4)  # fallback

# Record shoulder width at the widest point near shoulders
shoulder_width_zone = width_smooth[shoulders:min(shoulders + 40, rows)]
shoulder_width_px = np.max(shoulder_width_zone)
print(f"Shoulders: row {shoulders}, width: {shoulder_width_px:.0f} px")

# NAVEL: anatomically the navel is at approximately 60% of height from top
# (or 40% of height from soles).  On the Shroud the hands cross at ~45% which
# is the pubic symphysis.  The navel is ABOVE the hands.
# Search the depth/width profiles for the navel.  Use the width-narrowing
# (waist) which should be slightly above the hand crossing.
hand_crossing = int(TOTAL_HEIGHT_PX * 0.45)
navel_search_start = int(TOTAL_HEIGHT_PX * 0.32)
navel_search_end = int(TOTAL_HEIGHT_PX * 0.48)
navel_width_region = width_smooth[navel_search_start:navel_search_end]

# The waist (narrowest point) should approximate the navel level
waist_valleys, _ = find_peaks(-navel_width_region, distance=15)
if len(waist_valleys) > 0:
    navel = navel_search_start + waist_valleys[0]
else:
    navel = navel_search_start + np.argmin(navel_width_region)

# Anatomical sanity: navel should be ~55-65% from top of head
navel_frac = (navel - head_top) / (TOTAL_HEIGHT_PX - head_top)
if navel_frac < 0.45 or navel_frac > 0.70:
    # Fallback to anatomical estimate: 58% from top
    navel = head_top + int(0.58 * (TOTAL_HEIGHT_PX - head_top))
    print(f"Navel (anatomical fallback): row {navel}")
else:
    print(f"Navel/waist (detected): row {navel}")

print(f"  Navel at {(navel - head_top) / (TOTAL_HEIGHT_PX - head_top) * 100:.1f}% from head top")
print(f"  Hand crossing (reference): row {hand_crossing} (45.0% of frame)")

# KNEE: depth feature around 72-82% from head top (knees at ~mid-leg)
knee_search_start = int(TOTAL_HEIGHT_PX * 0.62)
knee_search_end = int(TOTAL_HEIGHT_PX * 0.78)
knee_depth_region = center_smooth[knee_search_start:knee_search_end]
# Knee shows as a local peak or inflection in depth
knee_peaks, _ = find_peaks(knee_depth_region, distance=15, prominence=1)
if len(knee_peaks) > 0:
    knee = knee_search_start + knee_peaks[0]
else:
    # Look for gradient zero-crossing
    knee_grad = depth_grad_smooth[knee_search_start:knee_search_end]
    knee_zeros = np.where(np.diff(np.sign(knee_grad)))[0]
    if len(knee_zeros) > 0:
        knee = knee_search_start + knee_zeros[0]
    else:
        knee = int(TOTAL_HEIGHT_PX * 0.70)
print(f"Knee: row {knee} ({knee / TOTAL_HEIGHT_PX * 100:.1f}% of frame)")

# FEET: end of body signal
feet_region = center_smooth[int(TOTAL_HEIGHT_PX * 0.85):]
feet_noise = np.percentile(feet_region[-20:], 50)
feet_thresh = feet_noise + 0.15 * (feet_region.max() - feet_noise)
feet_candidates = np.where(feet_region > feet_thresh)[0]
if len(feet_candidates) > 0:
    feet = int(TOTAL_HEIGHT_PX * 0.85) + feet_candidates[-1]
else:
    feet = TOTAL_HEIGHT_PX - 1
feet = min(feet, TOTAL_HEIGHT_PX - 1)
print(f"Feet: row {feet}")

# Landmarks dictionary
landmarks = {
    'Head top': head_top,
    'Chin': chin,
    'Shoulders': shoulders,
    'Navel': navel,
    'Knee': knee,
    'Feet': feet
}

# ── Step 5: Compute body segment ratios ─────────────────────────────────────
print("\n--- Body Segment Measurements ---")

total_h = feet - head_top
head_h = chin - head_top
neck_h = shoulders - chin
torso_h = navel - shoulders
upper_leg_h = knee - navel
lower_leg_h = feet - knee

segments = {
    'Head': (head_top, chin, head_h),
    'Neck': (chin, shoulders, neck_h),
    'Torso': (shoulders, navel, torso_h),
    'Upper leg': (navel, knee, upper_leg_h),
    'Lower leg': (knee, feet, lower_leg_h),
}

for name, (start, end, length) in segments.items():
    cm = length * MM_PER_PX / 10
    pct = length / total_h * 100
    print(f"  {name:12s}: rows {start:3d}-{end:3d} = {length:3d} px = {cm:5.1f} cm ({pct:5.1f}%)")

print(f"  {'TOTAL':12s}: rows {head_top:3d}-{feet:3d} = {total_h:3d} px = {total_h * MM_PER_PX / 10:.1f} cm")

# Compute ratios
shroud_ratios = {
    'Head / Total': head_h / total_h,
    'Head+Neck / Total': (head_h + neck_h) / total_h,
    'Torso / Total': torso_h / total_h,
    'Legs / Total': (upper_leg_h + lower_leg_h) / total_h,
    'Upper / Lower': (head_h + neck_h + torso_h) / (upper_leg_h + lower_leg_h),
    'Navel position': (navel - head_top) / total_h,
}

# Shoulder width in cm (if 2D available)
if depth_2d is not None:
    sw_cm = shoulder_width_px * MM_PER_PX / 10
    shroud_ratios['Shoulder width (cm)'] = sw_cm
    print(f"\n  Shoulder width: {shoulder_width_px:.0f} px = {sw_cm:.1f} cm")

# ── Step 6: Reference proportions ──────────────────────────────────────────
vitruvian = {
    'Head / Total': 1.0 / 8.0,          # 0.125
    'Head+Neck / Total': 1.5 / 8.0,     # 0.1875
    'Torso / Total': 3.0 / 8.0,         # 0.375
    'Legs / Total': 4.0 / 8.0,          # 0.500
    'Upper / Lower': 0.50 / 0.50,       # 1.0 (but classically lower body slightly longer)
    'Navel position': 0.618,            # Golden ratio
}

modern_male = {
    'Head / Total': 1.0 / 7.5,          # 0.133
    'Head+Neck / Total': 1.6 / 7.5,     # 0.213
    'Torso / Total': 2.8 / 7.5,         # 0.373
    'Legs / Total': 3.6 / 7.5,          # 0.480
    'Upper / Lower': 0.52 / 0.48,       # sitting height ratio ~0.52
    'Navel position': 0.60,             # slightly above 0.6
}

# First-century Mediterranean estimates (based on skeletal studies)
mediterranean_1c = {
    'Head / Total': 1.0 / 7.6,
    'Head+Neck / Total': 1.5 / 7.6,
    'Torso / Total': 2.9 / 7.6,
    'Legs / Total': 3.7 / 7.6,
    'Upper / Lower': 0.51 / 0.49,
    'Navel position': 0.61,
}

# ── Print comparison table ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'Ratio':<22s} {'Shroud':>8s} {'Vitruvian':>10s} {'Modern':>8s} {'1st-C Med':>10s}")
print("-" * 72)
ratio_keys = ['Head / Total', 'Head+Neck / Total', 'Torso / Total',
              'Legs / Total', 'Upper / Lower', 'Navel position']
for key in ratio_keys:
    sv = shroud_ratios.get(key, 0)
    vv = vitruvian.get(key, 0)
    mv = modern_male.get(key, 0)
    mc = mediterranean_1c.get(key, 0)
    print(f"  {key:<20s} {sv:8.4f} {vv:10.4f} {mv:8.4f} {mc:10.4f}")
print("=" * 72)

# Head-count analysis
head_count = total_h / head_h if head_h > 0 else 0
print(f"\nBody = {head_count:.2f} head-lengths (Vitruvian ideal: 8.00, modern avg: 7.50)")

# Golden ratio check
print(f"Navel at {shroud_ratios['Navel position']:.4f} of height (golden ratio = 0.6180)")
gr_diff = abs(shroud_ratios['Navel position'] - 0.618)
print(f"  Deviation from golden ratio: {gr_diff:.4f} ({gr_diff/0.618*100:.1f}%)")

# ── Step 7: Visualizations ──────────────────────────────────────────────────
print("\n--- Generating Visualizations ---")

# ────────────────────────────────────────────────────────────────────────────
# Figure A: Annotated depth profile with landmarks
# ────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 10), facecolor=BG_COLOR,
                         gridspec_kw={'width_ratios': [1.5, 1, 1]})

# Panel 1: Depth profile with landmarks
ax1 = axes[0]
ax1.set_facecolor(BG_COLOR)
y_axis = np.arange(rows)
ax1.plot(center_smooth, y_axis, color=GOLD, linewidth=1.5, label='Depth profile')
ax1.set_ylim(rows, 0)  # top to bottom
ax1.set_xlabel('Depth intensity', color=LIGHT_GOLD, fontsize=10)
ax1.set_ylabel('Row (pixels)', color=LIGHT_GOLD, fontsize=10)
ax1.set_title('Vertical Depth Profile with Landmarks', color=GOLD, fontsize=12, pad=10)
ax1.tick_params(colors=GRAY)
for spine in ax1.spines.values():
    spine.set_color(DARK_GRAY)

# Annotate landmarks
landmark_colors = {
    'Head top': '#ff6b6b',
    'Chin': '#ffa07a',
    'Shoulders': '#87ceeb',
    'Navel': '#98fb98',
    'Knee': '#dda0dd',
    'Feet': '#f0e68c',
}

for name, row in landmarks.items():
    color = landmark_colors[name]
    ax1.axhline(y=row, color=color, linestyle='--', alpha=0.7, linewidth=1)
    cm_val = (row - head_top) * MM_PER_PX / 10
    ax1.annotate(f'{name} ({cm_val:.0f} cm)',
                xy=(center_smooth.max() * 0.95, row),
                fontsize=8, color=color, va='bottom',
                fontweight='bold')

# Add segment length annotations on the left side
seg_colors = ['#ff6b6b', '#ffa07a', '#87ceeb', '#98fb98', '#dda0dd']
for i, (name, (start, end, length)) in enumerate(segments.items()):
    mid = (start + end) / 2
    cm = length * MM_PER_PX / 10
    ax1.annotate(f'{name}\n{cm:.1f} cm',
                xy=(center_smooth.min() - 5, mid),
                fontsize=7, color=seg_colors[i % len(seg_colors)],
                va='center', ha='right',
                fontweight='bold')

# Panel 2: Body width profile
ax2 = axes[1]
ax2.set_facecolor(BG_COLOR)
ax2.plot(width_smooth, y_axis, color='#87ceeb', linewidth=1.5, label='Body width')
ax2.set_ylim(rows, 0)
ax2.set_xlabel('Width (px)', color=LIGHT_GOLD, fontsize=10)
ax2.set_title('Body Width Profile', color=GOLD, fontsize=12, pad=10)
ax2.tick_params(colors=GRAY)
for spine in ax2.spines.values():
    spine.set_color(DARK_GRAY)

for name, row in landmarks.items():
    ax2.axhline(y=row, color=landmark_colors[name], linestyle='--', alpha=0.5, linewidth=0.8)

# Panel 3: Depth gradient
ax3 = axes[2]
ax3.set_facecolor(BG_COLOR)
ax3.plot(depth_grad_smooth, y_axis, color='#ff9999', linewidth=1, label='1st derivative')
ax3.axvline(x=0, color=GRAY, linestyle='-', alpha=0.3)
ax3.set_ylim(rows, 0)
ax3.set_xlabel('Depth gradient', color=LIGHT_GOLD, fontsize=10)
ax3.set_title('Depth Gradient', color=GOLD, fontsize=12, pad=10)
ax3.tick_params(colors=GRAY)
for spine in ax3.spines.values():
    spine.set_color(DARK_GRAY)

for name, row in landmarks.items():
    ax3.axhline(y=row, color=landmark_colors[name], linestyle='--', alpha=0.5, linewidth=0.8)

fig.suptitle('Body Proportions: Depth Profile Analysis',
             color=GOLD, fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_a = 'output/analysis/body_proportions_profile.png'
fig.savefig(out_a, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out_a}")

# ────────────────────────────────────────────────────────────────────────────
# Figure B: Comparison bar chart
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

bar_keys = ratio_keys
n_groups = len(bar_keys)
n_bars = 4
bar_width = 0.18
indices = np.arange(n_groups)

colors_bars = [GOLD, '#87ceeb', '#98fb98', '#dda0dd']
labels = ['Shroud', 'Vitruvian', 'Modern Male', '1st-C Mediterranean']
datasets = [shroud_ratios, vitruvian, modern_male, mediterranean_1c]

for i, (data, color, label) in enumerate(zip(datasets, colors_bars, labels)):
    values = [data.get(k, 0) for k in bar_keys]
    offset = (i - n_bars / 2 + 0.5) * bar_width
    bars = ax.bar(indices + offset, values, bar_width,
                  color=color, alpha=0.85, label=label,
                  edgecolor='#444444', linewidth=0.5)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=6, color=color, fontweight='bold')

ax.set_xticks(indices)
ax.set_xticklabels([k.replace(' / ', '\n/ ') for k in bar_keys],
                   color=LIGHT_GOLD, fontsize=9)
ax.set_ylabel('Ratio', color=LIGHT_GOLD, fontsize=11)
ax.set_title('Body Proportion Ratios: Shroud vs Reference Standards',
             color=GOLD, fontsize=14, fontweight='bold', pad=15)
ax.legend(facecolor=DARK_GRAY, edgecolor=GOLD, labelcolor=LIGHT_GOLD,
          fontsize=9, loc='upper right')
ax.tick_params(colors=GRAY)
ax.set_ylim(0, max(max(d.get(k, 0) for k in bar_keys) for d in datasets) * 1.15)
for spine in ax.spines.values():
    spine.set_color(DARK_GRAY)
ax.grid(axis='y', alpha=0.15, color=GRAY)

plt.tight_layout()
out_b = 'output/analysis/body_proportions_comparison.png'
fig.savefig(out_b, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out_b}")

# ────────────────────────────────────────────────────────────────────────────
# Figure C: Schematic body diagram with stacked segments
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 14), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Draw three columns: Shroud, Vitruvian ideal, Modern
col_centers = [0.2, 0.5, 0.8]
col_labels = ['SHROUD\nMeasurement', 'VITRUVIAN\nIdeal (8 heads)', 'MODERN\nMale Average']
col_width = 0.15

# Define segment proportions for each
segment_names = ['Head', 'Neck', 'Torso', 'Upper Leg', 'Lower Leg']
segment_colors = ['#ff6b6b', '#ffa07a', '#87ceeb', '#98fb98', '#dda0dd']

# Shroud proportions (from measurements)
shroud_segs = [head_h / total_h, neck_h / total_h, torso_h / total_h,
               upper_leg_h / total_h, lower_leg_h / total_h]

# Vitruvian: head=1/8, neck=0.5/8, torso=2.5/8, upper leg=2/8, lower leg=2/8
vitruvian_segs = [1/8, 0.5/8, 2.5/8, 2.0/8, 2.0/8]

# Modern: head=1/7.5, neck=0.6/7.5, torso=2.4/7.5, upper leg=1.8/7.5, lower leg=1.7/7.5
modern_segs = [1/7.5, 0.6/7.5, 2.4/7.5, 1.8/7.5, 1.7/7.5]

all_segs = [shroud_segs, vitruvian_segs, modern_segs]

# Total figure height in data coordinates
fig_top = 10.0
fig_bot = 0.5
fig_h = fig_top - fig_bot

for col_idx, (cx, label, segs) in enumerate(zip(col_centers, col_labels, all_segs)):
    # Column title
    ax.text(cx, fig_top + 0.6, label, ha='center', va='bottom',
            color=GOLD if col_idx == 0 else LIGHT_GOLD,
            fontsize=11, fontweight='bold')

    # Draw stacked rectangles from top to bottom
    y_pos = fig_top
    for seg_idx, (seg_name, seg_frac, seg_color) in enumerate(zip(
            segment_names, segs, segment_colors)):
        seg_h = seg_frac * fig_h
        rect = FancyBboxPatch(
            (cx - col_width / 2, y_pos - seg_h),
            col_width, seg_h,
            boxstyle="round,pad=0.005",
            facecolor=seg_color,
            edgecolor='#444444',
            alpha=0.8 if col_idx == 0 else 0.5,
            linewidth=1.5 if col_idx == 0 else 1
        )
        ax.add_patch(rect)

        # Label inside the box
        if seg_h > 0.3:
            cm_val = seg_frac * TOTAL_HEIGHT_M * 100
            pct_str = f'{seg_frac*100:.1f}%'
            if col_idx == 0:
                txt = f'{seg_name}\n{cm_val:.0f} cm\n({pct_str})'
            else:
                txt = f'{seg_name}\n({pct_str})'
            ax.text(cx, y_pos - seg_h / 2, txt,
                    ha='center', va='center',
                    fontsize=7, color='#1a1a1a', fontweight='bold')
        elif seg_h > 0.15:
            ax.text(cx, y_pos - seg_h / 2, segment_names[seg_idx],
                    ha='center', va='center',
                    fontsize=6, color='#1a1a1a', fontweight='bold')

        y_pos -= seg_h

    # Total height label
    ax.text(cx, fig_bot - 0.3,
            f'Total: {TOTAL_HEIGHT_M*100:.0f} cm' if col_idx == 0 else '',
            ha='center', va='top', color=LIGHT_GOLD, fontsize=9)

# Draw connecting lines between shroud and vitruvian for comparison
shroud_cx = col_centers[0] + col_width / 2 + 0.01
vitruv_cx = col_centers[1] - col_width / 2 - 0.01

y_s = fig_top
y_v = fig_top
for seg_idx in range(len(segment_names)):
    s_h = shroud_segs[seg_idx] * fig_h
    v_h = vitruvian_segs[seg_idx] * fig_h
    y_s -= s_h
    y_v -= v_h
    # Draw dashed connecting line at segment boundary
    ax.plot([shroud_cx, vitruv_cx], [y_s, y_v],
            color=GRAY, linestyle=':', alpha=0.4, linewidth=0.8)

# Add legend for segments
for i, (name, color) in enumerate(zip(segment_names, segment_colors)):
    ax.add_patch(plt.Rectangle((0.92, fig_top - 0.5 - i * 0.45), 0.04, 0.3,
                                facecolor=color, edgecolor='#444444', alpha=0.8))
    ax.text(0.97, fig_top - 0.35 - i * 0.45, name,
            color=LIGHT_GOLD, fontsize=8, va='center')

# Head-count scale on far left
for i in range(9):
    y = fig_top - i * (fig_h / 8)
    ax.plot([0.03, 0.07], [y, y], color=GRAY, linewidth=0.5)
    if i < 8:
        ax.text(0.05, y - fig_h / 16, f'{i+1}', ha='center', va='center',
                color=GRAY, fontsize=7)
ax.text(0.05, fig_top + 0.3, 'Head\nunits', ha='center', va='bottom',
        color=GRAY, fontsize=7)

ax.set_xlim(0, 1.05)
ax.set_ylim(fig_bot - 0.8, fig_top + 1.5)
ax.set_aspect('auto')
ax.axis('off')

ax.set_title('Body Proportions: Segment Comparison Diagram',
             color=GOLD, fontsize=14, fontweight='bold', pad=20)

# Summary text box
summary_lines = [
    f"Shroud figure: {head_count:.2f} head-lengths (Vitruvian: 8.00, Modern: 7.50)",
    f"Navel at {shroud_ratios['Navel position']*100:.1f}% of height (Golden ratio: 61.8%)",
    f"Upper/Lower ratio: {shroud_ratios['Upper / Lower']:.3f}",
]
summary_text = '\n'.join(summary_lines)
ax.text(0.5, fig_bot - 0.5, summary_text,
        ha='center', va='top', color=LIGHT_GOLD, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=DARK_GRAY,
                  edgecolor=GOLD, alpha=0.8))

plt.tight_layout()
out_c = 'output/analysis/body_proportions_figure.png'
fig.savefig(out_c, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out_c}")

# ── Copy to docs/images ────────────────────────────────────────────────────
for src_path in [out_a, out_b, out_c]:
    dst = os.path.join('docs/images', os.path.basename(src_path))
    shutil.copy2(src_path, dst)
    print(f"  Copied: {dst}")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Total body height: {total_h} px = {total_h * MM_PER_PX / 10:.1f} cm")
print(f"  Head-length count: {head_count:.2f}")
print(f"  Navel position: {shroud_ratios['Navel position']:.4f} "
      f"(golden ratio deviation: {gr_diff:.4f})")
print()

# Deviation analysis
print("  Deviation from Vitruvian:")
for key in ratio_keys:
    sv = shroud_ratios.get(key, 0)
    vv = vitruvian.get(key, 0)
    diff = sv - vv
    pct = abs(diff / vv * 100) if vv else 0
    direction = '+' if diff > 0 else ''
    print(f"    {key:<22s}: {direction}{diff:.4f} ({pct:.1f}%)")

print()
print("  Deviation from Modern Male:")
for key in ratio_keys:
    sv = shroud_ratios.get(key, 0)
    mv = modern_male.get(key, 0)
    diff = sv - mv
    pct = abs(diff / mv * 100) if mv else 0
    direction = '+' if diff > 0 else ''
    print(f"    {key:<22s}: {direction}{diff:.4f} ({pct:.1f}%)")

if 'Shoulder width (cm)' in shroud_ratios:
    print(f"\n  Shoulder width: {shroud_ratios['Shoulder width (cm)']:.1f} cm "
          f"(modern male avg: ~46 cm)")

print("\n  Analysis complete.")
print("=" * 60)
