import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import os

print("=" * 60)
print("BODY PROPORTIONS ANALYSIS - Shroud of Turin")
print("=" * 60)

# Create output directories
output_dir = 'output/analysis'
docs_images_dir = 'docs/images'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(docs_images_dir, exist_ok=True)

# Style configuration
plt.style.use('dark_background')
FIG_BG = '#1a1a1a'
ACCENT = '#c4a35a'
TEXT_COLOR = 'white'

# Calibration constants
TOTAL_HEIGHT_PX = 618
REAL_HEIGHT_M = 1.76
MM_PER_PX = (REAL_HEIGHT_M * 1000) / TOTAL_HEIGHT_PX
PX_PER_MM = 1 / MM_PER_PX

print(f"\nCalibration:")
print(f"  Total body height: {TOTAL_HEIGHT_PX}px = {REAL_HEIGHT_M}m")
print(f"  Scale: {MM_PER_PX:.3f}mm/px")

# Load image
image_path = 'data/source/shroud_full_negatives.jpg'
print(f"\nLoading image: {image_path}")
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
print(f"  Image shape: {img.shape}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
print(f"  Grayscale shape: {gray.shape}")

# Extract vertical center strip (middle 20% of width)
center_start = int(width * 0.40)
center_end = int(width * 0.60)
center_strip = gray[:, center_start:center_end]
print(f"\nExtracting center strip: columns {center_start} to {center_end}")

# Average horizontally to get 1D depth profile
depth_profile = np.mean(center_strip, axis=1)
print(f"  Depth profile length: {len(depth_profile)}")

# Smooth the profile
depth_smooth = savgol_filter(depth_profile, window_length=15, polyorder=3)
depth_smooth = gaussian_filter1d(depth_smooth, sigma=3)

# Invert for peaks (body is darker than background)
inverted = 255 - depth_smooth

# Normalize
inverted_norm = (inverted - np.min(inverted)) / (np.max(inverted) - np.min(inverted))

# Find peaks (body features)
peaks, properties = find_peaks(inverted_norm, height=0.3, distance=30, prominence=0.1)

print(f"\nPeak detection (scipy.signal.find_peaks):")
print(f"  Found {len(peaks)} significant peaks")
for i, p in enumerate(peaks):
    print(f"    Peak {i+1}: row {p}, height {inverted_norm[p]:.3f}")

# Gradient analysis for additional features
gradient = np.gradient(depth_smooth)
gradient_smooth = gaussian_filter1d(gradient, sigma=5)

# Locate landmarks
print("\n" + "-" * 60)
print("LANDMARK DETECTION")
print("-" * 60)

# Top of head - first significant rise from baseline
head_region = inverted_norm[:150]
baseline_head = np.mean(inverted_norm[:20])
head_threshold = baseline_head + 0.15
top_head_row = np.where(head_region > head_threshold)[0]
if len(top_head_row) > 0:
    top_head_row = int(top_head_row[0])
else:
    top_head_row = 10

print(f"\nTop of head: row {top_head_row} ({top_head_row * MM_PER_PX:.1f}mm = {top_head_row * MM_PER_PX / 1000:.3f}m)")

# Chin - local minimum between head top and shoulders
chin_search_start = 50
chin_search_end = 130
chin_region = inverted_norm[chin_search_start:chin_search_end]
min_idx_in_region = np.argmin(chin_region)
chin_row = chin_search_start + min_idx_in_region

# Verify it's actually a chin (not noise)
if chin_row < top_head_row + 30 or chin_row > 150:
    chin_row = top_head_row + 60

print(f"Chin (face-neck valley): row {chin_row} ({chin_row * MM_PER_PX:.1f}mm = {chin_row * MM_PER_PX / 1000:.3f}m)")

# Shoulders - width maximum in expected region
shoulder_search_start = 80
shoulder_search_end = 180
shoulder_region = inverted_norm[shoulder_search_start:shoulder_search_end]
shoulder_peaks, _ = find_peaks(shoulder_region, height=0.4, distance=20)

if len(shoulder_peaks) > 0:
    shoulder_peak_vals = [(sp, shoulder_region[sp]) for sp in shoulder_peaks]
    shoulder_peak_vals.sort(key=lambda x: x[1], reverse=True)
    shoulder_row = shoulder_search_start + shoulder_peak_vals[0][0]
else:
    shoulder_row = shoulder_search_start + np.argmax(shoulder_region)

print(f"Shoulders (width maximum): row {shoulder_row} ({shoulder_row * MM_PER_PX:.1f}mm = {shoulder_row * MM_PER_PX / 1000:.3f}m)")

# Hand crossing - known anatomical position (45% from top)
hand_crossing_ratio = 0.45
hand_row = int(top_head_row + (TOTAL_HEIGHT_PX - top_head_row) * hand_crossing_ratio)
hand_row = min(hand_row, TOTAL_HEIGHT_PX - 10)

print(f"Hand crossing (known at 45%): row {hand_row} ({hand_row * MM_PER_PX:.1f}mm = {hand_row * MM_PER_PX / 1000:.3f}m)")

# Navel/waist - narrowing region
waist_search_start = 200
waist_search_end = 350
waist_region = inverted_norm[waist_search_start:waist_search_end]

# Find local minimum (narrowest point = navel)
waist_valleys, _ = find_peaks(-waist_region, distance=20, prominence=0.05)
if len(waist_valleys) > 0:
    valley_vals = [(wv, waist_region[wv]) for wv in waist_valleys]
    valley_vals.sort(key=lambda x: x[1])
    navel_row = waist_search_start + valley_vals[0][0]
else:
    navel_row = waist_search_start + np.argmin(waist_region)

# Constrain to reasonable range
navel_row = max(waist_search_start, min(navel_row, waist_search_end))

print(f"Navel/waist (narrowing): row {navel_row} ({navel_row * MM_PER_PX:.1f}mm = {navel_row * MM_PER_PX / 1000:.3f}m)")

# Knees - depth feature in lower body
knee_search_start = 380
knee_search_end = 500
knee_region = inverted_norm[knee_search_start:knee_search_end]

# Look for local maxima (knees as protrusions)
knee_peaks, _ = find_peaks(knee_region, height=0.35, distance=30)
if len(knee_peaks) > 0:
    knee_peak_vals = [(kp, knee_region[kp]) for kp in knee_peaks]
    knee_peak_vals.sort(key=lambda x: x[1], reverse=True)
    knee_row = knee_search_start + knee_peak_vals[0][0]
else:
    # Fall back to minimum (knee crease)
    knee_row = knee_search_start + np.argmin(knee_region)

print(f"Knees (depth feature): row {knee_row} ({knee_row * MM_PER_PX:.1f}mm = {knee_row * MM_PER_PX / 1000:.3f}m)")

# Feet - signal end
feet_search_start = 550
feet_region = inverted_norm[feet_search_start:]
baseline_end = np.mean(inverted_norm[-20:])

# Find where signal drops below threshold
feet_candidates = np.where(feet_region < baseline_end + 0.1)[0]
if len(feet_candidates) > 0:
    feet_row = feet_search_start + feet_candidates[0]
else:
    feet_row = TOTAL_HEIGHT_PX - 10

print(f"Feet (signal end): row {feet_row} ({feet_row * MM_PER_PX:.1f}mm = {feet_row * MM_PER_PX / 1000:.3f}m)")

# Calculate body segment heights
total_body_height = feet_row - top_head_row
head_segment = chin_row - top_head_row
neck_segment = shoulder_row - chin_row
upper_torso = hand_row - shoulder_row
lower_torso = navel_row - hand_row
upper_legs = knee_row - navel_row
lower_legs = feet_row - knee_row

print("\n" + "-" * 60)
print("BODY SEGMENT ANALYSIS")
print("-" * 60)
print(f"\nTotal body height: {total_body_height}px ({total_body_height * MM_PER_PX:.1f}mm = {total_body_height * MM_PER_PX / 1000:.3f}m)")

print(f"\nHead segment: {head_segment}px ({head_segment * MM_PER_PX:.1f}mm)")
print(f"Neck segment: {neck_segment}px ({neck_segment * MM_PER_PX:.1f}mm)")
print(f"Upper torso: {upper_torso}px ({upper_torso * MM_PER_PX:.1f}mm)")
print(f"Lower torso: {lower_torso}px ({lower_torso * MM_PER_PX:.1f}mm)")
print(f"Upper legs: {upper_legs}px ({upper_legs * MM_PER_PX:.1f}mm)")
print(f"Lower legs: {lower_legs}px ({lower_legs * MM_PER_PX:.1f}mm)")

# Calculate ratios for comparison with Vitruvian/Modern standards
head_to_total = head_segment / total_body_height
torso_to_total = (neck_segment + upper_torso + lower_torso) / total_body_height
legs_to_total = (upper_legs + lower_legs) / total_body_height
navel_position = (navel_row - top_head_row) / total_body_height

# Extended Vitruvian ratios
upper_body_to_total = (head_segment + neck_segment + upper_torso) / total_body_height
lower_body_to_total = (lower_torso + upper_legs + lower_legs) / total_body_height

print("\n" + "-" * 60)
print("PROPORTION RATIOS")
print("-" * 60)

print(f"\n{'Ratio':<25} {'Shroud':<12} {'Vitruvian':<12} {'Modern':<12} {'Diff from Vitruvian':<20}")
print("-" * 75)

print(f"{'Head / Total':<25} {head_to_total:.4f}      0.1250        0.1200        {(head_to_total - 0.125) * 100:.2f}%")
print(f"{'Torso / Total':<25} {torso_to_total:.4f}      0.3750        0.3800        {(torso_to_total - 0.375) * 100:.2f}%")
print(f"{'Legs / Total':<25} {legs_to_total:.4f}      0.5000        0.5000        {(legs_to_total - 0.500) * 100:.2f}%")
print(f"{'Upper / Lower Torso':<25} {upper_torso/(lower_torso+0.01):.4f}      1.0000        1.0500        {(upper_torso/(lower_torso+0.01) - 1.0) * 100:.2f}%")

print("\n" + "-" * 60)
print("NAVEL POSITION ANALYSIS")
print("-" * 60)
golden_ratio = (1 + 5**0.5) / 2 - 1
print(f"\nNavel position from top: {navel_position:.4f}")
print(f"Golden ratio from top:   {golden_ratio:.4f}")
print(f"Difference:              {(navel_position - golden_ratio) * 100:.2f}%")

print("\n" + "=" * 60)
print("Figure Generation")
print("=" * 60)

# Create pixel-to-mm conversion for x-axis
y_values = np.arange(len(depth_profile))
y_mm = y_values * MM_PER_PX
y_cm = y_mm / 10

# Figure 1: Annotated vertical profile
fig1, ax1 = plt.subplots(figsize=(12, 16))
fig1.patch.set_facecolor(FIG_BG)
ax1.set_facecolor(FIG_BG)

ax1.plot(inverted_norm, y_cm, color=ACCENT, linewidth=1.5, label='Body profile')

landmarks = {
    'Top of Head': (top_head_row, 'red'),
    'Chin': (chin_row, 'orange'),
    'Shoulders': (shoulder_row, 'yellow'),
    'Hand Crossing': (hand_row, 'cyan'),
    'Navel': (navel_row, 'lime'),
    'Knees': (knee_row, 'magenta'),
    'Feet': (feet_row, 'blue')
}

for name, (row, color) in landmarks.items():
    y_pos = row * MM_PER_PX / 10
    ax1.axhline(y=y_pos, color=color, linestyle='--', alpha=0.7, linewidth=1)
    ax1.scatter([inverted_norm[row]], [y_pos], color=color, s=100, zorder=5)
    ax1.annotate(name, xy=(inverted_norm[row] + 0.05, y_pos),
                 fontsize=9, color=color, va='center', fontweight='bold')

ax1.set_xlabel('Normalized Body Width', fontsize=12, color=TEXT_COLOR)
ax1.set_ylabel('Height (cm)', fontsize=12, color=TEXT_COLOR)
ax1.set_title('Shroud Body Profile with Anatomical Landmarks\n(Center Strip Analysis)', 
              fontsize=14, color=ACCENT, fontweight='bold')
ax1.legend(loc='upper right', facecolor=FIG_BG, edgecolor=ACCENT)
ax1.grid(True, alpha=0.3, color=ACCENT)
ax1.set_xlim(-0.05, 1.1)
ax1.set_ylim(0, 180)
ax1.invert_yaxis()
ax1.tick_params(colors=TEXT_COLOR)
for spine in ax1.spines.values():
    spine.set_color(ACCENT)

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'body_props_profile.png'), dpi=150, facecolor=FIG_BG)
fig1.savefig(os.path.join(docs_images_dir, 'body_props_profile.png'), dpi=150, facecolor=FIG_BG)
print("Saved: body_props_profile.png")
plt.close(fig1)

# Figure 2: Grouped bar chart comparison
fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.patch.set_facecolor(FIG_BG)
ax2.set_facecolor(FIG_BG)

categories = ['Head/\nTotal', 'Torso/\nTotal', 'Legs/\nTotal', 'Upper/\nLower Torso', 'Navel\nPosition']

shrouds_values = [head_to_total, torso_to_total, legs_to_total, 
                   upper_torso/(lower_torso+0.01), navel_position]
vitruvian_values = [0.125, 0.375, 0.500, 1.000, golden_ratio]
modern_values = [0.120, 0.380, 0.500, 1.050, golden_ratio]

x = np.arange(len(categories))
width = 0.25

bars1 = ax2.bar(x - width, shrouds_values, width, label='Shroud', color='#c4a35a', edgecolor='white')
bars2 = ax2.bar(x, vitruvian_values, width, label='Vitruvian', color='#4a90d9', edgecolor='white')
bars3 = ax2.bar(x + width, modern_values, width, label='Modern Average', color='#7ed957', edgecolor='white')

ax2.set_xlabel('Proportion Category', fontsize=12, color=TEXT_COLOR)
ax2.set_ylabel('Ratio Value', fontsize=12, color=TEXT_COLOR)
ax2.set_title('Body Proportions: Shroud vs Vitruvian vs Modern', fontsize=14, color=ACCENT, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=10)
ax2.legend(loc='upper right', facecolor=FIG_BG, edgecolor=ACCENT)
ax2.grid(True, alpha=0.3, axis='y', color=ACCENT)
ax2.set_ylim(0, 0.7)
ax2.tick_params(colors=TEXT_COLOR)
for spine in ax2.spines.values():
    spine.set_color(ACCENT)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'body_props_comparison.png'), dpi=150, facecolor=FIG_BG)
fig2.savefig(os.path.join(docs_images_dir, 'body_props_comparison.png'), dpi=150, facecolor=FIG_BG)
print("Saved: body_props_comparison.png")
plt.close(fig2)

# Figure 3: Stacked colored rectangle diagram
fig3, ax3 = plt.subplots(figsize=(14, 10))
fig3.patch.set_facecolor(FIG_BG)
ax3.set_facecolor(FIG_BG)

# Normalize heights for display
total_display_height = 100
segments = [
    ('Head', head_segment, '#ff6b6b', top_head_row),
    ('Neck', neck_segment, '#feca57', chin_row),
    ('Upper Torso', upper_torso, '#48dbfb', shoulder_row),
    ('Lower Torso', lower_torso, '#1dd1a1', hand_row),
    ('Upper Legs', upper_legs, '#ff9ff3', navel_row),
    ('Lower Legs/Ft', lower_legs, '#a29bfe', knee_row)
]

# Calculate percentages
segment_heights = [seg[1] for seg in segments]
segment_pcts = [h / total_body_height * 100 for h in segment_heights]

# Create stacked horizontal bars
y_pos = 0
colors = [seg[2] for seg in segments]
heights_pct = segment_pcts

left = 0.1
bar_height = 0.6

for i, (name, px, color, row) in enumerate(segments):
    pct = segment_pcts[i]
    width = pct / 100 * 0.6
    rect = plt.Rectangle((left, y_pos), width, bar_height, 
                          facecolor=color, edgecolor='white', linewidth=2)
    ax3.add_patch(rect)
    
    # Add label
    mm_val = px * MM_PER_PX
    label_text = f'{name}\n{px}px\n{mm_val:.0f}mm\n({pct:.1f}%)'
    ax3.text(left + width + 0.02, y_pos + bar_height/2, label_text,
             fontsize=9, color=TEXT_COLOR, va='center', ha='left')
    
    y_pos += bar_height + 0.15

ax3.set_xlim(0, 1.2)
ax3.set_ylim(-0.2, y_pos + 0.2)
ax3.set_title('Shroud Body Proportions - Stacked Segment Analysis\n(Total Height: 618px = 1760mm = 1.76m)', 
              fontsize=14, color=ACCENT, fontweight='bold')
ax3.set_xlabel('Proportion', fontsize=12, color=TEXT_COLOR)
ax3.set_ylabel('')
ax3.set_yticks([])
ax3.tick_params(colors=TEXT_COLOR)
for spine in ax3.spines.values():
    spine.set_color(ACCENT)

# Add legend
legend_text = f"""
Calibration: 618px = 1.76m ({MM_PER_PX:.2f}mm/px)
Segment widths show proportional body height
"""
ax3.text(0.7, 0.95, legend_text, transform=ax3.transAxes,
         fontsize=10, color=TEXT_COLOR, va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor=FIG_BG, edgecolor=ACCENT, alpha=0.8))

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'body_props_figure.png'), dpi=150, facecolor=FIG_BG)
fig3.savefig(os.path.join(docs_images_dir, 'body_props_figure.png'), dpi=150, facecolor=FIG_BG)
print("Saved: body_props_figure.png")
plt.close(fig3)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nAll figures saved to:")
print(f"  - {output_dir}/")
print(f"  - {docs_images_dir}/")
print("\nSummary of Key Findings:")
print(f"  - Head proportion: {head_to_total:.4f} (Vitruvian: 0.1250, diff: {(head_to_total - 0.125)*100:+.2f}%)")
print(f"  - Torso proportion: {torso_to_total:.4f} (Vitruvian: 0.3750, diff: {(torso_to_total - 0.375)*100:+.2f}%)")
print(f"  - Legs proportion: {legs_to_total:.4f} (Vitruvian: 0.5000, diff: {(legs_to_total - 0.5)*100:+.2f}%)")
print(f"  - Navel position: {navel_position:.4f} (Golden ratio: {golden_ratio:.4f}, diff: {(navel_position - golden_ratio)*100:+.2f}%)")