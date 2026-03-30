"""Task I: Deep scourge mark pattern analysis — dumbbell detection, spacing, 3D mapping."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
from scipy.stats import pearsonr
from matplotlib.colors import TwoSlopeNorm

print("=== Deep Scourge Mark Pattern Analysis ===")

# Load full-body frontal
img = cv2.imread('data/source/shroud_full_negatives.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
frontal = img[:, :w//2]
frontal = frontal[:, :int(frontal.shape[1] * 0.95)]
print(f"Frontal: {frontal.shape}")

# CLAHE depth
norm_img = cv2.normalize(frontal, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_raw = clahe.apply(norm_img)

# Multiple smoothing scales for multi-scale residual
sigma_body = 25  # large-scale body shape
depth_smooth_body = gaussian_filter(depth_raw.astype(np.float64), sigma=sigma_body)

sigma_medium = 8  # medium features
depth_smooth_medium = gaussian_filter(depth_raw.astype(np.float64), sigma=sigma_medium)

# Band-pass residual: medium-smooth minus body-smooth isolates mid-frequency features
# This should capture scourge marks (5-15mm) while rejecting cloth weave (1-2mm) and body shape (>30mm)
bandpass = depth_smooth_medium - depth_smooth_body
print(f"Bandpass residual: range [{bandpass.min():.1f}, {bandpass.max():.1f}], std={bandpass.std():.1f}")

# === Focus on body regions ===
bh, bw = frontal.shape

regions = {
    'upper_back_chest': (int(bh*0.15), int(bh*0.30), 0, bw),
    'lower_torso': (int(bh*0.30), int(bh*0.45), 0, bw),
    'upper_legs': (int(bh*0.50), int(bh*0.70), 0, bw),
    'lower_legs': (int(bh*0.70), int(bh*0.88), 0, bw),
}

all_candidates = []

for region_name, (y1, y2, x1, x2) in regions.items():
    region_bp = bandpass[y1:y2, x1:x2]
    region_src = frontal[y1:y2, x1:x2]

    # Threshold: features > 1.5 std above mean in this region
    mean_r = region_bp.mean()
    std_r = region_bp.std()
    thresh = mean_r + 1.5 * std_r

    binary = (region_bp > thresh).astype(np.uint8) * 255

    # Morphological opening to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Connected components
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

    # Filter: flagrum dumbbell marks are elongated (aspect ratio 1.5-4.0) and small (10-120 px area)
    # At ~12 px/cm, a 10mm mark = 12px. Dumbbell = two tips connected = ~15-30px length
    region_candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if 8 <= area <= 150:
            aspect = max(cw, ch) / (min(cw, ch) + 1e-8)
            # Dumbbell shape: elongated (aspect > 1.3) but not too thin (aspect < 5)
            if 1.3 <= aspect <= 5.0:
                cx, cy = centroids[i]
                region_candidates.append({
                    'region': region_name,
                    'cx': cx + x1, 'cy': cy + y1,
                    'local_cx': cx, 'local_cy': cy,
                    'area': area, 'width': cw, 'height': ch,
                    'aspect': aspect,
                })

    all_candidates.extend(region_candidates)
    print(f"  {region_name}: {num_labels-1} raw components, {len(region_candidates)} dumbbell candidates")

print(f"\nTotal dumbbell-shaped candidates: {len(all_candidates)}")

# === Spacing analysis ===
# Roman flagrum had 2-3 tips, typically leaving marks at ~2-3cm spacing
# At ~12 px/cm, that's ~24-36 px between paired marks
# Look for pairs of candidates within 20-45 px of each other
pairs = []
for i in range(len(all_candidates)):
    for j in range(i+1, len(all_candidates)):
        c1, c2 = all_candidates[i], all_candidates[j]
        dist = np.sqrt((c1['cx'] - c2['cx'])**2 + (c1['cy'] - c2['cy'])**2)
        if 18 <= dist <= 50:  # ~1.5-4cm range
            pairs.append((i, j, dist))

print(f"Paired marks (18-50px / ~1.5-4cm spacing): {len(pairs)}")

# === Visualization 1: Full body with all regions marked ===
fig, axes = plt.subplots(1, 3, figsize=(21, 14))
fig.patch.set_facecolor('#1a1a1a')

# Source with region boxes
src_rgb = cv2.cvtColor(frontal, cv2.COLOR_GRAY2RGB)
for name, (y1, y2, x1, x2) in regions.items():
    cv2.rectangle(src_rgb, (x1, y1), (x2-1, y2-1), (0, 200, 100), 2)

axes[0].imshow(src_rgb)
axes[0].set_title('Analysis Regions', color='white', fontsize=12)
axes[0].axis('off')

# Bandpass residual
vmax = max(abs(bandpass.min()), abs(bandpass.max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
axes[1].imshow(bandpass, cmap='RdBu_r', norm=norm)
axes[1].set_title('Bandpass Residual\n(mid-frequency features)', color='white', fontsize=12)
axes[1].axis('off')

# Candidates on depth
depth_rgb = cv2.cvtColor(depth_raw, cv2.COLOR_GRAY2RGB)
for c in all_candidates:
    cx, cy = int(c['cx']), int(c['cy'])
    cv2.circle(depth_rgb, (cx, cy), 4, (0, 255, 0), 1)
# Draw pairs
for i, j, dist in pairs:
    c1, c2 = all_candidates[i], all_candidates[j]
    pt1 = (int(c1['cx']), int(c1['cy']))
    pt2 = (int(c2['cx']), int(c2['cy']))
    cv2.line(depth_rgb, pt1, pt2, (255, 100, 0), 1)

axes[2].imshow(depth_rgb)
axes[2].set_title(f'{len(all_candidates)} Dumbbell Candidates\n{len(pairs)} Paired Marks (orange lines)',
                  color='white', fontsize=12)
axes[2].axis('off')

for ax in axes:
    ax.set_facecolor('#1a1a1a')
fig.suptitle('Scourge Mark Pattern Analysis — Bandpass + Shape Filtering', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/scourge_deep_overview.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: scourge_deep_overview.png")

# === Visualization 2: Region detail panels ===
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor('#1a1a1a')

for ax, (region_name, (y1, y2, x1, x2)) in zip(axes.flat, regions.items()):
    region_rgb = cv2.cvtColor(frontal[y1:y2, x1:x2], cv2.COLOR_GRAY2RGB)
    region_cands = [c for c in all_candidates if c['region'] == region_name]
    for c in region_cands:
        cx, cy = int(c['local_cx']), int(c['local_cy'])
        cv2.circle(region_rgb, (cx, cy), 5, (0, 255, 0), 1)
        if c['aspect'] > 2.0:  # highlight strongly elongated
            cv2.circle(region_rgb, (cx, cy), 7, (255, 200, 0), 1)

    ax.imshow(region_rgb)
    n_cands = len(region_cands)
    n_elongated = sum(1 for c in region_cands if c['aspect'] > 2.0)
    ax.set_title(f'{region_name}\n{n_cands} candidates ({n_elongated} highly elongated)',
                 color='white', fontsize=11)
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')

fig.suptitle('Regional Scourge Mark Candidates\n(green=dumbbell shape, yellow=strongly elongated)',
             color='#c4a35a', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/scourge_regions_detail.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: scourge_regions_detail.png")

# === Visualization 3: Spacing histogram ===
if len(pairs) > 0:
    pair_dists = [d for _, _, d in pairs]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#222')
    ax.hist(pair_dists, bins=20, color='#c4a35a', edgecolor='#1a1a1a', alpha=0.8)
    ax.axvline(x=24, color='red', linestyle='--', alpha=0.7, label='Expected 2cm spacing')
    ax.axvline(x=36, color='blue', linestyle='--', alpha=0.7, label='Expected 3cm spacing')
    ax.set_xlabel('Distance between paired marks (pixels)', color='white')
    ax.set_ylabel('Count', color='white')
    ax.set_title('Spacing Distribution of Paired Mark Candidates', color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#333', labelcolor='white')
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('output/analysis/scourge_spacing_histogram.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print("Saved: scourge_spacing_histogram.png")

# === Regional distribution ===
print("\n--- Regional Distribution ---")
for region_name in regions:
    cands = [c for c in all_candidates if c['region'] == region_name]
    elongated = [c for c in cands if c['aspect'] > 2.0]
    print(f"  {region_name:20s}: {len(cands):4d} candidates, {len(elongated):3d} highly elongated")

print(f"\n--- Summary ---")
print(f"Total dumbbell candidates: {len(all_candidates)}")
print(f"Total paired marks (flagrum spacing): {len(pairs)}")
print(f"Literature expectation: ~120 marks on full body")
print(f"Our filtered detection: {len(all_candidates)} (improved over {832} from Wave 2)")
if len(pairs) > 0:
    print(f"Mean pair spacing: {np.mean([d for _,_,d in pairs]):.1f} px")

print("\n=== Deep Scourge Analysis Complete ===")
