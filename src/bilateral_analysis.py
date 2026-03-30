"""Bilateral (left vs right) comparison of Shroud face depth at horizontal slices.

Builds a wound catalog by analyzing asymmetry at brow, eye, cheek, mouth, and jaw levels.
"""
import os
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# --- Config ---
BG = '#1a1a1a'
GOLD = '#c4a35a'
OUT_DIR = 'output/analysis'
DOCS_DIR = 'docs/images'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

SLICE_LEVELS = {
    'Brow':  40,
    'Eye':   50,
    'Cheek': 65,
    'Mouth': 80,
    'Jaw':   95,
}
BAND = 3          # +-rows to average
MID = 75          # midpoint column
SIZE = 150

# --- Load and resize ---
raw = np.load('data/processed/depth_map_smooth_15.npy')
depth = cv2.resize(raw.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float64)
print(f"Depth map loaded and resized to {depth.shape}, range [{depth.min():.1f}, {depth.max():.1f}]")

# --- Extract slice profiles ---
profiles = {}
for name, row in SLICE_LEVELS.items():
    r0 = max(0, row - BAND)
    r1 = min(SIZE, row + BAND + 1)
    band = depth[r0:r1, :].mean(axis=0)  # average across band rows
    left_half = band[:MID][::-1]          # center -> left edge (flipped so index 0 = center)
    right_half = band[MID:]               # center -> right edge
    # Trim to same length
    n = min(len(left_half), len(right_half))
    left_half = left_half[:n]
    right_half = right_half[:n]
    diff = left_half - right_half
    std_val = np.std(band)
    threshold = std_val
    significant = np.abs(diff) > threshold
    profiles[name] = {
        'row': row,
        'band': band,
        'left': left_half,
        'right': right_half,
        'diff': diff,
        'std': std_val,
        'threshold': threshold,
        'significant': significant,
        'full_band_r0': r0,
        'full_band_r1': r1,
    }

# ============================================================
# Figure 1: bilateral_slices.png — 5 rows, depth map + profiles
# ============================================================
fig, axes = plt.subplots(5, 2, figsize=(14, 20),
                         gridspec_kw={'width_ratios': [1, 2]})
fig.patch.set_facecolor(BG)
fig.suptitle('Bilateral Symmetry Analysis — Horizontal Slices',
             color=GOLD, fontsize=18, fontweight='bold', y=0.98)

for i, (name, info) in enumerate(profiles.items()):
    ax_map = axes[i, 0]
    ax_prof = axes[i, 1]
    row = info['row']

    # Left panel: depth map with slice highlighted
    ax_map.imshow(depth, cmap='inferno', aspect='equal')
    ax_map.axhline(y=row, color=GOLD, linewidth=1.5, linestyle='--')
    ax_map.add_patch(Rectangle((0, info['full_band_r0']), SIZE,
                                info['full_band_r1'] - info['full_band_r0'],
                                linewidth=0, facecolor=GOLD, alpha=0.18))
    ax_map.set_title(f'{name} (row {row})', color='white', fontsize=12)
    ax_map.set_facecolor(BG)
    ax_map.tick_params(colors='white', labelsize=8)
    for spine in ax_map.spines.values():
        spine.set_color('#444')

    # Right panel: overlaid left (red) and mirrored right (blue)
    n = len(info['left'])
    x = np.arange(n)
    ax_prof.plot(x, info['left'], color='#e74c3c', linewidth=1.5, label='Left (center→edge)')
    ax_prof.plot(x, info['right'], color='#3498db', linewidth=1.5, label='Right (mirrored)')

    # Shade significant divergence regions
    diff = info['diff']
    sig = info['significant']
    # Find contiguous regions
    regions = []
    in_region = False
    for j in range(len(sig)):
        if sig[j] and not in_region:
            start = j
            in_region = True
        elif not sig[j] and in_region:
            regions.append((start, j))
            in_region = False
    if in_region:
        regions.append((start, len(sig)))

    for (s, e) in regions:
        mean_diff = diff[s:e].mean()
        color = '#e74c3c' if mean_diff > 0 else '#3498db'
        alpha = 0.25
        ax_prof.axvspan(s, e, color=color, alpha=alpha)
        label_text = 'swelling' if mean_diff > 0 else 'depression'
        mid_x = (s + e) / 2
        y_pos = max(info['left'][s:e].max(), info['right'][s:e].max()) + info['std'] * 0.3
        ax_prof.annotate(label_text, (mid_x, y_pos), color=color,
                         fontsize=8, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round,pad=0.2', fc=BG, ec=color, alpha=0.8))

    ax_prof.set_title(f'{name} Level — Left vs Mirrored Right', color='white', fontsize=12)
    ax_prof.set_xlabel('Distance from center (px)', color='#ccc', fontsize=9)
    ax_prof.set_ylabel('Depth intensity', color='#ccc', fontsize=9)
    ax_prof.legend(fontsize=8, loc='upper right',
                   facecolor=BG, edgecolor='#555', labelcolor='white')
    ax_prof.set_facecolor(BG)
    ax_prof.tick_params(colors='white', labelsize=8)
    for spine in ax_prof.spines.values():
        spine.set_color('#444')
    ax_prof.grid(True, alpha=0.15, color='white')

plt.tight_layout(rect=[0, 0, 1, 0.96])
path1 = os.path.join(OUT_DIR, 'bilateral_slices.png')
plt.savefig(path1, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {path1}")

# ============================================================
# Figure 2: bilateral_catalog.png — Summary heat map
# ============================================================
# Build matrix: rows = slice levels, cols = distance from center
max_n = min(len(p['diff']) for p in profiles.values())
catalog_matrix = np.zeros((len(profiles), max_n))
slice_names = list(profiles.keys())
for i, name in enumerate(slice_names):
    catalog_matrix[i, :] = profiles[name]['diff'][:max_n]

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

vmax = np.max(np.abs(catalog_matrix))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(catalog_matrix, cmap='RdBu_r', norm=norm, aspect='auto',
               interpolation='bilinear')
ax.set_yticks(range(len(slice_names)))
ax.set_yticklabels(slice_names, color='white', fontsize=11)
ax.set_xlabel('Distance from center (px)', color='#ccc', fontsize=11)
ax.set_title('Bilateral Asymmetry Catalog — Left minus Right',
             color=GOLD, fontsize=15, fontweight='bold', pad=12)
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_color('#444')

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Red = left raised (swelling)  |  Blue = left depressed',
               color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

plt.tight_layout()
path2 = os.path.join(OUT_DIR, 'bilateral_catalog.png')
plt.savefig(path2, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {path2}")

# ============================================================
# Figure 3: bilateral_overlay.png — Depth map with colored overlay
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.imshow(depth, cmap='gray', alpha=0.7)

# Build full asymmetry overlay
overlay_r = np.zeros((SIZE, SIZE))  # red channel (swelling)
overlay_b = np.zeros((SIZE, SIZE))  # blue channel (depression)

for name, info in profiles.items():
    row = info['row']
    r0 = info['full_band_r0']
    r1 = info['full_band_r1']
    diff = info['diff']
    sig = info['significant']
    n = len(diff)

    for j in range(n):
        if not sig[j]:
            continue
        # Map j back to image columns
        # left_half was band[:MID][::-1], so j=0 is col MID-1, j=k is col MID-1-k
        left_col = MID - 1 - j
        # right_half was band[MID:], so j=0 is col MID, j=k is col MID+k
        right_col = MID + j

        val = diff[j]
        if val > 0:  # left raised = swelling on left
            overlay_r[r0:r1, left_col] = abs(val)
            overlay_b[r0:r1, right_col] = abs(val)  # depression on right
        else:  # left depressed
            overlay_b[r0:r1, left_col] = abs(val)
            overlay_r[r0:r1, right_col] = abs(val)

# Normalize and display
max_val = max(overlay_r.max(), overlay_b.max(), 1e-6)
rgba_overlay = np.zeros((SIZE, SIZE, 4))
rgba_overlay[:, :, 0] = overlay_r / max_val  # red
rgba_overlay[:, :, 2] = overlay_b / max_val  # blue
rgba_overlay[:, :, 3] = np.clip((overlay_r + overlay_b) / max_val * 0.7, 0, 0.7)

ax.imshow(rgba_overlay, interpolation='nearest')

# Add slice level markers
for name, info in profiles.items():
    ax.annotate(name, (SIZE + 2, info['row']), color=GOLD, fontsize=9,
                fontweight='bold', va='center', annotation_clip=False)
    ax.axhline(y=info['row'], color=GOLD, linewidth=0.5, linestyle=':', alpha=0.5)

ax.axvline(x=MID, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_title('Bilateral Asymmetry Overlay\nRed = swelling  |  Blue = depression',
             color=GOLD, fontsize=14, fontweight='bold', pad=10)
ax.set_xlim(-2, SIZE + 20)
ax.tick_params(colors='white', labelsize=8)
for spine in ax.spines.values():
    spine.set_color('#444')

plt.tight_layout()
path3 = os.path.join(OUT_DIR, 'bilateral_overlay.png')
plt.savefig(path3, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {path3}")

# ============================================================
# Wound Catalog — Text table of findings
# ============================================================
print("\n" + "=" * 80)
print("  BILATERAL WOUND CATALOG — Shroud of Turin Face Depth Analysis")
print("=" * 80)
print(f"{'Level':<8} {'Row':>4}  {'Max |Asym|':>10}  {'Position':>10}  {'Direction':<12}  Interpretation")
print("-" * 80)

anatomical_notes = {
    'Brow':  'Consistent with frontal swelling from blunt trauma or crown of thorns pressure',
    'Eye':   'Periorbital asymmetry suggests localized swelling around eye socket',
    'Cheek': 'Malar region asymmetry consistent with unilateral blow to the face',
    'Mouth': 'Perioral asymmetry may indicate trauma to jaw or lip area',
    'Jaw':   'Mandibular asymmetry suggesting impact or post-mortem displacement',
}

for name, info in profiles.items():
    diff = info['diff']
    abs_max_idx = np.argmax(np.abs(diff))
    max_val = diff[abs_max_idx]
    direction = 'swelling' if max_val > 0 else 'depression'
    # Convert index back to anatomical description
    pos_desc = f"center+{abs_max_idx}px"
    interp = anatomical_notes.get(name, '')

    print(f"{name:<8} {info['row']:>4}  {abs(max_val):>10.2f}  {pos_desc:>10}  {direction:<12}  {interp}")

print("-" * 80)

# Summary stats
all_diffs = np.concatenate([p['diff'] for p in profiles.values()])
print(f"\nOverall asymmetry stats:")
print(f"  Mean absolute difference: {np.abs(all_diffs).mean():.2f}")
print(f"  Max absolute difference:  {np.abs(all_diffs).max():.2f}")
print(f"  Std of differences:       {all_diffs.std():.2f}")

# Count significant regions per level
print(f"\nSignificant asymmetry regions (|diff| > 1 std):")
for name, info in profiles.items():
    n_sig = info['significant'].sum()
    n_total = len(info['significant'])
    pct = 100.0 * n_sig / n_total if n_total > 0 else 0
    print(f"  {name:<8}: {n_sig:>3}/{n_total} positions ({pct:.0f}%)")

# --- Copy to docs/images ---
for fname in ['bilateral_slices.png', 'bilateral_catalog.png', 'bilateral_overlay.png']:
    src = os.path.join(OUT_DIR, fname)
    dst = os.path.join(DOCS_DIR, fname)
    shutil.copy2(src, dst)
    print(f"Copied: {dst}")

print("\nBilateral analysis complete.")
