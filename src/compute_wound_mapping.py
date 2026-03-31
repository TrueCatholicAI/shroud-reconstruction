"""Wound mapping analysis — produces structured JSON results."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEPTH_ORIG = PROJECT / "data" / "final" / "depth_150x150_g15.npy"
DEPTH_HEALED = PROJECT / "data" / "final" / "depth_healed_150.npy"
OUT_DIR = PROJECT / "output" / "wound_mapping"
RESULTS_JSON = PROJECT / "output" / "task_results" / "wound_mapping_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

original = np.load(DEPTH_ORIG).astype(np.float64)
healed = np.load(DEPTH_HEALED).astype(np.float64)
print(f"Original: {original.shape}, Healed: {healed.shape}")

# Difference map: original minus healed
# Positive = original is deeper/higher than healed (swelling)
# Negative = original is lower than healed (depression)
diff = original - healed

# Statistics
abs_diff = np.abs(diff)
asymmetry_index = float(np.mean(abs_diff))
threshold = float(np.std(diff))
pct_exceeding = float(np.sum(abs_diff > threshold) / diff.size * 100)
max_positive = float(np.max(diff))
max_negative = float(np.min(diff))
mean_diff = float(np.mean(diff))

print(f"Asymmetry index (mean |diff|): {asymmetry_index:.4f}")
print(f"1-sigma threshold: {threshold:.4f}")
print(f"Pixels exceeding threshold: {pct_exceeding:.1f}%")
print(f"Max positive deviation (swelling): {max_positive:.4f}")
print(f"Max negative deviation (depression): {max_negative:.4f}")

BG = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = 'white'
image_paths = []

# Figure 1: 2x2 overview
fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor=BG)
for ax in axes.flat:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_color(WHITE)

im0 = axes[0, 0].imshow(original, cmap='inferno')
axes[0, 0].set_title('Original Depth', color=GOLD, fontweight='bold')
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(healed, cmap='inferno')
axes[0, 1].set_title('Healed Depth', color=GOLD, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

vmax = max(abs(max_positive), abs(max_negative))
im2 = axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1, 0].set_title('Difference (Original - Healed)', color=GOLD, fontweight='bold')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

thresh_map = np.where(abs_diff > threshold, diff, 0)
im3 = axes[1, 1].imshow(thresh_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1, 1].set_title(f'Thresholded (>{threshold:.1f}, {pct_exceeding:.1f}% pixels)',
                      color=GOLD, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

fig.suptitle('Wound Mapping: Original vs Healed Depth', color=GOLD, fontsize=14, fontweight='bold')
plt.tight_layout()
path = OUT_DIR / 'wound_overview.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Figure 2: 3D surface with wound coloring
fig = plt.figure(figsize=(10, 8), facecolor=BG)
ax = fig.add_subplot(111, projection='3d', facecolor=BG)

x = np.arange(original.shape[1])
y = np.arange(original.shape[0])
X, Y = np.meshgrid(x, y)

# Normalize diff to 0-1 for colormap
from matplotlib.colors import Normalize
norm = Normalize(vmin=-vmax, vmax=vmax)
cmap = plt.cm.RdBu_r
colors = cmap(norm(diff))

ax.plot_surface(X, Y, original, facecolors=colors, rstride=2, cstride=2,
                linewidth=0, antialiased=True, shade=False)
ax.set_xlabel('X', color=WHITE)
ax.set_ylabel('Y', color=WHITE)
ax.set_zlabel('Depth', color=WHITE)
ax.set_title('3D Surface with Wound Coloring', color=GOLD, fontsize=13, fontweight='bold')
ax.tick_params(colors=WHITE)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(elev=25, azim=-60)
plt.tight_layout()
path = OUT_DIR / 'wound_3d_surface.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Figure 3: Annotated diverging map with regions
fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, fraction=0.046, label='Depth Difference')

# Annotate key regions
regions = {
    'Left cheek': (35, 55, 25, 50),   # rows 35-55, cols 25-50
    'Right cheek': (35, 55, 100, 125),
    'Brow ridge': (25, 40, 30, 120),
    'Nose': (45, 75, 60, 90),
    'Mouth': (75, 95, 45, 105),
}
for name, (r0, r1, c0, c1) in regions.items():
    region_mean = float(np.mean(diff[r0:r1, c0:c1]))
    rect = plt.Rectangle((c0, r0), c1-c0, r1-r0, linewidth=1.5,
                          edgecolor=GOLD, facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(c1+2, (r0+r1)/2, f'{name}\n{region_mean:+.2f}',
            color=WHITE, fontsize=8, va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333', alpha=0.8))

ax.set_title('Wound Map with Annotated Regions', color=GOLD, fontsize=13, fontweight='bold')
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
plt.tight_layout()
path = OUT_DIR / 'wound_annotated.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Compute region-level stats for JSON
region_stats = {}
for name, (r0, r1, c0, c1) in regions.items():
    region_data = diff[r0:r1, c0:c1]
    region_stats[name] = {
        "mean_difference": round(float(np.mean(region_data)), 4),
        "max_difference": round(float(np.max(region_data)), 4),
        "min_difference": round(float(np.min(region_data)), 4),
        "std": round(float(np.std(region_data)), 4),
        "bbox_rows": [r0, r1],
        "bbox_cols": [c0, c1],
    }

results = {
    "depth_original": str(DEPTH_ORIG.relative_to(PROJECT)),
    "depth_healed": str(DEPTH_HEALED.relative_to(PROJECT)),
    "asymmetry_index": round(asymmetry_index, 4),
    "threshold_value": round(threshold, 4),
    "pct_exceeding_threshold": round(pct_exceeding, 2),
    "max_positive_deviation": round(max_positive, 4),
    "max_negative_deviation": round(max_negative, 4),
    "mean_difference": round(mean_diff, 4),
    "regions": region_stats,
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps(results, indent=2))
