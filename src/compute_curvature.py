"""Curvature analysis of facial depth map — produces structured JSON results."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEPTH_PATH = PROJECT / "data" / "final" / "depth_150x150_g15.npy"
OUT_DIR = PROJECT / "output" / "curvature"
RESULTS_JSON = PROJECT / "output" / "task_results" / "curvature_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

depth = np.load(DEPTH_PATH).astype(np.float64)
print(f"Loaded depth map: {depth.shape}")

# Compute second derivatives (curvature proxy)
dy, dx = np.gradient(depth)
dyy, _ = np.gradient(dy)
_, dxx = np.gradient(dx)

# Mean curvature approximation: H = (d2z/dx2 + d2z/dy2) / 2
mean_curvature = (dxx + dyy) / 2.0

# Gaussian curvature: K = d2z/dx2 * d2z/dy2 - (d2z/dxdy)^2
dxy, _ = np.gradient(dx)  # d2z/dxdy via gradient of dz/dx in y direction
# Actually let's compute it properly
_, dxy = np.gradient(dy)  # d2z/dydx
gaussian_curvature = dxx * dyy - dxy ** 2

# Absolute curvature magnitude for finding sharp features
abs_curvature = np.abs(mean_curvature)

# Find top 10 highest-curvature points (with spatial separation)
def find_top_peaks(arr, n=10, min_dist=8):
    """Find top N peaks with minimum spatial separation."""
    flat = arr.flatten()
    sorted_idx = np.argsort(flat)[::-1]
    peaks = []
    for idx in sorted_idx:
        y, x = divmod(int(idx), arr.shape[1])
        # Check distance from existing peaks
        too_close = False
        for py, px, _ in peaks:
            if abs(y - py) < min_dist and abs(x - px) < min_dist:
                too_close = True
                break
        if not too_close:
            peaks.append((y, x, float(arr[y, x])))
        if len(peaks) >= n:
            break
    return peaks

top_10 = find_top_peaks(abs_curvature, n=10, min_dist=8)

# Label anatomical regions based on position
def label_region(y, x):
    if y < 35:
        return "forehead" if 40 < x < 110 else "temple"
    elif y < 55:
        if x < 60:
            return "left eye region"
        elif x > 90:
            return "right eye region"
        else:
            return "nose bridge"
    elif y < 75:
        if x < 50:
            return "left cheek"
        elif x > 100:
            return "right cheek"
        elif 60 < x < 90:
            return "nose"
        else:
            return "mid-face"
    elif y < 95:
        if 55 < x < 95:
            return "mouth/lip"
        else:
            return "lower cheek"
    else:
        return "chin/jaw"

top_10_labeled = []
for y, x, val in top_10:
    region = label_region(y, x)
    top_10_labeled.append({
        "rank": len(top_10_labeled) + 1,
        "y": y,
        "x": x,
        "curvature": round(val, 4),
        "region": region,
    })
    print(f"  #{len(top_10_labeled)}: ({x},{y}) curv={val:.4f} — {region}")

BG = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = 'white'
image_paths = []

# Figure 1: Curvature heatmap
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_color(WHITE)

im0 = axes[0].imshow(depth, cmap='inferno')
axes[0].set_title('Original Depth Map', color=GOLD, fontweight='bold')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

vmax = np.percentile(np.abs(mean_curvature), 99)
im1 = axes[1].imshow(mean_curvature, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1].set_title('Mean Curvature', color=GOLD, fontweight='bold')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(abs_curvature, cmap='hot', vmin=0, vmax=vmax)
axes[2].set_title('Absolute Curvature', color=GOLD, fontweight='bold')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

fig.suptitle('Facial Curvature Analysis', color=GOLD, fontsize=14, fontweight='bold')
plt.tight_layout()
path = OUT_DIR / 'curvature_heatmap.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Figure 2: Labeled high-curvature landmarks on depth map
fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
ax.set_facecolor(BG)
ax.imshow(depth, cmap='inferno', alpha=0.8)
# Overlay curvature
ax.imshow(abs_curvature, cmap='hot', alpha=0.3, vmin=0, vmax=vmax)

for p in top_10_labeled:
    ax.plot(p['x'], p['y'], 'o', color=GOLD, markersize=10, markeredgecolor=WHITE, markeredgewidth=1.5)
    ax.annotate(f"#{p['rank']} {p['region']}\n({p['curvature']:.2f})",
                (p['x'], p['y']), xytext=(8, -8), textcoords='offset points',
                color=WHITE, fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#333', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color=GOLD, lw=0.8))

ax.set_title('Top 10 High-Curvature Landmarks', color=GOLD, fontsize=13, fontweight='bold')
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
plt.tight_layout()
path = OUT_DIR / 'curvature_landmarks.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Stats
curvature_stats = {
    "mean_curvature_mean": round(float(np.mean(mean_curvature)), 4),
    "mean_curvature_std": round(float(np.std(mean_curvature)), 4),
    "abs_curvature_mean": round(float(np.mean(abs_curvature)), 4),
    "abs_curvature_max": round(float(np.max(abs_curvature)), 4),
    "abs_curvature_99pct": round(float(np.percentile(abs_curvature, 99)), 4),
}

results = {
    "depth_map": str(DEPTH_PATH.relative_to(PROJECT)),
    "method": "Second derivative of depth (numpy.gradient applied twice)",
    "curvature_stats": curvature_stats,
    "top_10_landmarks": top_10_labeled,
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps({k: v for k, v in results.items() if k != 'top_10_landmarks'}, indent=2))
