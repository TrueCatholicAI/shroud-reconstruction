"""Bilateral asymmetry analysis — produces structured JSON results."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEPTH_PATH = PROJECT / "data" / "final" / "depth_150x150_g15.npy"
OUT_DIR = PROJECT / "output" / "bilateral"
RESULTS_JSON = PROJECT / "output" / "task_results" / "bilateral_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

depth = np.load(DEPTH_PATH).astype(np.float64)
print(f"Loaded depth map: {depth.shape}")

# 5 horizontal slices at landmark y-coordinates
SLICES = [
    {"level": "brow",   "y": 32},
    {"level": "eyes",   "y": 51},
    {"level": "cheeks", "y": 60},
    {"level": "mouth",  "y": 74},
    {"level": "jaw",    "y": 79},
]
HALF_BAND = 3  # average ±3 rows
MIDLINE = 75   # x midpoint

BG = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = 'white'

slice_results = []
image_paths = []

# Figure: 5-row panel — left=depth with slice line, right=overlaid profiles
fig, axes = plt.subplots(5, 2, figsize=(12, 18), facecolor=BG,
                          gridspec_kw={'width_ratios': [1, 1.5]})

for i, s in enumerate(SLICES):
    y = s["y"]
    level = s["level"]
    y_lo = max(0, y - HALF_BAND)
    y_hi = min(depth.shape[0], y + HALF_BAND + 1)
    band = depth[y_lo:y_hi, :].mean(axis=0)

    left_profile = band[:MIDLINE]          # columns 0..74
    right_profile = band[MIDLINE:][::-1]   # columns 75..149, flipped

    # Ensure same length
    min_len = min(len(left_profile), len(right_profile))
    left_profile = left_profile[:min_len]
    right_profile = right_profile[:min_len]

    diff = left_profile - right_profile  # positive = left is higher/swollen

    max_asym = float(np.max(np.abs(diff)))
    max_loc = int(np.argmax(np.abs(diff)))
    mean_asym = float(np.mean(np.abs(diff)))
    direction = "left" if diff[max_loc] > 0 else "right"

    slice_results.append({
        "level": level,
        "y_coord": y,
        "max_asymmetry": round(max_asym, 4),
        "max_asymmetry_location_px": max_loc,
        "mean_asymmetry": round(mean_asym, 4),
        "direction_of_max": direction,
        "profile_left": [round(float(v), 2) for v in left_profile],
        "profile_right": [round(float(v), 2) for v in right_profile],
    })

    print(f"{level:8s} (y={y}): max_asym={max_asym:.4f} at px {max_loc} ({direction}), "
          f"mean_asym={mean_asym:.4f}")

    # Left panel: depth map with slice line
    ax_map = axes[i, 0]
    ax_map.set_facecolor(BG)
    ax_map.imshow(depth, cmap='inferno')
    ax_map.axhline(y=y, color=GOLD, linewidth=1.5, linestyle='--')
    ax_map.axhspan(y_lo, y_hi, alpha=0.15, color=GOLD)
    ax_map.set_title(f'{level.capitalize()} (y={y})', color=GOLD, fontsize=11, fontweight='bold')
    ax_map.tick_params(colors=WHITE)
    for spine in ax_map.spines.values():
        spine.set_color(WHITE)

    # Right panel: overlaid profiles with shaded divergence
    ax_prof = axes[i, 1]
    ax_prof.set_facecolor(BG)
    x = np.arange(min_len)
    ax_prof.plot(x, left_profile, color='#f44336', linewidth=1.5, label='Left')
    ax_prof.plot(x, right_profile, color='#2196F3', linewidth=1.5, label='Right (flipped)')
    ax_prof.fill_between(x, left_profile, right_profile,
                          where=left_profile > right_profile,
                          alpha=0.3, color='#f44336', label='Left > Right')
    ax_prof.fill_between(x, left_profile, right_profile,
                          where=left_profile <= right_profile,
                          alpha=0.3, color='#2196F3', label='Right > Left')
    ax_prof.set_xlabel('Distance from midline (px)', color=WHITE, fontsize=9)
    ax_prof.set_ylabel('Depth', color=WHITE, fontsize=9)
    ax_prof.set_title(f'{level.capitalize()} — L/R Profiles', color=GOLD, fontsize=11, fontweight='bold')
    ax_prof.legend(facecolor='#333', edgecolor=GOLD, labelcolor=WHITE, fontsize=8)
    ax_prof.tick_params(colors=WHITE)
    for spine in ax_prof.spines.values():
        spine.set_color(WHITE)

fig.suptitle('Bilateral Asymmetry: 5-Level Horizontal Slice Analysis',
             color=GOLD, fontsize=14, fontweight='bold')
plt.tight_layout()
path = OUT_DIR / 'bilateral_slices.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))
print(f"Saved {path.name}")

# Figure 2: Heatmap — x=distance from center, y=slices, color=asymmetry
max_profile_len = max(len(s["profile_left"]) for s in slice_results)
heatmap = np.zeros((5, max_profile_len))
for i, s in enumerate(slice_results):
    diff = np.array(s["profile_left"]) - np.array(s["profile_right"])
    heatmap[i, :len(diff)] = diff

fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
ax.set_facecolor(BG)
vmax = np.max(np.abs(heatmap))
im = ax.imshow(heatmap, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_yticks(range(5))
ax.set_yticklabels([s["level"].capitalize() for s in slice_results], color=WHITE)
ax.set_xlabel('Distance from midline (px)', color=WHITE)
ax.set_title('Asymmetry Heatmap (Red = Left Swelling, Blue = Right Swelling)',
             color=GOLD, fontsize=12, fontweight='bold')
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
plt.colorbar(im, ax=ax, label='L-R Difference')
plt.tight_layout()
path = OUT_DIR / 'bilateral_heatmap.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))
print(f"Saved {path.name}")

# Figure 3: Wound overlay on depth map
fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.imshow(depth, cmap='gray', alpha=0.6)
# Compute full asymmetry map: original vs horizontally flipped
flipped = np.fliplr(depth)
asym_map = depth - flipped
# Threshold at 1 std
std = np.std(asym_map)
mask_pos = asym_map > std   # left swelling
mask_neg = asym_map < -std  # right swelling
overlay = np.zeros((*depth.shape, 4))
overlay[mask_pos] = [1, 0, 0, 0.5]   # red = swelling
overlay[mask_neg] = [0, 0, 1, 0.5]   # blue = depression
ax.imshow(overlay)
# Mark slice lines
for s in SLICES:
    ax.axhline(y=s["y"], color=GOLD, linewidth=0.8, linestyle='--', alpha=0.7)
ax.set_title('Wound Overlay (Red=Swelling, Blue=Depression)', color=GOLD, fontsize=11, fontweight='bold')
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
plt.tight_layout()
path = OUT_DIR / 'bilateral_overlay.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))
print(f"Saved {path.name}")

# Save results
results = {
    "depth_map": str(DEPTH_PATH.relative_to(PROJECT)),
    "midline_x": MIDLINE,
    "half_band_rows": HALF_BAND,
    "slices": slice_results,
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps({k: v for k, v in results.items() if k != 'slices'}, indent=2))
for s in slice_results:
    print(f"  {s['level']}: max_asym={s['max_asymmetry']}, mean_asym={s['mean_asymmetry']}, dir={s['direction_of_max']}")
