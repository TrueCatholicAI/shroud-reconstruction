"""Wavelet decomposition analysis — produces structured JSON results."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEPTH_PATH = PROJECT / "data" / "final" / "depth_150x150_g15.npy"
OUT_DIR = PROJECT / "output" / "wavelet"
RESULTS_JSON = PROJECT / "output" / "task_results" / "wavelet_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

# Load depth map
depth = np.load(DEPTH_PATH).astype(np.float64)
print(f"Loaded depth map: {depth.shape}, range [{depth.min():.1f}, {depth.max():.1f}]")

# Wavelet decomposition: Symlet-4, 4 levels
wavelet = 'sym4'
max_level = 4
coeffs = pywt.wavedec2(depth, wavelet, level=max_level)

# Compute energy at each level (normalize to total wavelet coefficient energy)
level_energies = {}

# Level 4 approximation
approx_energy = np.sum(coeffs[0] ** 2)
level_energies['L4_approx'] = approx_energy

# Detail coefficients for each level (coeffs[1] = level 4 detail, ..., coeffs[4] = level 1 detail)
for i in range(1, max_level + 1):
    level_num = max_level - i + 1
    cH, cV, cD = coeffs[i]
    detail_e = np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)
    level_energies[f'L{level_num}_detail'] = detail_e

total_coeff_energy = sum(level_energies.values())
energy_percentages = {k: round(v / total_coeff_energy * 100, 4) for k, v in level_energies.items()}

# SNR: signal = L3+L4 approximation, noise = L1 detail
signal_energy = approx_energy + level_energies['L4_detail'] + level_energies['L3_detail']
noise_energy = level_energies['L1_detail']
snr_db = round(10 * np.log10(signal_energy / noise_energy), 2) if noise_energy > 0 else float('inf')

print(f"Energy percentages: {json.dumps(energy_percentages, indent=2)}")
print(f"SNR: {snr_db} dB")

# Reconstruct frequency bands for visualization
def reconstruct_band(coeffs, keep_levels, max_level):
    """Reconstruct keeping only specified levels."""
    new_coeffs = [np.zeros_like(coeffs[0])]  # zero approx
    for i in range(1, max_level + 1):
        level_num = max_level - i + 1
        if level_num in keep_levels:
            new_coeffs.append(coeffs[i])
        else:
            new_coeffs.append(tuple(np.zeros_like(c) for c in coeffs[i]))
    return pywt.waverec2(new_coeffs, wavelet)

def reconstruct_approx_only(coeffs, max_level):
    """Reconstruct keeping only the approximation."""
    new_coeffs = [coeffs[0]]
    for i in range(1, max_level + 1):
        new_coeffs.append(tuple(np.zeros_like(c) for c in coeffs[i]))
    return pywt.waverec2(new_coeffs, wavelet)

low_band = reconstruct_approx_only(coeffs, max_level)[:150, :150]
mid_band = reconstruct_band(coeffs, {3, 4}, max_level)[:150, :150]
high_band = reconstruct_band(coeffs, {1, 2}, max_level)[:150, :150]

# Style constants
BG = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = 'white'

image_paths = []

# Figure 1: Band decomposition (2x3)
fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor=BG)
for ax in axes.flat:
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_color(WHITE)

panels = [
    (axes[0, 0], depth, 'Original Depth', 'inferno'),
    (axes[0, 1], low_band, f'L4 Approx ({energy_percentages["L4_approx"]:.1f}%)', 'inferno'),
    (axes[0, 2], mid_band, f'Mid Detail L3-4 ({energy_percentages["L4_detail"] + energy_percentages["L3_detail"]:.2f}%)', 'RdBu_r'),
    (axes[1, 0], high_band, f'High Detail L1-2 ({energy_percentages["L2_detail"] + energy_percentages["L1_detail"]:.2f}%)', 'RdBu_r'),
]
# Coefficient visualizations for remaining panels
cH4, cV4, cD4 = coeffs[1]
panels.append((axes[1, 1], cH4, 'L4 Horizontal Detail', 'RdBu_r'))
panels.append((axes[1, 2], cV4, 'L4 Vertical Detail', 'RdBu_r'))

for ax, data, title, cmap in panels:
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title, color=GOLD, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('Wavelet Frequency Band Decomposition (Sym4, 4 Levels)', color=GOLD, fontsize=14, fontweight='bold')
plt.tight_layout()
path = OUT_DIR / 'wavelet_decomp_bands.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))
print(f"Saved {path.name}")

# Figure 2: Energy bar chart
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
labels = list(energy_percentages.keys())
values = list(energy_percentages.values())
bars = ax.bar(labels, values, color=GOLD, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Energy %', color=WHITE, fontsize=12)
ax.set_title('Energy Distribution by Decomposition Level', color=GOLD, fontsize=13, fontweight='bold')
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
# Add value labels
for bar, val in zip(bars, values):
    if val > 1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', color=WHITE, fontsize=10)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.3f}%', ha='center', va='bottom', color=WHITE, fontsize=9)
ax.set_yscale('log')
plt.tight_layout()
path = OUT_DIR / 'wavelet_energy.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))
print(f"Saved {path.name}")

# Figure 3: Profile comparison at row 75 (mid-face)
row = 75
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(depth[row, :], color=WHITE, linewidth=1.5, label='Original', alpha=0.8)
ax.plot(low_band[row, :], color=GOLD, linewidth=2, label='Low-freq (L4 approx)')
ax.plot(mid_band[row, :], color='#4CAF50', linewidth=1.2, label='Mid-freq (L3-4 detail)')
ax.plot(high_band[row, :], color='#f44336', linewidth=1, label='High-freq (L1-2 detail)', alpha=0.7)
ax.set_xlabel('Column (px)', color=WHITE)
ax.set_ylabel('Depth Value', color=WHITE)
ax.set_title(f'Frequency Band Profiles at Row {row}', color=GOLD, fontsize=13, fontweight='bold')
ax.legend(facecolor='#333', edgecolor=GOLD, labelcolor=WHITE)
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
plt.tight_layout()
path = OUT_DIR / 'wavelet_profiles.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))
print(f"Saved {path.name}")

# Save results JSON
results = {
    "wavelet": wavelet,
    "decomposition_levels": max_level,
    "depth_map_shape": list(depth.shape),
    "total_coefficient_energy": round(total_coeff_energy, 2),
    "energy_percentages": energy_percentages,
    "snr_db": snr_db,
    "signal_definition": "L3+L4 approximation + detail",
    "noise_definition": "L1 detail coefficients",
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps(results, indent=2))
