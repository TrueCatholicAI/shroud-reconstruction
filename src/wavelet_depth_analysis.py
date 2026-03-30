"""
Wavelet Depth Analysis
======================
Decomposes the Shroud of Turin depth map into frequency bands using
2D discrete wavelet transform to separate facial signal from cloth texture.

Uses db4 wavelet with 4 levels of decomposition:
  - Level 4 approximation: gross face shape (lowest frequency)
  - Levels 3-4 details: anatomical features (nose, brow ridges, eye sockets)
  - Levels 1-2 details: fine detail / cloth texture (high frequency)
"""

import os
import shutil
import numpy as np
import cv2
import pywt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT, "data", "processed", "depth_map_smooth_15.npy")
OUT_DIR = os.path.join(PROJECT, "output", "analysis")
DOCS_DIR = os.path.join(PROJECT, "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────
BG_COLOR = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = '#ffffff'
GREY = '#888888'
TARGET_SIZE = (150, 150)
WAVELET = 'db4'
LEVELS = 4


def save_and_copy(fig, name):
    """Save figure to output/analysis and copy to docs/images."""
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    shutil.copy2(out, os.path.join(DOCS_DIR, name))
    print(f"  Saved: {out}")


# ── 1. Load and resize ────────────────────────────────────────────────────
print("=" * 65)
print("WAVELET DEPTH ANALYSIS  —  Shroud of Turin Depth Map")
print("=" * 65)

depth_raw = np.load(INPUT_PATH)
print(f"\nLoaded depth map: {depth_raw.shape}, dtype={depth_raw.dtype}")
depth = cv2.resize(depth_raw.astype(np.float64), TARGET_SIZE, interpolation=cv2.INTER_AREA)
print(f"Resized to {depth.shape}")

# ── 2. Wavelet decomposition ──────────────────────────────────────────────
coeffs = pywt.wavedec2(depth, WAVELET, level=LEVELS)
# coeffs = [cA4, (cH4,cV4,cD4), (cH3,cV3,cD3), (cH2,cV2,cD2), (cH1,cV1,cD1)]
print(f"\nWavelet: {WAVELET}, levels: {LEVELS}")
print(f"Coefficient shapes:")
print(f"  Approx (L4): {coeffs[0].shape}")
for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
    lvl = LEVELS - i + 1
    print(f"  Detail L{lvl}:  H={cH.shape}  V={cV.shape}  D={cD.shape}")


def zero_coeffs(coeffs):
    """Return a deep copy with all detail coefficients zeroed."""
    out = [np.zeros_like(coeffs[0])]
    for (cH, cV, cD) in coeffs[1:]:
        out.append((np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))
    return out


def reconstruct_band(coeffs, keep_approx=False, keep_levels=None):
    """Reconstruct image keeping only specified components.

    keep_approx : bool – keep the level-4 approximation coefficients
    keep_levels : set/list of ints – which detail levels to keep (1-4)
    """
    if keep_levels is None:
        keep_levels = set()
    else:
        keep_levels = set(keep_levels)

    new = zero_coeffs(coeffs)
    if keep_approx:
        new[0] = coeffs[0].copy()

    for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
        lvl = LEVELS - i + 1  # detail index → wavelet level
        if lvl in keep_levels:
            new[i] = (cH.copy(), cV.copy(), cD.copy())

    rec = pywt.waverec2(new, WAVELET)
    # waverec2 may return slightly larger array – crop to target
    return rec[:TARGET_SIZE[0], :TARGET_SIZE[1]]


# ── 3. Reconstruct frequency bands ────────────────────────────────────────
print("\nReconstructing frequency bands...")

low_freq = reconstruct_band(coeffs, keep_approx=True, keep_levels=[])
mid_freq = reconstruct_band(coeffs, keep_approx=False, keep_levels=[3, 4])
high_freq = reconstruct_band(coeffs, keep_approx=False, keep_levels=[1, 2])
mid_low = reconstruct_band(coeffs, keep_approx=True, keep_levels=[3, 4])
residual = depth - mid_low

print(f"  Low-freq  range: [{low_freq.min():.3f}, {low_freq.max():.3f}]")
print(f"  Mid-freq  range: [{mid_freq.min():.3f}, {mid_freq.max():.3f}]")
print(f"  High-freq range: [{high_freq.min():.3f}, {high_freq.max():.3f}]")
print(f"  Mid+Low   range: [{mid_low.min():.3f}, {mid_low.max():.3f}]")
print(f"  Residual  range: [{residual.min():.3f}, {residual.max():.3f}]")

# ── 4a. Visualization: wavelet_bands.png ───────────────────────────────────
print("\nGenerating wavelet_bands.png ...")

fig = plt.figure(figsize=(15, 10), facecolor=BG_COLOR)
gs = gridspec.GridSpec(2, 3, wspace=0.25, hspace=0.35)

panels = [
    (depth,     "Original Depth Map",    'inferno', None),
    (low_freq,  "Low Freq (Gross Shape)", 'inferno', None),
    (mid_freq,  "Mid Freq (Anatomical)",  'inferno', None),
    (high_freq, "High Freq (Texture/Noise)", 'inferno', None),
    (mid_low,   "Mid + Low Combined",    'inferno', None),
    (residual,  "Residual (Orig − Mid+Low)", 'RdBu_r', 'residual'),
]

for idx, (data, title, cmap, mode) in enumerate(panels):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    if mode == 'residual':
        vmax = max(abs(data.min()), abs(data.max()))
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(data, cmap=cmap)
    ax.set_title(title, color=GOLD, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=GREY, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GREY)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=GREY, labelsize=7)

fig.suptitle("Wavelet Frequency Band Decomposition  —  Shroud Depth Map",
             color=WHITE, fontsize=14, fontweight='bold', y=0.98)
save_and_copy(fig, "wavelet_bands.png")
plt.close(fig)

# ── 4b. Visualization: wavelet_3d_bands.png ────────────────────────────────
print("Generating wavelet_3d_bands.png ...")

# Downsample for faster 3D rendering
step = 3
Y, X = np.mgrid[0:TARGET_SIZE[0]:step, 0:TARGET_SIZE[1]:step]

fig = plt.figure(figsize=(18, 6), facecolor=BG_COLOR)
bands_3d = [
    (low_freq,  "Low Frequency (Gross Shape)"),
    (mid_freq,  "Mid Frequency (Anatomical)"),
    (high_freq, "High Frequency (Texture)"),
]

for i, (data, title) in enumerate(bands_3d):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.set_facecolor(BG_COLOR)
    Z = data[::step, ::step]
    surf = ax.plot_surface(X, Y, Z, cmap='inferno', linewidth=0,
                           antialiased=True, alpha=0.9)
    ax.set_title(title, color=GOLD, fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(colors=GREY, labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GREY)
    ax.yaxis.pane.set_edgecolor(GREY)
    ax.zaxis.pane.set_edgecolor(GREY)
    ax.view_init(elev=50, azim=-60)
    ax.invert_yaxis()

fig.suptitle("3D Surface Reconstruction per Frequency Band",
             color=WHITE, fontsize=14, fontweight='bold', y=0.98)
fig.subplots_adjust(wspace=0.05)
save_and_copy(fig, "wavelet_3d_bands.png")
plt.close(fig)

# ── 4c. Visualization: wavelet_energy.png ──────────────────────────────────
print("Generating wavelet_energy.png ...")

# Compute energy per band
energy_approx = np.sum(coeffs[0] ** 2)
detail_energies = []
for i, (cH, cV, cD) in enumerate(coeffs[1:], 1):
    lvl = LEVELS - i + 1
    e = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
    detail_energies.append((lvl, e))

total_energy = energy_approx + sum(e for _, e in detail_energies)

band_labels = ["Approx\n(L4)"]
band_energies = [energy_approx]
band_colors = [GOLD]

color_map = {4: '#e06040', 3: '#d48040', 2: '#60a0c0', 1: '#5080b0'}
for lvl, e in sorted(detail_energies, key=lambda x: -x[0]):
    band_labels.append(f"Detail\nL{lvl}")
    band_energies.append(e)
    band_colors.append(color_map.get(lvl, GREY))

band_pct = [100.0 * e / total_energy for e in band_energies]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
bars = ax.bar(band_labels, band_pct, color=band_colors, edgecolor='none', width=0.6)
for bar, pct in zip(bars, band_pct):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{pct:.2f}%", ha='center', va='bottom', color=WHITE, fontsize=10,
            fontweight='bold')

ax.set_ylabel("Energy (%)", color=WHITE, fontsize=12)
ax.set_title("Energy Distribution Across Wavelet Bands", color=GOLD,
             fontsize=14, fontweight='bold')
ax.tick_params(colors=GREY, labelsize=10)
for spine in ax.spines.values():
    spine.set_color(GREY)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(band_pct) * 1.15)

save_and_copy(fig, "wavelet_energy.png")
plt.close(fig)

# ── 4d. Visualization: wavelet_profiles.png ────────────────────────────────
print("Generating wavelet_profiles.png ...")

nose_row = 65
x_axis = np.arange(TARGET_SIZE[1])

fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=BG_COLOR,
                         gridspec_kw={'height_ratios': [2, 1]})

# Top: overlay all bands
ax = axes[0]
ax.set_facecolor(BG_COLOR)
ax.plot(x_axis, depth[nose_row, :], color=WHITE, linewidth=2, label='Original', alpha=0.9)
ax.plot(x_axis, low_freq[nose_row, :], color=GOLD, linewidth=2, label='Low Freq', alpha=0.85)
ax.plot(x_axis, mid_freq[nose_row, :], color='#e06040', linewidth=1.5, label='Mid Freq', alpha=0.85)
ax.plot(x_axis, high_freq[nose_row, :], color='#60a0c0', linewidth=1, label='High Freq', alpha=0.7)
ax.plot(x_axis, mid_low[nose_row, :], color='#80d080', linewidth=1.5, linestyle='--',
        label='Mid+Low', alpha=0.85)
ax.set_title(f"Horizontal Profile at Row {nose_row} (Nose Region)",
             color=GOLD, fontsize=13, fontweight='bold')
ax.set_ylabel("Depth Value", color=WHITE, fontsize=11)
ax.legend(facecolor='#2a2a2a', edgecolor=GREY, labelcolor=WHITE, fontsize=9)
ax.tick_params(colors=GREY)
for spine in ax.spines.values():
    spine.set_color(GREY)

# Bottom: residual along profile
ax2 = axes[1]
ax2.set_facecolor(BG_COLOR)
ax2.fill_between(x_axis, residual[nose_row, :], 0, color='#e06040', alpha=0.4)
ax2.plot(x_axis, residual[nose_row, :], color='#e06040', linewidth=1)
ax2.axhline(0, color=GREY, linewidth=0.5, linestyle='--')
ax2.set_title("Residual (Original − Mid+Low)", color=GOLD, fontsize=11, fontweight='bold')
ax2.set_xlabel("Pixel Position", color=WHITE, fontsize=11)
ax2.set_ylabel("Residual", color=WHITE, fontsize=11)
ax2.tick_params(colors=GREY)
for spine in ax2.spines.values():
    spine.set_color(GREY)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Wavelet Band Contributions  —  Nose Cross-Section",
             color=WHITE, fontsize=14, fontweight='bold')
save_and_copy(fig, "wavelet_profiles.png")
plt.close(fig)

# ── 5. Quantitative analysis ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("QUANTITATIVE ANALYSIS")
print("=" * 65)

print("\n--- Energy Distribution ---")
print(f"  {'Band':<20s} {'Energy':>14s}  {'Percentage':>10s}")
print(f"  {'-'*20} {'-'*14}  {'-'*10}")
print(f"  {'Approx (L4)':<20s} {energy_approx:>14.2f}  {100*energy_approx/total_energy:>9.2f}%")
for lvl, e in sorted(detail_energies, key=lambda x: -x[0]):
    print(f"  {'Detail L' + str(lvl):<20s} {e:>14.2f}  {100*e/total_energy:>9.2f}%")
print(f"  {'TOTAL':<20s} {total_energy:>14.2f}  {'100.00%':>10s}")

# Grouped energies
low_energy = energy_approx
mid_energy = sum(e for lvl, e in detail_energies if lvl in (3, 4))
high_energy = sum(e for lvl, e in detail_energies if lvl in (1, 2))

print(f"\n--- Grouped Band Energies ---")
print(f"  Low  (approx):     {100*low_energy/total_energy:>8.2f}%")
print(f"  Mid  (L3+L4 det):  {100*mid_energy/total_energy:>8.2f}%")
print(f"  High (L1+L2 det):  {100*high_energy/total_energy:>8.2f}%")

signal_energy = low_energy + mid_energy
snr = 10 * np.log10(signal_energy / high_energy) if high_energy > 0 else float('inf')
print(f"\n--- Signal-to-Noise Ratio ---")
print(f"  Signal (mid+low):  {signal_energy:.2f}")
print(f"  Noise  (high):     {high_energy:.2f}")
print(f"  SNR:               {snr:.2f} dB")

# Spatial frequency cutoffs
# At each level, the wavelet captures frequencies around fs / 2^level
# where fs = pixels across face width = 150 pixels
face_width_px = TARGET_SIZE[1]
print(f"\n--- Spatial Frequency Cutoffs ---")
print(f"  Face width: {face_width_px} pixels")
print(f"  {'Level':<10s} {'Freq Range (cycles/face-width)':>35s}")
print(f"  {'-'*10} {'-'*35}")
for lvl in range(1, LEVELS + 1):
    f_low = face_width_px / (2 ** (lvl + 1))
    f_high = face_width_px / (2 ** lvl)
    print(f"  L{lvl:<8d} {f_low:>12.1f}  –  {f_high:>8.1f} cycles/face-width")
f_approx = face_width_px / (2 ** (LEVELS + 1))
print(f"  {'Approx':<10s} {'0':>12s}  –  {f_approx:>8.1f} cycles/face-width")

print(f"\n{'=' * 65}")
print("Wavelet depth analysis complete.")
print(f"Output saved to: {OUT_DIR}")
print(f"Copies in:       {DOCS_DIR}")
print(f"{'=' * 65}")
