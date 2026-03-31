import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
import os

output_dir = 'output/analysis'
docs_images_dir = 'docs/images'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(docs_images_dir, exist_ok=True)

print("=" * 60)
print("WAVELET DECOMPOSITION ANALYSIS")
print("=" * 60)

print("\n[1] Loading depth map...")
depth_map = np.load('data/processed/depth_map_smooth_15.npy')
print(f"    Original shape: {depth_map.shape}")

print("\n[2] Resizing to (150, 150) with INTER_AREA...")
depth_map_resized = cv2.resize(depth_map, (150, 150), interpolation=cv2.INTER_AREA)
print(f"    Resized shape: {depth_map_resized.shape}")

print("\n[3] Performing wavelet decomposition (db4, level 4)...")
coeffs = pywt.wavedec2(depth_map_resized, 'db4', level=4)
print(f"    Number of coefficient arrays: {len(coeffs)}")
print(f"    Level 0 (approx) shape: {coeffs[0].shape}")
for i in range(1, len(coeffs)):
    cH, cV, cD = coeffs[i]
    print(f"    Level {i} (details) shapes: cH={cH.shape}, cV={cV.shape}, cD={cD.shape}")

print("\n[4] Reconstructing frequency bands...")

def create_band_coeffs(coeffs, keep_levels):
    """Create coefficient structure for specific bands."""
    result = []
    for level in range(len(coeffs)):
        if level == 0:
            if 0 in keep_levels:
                result.append(coeffs[0].copy())
            else:
                result.append(np.zeros_like(coeffs[0]))
        else:
            if level in keep_levels:
                result.append(coeffs[level])
            else:
                cH, cV, cD = coeffs[level]
                result.append((np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))
    return result

low_coeffs = create_band_coeffs(coeffs, [0])
mid_coeffs = create_band_coeffs(coeffs, [0, 1])
high_coeffs = create_band_coeffs(coeffs, [3, 4])

low_band = pywt.waverec2(low_coeffs, 'db4')
mid_band = pywt.waverec2(mid_coeffs, 'db4')
high_band = pywt.waverec2(high_coeffs, 'db4')

if low_band.shape != depth_map_resized.shape:
    low_band = cv2.resize(low_band, (depth_map_resized.shape[1], depth_map_resized.shape[0]))
if mid_band.shape != depth_map_resized.shape:
    mid_band = cv2.resize(mid_band, (depth_map_resized.shape[1], depth_map_resized.shape[0]))
if high_band.shape != depth_map_resized.shape:
    high_band = cv2.resize(high_band, (depth_map_resized.shape[1], depth_map_resized.shape[0]))

print(f"    Low band shape: {low_band.shape}")
print(f"    Mid band shape: {mid_band.shape}")
print(f"    High band shape: {high_band.shape}")

print("\n[5] Computing energy per band...")

def compute_band_energy_from_decomp(coeffs, level_indices):
    """Compute energy from decomposition coefficients for specific levels."""
    energy = 0.0
    for level in level_indices:
        if level == 0:
            energy += np.sum(coeffs[0] ** 2)
        else:
            cH, cV, cD = coeffs[level]
            energy += np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)
    return energy

energy_approx = compute_band_energy_from_decomp(coeffs, [0])
energy_l4_details = compute_band_energy_from_decomp(coeffs, [1])
energy_l3_details = compute_band_energy_from_decomp(coeffs, [2])
energy_l2_details = compute_band_energy_from_decomp(coeffs, [3])
energy_l1_details = compute_band_energy_from_decomp(coeffs, [4])

total_decomp_energy = energy_approx + energy_l4_details + energy_l3_details + energy_l2_details + energy_l1_details

energy_low = compute_band_energy_from_decomp(coeffs, [0])
energy_mid = energy_l4_details + energy_l3_details
energy_high = energy_l2_details + energy_l1_details

energy_low_recon = np.sum(low_band ** 2)
energy_mid_recon = np.sum(mid_band ** 2)
energy_high_recon = np.sum(high_band ** 2)

original_energy = np.sum(depth_map_resized ** 2)

print(f"\n    Decomposition level energies:")
print(f"    Level 0 (approx):      {energy_approx:>12.4f}  ({100*energy_approx/total_decomp_energy:>6.2f}%)")
print(f"    Level 1 (finest):      {energy_l1_details:>12.4f}  ({100*energy_l1_details/total_decomp_energy:>6.2f}%)")
print(f"    Level 2:               {energy_l2_details:>12.4f}  ({100*energy_l2_details/total_decomp_energy:>6.2f}%)")
print(f"    Level 3:               {energy_l3_details:>12.4f}  ({100*energy_l3_details/total_decomp_energy:>6.2f}%)")
print(f"    Level 4 (coarsest):    {energy_l4_details:>12.4f}  ({100*energy_l4_details/total_decomp_energy:>6.2f}%)")
print(f"    Total:                 {total_decomp_energy:>12.4f}")

print(f"\n    Band energies (from decomposition):")
print(f"    Low  (Level 4 approx only):  {energy_low:>12.4f}  ({100*energy_low/total_decomp_energy:>6.2f}%)")
print(f"    Mid  (Levels 3-4):            {energy_mid:>12.4f}  ({100*energy_mid/total_decomp_energy:>6.2f}%)")
print(f"    High (Levels 1-2):             {energy_high:>12.4f}  ({100*energy_high/total_decomp_energy:>6.2f}%)")

print(f"\n    Band energies (from reconstruction):")
print(f"    Low:  {energy_low_recon:>12.4f}  ({100*energy_low_recon/original_energy:>6.2f}% of original)")
print(f"    Mid:  {energy_mid_recon:>12.4f}  ({100*energy_mid_recon/original_energy:>6.2f}% of original)")
print(f"    High: {energy_high_recon:>12.4f}  ({100*energy_high_recon/original_energy:>6.2f}% of original)")

print("\n[6] Computing SNR (dB)...")

def compute_snr(signal_band, noise_bands, original):
    """Compute SNR in dB."""
    signal_power = np.sum(signal_band ** 2)
    noise_power = np.sum((original - signal_band) ** 2)
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    return float('inf')

snr_low = compute_snr(low_band, None, depth_map_resized)
snr_mid = compute_snr(mid_band, None, depth_map_resized)
snr_high = compute_snr(high_band, None, depth_map_resized)

combined_band = low_band + mid_band + high_band
snr_combined = compute_snr(combined_band, None, depth_map_resized)

print(f"    SNR Low band:   {snr_low:>8.2f} dB")
print(f"    SNR Mid band:   {snr_mid:>8.2f} dB")
print(f"    SNR High band:  {snr_high:>8.2f} dB")
print(f"    SNR All bands:  {snr_combined:>8.2f} dB")

print("\n[7] Generating visualizations...")

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('#1a1a1a')
for ax in axes.flat:
    ax.set_facecolor('#2a2a2a')

im_orig = axes[0, 0].imshow(depth_map_resized, cmap='inferno')
axes[0, 0].set_title('Original Depth Map', color='white', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
plt.colorbar(im_orig, ax=axes[0, 0], fraction=0.046, pad=0.04)

im_low = axes[0, 1].imshow(low_band, cmap='inferno')
axes[0, 1].set_title('Low Band\n(Level 4 Approximation)', color='white', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im_low, ax=axes[0, 1], fraction=0.046, pad=0.04)

im_mid = axes[0, 2].imshow(mid_band, cmap='inferno')
axes[0, 2].set_title('Mid Band\n(Levels 3-4 Details)', color='white', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')
plt.colorbar(im_mid, ax=axes[0, 2], fraction=0.046, pad=0.04)

im_high = axes[1, 0].imshow(high_band, cmap='inferno')
axes[1, 0].set_title('High Band\n(Levels 1-2 Details)', color='white', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im_high, ax=axes[1, 0], fraction=0.046, pad=0.04)

im_combined = axes[1, 1].imshow(combined_band, cmap='inferno')
axes[1, 1].set_title('Combined Bands\n(Low + Mid + High)', color='white', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
plt.colorbar(im_combined, ax=axes[1, 1], fraction=0.046, pad=0.04)

axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, f'SNR Results\n\nLow:   {snr_low:.2f} dB\nMid:   {snr_mid:.2f} dB\nHigh:  {snr_high:.2f} dB',
                ha='center', va='center', color='#c4a35a', fontsize=14,
                transform=axes[1, 2].transAxes, fontweight='bold')
axes[1, 2].set_title('Signal-to-Noise Ratios', color='white', fontsize=12, fontweight='bold')

plt.tight_layout(pad=2.0)
fig.savefig(os.path.join(output_dir, 'wavelet_decomp_bands.png'), dpi=150, facecolor='#1a1a1a', edgecolor='none')
fig.savefig(os.path.join(docs_images_dir, 'wavelet_decomp_bands.png'), dpi=150, facecolor='#1a1a1a', edgecolor='none')
print(f"    Saved: wavelet_decomp_bands.png")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a1a')

energy_pcts = [100*energy_low/total_decomp_energy, 100*energy_mid/total_decomp_energy, 100*energy_high/total_decomp_energy]
bands = ['Low\n(Level 4)', 'Mid\n(Levels 3-4)', 'High\n(Levels 1-2)']
colors = ['#c4a35a', '#3498db', '#e74c3c']

ax = axes[0]
bars = ax.bar(bands, energy_pcts, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
ax.set_ylabel('Energy Percentage (%)', color='white', fontsize=12)
ax.set_title('Band Energy Distribution', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white', labelsize=11)
for spine in ax.spines.values():
    spine.set_color('white')
ax.set_facecolor('#2a2a2a')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, max(energy_pcts) * 1.15)
for bar, pct in zip(bars, energy_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{pct:.1f}%',
            ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

ax = axes[1]
snr_values = [snr_low, snr_mid, snr_high]
bars = ax.bar(bands, snr_values, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
ax.set_ylabel('SNR (dB)', color='white', fontsize=12)
ax.set_title('Signal-to-Noise Ratio by Band', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='white', labelsize=11)
for spine in ax.spines.values():
    spine.set_color('white')
ax.set_facecolor('#2a2a2a')
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=0, color='white', linewidth=0.5, linestyle='-')
for bar, snr in zip(bars, snr_values):
    offset = 0.5 if snr >= 0 else -1.5
    ax.text(bar.get_x() + bar.get_width()/2, snr + offset, f'{snr:.1f} dB',
            ha='center', va='bottom' if snr >= 0 else 'top', color='white', fontsize=11, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'wavelet_decomp_energy.png'), dpi=150, facecolor='#1a1a1a', edgecolor='none')
fig.savefig(os.path.join(docs_images_dir, 'wavelet_decomp_energy.png'), dpi=150, facecolor='#1a1a1a', edgecolor='none')
print(f"    Saved: wavelet_decomp_energy.png")
plt.close()

row_idx = 65
if row_idx >= depth_map_resized.shape[0]:
    row_idx = depth_map_resized.shape[0] // 2

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#1a1a1a')

x = np.arange(depth_map_resized.shape[1])
ax.plot(x, depth_map_resized[row_idx, :], label='Original', color='#c4a35a', linewidth=2.5, zorder=5)
ax.plot(x, low_band[row_idx, :], label='Low Band', color='#3498db', linewidth=1.8, alpha=0.85)
ax.plot(x, mid_band[row_idx, :], label='Mid Band', color='#2ecc71', linewidth=1.8, alpha=0.85)
ax.plot(x, high_band[row_idx, :], label='High Band', color='#e74c3c', linewidth=1.8, alpha=0.85)

ax.set_xlabel('Column Index', color='white', fontsize=12)
ax.set_ylabel('Depth Value', color='white', fontsize=12)
ax.set_title(f'Wavelet Decomposition Row Profiles (Row {row_idx})', color='white', fontsize=14, fontweight='bold')
legend = ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
legend.get_frame().set_facecolor('#2a2a2a')
for text in legend.get_texts():
    text.set_color('white')
ax.tick_params(colors='white', labelsize=10)
ax.grid(True, alpha=0.3)
for spine in ax.spines.values():
    spine.set_color('white')
ax.set_facecolor('#2a2a2a')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'wavelet_decomp_profiles.png'), dpi=150, facecolor='#1a1a1a', edgecolor='none')
fig.savefig(os.path.join(docs_images_dir, 'wavelet_decomp_profiles.png'), dpi=150, facecolor='#1a1a1a', edgecolor='none')
print(f"    Saved: wavelet_decomp_profiles.png")
plt.close()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nEnergy Distribution:")
print(f"  Low Band  (Level 4 approx):  {100*energy_low/total_decomp_energy:>6.2f}%")
print(f"  Mid Band  (Levels 3-4):       {100*energy_mid/total_decomp_energy:>6.2f}%")
print(f"  High Band (Levels 1-2):      {100*energy_high/total_decomp_energy:>6.2f}%")
print(f"\nSNR Results (dB):")
print(f"  Low Band:   {snr_low:>8.2f} dB")
print(f"  Mid Band:   {snr_mid:>8.2f} dB")
print(f"  High Band:  {snr_high:>8.2f} dB")
print(f"\nOutput files:")
print(f"  - {output_dir}/wavelet_decomp_bands.png")
print(f"  - {output_dir}/wavelet_decomp_energy.png")
print(f"  - {output_dir}/wavelet_decomp_profiles.png")
print(f"  - {docs_images_dir}/wavelet_decomp_bands.png")
print(f"  - {docs_images_dir}/wavelet_decomp_energy.png")
print(f"  - {docs_images_dir}/wavelet_decomp_profiles.png")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)