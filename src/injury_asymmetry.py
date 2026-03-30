"""Phase 1.2: Injury/asymmetry difference maps for Enrie and Miller studies."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom
from scipy.stats import pearsonr

# --- Enrie Study 1 ---
full_depth = np.load('data/processed/depth_map_smooth_15.npy')
h, w = full_depth.shape
enrie_150 = zoom(full_depth.astype(np.float64), (150/h, 150/w), order=1)
healed_e = np.load('data/final/depth_healed_150.npy').astype(np.float64)
diff_e = enrie_150 - healed_e

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(enrie_150, cmap='inferno')
axes[0].set_title('Original Depth', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(healed_e, cmap='inferno')
axes[1].set_title('Healed (Symmetrized)', color='white', fontsize=13)
axes[1].axis('off')

vmax_e = max(abs(diff_e.min()), abs(diff_e.max()))
norm_e = TwoSlopeNorm(vmin=-vmax_e, vcenter=0, vmax=vmax_e)
im = axes[2].imshow(diff_e, cmap='RdBu_r', norm=norm_e)
axes[2].set_title('Asymmetry / Trauma Signal', color='white', fontsize=13)
axes[2].axis('off')

cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label('Red = raised vs expected | Blue = depressed', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

fig.suptitle('Enrie Study 1 - Injury/Asymmetry Difference Map', color='#c4a35a', fontsize=16, y=0.98)
for ax in axes:
    ax.set_facecolor('#1a1a1a')
plt.tight_layout()
plt.savefig('output/analysis/injury_asymmetry_map_enrie.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()

print("Enrie asymmetry map:")
print(f"  Diff range: [{diff_e.min():.1f}, {diff_e.max():.1f}]")
print(f"  Mean absolute: {np.abs(diff_e).mean():.2f}, Std: {diff_e.std():.2f}")
print(f"  Saved: output/analysis/injury_asymmetry_map_enrie.png")

# --- Miller Study 2 ---
orig_m = np.load('output/study2_miller/depth_150x150_g15.npy').astype(np.float64)
healed_m = np.load('output/study2_miller/depth_healed_150.npy').astype(np.float64)
diff_m = orig_m - healed_m

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a1a')

axes[0].imshow(orig_m, cmap='inferno')
axes[0].set_title('Original Depth', color='white', fontsize=13)
axes[0].axis('off')

axes[1].imshow(healed_m, cmap='inferno')
axes[1].set_title('Healed (Symmetrized)', color='white', fontsize=13)
axes[1].axis('off')

vmax_m = max(abs(diff_m.min()), abs(diff_m.max()))
norm_m = TwoSlopeNorm(vmin=-vmax_m, vcenter=0, vmax=vmax_m)
im = axes[2].imshow(diff_m, cmap='RdBu_r', norm=norm_m)
axes[2].set_title('Asymmetry / Trauma Signal', color='white', fontsize=13)
axes[2].axis('off')

cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label('Red = raised vs expected | Blue = depressed', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

fig.suptitle('Miller Study 2 - Injury/Asymmetry Difference Map', color='#c4a35a', fontsize=16, y=0.98)
for ax in axes:
    ax.set_facecolor('#1a1a1a')
plt.tight_layout()
plt.savefig('output/analysis/injury_asymmetry_map_miller.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()

print("\nMiller asymmetry map:")
print(f"  Diff range: [{diff_m.min():.1f}, {diff_m.max():.1f}]")
print(f"  Mean absolute: {np.abs(diff_m).mean():.2f}, Std: {diff_m.std():.2f}")
print(f"  Saved: output/analysis/injury_asymmetry_map_miller.png")

# --- Cross-study comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1a1a1a')

vmax_both = max(vmax_e, vmax_m)
norm_both = TwoSlopeNorm(vmin=-vmax_both, vcenter=0, vmax=vmax_both)

axes[0].imshow(diff_e, cmap='RdBu_r', norm=norm_both)
axes[0].set_title('Enrie 1931 - Asymmetry', color='white', fontsize=13)
axes[0].axis('off')

im = axes[1].imshow(diff_m, cmap='RdBu_r', norm=norm_both)
axes[1].set_title('Miller 1978 - Asymmetry', color='white', fontsize=13)
axes[1].axis('off')

cbar = plt.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
cbar.set_label('Red = swollen/raised | Blue = depressed', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

fig.suptitle('Cross-Study Asymmetry Comparison', color='#c4a35a', fontsize=16, y=0.98)
for ax in axes:
    ax.set_facecolor('#1a1a1a')
plt.tight_layout()
plt.savefig('output/analysis/injury_asymmetry_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: output/analysis/injury_asymmetry_comparison.png")

# Correlation
r, p = pearsonr(diff_e.flatten(), diff_m.flatten())
print(f"\nCross-study asymmetry correlation: r={r:.3f}, p={p:.2e}")
