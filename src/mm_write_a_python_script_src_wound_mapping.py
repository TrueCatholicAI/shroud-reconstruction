import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, FancyArrow
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

depth_map_path = os.path.join(project_root, 'data', 'processed', 'depth_map_smooth_15.npy')
output_dir = os.path.join(project_root, 'output', 'analysis')
docs_output_dir = os.path.join(project_root, 'docs', 'images')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(docs_output_dir, exist_ok=True)

print("="*70)
print("WOUND MAPPING ASYMMETRY ANALYSIS")
print("="*70)

print("\n[1] Loading depth map...")
depth_map = np.load(depth_map_path)
print(f"    Loaded shape: {depth_map.shape}")
print(f"    Data type: {depth_map.dtype}")
print(f"    Value range: [{depth_map.min():.6f}, {depth_map.max():.6f}]")

print("\n[2] Resizing to 150x150 with INTER_AREA...")
depth_map_resized = cv2.resize(depth_map, (150, 150), interpolation=cv2.INTER_AREA)
print(f"    Resized shape: {depth_map_resized.shape}")

print("\n[3] Creating asymmetry map...")
mirrored = np.fliplr(depth_map_resized)
asymmetry_raw = depth_map_resized - mirrored
print(f"    Asymmetry range: [{asymmetry_raw.min():.6f}, {asymmetry_raw.max():.6f}]")

depth_range = depth_map_resized.max() - depth_map_resized.min()
asymmetry_normalized = asymmetry_raw / depth_range
print(f"    Depth range for normalization: {depth_range:.6f}")

print("\n[4] Computing regional measurements...")

left_cheekbone = asymmetry_raw[55:75, 25:55]
left_cheekbone_mean = np.mean(left_cheekbone)
left_cheekbone_std = np.std(left_cheekbone)
print(f"    Left Cheekbone (rows 55-75, cols 25-55):")
print(f"      Mean asymmetry: {left_cheekbone_mean:.6f}")
print(f"      Std asymmetry: {left_cheekbone_std:.6f}")

nasal_region = asymmetry_raw[50:80, 65:85]
nasal_deviation = np.std(nasal_region)
nasal_mean = np.mean(nasal_region)
print(f"    Nasal Region (rows 50-80, cols 65-85):")
print(f"      Mean asymmetry: {nasal_mean:.6f}")
print(f"      Deviation (std): {nasal_deviation:.6f}")

left_brow = asymmetry_raw[35:50, 20:50]
right_brow = asymmetry_raw[35:50, 100:130]
left_brow_mean = np.mean(left_brow)
right_brow_mean = np.mean(right_brow)
brow_ridge_asymmetry = left_brow_mean - right_brow_mean
print(f"    Brow Ridge (rows 35-50):")
print(f"      Left brow (cols 20-50) mean: {left_brow_mean:.6f}")
print(f"      Right brow (cols 100-130) mean: {right_brow_mean:.6f}")
print(f"      Asymmetry (L-R): {brow_ridge_asymmetry:.6f}")

print("\n[5] Computing overall asymmetry index...")
mean_abs_asymmetry = np.mean(np.abs(asymmetry_raw))
mean_depth = np.mean(depth_map_resized)
asymmetry_index = mean_abs_asymmetry / mean_depth
print(f"    Mean |asymmetry|: {mean_abs_asymmetry:.6f}")
print(f"    Mean depth: {mean_depth:.6f}")
print(f"    Asymmetry Index: {asymmetry_index:.6f}")

threshold = np.std(asymmetry_raw)
asymmetry_thresholded = np.abs(asymmetry_raw) > threshold
thresholded_pixels = np.sum(asymmetry_thresholded)
thresholded_percent = 100 * thresholded_pixels / asymmetry_thresholded.size
print(f"\n[6] Threshold analysis (1 STD = {threshold:.6f}):")
print(f"    Pixels above threshold: {thresholded_pixels}")
print(f"    Percentage: {thresholded_percent:.2f}%")

print("\n[7] Generating figures...")

print("    Creating wound_analysis_overview.png...")
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.patch.set_facecolor('#1a1a1a')
for ax in axes1.flat:
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#666666')

im_orig = axes1[0, 0].imshow(depth_map_resized, cmap='gray', aspect='equal')
axes1[0, 0].set_title('Original Depth Map', color='white', fontsize=12, fontweight='bold')
axes1[0, 0].set_xlabel('X (pixels)', color='white', fontsize=10)
axes1[0, 0].set_ylabel('Y (pixels)', color='white', fontsize=10)

im_mirror = axes1[0, 1].imshow(mirrored, cmap='gray', aspect='equal')
axes1[0, 1].set_title('Horizontally Mirrored', color='white', fontsize=12, fontweight='bold')
axes1[0, 1].set_xlabel('X (pixels)', color='white', fontsize=10)
axes1[0, 1].set_ylabel('Y (pixels)', color='white', fontsize=10)

vmax = max(abs(asymmetry_raw.min()), abs(asymmetry_raw.max()))
im_asym = axes1[1, 0].imshow(asymmetry_raw, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
axes1[1, 0].set_title('Raw Asymmetry Map (L - R)', color='white', fontsize=12, fontweight='bold')
axes1[1, 0].set_xlabel('X (pixels)', color='white', fontsize=10)
axes1[1, 0].set_ylabel('Y (pixels)', color='white', fontsize=10)
cbar1 = plt.colorbar(im_asym, ax=axes1[1, 0], fraction=0.046, pad=0.04)
cbar1.ax.yaxis.set_tick_params(color='white')
cbar1.ax.set_ylabel('Asymmetry Value', color='white', fontsize=9)
plt.setp(plt.getp(cbar1.ax, 'yticklabels'), color='white')

im_thresh = axes1[1, 1].imshow(asymmetry_thresholded.astype(float), cmap='hot', aspect='equal')
axes1[1, 1].set_title(f'Thresholded (> 1 STD = {threshold:.4f})', color='white', fontsize=12, fontweight='bold')
axes1[1, 1].set_xlabel('X (pixels)', color='white', fontsize=10)
axes1[1, 1].set_ylabel('Y (pixels)', color='white', fontsize=10)
cbar2 = plt.colorbar(im_thresh, ax=axes1[1, 1], fraction=0.046, pad=0.04)
cbar2.ax.yaxis.set_tick_params(color='white')
cbar2.ax.set_ylabel('Above Threshold (0/1)', color='white', fontsize=9)
plt.setp(plt.getp(cbar2.ax, 'yticklabels'), color='white')

plt.suptitle('Wound Mapping Asymmetry Overview', color='#c4a35a', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

fig1.savefig(os.path.join(output_dir, 'wound_analysis_overview.png'), dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
fig1.savefig(os.path.join(docs_output_dir, 'wound_analysis_overview.png'), dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
plt.close(fig1)
print("      Saved to output/analysis/wound_analysis_overview.png")
print("      Saved to docs/images/wound_analysis_overview.png")

print("    Creating wound_analysis_3d.png...")
fig2 = plt.figure(figsize=(14, 10))
fig2.patch.set_facecolor('#1a1a1a')
ax2_3d = fig2.add_subplot(111, projection='3d')

x = np.arange(150)
y = np.arange(150)
X, Y = np.meshgrid(x, y)

norm_asym = (asymmetry_normalized - asymmetry_normalized.min()) / (asymmetry_normalized.max() - asymmetry_normalized.min())
colors = plt.cm.RdBu_r(norm_asym)

surf = ax2_3d.plot_surface(X, Y, depth_map_resized, facecolors=colors, linewidth=0, antialiased=True, alpha=0.9, rstride=2, cstride=2)

ax2_3d.set_title('3D Depth Surface with Asymmetry Coloring', color='#c4a35a', fontsize=14, fontweight='bold', pad=20)
ax2_3d.set_xlabel('X (pixels)', color='white', fontsize=11, labelpad=10)
ax2_3d.set_ylabel('Y (pixels)', color='white', fontsize=11, labelpad=10)
ax2_3d.set_zlabel('Depth', color='white', fontsize=11, labelpad=10)
ax2_3d.tick_params(colors='white', labelsize=9)
ax2_3d.xaxis.pane.fill = False
ax2_3d.yaxis.pane.fill = False
ax2_3d.zaxis.pane.fill = False
ax2_3d.xaxis.pane.set_edgecolor('#444444')
ax2_3d.yaxis.pane.set_edgecolor('#444444')
ax2_3d.zaxis.pane.set_edgecolor('#444444')
ax2_3d.grid(True, color='#444444', alpha=0.5)

mappable = plt.cm.ScalarMappable(cmap='RdBu_r')
mappable.set_array(asymmetry_normalized)
cbar3 = fig2.colorbar(mappable, ax=ax2_3d, shrink=0.6, aspect=20, pad=0.1)
cbar3.ax.yaxis.set_tick_params(color='white')
cbar3.ax.set_ylabel('Normalized Asymmetry', color='white', fontsize=10)
plt.setp(plt.getp(cbar3.ax, 'yticklabels'), color='white')

fig2.savefig(os.path.join(output_dir, 'wound_analysis_3d.png'), dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
fig2.savefig(os.path.join(docs_output_dir, 'wound_analysis_3d.png'), dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
plt.close(fig2)
print("      Saved to output/analysis/wound_analysis_3d.png")
print("      Saved to docs/images/wound_analysis_3d.png")

print("    Creating wound_analysis_annotated.png...")
fig3, ax3 = plt.subplots(figsize=(12, 10))
fig3.patch.set_facecolor('#1a1a1a')
ax3.set_facecolor('#1a1a1a')

im_ann = ax3.imshow(asymmetry_raw, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
ax3.set_title('Asymmetry Map with Anatomical Regions', color='#c4a35a', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('X (pixels)', color='white', fontsize=11)
ax3.set_ylabel('Y (pixels)', color='white', fontsize=11)
ax3.tick_params(colors='white', labelsize=10)
for spine in ax3.spines.values():
    spine.set_color('#666666')

cbar4 = plt.colorbar(im_ann, ax=ax3, fraction=0.046, pad=0.04)
cbar4.ax.yaxis.set_tick_params(color='white')
cbar4.ax.set_ylabel('Asymmetry Value', color='white', fontsize=10)
plt.setp(plt.getp(cbar4.ax, 'yticklabels'), color='white')

ax3.add_patch(Rectangle((25, 55), 30, 20, fill=False, edgecolor='#c4a35a', linewidth=2.5, linestyle='-'))
ax3.annotate('Left Cheekbone\n(asymmetry: {:.4f})'.format(left_cheekbone_mean),
             xy=(40, 65), xytext=(40, 40),
             fontsize=9, color='white', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor='#c4a35a', alpha=0.9),
             arrowprops=dict(arrowstyle='->', color='#c4a35a', lw=1.5))

ax3.add_patch(Rectangle((65, 50), 20, 30, fill=False, edgecolor='#c4a35a', linewidth=2.5, linestyle='-'))
ax3.annotate('Nasal Region\n(deviation: {:.4f})'.format(nasal_deviation),
             xy=(75, 65), xytext=(110, 55),
             fontsize=9, color='white', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor='#c4a35a', alpha=0.9),
             arrowprops=dict(arrowstyle='->', color='#c4a35a', lw=1.5))

ax3.add_patch(Rectangle((20, 35), 30, 15, fill=False, edgecolor='#c4a35a', linewidth=2.5, linestyle='-'))
ax3.annotate('Left Brow\n(asymmetry: {:.4f})'.format(left_brow_mean),
             xy=(35, 42), xytext=(35, 15),
             fontsize=9, color='white', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor='#c4a35a', alpha=0.9),
             arrowprops=dict(arrowstyle='->', color='#c4a35a', lw=1.5))

ax3.add_patch(Rectangle((100, 35), 30, 15, fill=False, edgecolor='#c4a35a', linewidth=2.5, linestyle='-'))
ax3.annotate('Right Brow\n(asymmetry: {:.4f})'.format(right_brow_mean),
             xy=(115, 42), xytext=(115, 15),
             fontsize=9, color='white', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor='#c4a35a', alpha=0.9),
             arrowprops=dict(arrowstyle='->', color='#c4a35a', lw=1.5))

ax3.annotate('Brow Asymmetry\n(L-R: {:.4f})'.format(brow_ridge_asymmetry),
             xy=(75, 42), xytext=(75, 5),
             fontsize=10, color='#c4a35a', ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a', edgecolor='#c4a35a', alpha=0.9))

ax3.annotate('', xy=(55, 42), xytext=(85, 42),
             arrowprops=dict(arrowstyle='<->', color='white', lw=2, shrinkA=5, shrinkB=5))
ax3.text(70, 48, 'vs', fontsize=8, color='white', ha='center', va='bottom')

plt.tight_layout()

fig3.savefig(os.path.join(output_dir, 'wound_analysis_annotated.png'), dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
fig3.savefig(os.path.join(docs_output_dir, 'wound_analysis_annotated.png'), dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
plt.close(fig3)
print("      Saved to output/analysis/wound_analysis_annotated.png")
print("      Saved to docs/images/wound_analysis_annotated.png")

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print(f"\nInput Data:")
print(f"  Source: {depth_map_path}")
print(f"  Original shape: {depth_map.shape}")
print(f"  Processing size: 150x150 pixels")

print(f"\nDepth Statistics:")
print(f"  Min depth: {depth_map_resized.min():.6f}")
print(f"  Max depth: {depth_map_resized.max():.6f}")
print(f"  Mean depth: {mean_depth:.6f}")
print(f"  Depth range: {depth_range:.6f}")

print(f"\nAsymmetry Metrics:")
print(f"  Raw asymmetry range: [{asymmetry_raw.min():.6f}, {asymmetry_raw.max():.6f}]")
print(f"  Mean asymmetry: {asymmetry_raw.mean():.6f}")
print(f"  Asymmetry std: {asymmetry_raw.std():.6f}")
print(f"  Mean |asymmetry|: {mean_abs_asymmetry:.6f}")

print(f"\nRegional Analysis:")
print(f"  Left cheekbone mean asymmetry: {left_cheekbone_mean:+.6f}")
print(f"  Nasal region deviation: {nasal_deviation:.6f}")
print(f"  Brow asymmetry (L-R): {brow_ridge_asymmetry:+.6f}")

print(f"\nOverall Index:")
print(f"  Asymmetry Index (mean|asy| / mean depth): {asymmetry_index:.6f}")

print(f"\nThreshold Analysis:")
print(f"  Threshold (1 STD): {threshold:.6f}")
print(f"  Significant pixels: {thresholded_pixels} ({thresholded_percent:.2f}%)")

print(f"\nOutput Files:")
print(f"  1. {os.path.join(output_dir, 'wound_analysis_overview.png')}")
print(f"  2. {os.path.join(output_dir, 'wound_analysis_3d.png')}")
print(f"  3. {os.path.join(output_dir, 'wound_analysis_annotated.png')}")
print(f"  4. {os.path.join(docs_output_dir, 'wound_analysis_overview.png')}")
print(f"  5. {os.path.join(docs_output_dir, 'wound_analysis_3d.png')}")
print(f"  6. {os.path.join(docs_output_dir, 'wound_analysis_annotated.png')}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)