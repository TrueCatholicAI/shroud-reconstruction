"""
Computational forensic wound mapping from the Shroud of Turin facial asymmetry data.

Loads the Enrie depth map, computes left-right asymmetry to identify regions of
facial swelling and trauma, and produces annotated forensic visualizations.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEPTH_PATH = ROOT / "data" / "processed" / "depth_map_smooth_15.npy"
OUT_DIR = ROOT / "output" / "analysis"
DOCS_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────
BG_COLOR = '#1a1a1a'
GOLD = '#c4a35a'
TEXT_COLOR = '#e0e0e0'
LABEL_COLOR = '#ffffff'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'axes.edgecolor': GOLD,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'savefig.facecolor': BG_COLOR,
    'savefig.edgecolor': BG_COLOR,
    'font.size': 10,
})


def save_fig(fig, name):
    """Save figure to both output/analysis/ and docs/images/."""
    for d in (OUT_DIR, DOCS_DIR):
        fig.savefig(d / name, dpi=180, bbox_inches='tight', pad_inches=0.3)
    print(f"  Saved {name}")


# ── 1. Load and resize ────────────────────────────────────────────────────
print("=" * 65)
print("FORENSIC WOUND MAPPING  --  Shroud of Turin Facial Asymmetry")
print("=" * 65)

raw = np.load(DEPTH_PATH)
print(f"\nOriginal depth map shape: {raw.shape}, dtype: {raw.dtype}")

# Resize to 150x150 using PIL
img_pil = Image.fromarray(raw)
img_pil = img_pil.resize((150, 150), Image.LANCZOS)
depth = np.array(img_pil, dtype=np.float64)
print(f"Resized depth map: {depth.shape}, range [{depth.min():.1f}, {depth.max():.1f}]")

# ── 2. Asymmetry map ──────────────────────────────────────────────────────
mirrored = np.fliplr(depth)
asymmetry = depth - mirrored  # positive = left side raised (swelling)

asym_std = np.std(asymmetry)
asym_mean_abs = np.mean(np.abs(asymmetry))
overall_index = asym_mean_abs / np.mean(depth)

print(f"\nAsymmetry statistics:")
print(f"  Mean absolute asymmetry : {asym_mean_abs:.3f}")
print(f"  Std dev of asymmetry    : {asym_std:.3f}")
print(f"  Asymmetry range         : [{asymmetry.min():.2f}, {asymmetry.max():.2f}]")
print(f"  Overall asymmetry index : {overall_index:.4f}  ({overall_index*100:.2f}%)")

# Threshold mask: |asymmetry| > 1 std dev
threshold = asym_std
wound_mask = np.abs(asymmetry) > threshold
wound_map = np.where(wound_mask, asymmetry, np.nan)

print(f"  Threshold (1 std)       : {threshold:.3f}")
print(f"  Pixels above threshold  : {wound_mask.sum()} / {wound_mask.size} "
      f"({wound_mask.sum()/wound_mask.size*100:.1f}%)")

# ── 3a. Overview (2x2 grid) ───────────────────────────────────────────────
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Forensic Wound Mapping: Facial Asymmetry Analysis",
             fontsize=16, color=GOLD, fontweight='bold', y=0.96)

# Original depth
ax = axes[0, 0]
im0 = ax.imshow(depth, cmap='inferno', origin='upper')
ax.set_title("Original Depth Map (Enrie)", color=LABEL_COLOR, fontsize=12)
cb0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
cb0.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cb0.outline.set_edgecolor(GOLD)
plt.setp(cb0.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
ax.set_xticks([]); ax.set_yticks([])

# Mirrored depth
ax = axes[0, 1]
im1 = ax.imshow(mirrored, cmap='inferno', origin='upper')
ax.set_title("Horizontally Mirrored Depth", color=LABEL_COLOR, fontsize=12)
cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cb1.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cb1.outline.set_edgecolor(GOLD)
plt.setp(cb1.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
ax.set_xticks([]); ax.set_yticks([])

# Raw asymmetry
ax = axes[1, 0]
vmax_asym = max(abs(asymmetry.min()), abs(asymmetry.max()))
im2 = ax.imshow(asymmetry, cmap='RdBu_r', origin='upper',
                vmin=-vmax_asym, vmax=vmax_asym)
ax.set_title("Raw Asymmetry (Original - Mirrored)", color=LABEL_COLOR, fontsize=12)
cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cb2.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cb2.outline.set_edgecolor(GOLD)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
cb2.set_label("+ left raised / - left depressed", color=TEXT_COLOR, fontsize=9)
ax.set_xticks([]); ax.set_yticks([])

# Thresholded wound map
ax = axes[1, 1]
# Show depth as dim background
ax.imshow(depth, cmap='gray', origin='upper', alpha=0.3)
im3 = ax.imshow(wound_map, cmap='RdBu_r', origin='upper',
                vmin=-vmax_asym, vmax=vmax_asym)
ax.set_title(f"Wound Map (|asym| > 1 std = {threshold:.1f})",
             color=LABEL_COLOR, fontsize=12)
cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
cb3.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cb3.outline.set_edgecolor(GOLD)
plt.setp(cb3.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout(rect=[0, 0, 1, 0.94])
save_fig(fig, "wound_mapping_overview.png")
plt.close(fig)

# ── 3b. 3D surface with asymmetry overlay ─────────────────────────────────
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG_COLOR)

# Create coordinate grids
Y_grid, X_grid = np.mgrid[0:150, 0:150]

# Normalize asymmetry for colormap
norm = Normalize(vmin=-vmax_asym, vmax=vmax_asym)
cmap_rdbu = cm.get_cmap('RdBu_r')
face_colors = cmap_rdbu(norm(asymmetry))

# Plot surface (invert Y so top of face is at top)
ax.plot_surface(X_grid, Y_grid, depth,
                facecolors=face_colors,
                rstride=2, cstride=2,
                antialiased=True, shade=False)

ax.set_title("3D Facial Surface with Asymmetry Overlay",
             fontsize=14, color=GOLD, fontweight='bold', pad=20)
ax.set_xlabel("X", labelpad=10)
ax.set_ylabel("Y", labelpad=10)
ax.set_zlabel("Depth", labelpad=10)

# Angled view
ax.view_init(elev=35, azim=-55)
ax.invert_zaxis()

# Style 3d axes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor(BG_COLOR)
ax.yaxis.pane.set_edgecolor(BG_COLOR)
ax.zaxis.pane.set_edgecolor(BG_COLOR)
ax.tick_params(axis='x', colors=TEXT_COLOR)
ax.tick_params(axis='y', colors=TEXT_COLOR)
ax.tick_params(axis='z', colors=TEXT_COLOR)

# Add colorbar
mappable = cm.ScalarMappable(norm=norm, cmap=cmap_rdbu)
cb = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.1, shrink=0.6)
cb.set_label("Asymmetry (+ swelling / - depression)", color=TEXT_COLOR, fontsize=10)
cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cb.outline.set_edgecolor(GOLD)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

save_fig(fig, "wound_mapping_3d.png")
plt.close(fig)

# ── 3c. Annotated asymmetry map ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle("Annotated Forensic Wound Map",
             fontsize=16, color=GOLD, fontweight='bold', y=0.96)

# Show asymmetry
im = ax.imshow(asymmetry, cmap='RdBu_r', origin='upper',
               vmin=-vmax_asym, vmax=vmax_asym)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Asymmetry (+ left raised / - left depressed)",
             color=TEXT_COLOR, fontsize=10)
cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
cb.outline.set_edgecolor(GOLD)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

# Region definitions
regions = {
    'left_cheek': {'rows': slice(55, 75), 'cols': slice(25, 55),
                   'label_prefix': 'Left cheekbone swelling'},
    'nasal':      {'rows': slice(50, 80), 'cols': slice(60, 75),
                   'label_prefix': 'Nasal deviation'},
    'brow_left':  {'rows': slice(35, 50), 'cols': slice(20, 50)},
    'brow_right': {'rows': slice(35, 50), 'cols': slice(100, 130)},
}

# Annotation style
ann_kwargs = dict(fontsize=10, color=LABEL_COLOR, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.4', fc='#333333', ec=GOLD,
                            alpha=0.9))
arrow_props = dict(arrowstyle='->', color=GOLD, lw=2)

# --- Left cheekbone ---
r = regions['left_cheek']
cheek_val = np.mean(asymmetry[r['rows'], r['cols']])
rect = plt.Rectangle((r['cols'].start, r['rows'].start),
                      r['cols'].stop - r['cols'].start,
                      r['rows'].stop - r['rows'].start,
                      linewidth=2, edgecolor=GOLD, facecolor='none', linestyle='--')
ax.add_patch(rect)
center_y = (r['rows'].start + r['rows'].stop) / 2
center_x = (r['cols'].start + r['cols'].stop) / 2
ax.annotate(f"Left cheekbone swelling:\n{cheek_val:+.2f}",
            xy=(center_x, center_y), xytext=(5, 120),
            arrowprops=arrow_props, **ann_kwargs)

# --- Nasal region ---
r = regions['nasal']
nasal_val = np.mean(asymmetry[r['rows'], r['cols']])
rect = plt.Rectangle((r['cols'].start, r['rows'].start),
                      r['cols'].stop - r['cols'].start,
                      r['rows'].stop - r['rows'].start,
                      linewidth=2, edgecolor=GOLD, facecolor='none', linestyle='--')
ax.add_patch(rect)
center_y = (r['rows'].start + r['rows'].stop) / 2
center_x = (r['cols'].start + r['cols'].stop) / 2
ax.annotate(f"Nasal deviation:\n{nasal_val:+.2f}",
            xy=(center_x, center_y), xytext=(110, 30),
            arrowprops=arrow_props, **ann_kwargs)

# --- Brow ridge comparison ---
brow_left_val = np.mean(asymmetry[regions['brow_left']['rows'],
                                   regions['brow_left']['cols']])
brow_right_val = np.mean(asymmetry[regions['brow_right']['rows'],
                                    regions['brow_right']['cols']])
brow_asym = abs(brow_left_val - brow_right_val)

# Draw both brow regions
for key in ('brow_left', 'brow_right'):
    r = regions[key]
    rect = plt.Rectangle((r['cols'].start, r['rows'].start),
                          r['cols'].stop - r['cols'].start,
                          r['rows'].stop - r['rows'].start,
                          linewidth=2, edgecolor=GOLD, facecolor='none',
                          linestyle='--')
    ax.add_patch(rect)

# Annotate brow from midpoint between the two regions
brow_mid_x = 75
brow_mid_y = 42
ax.annotate(f"Brow asymmetry:\n{brow_asym:.2f}\n(L: {brow_left_val:+.2f}, R: {brow_right_val:+.2f})",
            xy=(brow_mid_x, brow_mid_y), xytext=(110, 110),
            arrowprops=arrow_props, **ann_kwargs)

ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Regions of forensic interest", color=TEXT_COLOR, fontsize=12, pad=10)

fig.tight_layout(rect=[0, 0, 1, 0.94])
save_fig(fig, "wound_mapping_annotated.png")
plt.close(fig)

# ── 4. Print numeric findings ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("NUMERIC FINDINGS")
print("=" * 65)

print(f"\n  Depth map range          : {depth.min():.1f} - {depth.max():.1f}")
print(f"  Mean depth               : {np.mean(depth):.2f}")
print(f"  Asymmetry std dev        : {asym_std:.3f}")
print(f"  Mean absolute asymmetry  : {asym_mean_abs:.3f}")
print(f"  Overall asymmetry index  : {overall_index:.4f}  ({overall_index*100:.2f}%)")

print(f"\n  Left cheekbone (rows 55-75, cols 25-55):")
print(f"    Mean asymmetry         : {cheek_val:+.3f}")
print(f"    Interpretation         : {'Swelling (left raised)' if cheek_val > 0 else 'Depression (left depressed)'}")

print(f"\n  Nasal region (rows 50-80, cols 60-75):")
print(f"    Mean asymmetry         : {nasal_val:+.3f}")
print(f"    Interpretation         : {'Deviation toward left' if nasal_val > 0 else 'Deviation toward right'}")

print(f"\n  Brow ridge:")
print(f"    Left brow mean asym    : {brow_left_val:+.3f}")
print(f"    Right brow mean asym   : {brow_right_val:+.3f}")
print(f"    Brow asymmetry (|L-R|) : {brow_asym:.3f}")

print(f"\n  Wound map coverage       : {wound_mask.sum()} pixels "
      f"({wound_mask.sum()/wound_mask.size*100:.1f}% of face)")

print("\n" + "=" * 65)
print("All outputs saved to:")
print(f"  {OUT_DIR}")
print(f"  {DOCS_DIR}")
print("=" * 65)
