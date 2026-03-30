"""Phase 2.2: Higher-resolution Miller depth maps using FFT-filtered source."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image

print("=== Higher-Resolution Miller Depth Maps ===")

# Load FFT-filtered Miller face crop
filtered = cv2.imread('output/fft_weave/miller_filtered.png', cv2.IMREAD_GRAYSCALE)
print(f"FFT-filtered face crop: {filtered.shape}")

# CLAHE depth extraction
norm_img = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
depth_full = clahe.apply(norm_img)

configs = [
    (300, 21, "300x300_g21"),
    (500, 31, "500x500_g31"),
]

for ds_size, gauss_size, label in configs:
    print(f"\n--- Processing {label} ---")
    h, w = depth_full.shape

    # Downsample
    depth_ds = zoom(depth_full.astype(np.float32), (ds_size/h, ds_size/w), order=1)

    # Gaussian smoothing (kernel size -> sigma approximation)
    sigma = gauss_size / 6.0
    depth_smooth = gaussian_filter(depth_ds, sigma=sigma)
    depth_smooth = np.clip(depth_smooth, 0, 255).astype(np.uint8)

    print(f"  Shape: {depth_smooth.shape}, range: [{depth_smooth.min()}, {depth_smooth.max()}]")

    # Save array and image
    np.save(f'output/highres_miller/depth_{label}.npy', depth_smooth)
    cv2.imwrite(f'output/highres_miller/depth_{label}.png', depth_smooth)

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.imshow(depth_smooth, cmap='inferno')
    ax.set_title(f'Miller FFT-Filtered Depth — {label}', color='white', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'output/highres_miller/heatmap_{label}.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    # 3D surface (angled view)
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('#1a1a1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a1a')

    rows_ds, cols_ds = depth_smooth.shape
    X = np.arange(cols_ds)
    Y = np.arange(rows_ds)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, depth_smooth.astype(float),
                    cmap='inferno', linewidth=0, antialiased=True,
                    rstride=max(1, rows_ds//100), cstride=max(1, cols_ds//100))
    ax.set_zlim(0, 280)
    ax.view_init(elev=35, azim=135)
    ax.set_title(f'VP-8 3D Surface — {label}', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tight_layout()
    plt.savefig(f'output/highres_miller/3d_surface_{label}.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print(f"  Saved: heatmap_{label}.png, 3d_surface_{label}.png")

    # ControlNet depth input (512x512)
    depth_512 = cv2.resize(depth_smooth, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'output/highres_miller/controlnet_depth_{label}_512.png', depth_512)
    print(f"  Saved: controlnet_depth_{label}_512.png")

# === Comparison grid: 150 vs 300 vs 500 ===
depth_150 = np.load('output/study2_miller/depth_150x150_g15.npy')
depth_300 = np.load('output/highres_miller/depth_300x300_g21.npy')
depth_500 = np.load('output/highres_miller/depth_500x500_g31.npy')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a1a')

for ax, d, title in zip(axes,
                         [depth_150, depth_300, depth_500],
                         ['150x150 + G15 (Study 2)', '300x300 + G21 (FFT-filtered)', '500x500 + G31 (FFT-filtered)']):
    ax.imshow(d, cmap='inferno')
    ax.set_title(title, color='white', fontsize=12)
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')

fig.suptitle('Miller Depth Map Resolution Comparison', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/highres_miller/resolution_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: resolution_comparison.png")

print("\n=== Higher-Resolution Processing Complete ===")
print("Now running GPU reconstructions...")
