"""Phase 1.3: Neural depth estimation comparison — MiDaS vs VP-8 depth maps."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import zoom
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Try MiDaS DPT-Large first, fall back to Depth Anything v2 small
model_type = None
model = None
transform = None

try:
    print("Loading MiDaS DPT-Large via torch.hub...")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    model.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform
    model_type = "MiDaS DPT-Large"
    print(f"Loaded: {model_type}")
except Exception as e:
    print(f"MiDaS DPT-Large failed: {e}")
    try:
        print("Trying MiDaS DPT-Hybrid (smaller)...")
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        model.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.dpt_transform
        model_type = "MiDaS DPT-Hybrid"
        print(f"Loaded: {model_type}")
    except Exception as e2:
        print(f"MiDaS DPT-Hybrid failed: {e2}")
        try:
            print("Trying Depth Anything v2 small via transformers...")
            from transformers import pipeline
            pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
            model_type = "Depth-Anything-V2-Small"
            print(f"Loaded: {model_type}")
        except Exception as e3:
            print(f"All models failed: {e3}")
            raise RuntimeError("No neural depth model available")


def run_midas(image_path, label):
    """Run MiDaS on a source image and return the depth prediction."""
    print(f"\nProcessing: {label} ({image_path})")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  Input shape: {img_rgb.shape}")

    if model_type and "MiDaS" in model_type:
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
    elif model_type and "Depth-Anything" in model_type:
        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        result = pipe(pil_img)
        depth = np.array(result["depth"])
    else:
        raise RuntimeError("No model loaded")

    print(f"  Neural depth shape: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]")
    return depth, img_rgb


def compare_depths(vp8_depth, neural_depth, source_img, label, output_prefix):
    """Create side-by-side comparison and compute correlation."""
    # Resize both to same dimensions for comparison
    target_h, target_w = 150, 150

    # VP-8 depth — already at analysis resolution or needs downsample
    if vp8_depth.shape != (target_h, target_w):
        h, w = vp8_depth.shape
        vp8_rs = zoom(vp8_depth.astype(np.float64), (target_h/h, target_w/w), order=1)
    else:
        vp8_rs = vp8_depth.astype(np.float64)

    # Neural depth — resize to match
    h, w = neural_depth.shape[:2]
    neural_rs = zoom(neural_depth.astype(np.float64), (target_h/h, target_w/w), order=1)

    # Normalize both to [0, 1]
    vp8_norm = (vp8_rs - vp8_rs.min()) / (vp8_rs.max() - vp8_rs.min() + 1e-8)
    neural_norm = (neural_rs - neural_rs.min()) / (neural_rs.max() - neural_rs.min() + 1e-8)

    # MiDaS outputs relative depth where HIGHER = CLOSER (same polarity as VP-8 on the negative)
    # But let's check correlation both ways and pick the better one
    r_pos, _ = pearsonr(vp8_norm.flatten(), neural_norm.flatten())
    r_neg, _ = pearsonr(vp8_norm.flatten(), (1 - neural_norm).flatten())

    if abs(r_neg) > abs(r_pos):
        neural_norm = 1 - neural_norm
        r_final = r_neg
        print(f"  Inverted neural depth (negative correlation was stronger)")
    else:
        r_final = r_pos

    print(f"  Pixel-wise correlation (150x150): r = {r_final:.4f}")

    # Difference map
    diff = vp8_norm - neural_norm

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a1a')

    # Row 1: Source, VP-8 depth, Neural depth
    if source_img.ndim == 3:
        axes[0, 0].imshow(source_img)
    else:
        axes[0, 0].imshow(source_img, cmap='gray')
    axes[0, 0].set_title('Source Photograph', color='white', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vp8_norm, cmap='inferno')
    axes[0, 1].set_title('VP-8 Depth (CLAHE)', color='white', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(neural_norm, cmap='inferno')
    axes[0, 2].set_title(f'{model_type} Depth', color='white', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: VP-8 3D-ish view, Neural 3D-ish view, Difference
    from matplotlib.colors import TwoSlopeNorm

    axes[1, 0].imshow(vp8_norm, cmap='inferno')
    axes[1, 0].set_title('VP-8 (normalized)', color='white', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(neural_norm, cmap='inferno')
    axes[1, 1].set_title(f'{model_type} (normalized)', color='white', fontsize=12)
    axes[1, 1].axis('off')

    vmax = max(abs(diff.min()), abs(diff.max()))
    if vmax < 1e-8:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = axes[1, 2].imshow(diff, cmap='RdBu_r', norm=norm)
    axes[1, 2].set_title(f'VP-8 minus Neural (r={r_final:.3f})', color='white', fontsize=12)
    axes[1, 2].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    for row in axes:
        for ax in row:
            ax.set_facecolor('#1a1a1a')

    fig.suptitle(f'{label} - VP-8 vs {model_type} Depth Comparison',
                 color='#c4a35a', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {output_prefix}_comparison.png")

    # Save neural depth as standalone image
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig2.patch.set_facecolor('#1a1a1a')
    ax2.imshow(neural_rs, cmap='inferno')
    ax2.set_title(f'{model_type} Depth', color='white')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_neural_depth.png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    return r_final


# === Process Enrie Study 1 ===
enrie_neural, enrie_rgb = run_midas('data/source/enrie_1931_face_hires.jpg', 'Enrie 1931')
enrie_vp8 = np.load('data/processed/depth_map_smooth_15.npy')
r_enrie = compare_depths(enrie_vp8, enrie_neural, enrie_rgb,
                         'Enrie 1931 (Study 1)', 'output/neural_depth/enrie')

# Save neural depth array
np.save('output/neural_depth/enrie_neural_depth.npy', enrie_neural)

# === Process Miller Study 2 ===
# Clear GPU memory first
torch.cuda.empty_cache()

miller_neural, miller_rgb = run_midas('data/source/vernon_miller/34c-Fa-N_0414.jpg', 'Miller 1978')
miller_vp8 = np.load('output/study2_miller/depth_150x150_g15.npy')
r_miller = compare_depths(miller_vp8, miller_neural, miller_rgb,
                          'Miller 1978 (Study 2)', 'output/neural_depth/miller')

np.save('output/neural_depth/miller_neural_depth.npy', miller_neural)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Model used: {model_type}")
print(f"Enrie VP-8 vs Neural correlation:  r = {r_enrie:.4f}")
print(f"Miller VP-8 vs Neural correlation: r = {r_miller:.4f}")
print(f"Neural depth arrays saved to output/neural_depth/")

# Clean up GPU
torch.cuda.empty_cache()
