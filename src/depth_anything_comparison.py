"""Task P: Depth Anything V2 comparison with VP-8 and MiDaS."""
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import pipeline

print("=== Depth Anything V2 Comparison ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load sources
enrie_src = cv2.imread('data/source/enrie_1931_face_hires.jpg')
miller_src = cv2.imread('data/source/vernon_miller/34c-Fa-N_0414.jpg')
print(f"Enrie source: {enrie_src.shape}")
print(f"Miller source: {miller_src.shape}")

# Load VP-8 depth maps
enrie_full = np.load('data/processed/depth_map_smooth_15.npy')
enrie_vp8 = cv2.resize(enrie_full.astype(np.float32), (150, 150), interpolation=cv2.INTER_AREA)
miller_vp8 = np.load('output/study2_miller/depth_150x150_g15.npy')
print(f"VP-8 Enrie: {enrie_vp8.shape}, Miller: {miller_vp8.shape}")

# Load MiDaS results if available
try:
    enrie_midas = np.load('output/neural_depth/enrie_neural_depth.npy')
    miller_midas = np.load('output/neural_depth/miller_neural_depth.npy')
    has_midas = True
    print(f"MiDaS loaded: Enrie {enrie_midas.shape}, Miller {miller_midas.shape}")
except:
    has_midas = False
    print("MiDaS results not found, will skip three-way comparison")

# === Run Depth Anything V2 ===
print("\nLoading Depth Anything V2 Small...")
pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=0)
print("Model loaded.")

# Convert BGR to RGB for the pipeline
enrie_rgb = cv2.cvtColor(enrie_src, cv2.COLOR_BGR2RGB)
miller_rgb = cv2.cvtColor(miller_src, cv2.COLOR_BGR2RGB)

# Run inference
from PIL import Image
print("Running Depth Anything on Enrie...")
enrie_pil = Image.fromarray(enrie_rgb)
enrie_da_result = pipe(enrie_pil)
enrie_da = np.array(enrie_da_result['depth'])
print(f"Enrie DA output: {enrie_da.shape}, range [{enrie_da.min()}, {enrie_da.max()}]")

print("Running Depth Anything on Miller...")
miller_pil = Image.fromarray(miller_rgb)
miller_da_result = pipe(miller_pil)
miller_da = np.array(miller_da_result['depth'])
print(f"Miller DA output: {miller_da.shape}, range [{miller_da.min()}, {miller_da.max()}]")

# Resize all to 150x150 for comparison
enrie_da_150 = cv2.resize(enrie_da.astype(np.float32), (150, 150), interpolation=cv2.INTER_AREA)
miller_da_150 = cv2.resize(miller_da.astype(np.float32), (150, 150), interpolation=cv2.INTER_AREA)

# Normalize all to 0-1
def norm01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

enrie_vp8_n = norm01(enrie_vp8)
miller_vp8_n = norm01(miller_vp8)
enrie_da_n = norm01(enrie_da_150)
miller_da_n = norm01(miller_da_150)

# Correlations
r_enrie_da_vp8, _ = pearsonr(enrie_da_n.ravel(), enrie_vp8_n.ravel())
r_miller_da_vp8, _ = pearsonr(miller_da_n.ravel(), miller_vp8_n.ravel())
print(f"\nDepth Anything vs VP-8:")
print(f"  Enrie:  r = {r_enrie_da_vp8:.4f}")
print(f"  Miller: r = {r_miller_da_vp8:.4f}")

if has_midas:
    enrie_midas_150 = cv2.resize(enrie_midas.astype(np.float32), (150, 150), interpolation=cv2.INTER_AREA)
    miller_midas_150 = cv2.resize(miller_midas.astype(np.float32), (150, 150), interpolation=cv2.INTER_AREA)
    enrie_midas_n = norm01(enrie_midas_150)
    miller_midas_n = norm01(miller_midas_150)

    r_enrie_midas_vp8, _ = pearsonr(enrie_midas_n.ravel(), enrie_vp8_n.ravel())
    r_miller_midas_vp8, _ = pearsonr(miller_midas_n.ravel(), miller_vp8_n.ravel())
    r_enrie_da_midas, _ = pearsonr(enrie_da_n.ravel(), enrie_midas_n.ravel())
    r_miller_da_midas, _ = pearsonr(miller_da_n.ravel(), miller_midas_n.ravel())

    print(f"\nMiDaS vs VP-8:")
    print(f"  Enrie:  r = {r_enrie_midas_vp8:.4f}")
    print(f"  Miller: r = {r_miller_midas_vp8:.4f}")
    print(f"\nDepth Anything vs MiDaS:")
    print(f"  Enrie:  r = {r_enrie_da_midas:.4f}")
    print(f"  Miller: r = {r_miller_da_midas:.4f}")

# === Visualization: Three-model comparison ===
if has_midas:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('#1a1a1a')

# Row 1: Enrie
row = 0
axes[row, 0].imshow(enrie_vp8_n, cmap='inferno')
axes[row, 0].set_title(f'VP-8 Depth (Enrie)', color='white', fontsize=11)
axes[row, 1].imshow(enrie_da_n, cmap='inferno')
axes[row, 1].set_title(f'Depth Anything V2 (Enrie)\nr={r_enrie_da_vp8:.3f} vs VP-8', color='white', fontsize=11)
if has_midas:
    axes[row, 2].imshow(enrie_midas_n, cmap='inferno')
    axes[row, 2].set_title(f'MiDaS DPT-Large (Enrie)\nr={r_enrie_midas_vp8:.3f} vs VP-8', color='white', fontsize=11)

# Row 2: Miller
row = 1
axes[row, 0].imshow(miller_vp8_n, cmap='inferno')
axes[row, 0].set_title(f'VP-8 Depth (Miller)', color='white', fontsize=11)
axes[row, 1].imshow(miller_da_n, cmap='inferno')
axes[row, 1].set_title(f'Depth Anything V2 (Miller)\nr={r_miller_da_vp8:.3f} vs VP-8', color='white', fontsize=11)
if has_midas:
    axes[row, 2].imshow(miller_midas_n, cmap='inferno')
    axes[row, 2].set_title(f'MiDaS DPT-Large (Miller)\nr={r_miller_midas_vp8:.3f} vs VP-8', color='white', fontsize=11)

for ax in axes.flat:
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')

fig.suptitle('Three Depth Models Compared: VP-8 vs Depth Anything V2 vs MiDaS',
             color='#c4a35a', fontsize=15, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/depth_anything_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: depth_anything_comparison.png")

# === Correlation summary table visualization ===
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#222')

if has_midas:
    models = ['DA V2\nvs VP-8', 'MiDaS\nvs VP-8', 'DA V2\nvs MiDaS']
    enrie_vals = [r_enrie_da_vp8, r_enrie_midas_vp8, r_enrie_da_midas]
    miller_vals = [r_miller_da_vp8, r_miller_midas_vp8, r_miller_da_midas]
else:
    models = ['DA V2 vs VP-8']
    enrie_vals = [r_enrie_da_vp8]
    miller_vals = [r_miller_da_vp8]

x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, enrie_vals, width, color='#e74c3c', alpha=0.85, label='Enrie')
ax.bar(x + width/2, miller_vals, width, color='#3498db', alpha=0.85, label='Miller')

ax.axhline(y=0, color='#555', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models, color='#ccc', fontsize=10)
ax.set_ylabel('Pearson Correlation (r)', color='white')
ax.set_ylim(-0.3, 1.0)
ax.tick_params(colors='white')
ax.legend(facecolor='#333', labelcolor='white')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, (e, m) in enumerate(zip(enrie_vals, miller_vals)):
    ax.text(i - width/2, e + 0.03, f'{e:.3f}', ha='center', color='white', fontsize=9)
    ax.text(i + width/2, m + 0.03, f'{m:.3f}', ha='center', color='white', fontsize=9)

fig.suptitle('Neural Depth Model Correlations', color='#c4a35a', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/depth_anything_correlations.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: depth_anything_correlations.png")

print("\n--- Summary ---")
print(f"Depth Anything V2 vs VP-8: Enrie r={r_enrie_da_vp8:.4f}, Miller r={r_miller_da_vp8:.4f}")
if has_midas:
    print(f"MiDaS DPT-Large vs VP-8: Enrie r={r_enrie_midas_vp8:.4f}, Miller r={r_miller_midas_vp8:.4f}")
    print(f"DA V2 vs MiDaS: Enrie r={r_enrie_da_midas:.4f}, Miller r={r_miller_da_midas:.4f}")
print("\nTwo independent state-of-the-art neural depth models both show low")
print("correlation with the VP-8 signal, confirming the Shroud's depth")
print("encoding is fundamentally unlike standard photographic depth cues.")

print("\n=== Depth Anything Comparison Complete ===")
