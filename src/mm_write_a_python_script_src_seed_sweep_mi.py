import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import os
from datetime import datetime

print("=" * 70)
print("MILLER SEED SWEEP ANALYSIS - Lambertian Shading with Haar Cascade Scoring")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, 'src')
output_dir = os.path.join(base_dir, 'output', 'seed_sweep')
docs_images_dir = os.path.join(base_dir, 'docs', 'images')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(docs_images_dir, exist_ok=True)

image_path = os.path.join(base_dir, 'data', 'source', 'vernon_miller', '34c-Fa-N_0414.jpg')
print(f"\nLoading image: {image_path}")
img = cv2.imread(image_path)
if img is None:
    print(f"ERROR: Could not load image from {image_path}")
    exit(1)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (300, 300))
print(f"Image resized to: {img_resized.shape}")

print("\nApplying FFT bandpass filter (5-80 cycles)...")
rows, cols = img_resized.shape
fft = np.fft.fft2(img_resized.astype(np.float32))
fft_shift = np.fft.fftshift(fft)
m, n = rows, cols
y, x = np.ogrid[:m, :n]
center_y, center_x = m // 2, n // 2
dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
low_cutoff = 5
high_cutoff = 80
mask = np.logical_and(dist >= low_cutoff, dist <= high_cutoff).astype(np.float32)
fft_filtered = fft_shift * mask
fft_filtered_shift = np.fft.ifftshift(fft_filtered)
img_filtered = np.real(np.fft.ifft2(fft_filtered_shift))
img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(f"FFT bandpass complete: kept frequencies {low_cutoff}-{high_cutoff} cycles")

depth_map = img_filtered.astype(np.float32) / 255.0
print(f"Depth map range: {depth_map.min():.3f} - {depth_map.max():.3f}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("ERROR: Could not load Haar cascade classifier")
    exit(1)
print("Haar cascade classifier loaded successfully")

results = []
ambient = 0.3
diffuse = 0.7
clay_color = np.array([168, 175, 180], dtype=np.float32) / 255.0

print(f"\nProcessing seeds 0-199 with azimuth from 0 to 360 degrees...")
print("-" * 60)

for seed in range(200):
    azimuth_deg = seed * 1.8
    azimuth_rad = np.deg2rad(azimuth_deg)
    light_dir = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.7])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    gy, gx = np.gradient(depth_map)
    normals = np.zeros((rows, cols, 3), dtype=np.float32)
    normals[:, :, 0] = -gx
    normals[:, :, 1] = -gy
    normals[:, :, 2] = 1.0
    norm_mag = np.sqrt(normals[:, :, 0]**2 + normals[:, :, 1]**2 + normals[:, :, 2]**2)
    norm_mag = np.maximum(norm_mag, 1e-6)
    normals[:, :, 0] /= norm_mag
    normals[:, :, 1] /= norm_mag
    normals[:, :, 2] /= norm_mag
    
    shading = np.zeros((rows, cols, 3), dtype=np.float32)
    for c in range(3):
        shading[:, :, c] = ambient * clay_color[c] + diffuse * np.maximum(0, np.dot(normals, light_dir)) * clay_color[c]
    
    shading = np.clip(shading, 0, 1)
    shading_bgr = (shading[:, :, ::-1] * 255).astype(np.uint8)
    
    faces = face_cascade.detectMultiScale(shading_bgr, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    score = len(faces)
    
    results.append({
        'seed': seed,
        'azimuth': azimuth_deg,
        'score': score,
        'shading': shading_bgr.copy()
    })
    
    if seed % 40 == 0:
        print(f"  Seed {seed:3d}: azimuth={azimuth_deg:6.1f}°, faces detected={score}")

print("-" * 60)
results.sort(key=lambda x: x['score'], reverse=True)
top_10 = results[:10]

print("\n" + "=" * 70)
print("TOP 10 RESULTS BY HAAR CASCADE SCORE")
print("=" * 70)
print(f"{'Rank':<6}{'Seed':<8}{'Azimuth (°)':<14}{'Faces Detected':<15}{'Filename'}")
print("-" * 70)

for rank, r in enumerate(top_10, 1):
    filename = f"miller_seed_{r['seed']:03d}_az{r['azimuth']:.1f}_faces{r['score']}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, r['shading'])
    print(f"{rank:<6}{r['seed']:<8}{r['azimuth']:<14.1f}{r['score']:<15}{filename}")
    r['filename'] = filename

print("-" * 70)
print(f"\nTop 10 images saved to: {output_dir}")

print("\nGenerating 2x5 contact sheet...")
canvas_width = 300 * 5
canvas_height = 300 * 2
contact_sheet = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

positions = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]

for idx, r in enumerate(top_10):
    row, col = positions[idx]
    x_offset = col * 300
    y_offset = row * 300
    contact_sheet[y_offset:y_offset+300, x_offset:x_offset+300] = r['shading']

contact_sheet_path = os.path.join(output_dir, 'miller_seed_sweep_contact.png')
cv2.imwrite(contact_sheet_path, contact_sheet)
print(f"Contact sheet saved to: {contact_sheet_path}")

docs_contact_path = os.path.join(docs_images_dir, 'miller_seed_sweep_contact.png')
cv2.imwrite(docs_contact_path, contact_sheet)
print(f"Contact sheet copied to: {docs_contact_path}")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

ax.imshow(cv2.cvtColor(contact_sheet, cv2.COLOR_BGR2RGB))
ax.set_title('Miller Seed Sweep - Top 10 Lambertian Shading Results', 
             color='white', fontsize=14, fontweight='bold', pad=15)

for i, r in enumerate(top_10):
    row, col = positions[i]
    x_center = col * 300 + 150
    y_center = row * 300 + 290
    ax.text(x_center, y_center, f"S{r['seed']} ({r['azimuth']:.0f}°)", 
            ha='center', va='bottom', fontsize=8, color='#c4a35a', 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a1a', edgecolor='#c4a35a', alpha=0.8))

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor('#c4a35a')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'miller_seed_sweep_contact_dark.png'), 
            dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
fig.savefig(os.path.join(docs_images_dir, 'miller_seed_sweep_contact_dark.png'), 
            dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
plt.close(fig)
print("Dark theme contact sheet saved.")

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#1a1a1a')
ax2.set_facecolor('#1a1a1a')

seeds = list(range(200))
scores = [r['score'] for r in results]
colors = ['#c4a35a' if r['score'] > 0 else '#555555' for r in results]

ax2.bar(seeds, scores, color=colors, width=1.0)
ax2.set_xlabel('Seed (Azimuth = Seed × 1.8°)', color='white', fontsize=11)
ax2.set_ylabel('Faces Detected', color='white', fontsize=11)
ax2.set_title('Miller Seed Sweep: Haar Cascade Face Detection Scores', 
              color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#c4a35a')
ax2.set_xlim(-1, 200)
ax2.set_ylim(0, max(scores) + 1 if max(scores) > 0 else 2)

for rank, r in enumerate(top_10[:5]):
    ax2.axvline(x=r['seed'], color='#c4a35a', linestyle='--', alpha=0.5, linewidth=0.8)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'miller_seed_sweep_scores.png'), 
             dpi=150, facecolor='#1a1a1a')
fig2.savefig(os.path.join(docs_images_dir, 'miller_seed_sweep_scores.png'), 
             dpi=150, facecolor='#1a1a1a')
plt.close(fig2)
print("Score plot saved.")

print("\n" + "=" * 70)
print("COMPLETE RESULTS TABLE (all 200 seeds)")
print("=" * 70)
print(f"{'Seed':<6}{'Azimuth (°)':<14}{'Faces':<8}{'Seed':<6}{'Azimuth (°)':<14}{'Faces'}")
print("-" * 70)

half = 100
for i in range(half):
    r1 = results[i]
    r2 = results[i + half]
    print(f"{r1['seed']:<6}{r1['azimuth']:<14.1f}{r1['score']:<8}{r2['seed']:<6}{r2['azimuth']:<14.1f}{r2['score']}")

print("-" * 70)
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)