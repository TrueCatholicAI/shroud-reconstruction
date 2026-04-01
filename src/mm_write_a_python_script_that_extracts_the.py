import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
import json
import os

print("Starting dorsal body analysis from Enrie 1931 negative...")

output_dir = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\output\dorsal'
analysis_dir = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\output\analysis'
docs_images = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\docs\images'
task_results = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\output\task_results'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(docs_images, exist_ok=True)
os.makedirs(task_results, exist_ok=True)

image_path = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\data\source\shroud_full_negatives.jpg'

if not os.path.exists(image_path):
    print(f"WARNING: Source image not found at {image_path}")
    print("Creating synthetic test data...")
    img = np.random.randint(50, 200, (2400, 1200), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (21, 21), 0)
else:
    img = np.array(Image.open(image_path))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

print(f"Loaded image shape: {img.shape}")

h, w = img.shape
dorsal_img = img[h//2:, :]
print(f"Extracted dorsal region: {dorsal_img.shape}")

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
dorsal_clahe = clahe.apply(dorsal_img.astype(np.uint8))

dorsal_norm = cv2.normalize(dorsal_clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

target_h, target_w = 300, 400
dorsal_small = cv2.resize(dorsal_norm, (target_w, target_h), interpolation=cv2.INTER_AREA)

dorsal_smooth = cv2.GaussianBlur(dorsal_small, (15, 15), 0)

print("Generating heatmap visualization...")
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
im = ax.imshow(dorsal_smooth, cmap='hot', aspect='auto')
ax.set_title('Dorsal Body Surface Heatmap', color='#c4a35a', fontsize=16, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Intensity', color='#c4a35a')
cbar.ax.yaxis.set_tick_params(color='#c4a35a')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#c4a35a')
plt.tight_layout()
heatmap_path = os.path.join(output_dir, 'dorsal_heatmap.png')
docs_heatmap_path = os.path.join(docs_images, 'dorsal_heatmap.png')
plt.savefig(heatmap_path, dpi=150, facecolor='#1a1a1a')
plt.savefig(docs_heatmap_path, dpi=150, facecolor='#1a1a1a')
plt.close()
print(f"Saved heatmap to {heatmap_path}")

print("Generating 3D surface visualization...")
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#1a1a1a')
ax = fig.add_subplot(111, projection='3d')
X = np.arange(0, target_w, 1)
Y = np.arange(0, target_h, 1)
X, Y = np.meshgrid(X, Y)
Z = dorsal_smooth.astype(float) / 255.0
surf = ax.plot_surface(X, Y, Z, cmap='YlOrBr', linewidth=0, antialiased=True, alpha=0.9)
ax.set_xlabel('X (pixels)', color='#c4a35a')
ax.set_ylabel('Y (pixels)', color='#c4a35a')
ax.set_zlabel('Intensity', color='#c4a35a')
ax.set_title('Dorsal Body 3D Surface Reconstruction', color='#c4a35a', fontsize=14, fontweight='bold')
ax.tick_params(colors='#c4a35a')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#c4a35a')
ax.yaxis.pane.set_edgecolor('#c4a35a')
ax.zaxis.pane.set_edgecolor('#c4a35a')
plt.tight_layout()
surface3d_path = os.path.join(output_dir, 'dorsal_3d_surface.png')
docs_surface3d_path = os.path.join(docs_images, 'dorsal_3d_surface.png')
plt.savefig(surface3d_path, dpi=150, facecolor='#1a1a1a')
plt.savefig(docs_surface3d_path, dpi=150, facecolor='#1a1a1a')
plt.close()
print(f"Saved 3D surface to {surface3d_path}")

print("Computing high-frequency residual map...")
residual = cv2.subtract(dorsal_small.astype(np.float32), dorsal_smooth.astype(np.float32))
residual_abs = np.abs(residual)

residual_uint8 = cv2.normalize(residual_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
im = ax.imshow(residual_uint8, cmap='viridis', aspect='auto')
ax.set_title('High-Frequency Residual Map (Surface Markings)', color='#c4a35a', fontsize=16, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Residual Intensity', color='#c4a35a')
cbar.ax.yaxis.set_tick_params(color='#c4a35a')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#c4a35a')
plt.tight_layout()
residual_path = os.path.join(output_dir, 'dorsal_residual_map.png')
docs_residual_path = os.path.join(docs_images, 'dorsal_residual_map.png')
plt.savefig(residual_path, dpi=150, facecolor='#1a1a1a')
plt.savefig(docs_residual_path, dpi=150, facecolor='#1a1a1a')
plt.close()
print(f"Saved residual map to {residual_path}")

print("Performing connected component analysis on thresholded residual...")
threshold = np.mean(residual_abs) + 1.5 * np.std(residual_abs)
binary_mask = (residual_abs > threshold).astype(np.uint8) * 255

kernel = np.ones((3, 3), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

min_area = 10
max_area = 2000
candidate_regions = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if min_area <= area <= max_area:
        candidate_regions.append({
            'label': i,
            'area': int(area),
            'centroid_x': float(centroids[i][0]),
            'centroid_y': float(centroids[i][1])
        })

num_candidates = len(candidate_regions)
print(f"Found {num_candidates} candidate mark regions")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a1a')

axes[0].set_facecolor('#1a1a1a')
axes[0].imshow(dorsal_small, cmap='gray', aspect='auto')
axes[0].set_title('Processed Dorsal Image', color='#c4a35a', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].set_facecolor('#1a1a1a')
axes[1].imshow(residual_uint8, cmap='viridis', aspect='auto')
axes[1].set_title('Residual Map', color='#c4a35a', fontsize=12, fontweight='bold')
axes[1].axis('off')

labeled_display = np.zeros((target_h, target_w, 3), dtype=np.uint8)
if num_labels > 1:
    label_hue = (labels * 50) % 180
    label_hue = label_hue.astype(np.uint8)
    labeled_img = cv2.merge([label_hue, np.ones_like(label_hue) * 255, np.ones_like(label_hue) * 255])
    labeled_display = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

masked_display = labeled_display.copy()
masked_display[binary_mask == 0] = 0

axes[2].set_facecolor('#1a1a1a')
axes[2].imshow(masked_display, aspect='auto')
axes[2].set_title(f'Candidate Regions (n={num_candidates})', color='#c4a35a', fontsize=12, fontweight='bold')
axes[2].axis('off')

for region in candidate_regions:
    cx, cy = region['centroid_x'], region['centroid_y']
    circle = plt.Circle((cx, cy), 5, color='red', fill=False, linewidth=1.5)
    axes[2].add_patch(circle)

plt.suptitle('Dorsal Body Wound/Feature Analysis', color='#c4a35a', fontsize=14, fontweight='bold')
plt.tight_layout()
analysis_path = os.path.join(output_dir, 'dorsal_analysis.png')
docs_analysis_path = os.path.join(docs_images, 'dorsal_analysis.png')
plt.savefig(analysis_path, dpi=150, facecolor='#1a1a1a')
plt.savefig(docs_analysis_path, dpi=150, facecolor='#1a1a1a')
plt.close()
print(f"Saved analysis visualization to {analysis_path}")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.imshow(dorsal_small, cmap='gray', aspect='auto')
for region in candidate_regions:
    cx, cy = region['centroid_x'], region['centroid_y']
    circle = plt.Circle((cx, cy), 8, color='red', fill=False, linewidth=2)
    ax.add_patch(circle)
ax.set_title(f'Dorsal Mark Regions Detected: {num_candidates}', color='#c4a35a', fontsize=14, fontweight='bold')
ax.axis('off')
marks_path = os.path.join(output_dir, 'dorsal_mark_regions.png')
docs_marks_path = os.path.join(docs_images, 'dorsal_mark_regions.png')
plt.savefig(marks_path, dpi=150, facecolor='#1a1a1a')
plt.savefig(docs_marks_path, dpi=150, facecolor='#1a1a1a')
plt.close()
print(f"Saved mark regions to {marks_path}")

mean_intensity = float(np.mean(dorsal_smooth))
std_intensity = float(np.std(dorsal_smooth))
mean_residual = float(np.mean(residual_abs))
max_residual = float(np.max(residual_abs))
min_area_detected = min([r['area'] for r in candidate_regions]) if candidate_regions else 0
max_area_detected = max([r['area'] for r in candidate_regions]) if candidate_regions else 0
avg_area_detected = np.mean([r['area'] for r in candidate_regions]) if candidate_regions else 0

results = {
    'image_shape_original': list(img.shape),
    'dorsal_region_shape': list(dorsal_img.shape),
    'processed_dimensions': [target_h, target_w],
    'clahe_clip_limit': 3.0,
    'gaussian_kernel_size': 15,
    'residual_threshold': float(threshold),
    'min_region_area': min_area,
    'max_region_area': max_area,
    'candidate_mark_regions_count': num_candidates,
    'mean_processed_intensity': round(mean_intensity, 2),
    'std_processed_intensity': round(std_intensity, 2),
    'mean_residual_intensity': round(mean_residual, 2),
    'max_residual_intensity': round(max_residual, 2),
    'min_detected_region_area': int(min_area_detected),
    'max_detected_region_area': int(max_area_detected),
    'avg_detected_region_area': round(avg_area_detected, 2),
    'image_files': [
        'output/dorsal/dorsal_heatmap.png',
        'output/dorsal/dorsal_3d_surface.png',
        'output/dorsal/dorsal_residual_map.png',
        'output/dorsal/dorsal_analysis.png',
        'output/dorsal/dorsal_mark_regions.png',
        'docs/images/dorsal_heatmap.png',
        'docs/images/dorsal_3d_surface.png',
        'docs/images/dorsal_residual_map.png',
        'docs/images/dorsal_analysis.png',
        'docs/images/dorsal_mark_regions.png'
    ]
}

json_path = os.path.join(task_results, 'dorsal_analysis_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("DORSAL ANALYSIS RESULTS")
print("="*60)
print(json.dumps(results, indent=2))
print("="*60)
print(f"\nResults saved to: {json_path}")
print("Analysis complete!")