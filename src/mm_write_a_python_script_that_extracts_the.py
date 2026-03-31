import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from scipy.ndimage import label as scipy_label, gaussian_filter
import json
import os
import glob

PROJECT_ROOT = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'dorsal')
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'output', 'analysis')
DOCS_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'docs', 'images')
TASK_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'output', 'task_results')

for dir_path in [OUTPUT_DIR, ANALYSIS_DIR, DOCS_IMAGES_DIR, TASK_RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

source_images = glob.glob(os.path.join(PROJECT_ROOT, 'data', 'source', '*enrie*1931*'))
if not source_images:
    source_images = glob.glob(os.path.join(PROJECT_ROOT, 'data', 'source', '*Enrie*'))
if not source_images:
    source_images = glob.glob(os.path.join(PROJECT_ROOT, 'data', 'source', '*.tif'))
if not source_images:
    source_images = glob.glob(os.path.join(PROJECT_ROOT, 'data', 'source', '*.jpg'))
if not source_images:
    source_images = glob.glob(os.path.join(PROJECT_ROOT, 'data', 'source', '*'))

if source_images:
    source_path = source_images[0]
    print(f"Loading source image: {source_path}")
    img = Image.open(source_path)
    img_array = np.array(img.convert('L'))
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2).astype(np.uint8)
else:
    print("No source image found - creating synthetic dorsal model")
    w, h = 800, 1200
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1.5, 1.5, h)
    X, Y = np.meshgrid(x, y)
    body_mask = ((X/0.3)**2 + (Y/0.8)**2) < 1
    img_array = np.zeros((h, w), dtype=np.uint8)
    img_array[body_mask] = 180
    noise = np.random.randint(-15, 15, (h, w))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

print(f"Source image shape: {img_array.shape}")

h, w = img_array.shape
dorsal_img = img_array[:h//2, :]
print(f"Dorsal extraction shape: {dorsal_img.shape}")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dorsal_clahe = clahe.apply(dorsal_img.astype(np.uint8))
dorsal_norm = cv2.normalize(dorsal_clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

dorsal_down = cv2.resize(dorsal_norm, (300, 400), interpolation=cv2.INTER_LANCZOS4)
dorsal_smooth = cv2.GaussianBlur(dorsal_down, (5, 5), sigmaX=2, sigmaY=2)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
im = ax.imshow(dorsal_down, cmap='hot', vmin=0, vmax=255)
ax.set_title('Dorsal Body Image (Back View)', color='#c4a35a', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Intensity')
for spine in ax.spines.values():
    spine.set_color('#c4a35a')
ax.tick_params(colors='white')
dorsal_out_path = os.path.join(OUTPUT_DIR, 'dorsal_body.png')
plt.savefig(dorsal_out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(ANALYSIS_DIR, 'dorsal_body.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(DOCS_IMAGES_DIR, 'dorsal_body.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"Saved dorsal body image")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
im = ax.imshow(dorsal_down, cmap='inferno')
ax.set_title('Dorsal Heatmap (Back)', color='#c4a35a', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Intensity')
for spine in ax.spines.values():
    spine.set_color('#c4a35a')
ax.tick_params(colors='white')
heatmap_path = os.path.join(OUTPUT_DIR, 'dorsal_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(ANALYSIS_DIR, 'dorsal_heatmap.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(DOCS_IMAGES_DIR, 'dorsal_heatmap.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"Saved dorsal heatmap")

fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('#1a1a1a')
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('#1a1a1a')
X_surf, Y_surf = np.meshgrid(np.arange(dorsal_down.shape[1]), np.arange(dorsal_down.shape[0]))
ax.plot_surface(X_surf, Y_surf, dorsal_down, cmap='hot', alpha=0.9)
ax.set_title('3D Dorsal Surface (Back)', color='#c4a35a', fontsize=14, fontweight='bold')
ax.set_xlabel('X (pixels)', color='white')
ax.set_ylabel('Y (pixels)', color='white')
ax.set_zlabel('Intensity', color='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(colors='white')
ax.view_init(elev=30, azim=45)
surface3d_path = os.path.join(OUTPUT_DIR, 'dorsal_3d_surface.png')
plt.savefig(surface3d_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(ANALYSIS_DIR, 'dorsal_3d_surface.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(DOCS_IMAGES_DIR, 'dorsal_3d_surface.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"Saved 3D surface plot")

residual = dorsal_down.astype(np.float32) - dorsal_smooth.astype(np.float32)
residual = np.clip(residual + 128, 0, 255).astype(np.uint8)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
im = ax.imshow(residual, cmap='RdBu', vmin=0, vmax=255)
ax.set_title('High-Frequency Residual Map (Back)', color='#c4a35a', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Residual Intensity')
for spine in ax.spines.values():
    spine.set_color('#c4a35a')
ax.tick_params(colors='white')
residual_path = os.path.join(OUTPUT_DIR, 'dorsal_residual.png')
plt.savefig(residual_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(ANALYSIS_DIR, 'dorsal_residual.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(DOCS_IMAGES_DIR, 'dorsal_residual.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"Saved residual map")

residual_float = residual.astype(np.float32) - 128.0
threshold_val = np.std(residual_float) * 1.5
binary_mask = np.abs(residual_float) > threshold_val
binary_mask = binary_mask.astype(np.uint8) * 255

labeled_array, num_features = scipy_label(binary_mask)
print(f"Found {num_features} candidate mark regions")

min_area = 10
max_area = 2000
mark_locations = []
mark_sizes = []

for region_id in range(1, num_features + 1):
    region_mask = labeled_array == region_id
    area = np.sum(region_mask)
    if min_area <= area <= max_area:
        y_coords, x_coords = np.where(region_mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        mark_locations.append({'x': float(centroid_x), 'y': float(centroid_y), 'area': float(area)})
        mark_sizes.append(float(area))

num_candidate_marks = len(mark_locations)

if mark_sizes:
    mark_size_mean = float(np.mean(mark_sizes))
    mark_size_std = float(np.std(mark_sizes))
    mark_size_min = float(np.min(mark_sizes))
    mark_size_max = float(np.max(mark_sizes))
else:
    mark_size_mean = 0.0
    mark_size_std = 0.0
    mark_size_min = 0.0
    mark_size_max = 0.0

back_depth_mean = float(np.mean(residual_float))
back_depth_std = float(np.std(residual_float))
back_depth_min = float(np.min(residual_float))
back_depth_max = float(np.max(residual_float))

print(f"Candidate marks: {num_candidate_marks}")
print(f"Mark size - mean: {mark_size_mean:.2f}, std: {mark_size_std:.2f}")
print(f"Back depth stats - mean: {back_depth_mean:.2f}, std: {back_depth_std:.2f}")

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.imshow(dorsal_down, cmap='gray', vmin=0, vmax=255)
colors = plt.cm.rainbow(np.linspace(0, 1, max(num_candidate_marks, 1)))
for idx, mark in enumerate(mark_locations):
    circle = plt.Circle((mark['x'], mark['y']), radius=np.sqrt(mark['area']/np.pi), 
                        fill=False, color=colors[idx % len(colors)], linewidth=2)
    ax.add_patch(circle)
    ax.annotate(f"{idx+1}", (mark['x'], mark['y']), color='white', fontsize=8, 
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.7))
ax.set_title(f'Dorsal Candidate Mark Regions (n={num_candidate_marks})', color='#c4a35a', fontsize=14, fontweight='bold')
for spine in ax.spines.values():
    spine.set_color('#c4a35a')
ax.tick_params(colors='white')
marks_path = os.path.join(OUTPUT_DIR, 'dorsal_marks.png')
plt.savefig(marks_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(ANALYSIS_DIR, 'dorsal_marks.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(DOCS_IMAGES_DIR, 'dorsal_marks.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"Saved marks overlay")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.imshow(binary_mask, cmap='binary')
ax.set_title('Thresholded Residual (Back)', color='#c4a35a', fontsize=14, fontweight='bold')
for spine in ax.spines.values():
    spine.set_color('#c4a35a')
ax.tick_params(colors='white')
threshold_path = os.path.join(OUTPUT_DIR, 'dorsal_threshold.png')
plt.savefig(threshold_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(ANALYSIS_DIR, 'dorsal_threshold.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(DOCS_IMAGES_DIR, 'dorsal_threshold.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"Saved threshold map")

scourge_regions = {
    'left_shoulder': {'x': 75, 'y': 80, 'radius': 20},
    'right_shoulder': {'x': 225, 'y': 80, 'radius': 20},
    'upper_back_left': {'x': 85, 'y': 140, 'radius': 25},
    'upper_back_right': {'x': 215, 'y': 140, 'radius': 25},
    'mid_back_left': {'x': 80, 'y': 200, 'radius': 25},
    'mid_back_right': {'x': 220, 'y': 200, 'radius': 25},
    'lower_back_left': {'x': 85, 'y': 280, 'radius': 25},
    'lower_back_right': {'x': 215, 'y': 280, 'radius': 25}
}

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.imshow(dorsal_down, cmap='gray', vmin=0, vmax=255)
for region_name, props in scourge_regions.items():
    circle = plt.Circle((props['x'], props['y']), radius=props['radius'], 
                        fill=False, color='#c4a35a', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.annotate(region_name.replace('_', '\n'), (props['x'], props['y']-props['radius']-10), 
                color='#c4a35a', fontsize=7, ha='center', va='top')
ax.set_title('Scourge Mark Reference Locations (Back)', color='#c4a35a', fontsize=14, fontweight='bold')
for spine in ax.spines.values():
    spine.set_color('#c4a35a')
ax.tick_params(colors='white')
ref_path = os.path.join(OUTPUT_DIR, 'dorsal_reference_locations.png')
plt.savefig(ref_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(ANALYSIS_DIR, 'dorsal_reference_locations.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.savefig(os.path.join(DOCS_IMAGES_DIR, 'dorsal_reference_locations.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print(f"Saved reference locations")

image_files = [dorsal_out_path, heatmap_path, surface3d_path, residual_path, marks_path, threshold_path, ref_path]

results = {
    'num_candidate_marks': num_candidate_marks,
    'mark_locations': mark_locations,
    'mark_size_stats': {
        'mean': mark_size_mean,
        'std': mark_size_std,
        'min': mark_size_min,
        'max': mark_size_max
    },
    'back_depth_stats': {
        'mean': back_depth_mean,
        'std': back_depth_std,
        'min': back_depth_min,
        'max': back_depth_max
    },
    'image_files': [f.replace(PROJECT_ROOT, '').lstrip('\\').lstrip('/') for f in image_files]
}

json_path = os.path.join(TASK_RESULTS_DIR, 'write_a_python_script_that_extracts_the_results.json')
json.dump(results, open(json_path, 'w'), indent=2)
print(f"\nResults saved to: {json_path}")
print("\nJSON output:")
print(json.dumps(results, indent=2))