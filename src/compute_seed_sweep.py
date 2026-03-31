"""Seed sweep analysis — produces structured JSON results.

Renders Lambertian clay-shaded images from the Miller depth map at 100
different lighting angles, then runs Haar cascade face detection on each.
Reports honest results — if no faces are detected, that's what we report.
"""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "output" / "seed_sweep"
RESULTS_JSON = PROJECT / "output" / "task_results" / "seed_sweep_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

# Find the Miller depth map
MILLER_DIR = PROJECT / "output" / "highres_miller"
candidates = list(MILLER_DIR.glob("depth_300x300*.npy"))
if not candidates:
    raise FileNotFoundError(f"No 300x300 .npy files found in {MILLER_DIR}")
DEPTH_PATH = candidates[0]
print(f"Using Miller depth: {DEPTH_PATH.name}")

depth = np.load(DEPTH_PATH).astype(np.float64)
print(f"Loaded: {depth.shape}, range [{depth.min():.1f}, {depth.max():.1f}]")

# Normalize depth to 0-1
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

# Compute surface normals from depth gradient
dy, dx = np.gradient(depth_norm)
normals = np.stack([-dx, -dy, np.ones_like(depth_norm)], axis=-1)
norms = np.linalg.norm(normals, axis=-1, keepdims=True)
normals = normals / (norms + 1e-8)

# Haar cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Clay color
CLAY_RGB = np.array([180, 175, 168], dtype=np.float64) / 255.0
AMBIENT = 0.3
DIFFUSE = 0.7

N_SEEDS = 100
results_list = []

print(f"Running {N_SEEDS} lighting angles...")
for seed in range(N_SEEDS):
    # Spherical lighting: azimuth from seed, fixed elevation
    azimuth = seed * 3.6  # 0-360 degrees over 100 seeds
    elevation = 45.0  # fixed
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)

    light_dir = np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad),
    ])

    # Lambertian shading
    dot = np.sum(normals * light_dir, axis=-1)
    dot = np.clip(dot, 0, 1)
    shade = AMBIENT + DIFFUSE * dot

    # Apply clay color
    img = np.stack([shade * CLAY_RGB[0], shade * CLAY_RGB[1], shade * CLAY_RGB[2]], axis=-1)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Convert to grayscale for Haar detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Run face detection with multiple scale factors
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50),
    )

    num_detections = len(faces)
    # Score: use number of detections as primary, sum of confidence areas as secondary
    if num_detections > 0:
        score = float(num_detections)
        # Save image with detections drawn
        img_annotated = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        score = 0.0
        img_annotated = img

    results_list.append({
        "seed": seed,
        "azimuth_deg": round(azimuth, 1),
        "elevation_deg": elevation,
        "num_detections": num_detections,
        "haar_score": score,
    })

    # Save the rendered image (we'll prune later)
    img_path = OUT_DIR / f"render_seed{seed:03d}.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if seed % 20 == 0:
        print(f"  Seed {seed}: azimuth={azimuth:.1f}°, detections={num_detections}, score={score}")

# Sort by score (descending), then by seed
results_list.sort(key=lambda r: (-r["haar_score"], r["seed"]))

# Keep top 10 images, delete the rest
top_10 = results_list[:10]
top_seeds = {r["seed"] for r in top_10}
deleted = 0
for seed in range(N_SEEDS):
    img_path = OUT_DIR / f"render_seed{seed:03d}.png"
    if seed not in top_seeds and img_path.exists():
        img_path.unlink()
        deleted += 1

print(f"\nDeleted {deleted} non-top images, kept {len(top_seeds)}")

# Add image paths to top 10
top_10_with_paths = []
for r in top_10:
    r["image_path"] = str((OUT_DIR / f"render_seed{r['seed']:03d}.png").relative_to(PROJECT))
    top_10_with_paths.append(r)

# Detection rate
total_detections = sum(1 for r in results_list if r["haar_score"] > 0)
detection_rate = total_detections / N_SEEDS
mean_score = float(np.mean([r["haar_score"] for r in results_list]))

print(f"\nDetection rate: {total_detections}/{N_SEEDS} = {detection_rate*100:.1f}%")
print(f"Mean score: {mean_score:.4f}")
print(f"\nTop 10:")
for r in top_10:
    print(f"  Seed {r['seed']}: azimuth={r['azimuth_deg']}°, "
          f"detections={r['num_detections']}, score={r['haar_score']}")

# Generate contact sheet of top 10 (2x5)
BG = '#1a1a1a'
GOLD = '#c4a35a'
WHITE = 'white'
image_paths = []

if top_10[0]["haar_score"] > 0:
    # Only make contact sheet if there are actual detections
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), facecolor=BG)
    for idx, (ax, r) in enumerate(zip(axes.flat, top_10)):
        img_path = OUT_DIR / f"render_seed{r['seed']:03d}.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(f"Seed {r['seed']}\n{r['azimuth_deg']}° | det={r['num_detections']}",
                      color=GOLD, fontsize=9, fontweight='bold')
        ax.axis('off')
    # Fill remaining slots if fewer than 10
    for ax in axes.flat[len(top_10):]:
        ax.axis('off')
    fig.suptitle(f'Top 10 Seeds by Haar Detection ({detection_rate*100:.0f}% detection rate)',
                 color=GOLD, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / 'seed_sweep_contact.png'
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    image_paths.append(str(path.relative_to(PROJECT)))
else:
    # Zero detections — make a contact sheet anyway showing the top 10 by seed order
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), facecolor=BG)
    # Just show first 10 seeds
    for idx, ax in enumerate(axes.flat):
        img_path = OUT_DIR / f"render_seed{idx:03d}.png"
        if not img_path.exists():
            # Use any available render
            available = sorted(OUT_DIR.glob("render_seed*.png"))
            if idx < len(available):
                img_path = available[idx]
            else:
                ax.axis('off')
                continue
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(f"Seed {idx}\n{idx*3.6:.0f}°",
                      color=GOLD, fontsize=9, fontweight='bold')
        ax.axis('off')
    fig.suptitle(f'Seed Sweep Samples (0/{N_SEEDS} Haar detections — Lambertian renders '
                 f'are not photo-realistic enough for cascade detection)',
                 color=GOLD, fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / 'seed_sweep_contact.png'
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    image_paths.append(str(path.relative_to(PROJECT)))

# Azimuth distribution plot
fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
ax.set_facecolor(BG)
azimuths = [r["azimuth_deg"] for r in results_list]
scores = [r["haar_score"] for r in results_list]
ax.bar(azimuths, scores, width=3, color=GOLD, alpha=0.8)
ax.set_xlabel('Azimuth (°)', color=WHITE)
ax.set_ylabel('Haar Score', color=WHITE)
ax.set_title(f'Face Detection Score vs Lighting Angle ({N_SEEDS} seeds)',
             color=GOLD, fontsize=12, fontweight='bold')
ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_color(WHITE)
plt.tight_layout()
path = OUT_DIR / 'seed_sweep_scores.png'
fig.savefig(path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(path.relative_to(PROJECT)))

# Add image paths to all top_10 entries
for r in top_10_with_paths:
    if r["image_path"] not in image_paths:
        image_paths.append(r["image_path"])

results = {
    "depth_map": str(DEPTH_PATH.relative_to(PROJECT)),
    "depth_shape": list(depth.shape),
    "seeds_tested": N_SEEDS,
    "elevation_deg": 45.0,
    "azimuth_range_deg": "0-360",
    "clay_color_rgb": [180, 175, 168],
    "ambient": AMBIENT,
    "diffuse": DIFFUSE,
    "detection_rate": round(detection_rate, 4),
    "mean_score": round(mean_score, 4),
    "total_detections": total_detections,
    "top_10": top_10_with_paths,
    "note": ("Haar cascade on Lambertian-shaded depth maps. Low detection "
             "rates are expected — clay renders lack the texture and contrast "
             "that cascade classifiers are trained on."),
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
