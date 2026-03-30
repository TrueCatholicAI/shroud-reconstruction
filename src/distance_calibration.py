"""Task J: Cloth-to-body distance function calibration using anatomical landmarks."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

print("=== Cloth-to-Body Distance Function Calibration ===")

# Load both depth maps
# Enrie: downsample from full smooth depth to 150x150
enrie_full = np.load('data/processed/depth_map_smooth_15.npy')  # 3000x2388
enrie_depth = cv2.resize(enrie_full.astype(np.float32), (150, 150), interpolation=cv2.INTER_AREA)
miller_depth = np.load('output/study2_miller/depth_150x150_g15.npy')

print(f"Enrie depth: {enrie_depth.shape}, range [{enrie_depth.min():.0f}, {enrie_depth.max():.0f}]")
print(f"Miller depth: {miller_depth.shape}, range [{miller_depth.min():.0f}, {miller_depth.max():.0f}]")

# === Anatomical landmark positions with estimated cloth-to-body distances ===
# Based on forensic anthropology: known approximate distances from a face surface
# to a draped cloth at key anatomical points.
# The Shroud would drape over the highest points (nose tip, brow) and fall away
# at recessed areas (eye sockets, temples, neck sides).
#
# Estimated distances in mm for a male face (~23cm long, ~15cm wide):
# These are approximate and based on published draping studies (Jackson 1984)
landmarks_150 = {
    # (row, col) in 150x150 grid, estimated_distance_mm
    'nose_tip':        {'pos': (68, 75),  'dist_mm': 0},      # Contact point
    'nose_bridge':     {'pos': (58, 75),  'dist_mm': 3},      # Slight standoff
    'brow_center':     {'pos': (48, 75),  'dist_mm': 5},      # Forehead curves back
    'left_eye':        {'pos': (55, 62),  'dist_mm': 12},     # Eye socket depression
    'right_eye':       {'pos': (55, 88),  'dist_mm': 12},     # Eye socket depression
    'left_cheek':      {'pos': (70, 55),  'dist_mm': 8},      # Cheek surface
    'right_cheek':     {'pos': (70, 95),  'dist_mm': 8},      # Cheek surface
    'upper_lip':       {'pos': (78, 75),  'dist_mm': 5},      # Lips protrude slightly
    'chin':            {'pos': (92, 75),  'dist_mm': 10},     # Chin drops away
    'left_temple':     {'pos': (50, 45),  'dist_mm': 20},     # Temple recedes
    'right_temple':    {'pos': (50, 105), 'dist_mm': 20},     # Temple recedes
    'forehead_top':    {'pos': (35, 75),  'dist_mm': 15},     # Forehead curves away
    'left_jaw':        {'pos': (85, 50),  'dist_mm': 18},     # Jaw angle
    'right_jaw':       {'pos': (85, 100), 'dist_mm': 18},     # Jaw angle
    'neck_center':     {'pos': (105, 75), 'dist_mm': 25},     # Neck drops away
}

# Sample intensity at each landmark (use 3x3 average for robustness)
def sample_intensity(depth_map, row, col, window=3):
    h, w = depth_map.shape
    half = window // 2
    r1, r2 = max(0, row-half), min(h, row+half+1)
    c1, c2 = max(0, col-half), min(w, col+half+1)
    return float(np.mean(depth_map[r1:r2, c1:c2]))

print("\n--- Landmark Sampling ---")
enrie_data = []
miller_data = []

for name, info in landmarks_150.items():
    r, c = info['pos']
    d = info['dist_mm']
    e_val = sample_intensity(enrie_depth, r, c)
    m_val = sample_intensity(miller_depth, r, c)
    enrie_data.append((name, d, e_val))
    miller_data.append((name, d, m_val))
    print(f"  {name:18s}: dist={d:3d}mm  Enrie={e_val:6.1f}  Miller={m_val:6.1f}")

# Extract arrays
distances = np.array([d for _, d, _ in enrie_data])
enrie_intensities = np.array([v for _, _, v in enrie_data])
miller_intensities = np.array([v for _, _, v in miller_data])

# Normalize intensities to [0, 1]
enrie_norm = (enrie_intensities - enrie_intensities.min()) / (enrie_intensities.max() - enrie_intensities.min())
miller_norm = (miller_intensities - miller_intensities.min()) / (miller_intensities.max() - miller_intensities.min())

# Normalize distances to [0, 1]
dist_norm = distances / distances.max()

# === Fit candidate functions ===
# 1. Linear: I = a - b*d
def linear(d, a, b):
    return a - b * d

# 2. Exponential: I = a * exp(-k*d)
def exponential(d, a, k):
    return a * np.exp(-k * d)

# 3. Inverse square: I = a / (1 + k*d^2)
def inverse_sq(d, a, k):
    return a / (1 + k * d**2)

# 4. Gaussian: I = a * exp(-d^2 / (2*s^2))
def gaussian_func(d, a, s):
    return a * np.exp(-d**2 / (2 * s**2))

models = {
    'Linear': (linear, [1.0, 1.0]),
    'Exponential': (exponential, [1.0, 2.0]),
    'Inverse Square': (inverse_sq, [1.0, 5.0]),
    'Gaussian': (gaussian_func, [1.0, 0.5]),
}

print("\n--- Curve Fitting Results ---")
results = {}

for study_name, (intensities_norm, label) in [('Enrie', (enrie_norm, 'Enrie 1931')),
                                                ('Miller', (miller_norm, 'Miller 1978'))]:
    print(f"\n  {label}:")
    results[study_name] = {}
    for model_name, (func, p0) in models.items():
        try:
            popt, pcov = curve_fit(func, dist_norm, intensities_norm, p0=p0, maxfev=10000)
            predicted = func(dist_norm, *popt)
            ss_res = np.sum((intensities_norm - predicted) ** 2)
            ss_tot = np.sum((intensities_norm - np.mean(intensities_norm)) ** 2)
            r2 = 1 - ss_res / ss_tot
            results[study_name][model_name] = {'popt': popt, 'r2': r2, 'func': func}
            print(f"    {model_name:16s}: R2 = {r2:.4f}  params = {popt}")
        except Exception as e:
            print(f"    {model_name:16s}: FAILED ({e})")
            results[study_name][model_name] = {'r2': 0, 'func': func, 'popt': p0}

# === Visualization 1: Scatter + fits ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#1a1a1a')

colors = {'Linear': '#e74c3c', 'Exponential': '#3498db', 'Inverse Square': '#2ecc71', 'Gaussian': '#9b59b6'}
d_smooth = np.linspace(0, 1, 200)

for ax, (study_name, (intensities_norm, label)) in zip(axes,
    [('Enrie', (enrie_norm, 'Enrie 1931')), ('Miller', (miller_norm, 'Miller 1978'))]):

    ax.set_facecolor('#222')

    # Data points with landmark labels
    ax.scatter(dist_norm, intensities_norm, c='#c4a35a', s=60, zorder=5, edgecolors='white', linewidths=0.5)

    # Label a few key points
    for i, (name, _, _) in enumerate(enrie_data):
        if name in ['nose_tip', 'left_eye', 'chin', 'neck_center', 'left_temple']:
            ax.annotate(name.replace('_', ' '), (dist_norm[i], intensities_norm[i]),
                       textcoords='offset points', xytext=(8, 5),
                       fontsize=7, color='#aaa')

    # Fitted curves
    for model_name in models:
        if model_name in results[study_name] and results[study_name][model_name]['r2'] > 0:
            r = results[study_name][model_name]
            y_fit = r['func'](d_smooth, *r['popt'])
            ax.plot(d_smooth, y_fit, color=colors[model_name], linewidth=2,
                    label=f"{model_name} (R²={r['r2']:.3f})")

    ax.set_xlabel('Normalized Distance (cloth-to-body)', color='white')
    ax.set_ylabel('Normalized Intensity', color='white')
    ax.set_title(label, color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#333', labelcolor='white', fontsize=9)
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.15)

fig.suptitle('Intensity vs Cloth-to-Body Distance — Anatomical Landmark Calibration',
             color='#c4a35a', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/distance_calibration_fits.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: distance_calibration_fits.png")

# === Visualization 2: Landmark positions on depth maps ===
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('#1a1a1a')

for ax, (depth, label) in zip(axes, [(enrie_depth, 'Enrie 1931'), (miller_depth, 'Miller 1978')]):
    depth_rgb = cv2.cvtColor(cv2.normalize(depth.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2RGB)

    for name, info in landmarks_150.items():
        r, c = info['pos']
        d = info['dist_mm']
        # Color by distance: green=close, red=far
        frac = d / 25.0
        color = (int(255 * frac), int(255 * (1 - frac)), 0)
        cv2.circle(depth_rgb, (c, r), 4, color, -1)
        cv2.circle(depth_rgb, (c, r), 5, (255, 255, 255), 1)

    ax.imshow(depth_rgb)
    ax.set_title(f'{label} — Calibration Landmarks\n(green=contact, red=far)', color='white', fontsize=11)
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')

fig.suptitle('Anatomical Calibration Points on Depth Maps', color='#c4a35a', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/distance_calibration_landmarks.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: distance_calibration_landmarks.png")

# === Visualization 3: Residual comparison ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a1a')

for ax, (study_name, intensities_norm, label) in zip(axes,
    [('Enrie', enrie_norm, 'Enrie 1931'), ('Miller', miller_norm, 'Miller 1978')]):

    ax.set_facecolor('#222')
    x_pos = np.arange(len(landmarks_150))
    width = 0.2

    for i, model_name in enumerate(models):
        if model_name in results[study_name] and results[study_name][model_name]['r2'] > 0:
            r = results[study_name][model_name]
            predicted = r['func'](dist_norm, *r['popt'])
            residuals = intensities_norm - predicted
            ax.bar(x_pos + i * width, residuals, width, color=colors[model_name],
                   alpha=0.7, label=model_name)

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels([n.replace('_', '\n') for n in landmarks_150.keys()],
                       rotation=45, ha='right', fontsize=6, color='#aaa')
    ax.set_ylabel('Residual (observed - predicted)', color='white')
    ax.set_title(f'{label} — Fit Residuals by Landmark', color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.axhline(y=0, color='#555', linewidth=0.5)
    ax.legend(facecolor='#333', labelcolor='white', fontsize=8)
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Model Residuals at Each Anatomical Landmark', color='#c4a35a', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/distance_calibration_residuals.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: distance_calibration_residuals.png")

# === Summary ===
print("\n--- R² Summary ---")
print(f"{'Model':<18s} {'Enrie R²':>10s} {'Miller R²':>10s} {'Mean R²':>10s}")
print("-" * 50)
for model_name in models:
    e_r2 = results['Enrie'].get(model_name, {}).get('r2', 0)
    m_r2 = results['Miller'].get(model_name, {}).get('r2', 0)
    mean_r2 = (e_r2 + m_r2) / 2
    print(f"{model_name:<18s} {e_r2:>10.4f} {m_r2:>10.4f} {mean_r2:>10.4f}")

# Best model
best_enrie = max(results['Enrie'].items(), key=lambda x: x[1]['r2'])
best_miller = max(results['Miller'].items(), key=lambda x: x[1]['r2'])
print(f"\nBest fit Enrie: {best_enrie[0]} (R²={best_enrie[1]['r2']:.4f})")
print(f"Best fit Miller: {best_miller[0]} (R²={best_miller[1]['r2']:.4f})")

print("\nMETHODOLOGICAL NOTES:")
print("- Landmark positions are approximate (±3-5 pixels in 150x150 grid)")
print("- Cloth-to-body distances are estimated from published draping studies")
print("- Not ground truth — these are model-dependent estimates")
print("- The key result is which functional form best describes the relationship,")
print("  not the absolute distance values")

print("\n=== Distance Calibration Complete ===")
