"""Task E: Image formation distance function — fit intensity vs distance relationship."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from scipy.stats import pearsonr

print("=== Image Formation Distance Function Analysis ===")

# Load Enrie data
enrie_full = np.load('data/processed/depth_map_smooth_15.npy')  # 3000x2388
enrie_raw = np.load('data/processed/depth_map.npy')  # raw CLAHE
enrie_src = cv2.imread('data/source/enrie_1931_face_hires.jpg', cv2.IMREAD_GRAYSCALE)

# Load Miller data
miller_150 = np.load('output/study2_miller/depth_150x150_g15.npy')
miller_full_depth = np.load('output/study2_miller/depth_map.npy')
miller_src = cv2.imread('data/source/vernon_miller/34c-Fa-N_0414.jpg', cv2.IMREAD_GRAYSCALE)

print(f"Enrie source: {enrie_src.shape}")
print(f"Miller source: {miller_src.shape}")

# === Extract centerline intensity profiles ===
# For the Enrie 150x150 analysis map
h, w = enrie_full.shape
enrie_150 = zoom(enrie_full.astype(np.float64), (150/h, 150/w), order=1)

# Find midline (center column)
mid_x = 75  # center of 150x150

# Average a 5-pixel strip around midline for noise reduction
strip = 2
enrie_profile = enrie_150[:, mid_x-strip:mid_x+strip+1].mean(axis=1)
miller_profile = miller_150[:, mid_x-strip:mid_x+strip+1].mean(axis=1).astype(np.float64)

print(f"Enrie centerline: {enrie_profile.shape}, range [{enrie_profile.min():.1f}, {enrie_profile.max():.1f}]")
print(f"Miller centerline: {miller_profile.shape}, range [{miller_profile.min():.1f}, {miller_profile.max():.1f}]")

# === The VP-8 assumption: brightness IS distance ===
# In the Shroud negative:
#   - Brighter = closer to cloth (shorter distance)
#   - Darker = farther from cloth (longer distance)
#
# So: intensity I is proportional to some function of distance d
#   I = f(d) where d is estimated as max_intensity - observed_intensity
#
# We extract the nose ridge region (the strongest, most reliable feature)
# and fit various functions.

# Identify nose ridge region in the centerline profile
# The nose tip is the brightest peak in the center third of the profile
center_region = enrie_profile[50:100]  # rows 50-100 in 150x150
nose_peak_local = np.argmax(center_region)
nose_peak = nose_peak_local + 50
print(f"\nEnrie nose peak at row {nose_peak}, intensity {enrie_profile[nose_peak]:.1f}")

# Extract a broader region around the nose for fitting: brow to chin
# Roughly rows 30-130 in 150x150
fit_start, fit_end = 25, 130
y_range = np.arange(fit_start, fit_end)
enrie_fit_data = enrie_profile[fit_start:fit_end]
miller_fit_data = miller_profile[fit_start:fit_end]

# Convert intensity to "estimated distance"
# In the VP-8 model: higher intensity = closer = smaller distance
# distance = max_intensity - intensity (so nose tip has distance ~0)
enrie_max_I = enrie_fit_data.max()
enrie_distance = enrie_max_I - enrie_fit_data  # 0 at nose tip, higher at background
enrie_intensity = enrie_fit_data  # raw intensity

miller_max_I = miller_fit_data.max()
miller_distance = miller_max_I - miller_fit_data
miller_intensity = miller_fit_data

# Normalize distance to [0, 1]
enrie_d_norm = enrie_distance / (enrie_distance.max() + 1e-8)
enrie_i_norm = enrie_intensity / (enrie_intensity.max() + 1e-8)

miller_d_norm = miller_distance / (miller_distance.max() + 1e-8)
miller_i_norm = miller_intensity / (miller_intensity.max() + 1e-8)

# === Fit candidate functions ===
# All models: I = f(d) where I is normalized intensity, d is normalized distance
# f(d) = 1 - d  (linear, VP-8 assumption)
# f(d) = exp(-k*d)  (exponential decay)
# f(d) = 1/(1 + k*d^2)  (inverse square)
# f(d) = (1-d)^n  (power law)

def linear(d, a, b):
    return a - b * d

def exponential(d, a, k):
    return a * np.exp(-k * d)

def inverse_square(d, a, k):
    return a / (1 + k * d**2)

def power_law(d, a, n):
    return a * (1 - d + 1e-8)**n

models = {
    'Linear (VP-8)': (linear, [1.0, 1.0]),
    'Exponential': (exponential, [1.0, 2.0]),
    'Inverse Square': (inverse_square, [1.0, 5.0]),
    'Power Law': (power_law, [1.0, 1.5]),
}

print("\n--- Curve Fitting Results ---")
results = {}

for study_name, d_data, i_data in [('Enrie 1931', enrie_d_norm, enrie_i_norm),
                                      ('Miller 1978', miller_d_norm, miller_i_norm)]:
    print(f"\n{study_name}:")
    study_results = {}

    for model_name, (func, p0) in models.items():
        try:
            popt, pcov = curve_fit(func, d_data, i_data, p0=p0, maxfev=10000)
            predicted = func(d_data, *popt)
            residuals = i_data - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((i_data - i_data.mean())**2)
            r_squared = 1 - ss_res / ss_tot
            rmse = np.sqrt(np.mean(residuals**2))

            study_results[model_name] = {
                'params': popt,
                'r_squared': r_squared,
                'rmse': rmse,
                'predicted': predicted,
            }
            param_str = ', '.join([f'{p:.4f}' for p in popt])
            print(f"  {model_name:20s}: R2={r_squared:.4f}, RMSE={rmse:.4f}, params=[{param_str}]")
        except Exception as e:
            print(f"  {model_name:20s}: FAILED ({e})")
            study_results[model_name] = None

    results[study_name] = study_results

# === Visualization 1: Centerline profiles ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a1a')

axes[0].plot(enrie_profile, color='#c4a35a', linewidth=1.5)
axes[0].set_title('Enrie 1931 Centerline', color='white', fontsize=13)
axes[0].set_xlabel('Row (0=top)', color='white')
axes[0].set_ylabel('Intensity', color='white')
axes[0].axvline(x=nose_peak, color='red', linestyle='--', alpha=0.5, label='Nose peak')
axes[0].legend(facecolor='#333')

axes[1].plot(miller_profile, color='#c4a35a', linewidth=1.5)
axes[1].set_title('Miller 1978 Centerline', color='white', fontsize=13)
axes[1].set_xlabel('Row (0=top)', color='white')
axes[1].set_ylabel('Intensity', color='white')

for ax in axes:
    ax.set_facecolor('#222')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Nose Ridge Centerline Intensity Profiles', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/centerline_profiles.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: centerline_profiles.png")

# === Visualization 2: Curve fits ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#1a1a1a')

colors_fit = {'Linear (VP-8)': '#e74c3c', 'Exponential': '#3498db',
              'Inverse Square': '#2ecc71', 'Power Law': '#e67e22'}

for ax, (study_name, d_data, i_data) in zip(axes,
    [('Enrie 1931', enrie_d_norm, enrie_i_norm), ('Miller 1978', miller_d_norm, miller_i_norm)]):

    ax.scatter(d_data, i_data, c='#c4a35a', s=8, alpha=0.6, label='Data', zorder=1)

    d_smooth = np.linspace(0, 1, 200)
    for model_name in models:
        res = results[study_name].get(model_name)
        if res is not None:
            func = models[model_name][0]
            pred = func(d_smooth, *res['params'])
            ax.plot(d_smooth, pred, color=colors_fit[model_name], linewidth=2,
                    label=f"{model_name} (R2={res['r_squared']:.3f})", zorder=2)

    ax.set_title(study_name, color='white', fontsize=13)
    ax.set_xlabel('Normalized Distance (0=closest)', color='white')
    ax.set_ylabel('Normalized Intensity', color='white')
    ax.legend(fontsize=9, facecolor='#333', labelcolor='white')
    ax.set_facecolor('#222')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Intensity vs Distance: Candidate Function Fits', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/formation_function_fits.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: formation_function_fits.png")

# === Summary ===
print("\n" + "="*60)
print("FORMATION FUNCTION SUMMARY")
print("="*60)
for study_name in ['Enrie 1931', 'Miller 1978']:
    print(f"\n{study_name}:")
    best_r2 = -1
    best_model = None
    for model_name, res in results[study_name].items():
        if res is not None and res['r_squared'] > best_r2:
            best_r2 = res['r_squared']
            best_model = model_name
    print(f"  Best fit: {best_model} (R2={best_r2:.4f})")

    for model_name in ['Linear (VP-8)', 'Exponential', 'Inverse Square', 'Power Law']:
        res = results[study_name].get(model_name)
        if res:
            print(f"  {model_name:20s}: R2={res['r_squared']:.4f}")

print("\n=== Formation Analysis Complete ===")
