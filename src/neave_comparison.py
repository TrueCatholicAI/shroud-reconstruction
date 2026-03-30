"""Task Q: Comparison with Neave 2001 BBC reconstruction and population averages."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json

print("=== Neave Reconstruction Comparison ===")

# === Measurement data ===
# Shroud Study 1 (Enrie 1931) — from approved_measurements.json
# Shroud Study 2 (Miller 1978) — from study2 analysis
# Neave 2001 — Richard Neave's forensic reconstruction for BBC
#   Based on first-century Judean skulls from archaeological excavations
#   Published proportions from the documentary and subsequent publications
# Modern population averages — forensic anthropology reference ranges

measurements = {
    'Interpupillary Distance': {
        'study1': 5.45,
        'study2': 6.30,
        'neave': 6.2,       # Estimated from Neave reconstruction proportions
        'modern_low': 5.5,
        'modern_high': 7.5,
        'unit': 'cm',
    },
    'Inner Eye Distance': {
        'study1': 2.87,
        'study2': 2.56,
        'neave': 3.1,       # Wider nasal bridge typical of Middle Eastern populations
        'modern_low': 2.8,
        'modern_high': 3.5,
        'unit': 'cm',
    },
    'Nose Width': {
        'study1': 3.59,
        'study2': 3.15,
        'neave': 3.8,       # Broader nose typical of Levantine populations
        'modern_low': 2.5,
        'modern_high': 5.0,
        'unit': 'cm',
    },
    'Face Width': {
        'study1': 16.50,
        'study2': 12.60,
        'neave': 14.0,      # Based on first-century Judean skull bizygomatic breadth
        'modern_low': 12.0,
        'modern_high': 17.0,
        'unit': 'cm',
    },
    'Jaw Width': {
        'study1': 12.91,
        'study2': 11.03,
        'neave': 11.5,      # Moderate mandibular breadth
        'modern_low': 10.0,
        'modern_high': 15.0,
        'unit': 'cm',
    },
    'Mouth Width': {
        'study1': 4.30,
        'study2': 3.94,
        'neave': 5.0,       # Wider mouth typical of reconstructions from this region
        'modern_low': 4.0,
        'modern_high': 6.5,
        'unit': 'cm',
    },
    'Nose-to-Chin': {
        'study1': 9.91,     # Outlier due to cloth draping
        'study2': 7.09,
        'neave': 7.5,       # Lower face height from skull
        'modern_low': 7.0,
        'modern_high': 9.5,
        'unit': 'cm',
    },
}

# Print comparison table
print(f"\n{'Measurement':<25s} {'Study 1':>8s} {'Study 2':>8s} {'Neave':>8s} {'Modern Range':>14s}")
print("-" * 70)
for name, data in measurements.items():
    print(f"{name:<25s} {data['study1']:>7.2f} {data['study2']:>7.2f} {data['neave']:>7.1f}  {data['modern_low']:.1f}-{data['modern_high']:.1f} {data['unit']}")

# === Radar chart ===
categories = list(measurements.keys())
N = len(categories)

# Normalize all values to [0, 1] based on modern range
def normalize(val, low, high):
    return (val - low) / (high - low) if high > low else 0.5

study1_norm = [normalize(measurements[c]['study1'], measurements[c]['modern_low'], measurements[c]['modern_high']) for c in categories]
study2_norm = [normalize(measurements[c]['study2'], measurements[c]['modern_low'], measurements[c]['modern_high']) for c in categories]
neave_norm = [normalize(measurements[c]['neave'], measurements[c]['modern_low'], measurements[c]['modern_high']) for c in categories]
modern_mid_norm = [0.5] * N  # Center of range

# Close the polygon
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
study1_norm += study1_norm[:1]
study2_norm += study2_norm[:1]
neave_norm += neave_norm[:1]
modern_mid_norm += modern_mid_norm[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

# Fill modern range band
inner_ring = [0.0] * (N + 1)
outer_ring = [1.0] * (N + 1)
ax.fill(angles, outer_ring, alpha=0.08, color='white')
ax.fill(angles, [0.5] * (N + 1), alpha=0.05, color='white')

# Plot each source
ax.plot(angles, study1_norm, 'o-', color='#e74c3c', linewidth=2, markersize=6, label='Shroud Study 1 (Enrie)')
ax.fill(angles, study1_norm, alpha=0.1, color='#e74c3c')

ax.plot(angles, study2_norm, 's-', color='#3498db', linewidth=2, markersize=6, label='Shroud Study 2 (Miller)')
ax.fill(angles, study2_norm, alpha=0.1, color='#3498db')

ax.plot(angles, neave_norm, '^-', color='#2ecc71', linewidth=2, markersize=7, label='Neave 2001 Reconstruction')
ax.fill(angles, neave_norm, alpha=0.1, color='#2ecc71')

ax.plot(angles, modern_mid_norm, '--', color='#888', linewidth=1, alpha=0.5, label='Modern population midpoint')

# Labels
ax.set_xticks(angles[:-1])
short_labels = [c.replace(' Distance', '').replace('Interpupillary', 'IPD') for c in categories]
ax.set_xticklabels(short_labels, color='#ccc', fontsize=9)
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['Low', '', 'Mid', '', 'High'], color='#666', fontsize=7)
ax.spines['polar'].set_color('#444')
ax.grid(color='#333', linewidth=0.5)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#222', labelcolor='white',
          fontsize=9, framealpha=0.9)

fig.suptitle('Facial Proportions: Shroud vs Neave vs Modern Averages',
             color='#c4a35a', fontsize=15, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/neave_radar_chart.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: neave_radar_chart.png")

# === Bar chart comparison ===
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#222')

x = np.arange(N)
width = 0.22

bars1 = ax.bar(x - width * 1.2, [measurements[c]['study1'] for c in categories], width,
               color='#e74c3c', alpha=0.85, label='Shroud Study 1')
bars2 = ax.bar(x, [measurements[c]['study2'] for c in categories], width,
               color='#3498db', alpha=0.85, label='Shroud Study 2')
bars3 = ax.bar(x + width * 1.2, [measurements[c]['neave'] for c in categories], width,
               color='#2ecc71', alpha=0.85, label='Neave 2001')

# Modern range as error bars on Neave
for i, c in enumerate(categories):
    mid = (measurements[c]['modern_low'] + measurements[c]['modern_high']) / 2
    half_range = (measurements[c]['modern_high'] - measurements[c]['modern_low']) / 2
    ax.errorbar(i + width * 1.2, mid, yerr=half_range, fmt='none', ecolor='#888',
                elinewidth=1.5, capsize=4, capthick=1.5, alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels([c.replace(' Distance', '').replace('Interpupillary', 'IPD') for c in categories],
                    rotation=30, ha='right', color='#ccc', fontsize=9)
ax.set_ylabel('Measurement (cm)', color='white', fontsize=11)
ax.tick_params(colors='white')
ax.legend(facecolor='#333', labelcolor='white', fontsize=10)
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Measurement Comparison: Shroud Studies vs Neave Forensic Reconstruction',
             color='#c4a35a', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/neave_bar_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: neave_bar_comparison.png")

# === Concordance analysis ===
print("\n--- Concordance Analysis ---")
concordance_s1 = 0
concordance_s2 = 0
concordance_neave = 0
for name, data in measurements.items():
    in_s1 = data['modern_low'] <= data['study1'] <= data['modern_high']
    in_s2 = data['modern_low'] <= data['study2'] <= data['modern_high']
    in_neave = data['modern_low'] <= data['neave'] <= data['modern_high']
    if in_s1: concordance_s1 += 1
    if in_s2: concordance_s2 += 1
    if in_neave: concordance_neave += 1
    s1_flag = "OK" if in_s1 else "OUT"
    s2_flag = "OK" if in_s2 else "OUT"
    n_flag = "OK" if in_neave else "OUT"
    print(f"  {name:<25s}: S1={s1_flag:3s}  S2={s2_flag:3s}  Neave={n_flag:3s}")

print(f"\nStudy 1: {concordance_s1}/{N} in modern range")
print(f"Study 2: {concordance_s2}/{N} in modern range")
print(f"Neave:   {concordance_neave}/{N} in modern range")

# Distance between Shroud measurements and Neave
diffs_s1 = [abs(measurements[c]['study1'] - measurements[c]['neave']) for c in categories]
diffs_s2 = [abs(measurements[c]['study2'] - measurements[c]['neave']) for c in categories]
print(f"\nMean |Study 1 - Neave|: {np.mean(diffs_s1):.2f} cm")
print(f"Mean |Study 2 - Neave|: {np.mean(diffs_s2):.2f} cm")
print(f"Study 2 is closer to Neave in {sum(1 for a,b in zip(diffs_s2, diffs_s1) if a < b)}/{N} measurements")

print("\nMETHODOLOGICAL NOTES:")
print("- Neave measurements are ESTIMATED from his published reconstruction")
print("- Neave's reconstruction was based on first-century Judean skulls, not the Shroud")
print("- The comparison tests whether the Shroud face is anthropologically consistent")
print("  with what a first-century Judean male should look like")
print("- This is NOT a claim of identity — it is a consistency check")

print("\n=== Neave Comparison Complete ===")
