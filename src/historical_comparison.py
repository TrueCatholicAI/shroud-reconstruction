"""Task S: Historical depiction comparison — Shroud proportions vs art history."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== Historical Depiction Comparison ===")

# === Facial ratio data ===
# We compare normalized facial RATIOS (not absolute measurements) across:
# 1. Shroud Study 1 (Enrie 1931)
# 2. Shroud Study 2 (Miller 1978)
# 3. Neave 2001 forensic reconstruction (first-century Judean skulls)
# 4. Christ Pantocrator, St Catherine's Monastery, Sinai (6th century)
#    - Often claimed to derive from the Shroud; one of the oldest icons
#    - Facial proportions measured from published high-resolution photographs
# 5. Typical Renaissance depiction (composite from published art analysis)
# 6. Modern population average

# Key ratios (dimensionless, more robust than absolute measurements):
# IPD / Face Width — how wide-set the eyes are relative to face
# Nose Width / Face Width — nasal proportion
# Mouth Width / Face Width — mouth proportion
# Nose-to-Chin / Face Height — lower face proportion

# Data sources and estimated ratios:
sources = {
    'Shroud Study 1\n(Enrie 1931)': {
        'IPD/Face Width': 5.45 / 16.50,       # 0.330
        'Nose/Face Width': 3.59 / 16.50,      # 0.218
        'Mouth/Face Width': 4.30 / 16.50,     # 0.261
        'Jaw/Face Width': 12.91 / 16.50,      # 0.782
    },
    'Shroud Study 2\n(Miller 1978)': {
        'IPD/Face Width': 6.30 / 12.60,       # 0.500
        'Nose/Face Width': 3.15 / 12.60,      # 0.250
        'Mouth/Face Width': 3.94 / 12.60,     # 0.313
        'Jaw/Face Width': 11.03 / 12.60,      # 0.875
    },
    'Neave 2001\n(Judean skulls)': {
        'IPD/Face Width': 6.2 / 14.0,         # 0.443
        'Nose/Face Width': 3.8 / 14.0,        # 0.271
        'Mouth/Face Width': 5.0 / 14.0,       # 0.357
        'Jaw/Face Width': 11.5 / 14.0,        # 0.821
    },
    'Pantocrator\n(6th c. Sinai)': {
        # Estimated from published icon photographs
        # The Pantocrator has a long face, prominent nose, almond eyes
        'IPD/Face Width': 0.42,
        'Nose/Face Width': 0.22,
        'Mouth/Face Width': 0.28,
        'Jaw/Face Width': 0.78,
    },
    'Renaissance\n(composite)': {
        # Idealized European proportions from Renaissance masters
        # Da Vinci's Vitruvian proportions, Raphael, etc.
        'IPD/Face Width': 0.40,
        'Nose/Face Width': 0.20,
        'Mouth/Face Width': 0.30,
        'Jaw/Face Width': 0.75,
    },
    'Modern Average': {
        # Population averages from forensic anthropology
        'IPD/Face Width': 6.3 / 14.5,         # 0.434
        'Nose/Face Width': 3.5 / 14.5,        # 0.241
        'Mouth/Face Width': 5.0 / 14.5,       # 0.345
        'Jaw/Face Width': 12.0 / 14.5,        # 0.828
    },
}

ratios = ['IPD/Face Width', 'Nose/Face Width', 'Mouth/Face Width', 'Jaw/Face Width']

# Print table
print(f"\n{'Source':<24s}", end='')
for r in ratios:
    print(f"  {r:>16s}", end='')
print()
print("-" * 92)
for name, data in sources.items():
    clean_name = name.replace('\n', ' ')
    print(f"{clean_name:<24s}", end='')
    for r in ratios:
        print(f"  {data[r]:>16.3f}", end='')
    print()

# === Parallel coordinates plot ===
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#222')

colors = {
    'Shroud Study 1\n(Enrie 1931)': '#e74c3c',
    'Shroud Study 2\n(Miller 1978)': '#3498db',
    'Neave 2001\n(Judean skulls)': '#2ecc71',
    'Pantocrator\n(6th c. Sinai)': '#c4a35a',
    'Renaissance\n(composite)': '#9b59b6',
    'Modern Average': '#888888',
}

x_positions = np.arange(len(ratios))

for name, data in sources.items():
    values = [data[r] for r in ratios]
    linewidth = 3 if 'Shroud' in name else 2
    alpha = 1.0 if 'Shroud' in name else 0.7
    ax.plot(x_positions, values, 'o-', color=colors[name], linewidth=linewidth,
            markersize=8, label=name.replace('\n', ' '), alpha=alpha)

ax.set_xticks(x_positions)
ax.set_xticklabels(ratios, color='#ccc', fontsize=10)
ax.set_ylabel('Ratio', color='white', fontsize=11)
ax.tick_params(colors='white')
ax.legend(facecolor='#333', labelcolor='white', fontsize=9, loc='upper left',
          bbox_to_anchor=(0, 1))
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#333', linewidth=0.5)

fig.suptitle('Facial Proportions Across Historical Depictions and Forensic Data',
             color='#c4a35a', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/historical_parallel_coords.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: historical_parallel_coords.png")

# === Distance matrix (Euclidean distance in ratio space) ===
source_names = list(sources.keys())
n = len(source_names)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        vals_i = np.array([sources[source_names[i]][r] for r in ratios])
        vals_j = np.array([sources[source_names[j]][r] for r in ratios])
        dist_matrix[i, j] = np.sqrt(np.sum((vals_i - vals_j)**2))

# Heatmap
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor('#1a1a1a')
im = ax.imshow(dist_matrix, cmap='YlOrRd_r', aspect='equal')

short_names = [n.replace('\n', ' ').replace('(Enrie 1931)', '(S1)').replace('(Miller 1978)', '(S2)').replace('(Judean skulls)', '').replace('(6th c. Sinai)', '').replace('(composite)', '') for n in source_names]
ax.set_xticks(range(n))
ax.set_xticklabels(short_names, rotation=45, ha='right', color='#ccc', fontsize=8)
ax.set_yticks(range(n))
ax.set_yticklabels(short_names, color='#ccc', fontsize=8)

for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{dist_matrix[i,j]:.3f}', ha='center', va='center',
                color='white' if dist_matrix[i,j] > 0.05 else 'black', fontsize=8)

plt.colorbar(im, ax=ax, label='Euclidean distance in ratio space')
fig.suptitle('Proportional Similarity Between Sources', color='#c4a35a', fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig('output/analysis/historical_distance_matrix.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: historical_distance_matrix.png")

# === Key findings ===
print("\n--- Proportional Distances ---")
shroud1_idx = 0
shroud2_idx = 1
neave_idx = 2
panto_idx = 3
ren_idx = 4
modern_idx = 5

print(f"Shroud S1 <-> Pantocrator: {dist_matrix[shroud1_idx, panto_idx]:.3f}")
print(f"Shroud S2 <-> Pantocrator: {dist_matrix[shroud2_idx, panto_idx]:.3f}")
print(f"Shroud S1 <-> Neave: {dist_matrix[shroud1_idx, neave_idx]:.3f}")
print(f"Shroud S2 <-> Neave: {dist_matrix[shroud2_idx, neave_idx]:.3f}")
print(f"Shroud S1 <-> Renaissance: {dist_matrix[shroud1_idx, ren_idx]:.3f}")
print(f"Shroud S2 <-> Renaissance: {dist_matrix[shroud2_idx, ren_idx]:.3f}")
print(f"Pantocrator <-> Neave: {dist_matrix[panto_idx, neave_idx]:.3f}")
print(f"Pantocrator <-> Renaissance: {dist_matrix[panto_idx, ren_idx]:.3f}")
print(f"Neave <-> Modern Average: {dist_matrix[neave_idx, modern_idx]:.3f}")

# Closest to Pantocrator
panto_dists = [(short_names[i], dist_matrix[panto_idx, i]) for i in range(n) if i != panto_idx]
panto_dists.sort(key=lambda x: x[1])
print(f"\nClosest to Pantocrator: {panto_dists[0][0]} ({panto_dists[0][1]:.3f})")
print(f"Second closest: {panto_dists[1][0]} ({panto_dists[1][1]:.3f})")

print("\nNOTE: This analysis compares proportional RATIOS, not absolute sizes.")
print("Two faces can have identical ratios but very different sizes.")
print("The Pantocrator proportions are ESTIMATED from published photographs.")

print("\n=== Historical Comparison Complete ===")
