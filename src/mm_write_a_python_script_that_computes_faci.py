import matplotlib
matplotlib.use('Agg')

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction')

output_dir = project_root / 'output' / 'ratios'
output_dir.mkdir(parents=True, exist_ok=True)
task_output_dir = project_root / 'output' / 'task_results'
task_output_dir.mkdir(parents=True, exist_ok=True)
analysis_dir = project_root / 'output' / 'analysis'
analysis_dir.mkdir(parents=True, exist_ok=True)
docs_images_dir = project_root / 'docs' / 'images'
docs_images_dir.mkdir(parents=True, exist_ok=True)

landmarks_path = project_root / 'data' / 'measurements' / 'landmarks.json'
study2_paths = [
    project_root / 'data' / 'measurements' / 'landmarks_study2.json',
    project_root / 'data' / 'landmarks_study2.json',
    project_root / 'output' / 'landmarks_study2.json',
    project_root / 'output' / 'study2_landmarks.json',
]

study1_data = None
study2_data = None

if landmarks_path.exists():
    try:
        with open(landmarks_path, 'r') as f:
            study1_data = json.load(f)
        print(f"Loaded Study 1 landmarks from {landmarks_path}")
    except Exception as e:
        print(f"Error loading Study 1 landmarks: {e}")

for path in study2_paths:
    if path.exists():
        try:
            with open(path, 'r') as f:
                study2_data = json.load(f)
            print(f"Loaded Study 2 landmarks from {path}")
            break
        except Exception as e:
            print(f"Error loading Study 2 from {path}: {e}")

def compute_ratios(landmarks):
    results = {}
    
    if 'ipd' in landmarks and 'face_width' in landmarks:
        results['ipd_face_width'] = landmarks['ipd'] / landmarks['face_width'] if landmarks['face_width'] != 0 else 0
    else:
        results['ipd_face_width'] = 0
    
    if 'nose_width' in landmarks and 'face_width' in landmarks:
        results['nose_width_face_width'] = landmarks['nose_width'] / landmarks['face_width'] if landmarks['face_width'] != 0 else 0
    else:
        results['nose_width_face_width'] = 0
    
    if 'mouth_width' in landmarks and 'face_width' in landmarks:
        results['mouth_width_face_width'] = landmarks['mouth_width'] / landmarks['face_width'] if landmarks['face_width'] != 0 else 0
    else:
        results['mouth_width_face_width'] = 0
    
    if 'nose_to_chin' in landmarks and 'face_height' in landmarks:
        results['nose_to_chin_face_height'] = landmarks['nose_to_chin'] / landmarks['face_height'] if landmarks['face_height'] != 0 else 0
    else:
        results['nose_to_chin_face_height'] = 0
    
    if 'jaw_width' in landmarks and 'face_width' in landmarks:
        results['jaw_width_face_width'] = landmarks['jaw_width'] / landmarks['face_width'] if landmarks['face_width'] != 0 else 0
    else:
        results['jaw_width_face_width'] = 0
    
    return results

ratio_names = [
    'ipd_face_width',
    'nose_width_face_width',
    'mouth_width_face_width',
    'nose_to_chin_face_height',
    'jaw_width_face_width'
]

plot_labels = [
    'IPD / Face Width',
    'Nose Width / Face Width',
    'Mouth Width / Face Width',
    'Nose to Chin / Face Height',
    'Jaw Width / Face Width'
]

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

x = np.arange(len(ratio_names))
width = 0.35

colors = {'study1': '#c4a35a', 'study2': '#4a9eff'}

study1_ratios = None
study2_ratios = None

if study1_data:
    study1_ratios = compute_ratios(study1_data)
    study1_values = [study1_ratios.get(r, 0) for r in ratio_names]
    bars1 = ax.bar(x - width/2, study1_values, width, label='Study 1', color=colors['study1'], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars1, study1_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', color='white', fontsize=9)
else:
    print("Study 1 landmarks not found")

if study2_data:
    study2_ratios = compute_ratios(study2_data)
    study2_values = [study2_ratios.get(r, 0) for r in ratio_names]
    bars2 = ax.bar(x + width/2, study2_values, width, label='Study 2', color=colors['study2'], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, study2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', color='white', fontsize=9)
else:
    print("Study 2 landmarks not found (comparison not available)")

ax.set_xlabel('Facial Proportion Ratios', color='white', fontsize=12)
ax.set_ylabel('Ratio Value', color='white', fontsize=12)
ax.set_title('Facial Proportion Ratios: Scale-Independent Analysis', color='white', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(plot_labels, rotation=15, ha='right', color='white', fontsize=9)
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')

if study1_data and study2_data:
    ax.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
elif study1_data:
    ax.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='white', labelcolor='white')

ax.set_ylim(0, max([max([s for s in [study1_ratios.get(r, 0), study2_ratios.get(r, 0) if study2_ratios else 0]]) for r in ratio_names]) * 1.2 + 0.1 if study1_data else 1)
ax.grid(axis='y', alpha=0.3, color='gray')

plt.tight_layout()

ratio_bar_path = output_dir / 'ratio_comparison.png'
docs_bar_path = docs_images_dir / 'ratio_comparison.png'
analysis_bar_path = analysis_dir / 'ratio_comparison.png'

plt.savefig(ratio_bar_path, dpi=150, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
plt.savefig(docs_bar_path, dpi=150, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
plt.savefig(analysis_bar_path, dpi=150, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"Saved ratio comparison bar chart to {ratio_bar_path}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('#1a1a1a')
ax2.set_facecolor('#1a1a1a')

theta = np.linspace(0, 2*np.pi, 100)
radius = 1

circle = plt.Circle((0, 0), radius, fill=False, color='#c4a35a', linewidth=2, linestyle='--')
ax2.add_patch(circle)

if study1_ratios:
    study1_vals = [study1_ratios.get(r, 0) for r in ratio_names]
    study1_normalized = [v / max(study1_vals) if max(study1_vals) > 0 else 0 for v in study1_vals]
    angles = np.linspace(0, 2*np.pi, len(ratio_names), endpoint=False)
    xs1 = [v * np.cos(a) for v, a in zip(study1_normalized, angles)]
    ys1 = [v * np.sin(a) for v, a in zip(study1_normalized, angles)]
    ax2.plot(xs1 + [xs1[0]], ys1 + [ys1[0]], 'o-', color='#c4a35a', linewidth=2, markersize=8, label='Study 1')

if study2_ratios:
    study2_vals = [study2_ratios.get(r, 0) for r in ratio_names]
    study2_normalized = [v / max(study2_vals) if max(study2_vals) > 0 else 0 for v in study2_vals]
    angles = np.linspace(0, 2*np.pi, len(ratio_names), endpoint=False)
    xs2 = [v * np.cos(a) for v, a in zip(study2_normalized, angles)]
    ys2 = [v * np.sin(a) for v, a in zip(study2_normalized, angles)]
    ax2.plot(xs2 + [xs2[0]], ys2 + [ys2[0]], 's--', color='#4a9eff', linewidth=2, markersize=8, label='Study 2')

for i, (angle, name) in enumerate(zip(angles, plot_labels)):
    ax2.text(1.2 * np.cos(angle), 1.2 * np.sin(angle), name, ha='center', va='center', 
             color='white', fontsize=8, rotation=np.degrees(angle) - 90 if angle <= np.pi else np.degrees(angle) + 90)

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Facial Proportion Radar Profile', color='white', fontsize=14, fontweight='bold', pad=20)

if study1_data or study2_data:
    ax2.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='white', labelcolor='white')

plt.tight_layout()

radar_path = output_dir / 'ratio_radar.png'
docs_radar_path = docs_images_dir / 'ratio_radar.png'
analysis_radar_path = analysis_dir / 'ratio_radar.png'

plt.savefig(radar_path, dpi=150, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
plt.savefig(docs_radar_path, dpi=150, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
plt.savefig(analysis_radar_path, dpi=150, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"Saved ratio radar profile to {radar_path}")

results = {}

if study1_ratios:
    for k, v in study1_ratios.items():
        results[f'study1_{k}'] = round(v, 6)
    
    results['study1_data_available'] = True
    results['study1_source'] = str(landmarks_path)
else:
    results['study1_data_available'] = False

if study2_ratios:
    for k, v in study2_ratios.items():
        results[f'study2_{k}'] = round(v, 6)
    
    results['study2_data_available'] = True
    results['study2_source'] = str([p for p in study2_paths if p.exists()][0]) if any(p.exists() for p in study2_paths) else 'unknown'
    
    if study1_ratios:
        for r in ratio_names:
            s1_val = study1_ratios.get(r, 0)
            s2_val = study2_ratios.get(r, 0)
            if s1_val > 0:
                pct_diff = ((s2_val - s1_val) / s1_val) * 100
                results[f'comparison_{r}_percent_diff'] = round(pct_diff, 2)
            else:
                results[f'comparison_{r}_percent_diff'] = 0
else:
    results['study2_data_available'] = False

results['image_files'] = [
    str(ratio_bar_path.relative_to(project_root)),
    str(radar_path.relative_to(project_root))
]

results['ratio_definitions'] = {
    'ipd_face_width': 'Interpupillary distance divided by face width',
    'nose_width_face_width': 'Nose width divided by face width',
    'mouth_width_face_width': 'Mouth width divided by face width',
    'nose_to_chin_face_height': 'Nose to chin distance divided by face height',
    'jaw_width_face_width': 'Jaw width divided by face width'
}

output_json_path = task_output_dir / 'ratio_analysis_results.json'

with open(output_json_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("RATIO ANALYSIS RESULTS")
print("="*60)
print(json.dumps(results, indent=2))
print("="*60)
print(f"\nResults saved to: {output_json_path}")

if not study1_data:
    print("\nWARNING: Study 1 landmarks not found. Please ensure data/measurements/landmarks.json exists.")
if not study2_data:
    print("\nNOTE: Study 2 landmarks not found. Only Study 1 ratios computed.")
    print("Searched paths:")
    for p in study2_paths:
        print(f"  - {p}")