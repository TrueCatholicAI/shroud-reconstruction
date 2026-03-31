import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

def load_and_prepare_depth():
    print("Loading depth map from data/processed/depth_map_smooth_15.npy...")
    depth = np.load('data/processed/depth_map_smooth_15.npy')
    print(f"Original depth map shape: {depth.shape}")
    
    depth_resized = cv2.resize(depth, (150, 150), interpolation=cv2.INTER_LINEAR)
    print(f"Resized depth map shape: {depth_resized.shape}")
    
    return depth_resized

def define_slices():
    slices = {
        'Brow': 40,
        'Eye': 50,
        'Cheek': 65,
        'Mouth': 80,
        'Jaw': 95
    }
    return slices

def analyze_slice(depth, row, name):
    margin = 3
    start_row = max(0, row - margin)
    end_row = min(depth.shape[0], row + margin + 1)
    
    profile = np.mean(depth[start_row:end_row, :], axis=0)
    
    midpoint = 75
    left_half = profile[:midpoint]
    right_half = np.flip(profile[midpoint:])
    
    diff = left_half - right_half
    
    std_diff = np.std(diff)
    threshold = std_diff
    
    significant_mask = np.abs(diff) > threshold
    
    max_asymmetry = np.max(np.abs(diff))
    max_idx = np.argmax(np.abs(diff))
    location_px = max_idx
    
    direction = "swelling" if diff[max_idx] > 0 else "depression"
    
    interpretation = ""
    if name == 'Brow':
        if direction == "depression":
            interpretation = "Possible scourging or contusion"
        else:
            interpretation = "Bilateral swelling from trauma"
    elif name == 'Eye':
        if direction == "depression":
            interpretation = "Orbital trauma or swelling asymmetry"
        else:
            interpretation = "Periorbital edema"
    elif name == 'Cheek':
        if direction == "depression":
            interpretation = "Blunt force trauma pattern"
        else:
            interpretation = "Bilateral contusion asymmetry"
    elif name == 'Mouth':
        if direction == "depression":
            interpretation = "Perioral trauma or edema pattern"
        else:
            interpretation = "Swelling asymmetry"
    elif name == 'Jaw':
        if direction == "depression":
            interpretation = "Mandible trauma or post-mortem changes"
        else:
            interpretation = "Submandibular swelling"
    
    return {
        'name': name,
        'row': row,
        'profile': profile,
        'left_half': left_half,
        'right_half': right_half,
        'diff': diff,
        'threshold': threshold,
        'significant_mask': significant_mask,
        'max_asymmetry': max_asymmetry,
        'location_px': location_px,
        'direction': direction,
        'interpretation': interpretation
    }

def create_slices_figure(depth, results):
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    fig.patch.set_facecolor('#1a1a1a')
    
    for idx, result in enumerate(results):
        ax_left = axes[idx, 0]
        ax_right = axes[idx, 1]
        
        for spine in ax_left.spines.values():
            spine.set_color('white')
        for spine in ax_right.spines.values():
            spine.set_color('white')
        
        ax_left.imshow(depth, cmap='viridis', aspect='auto')
        ax_left.axhline(y=result['row'], color='#c4a35a', linewidth=2, linestyle='--')
        ax_left.set_title(f'{result["name"]} - Row {result["row"]}', color='white', fontsize=12, fontweight='bold')
        ax_left.set_xlabel('Column (px)', color='white')
        ax_left.set_ylabel('Row (px)', color='white')
        ax_left.tick_params(colors='white')
        
        x_left = np.arange(len(result['left_half']))
        x_right = np.arange(len(result['right_half']))
        
        ax_right.fill_between(x_left, result['left_half'], result['right_half'], 
                              where=result['significant_mask'][:len(x_left)], 
                              color='red', alpha=0.5, label='Significant diff')
        ax_right.plot(x_left, result['left_half'], 'r-', linewidth=2, label='Left')
        ax_right.plot(x_right, result['right_half'], 'b-', linewidth=2, label='Right (flipped)')
        ax_right.plot(x_left, result['diff'], 'g--', linewidth=1.5, label='Diff (L-R)')
        ax_right.axhline(y=result['threshold'], color='yellow', linestyle=':', alpha=0.7)
        ax_right.axhline(y=-result['threshold'], color='yellow', linestyle=':', alpha=0.7)
        
        ax_right.set_title(f'{result["name"]} Profile Analysis', color='white', fontsize=12, fontweight='bold')
        ax_right.set_xlabel('Distance from center (px)', color='white')
        ax_right.set_ylabel('Depth (normalized)', color='white')
        ax_right.tick_params(colors='white')
        ax_right.legend(loc='upper right', fontsize=8, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        ax_right.set_facecolor('#2a2a2a')
    
    plt.tight_layout()
    return fig

def create_catalog_heatmap(results):
    n_slices = len(results)
    n_cols = 75
    
    asymmetry_matrix = np.zeros((n_slices, n_cols))
    
    for idx, result in enumerate(results):
        asymmetry_matrix[idx, :] = result['diff']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    slice_names = [r['name'] for r in results]
    
    im = ax.imshow(asymmetry_matrix, cmap='RdBu_r', aspect='auto', 
                   extent=[-75, 0, n_slices, 0])
    
    ax.set_yticks(np.arange(n_slices) + 0.5)
    ax.set_yticklabels(slice_names, color='white', fontsize=12)
    
    ax.set_xlabel('Distance from center (px)', color='white', fontsize=12)
    ax.set_ylabel('Facial Region', color='white', fontsize=12)
    ax.set_title('Bilateral Wound Asymmetry Map\nRed = Left swelling / Right depression | Blue = Left depression / Right swelling', 
                 color='white', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Asymmetry (Left - Right)', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    plt.tight_layout()
    return fig

def create_overlay_figure(depth, results):
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    
    ax.imshow(depth_norm, cmap='gray', aspect='auto', vmin=0, vmax=1)
    
    composite_overlay = np.zeros((depth.shape[0], depth.shape[1], 3))
    
    for idx, result in enumerate(results):
        row = result['row']
        margin = 3
        start_row = max(0, row - margin)
        end_row = min(depth.shape[0], row + margin + 1)
        
        for r in range(start_row, end_row):
            for c in range(len(result['diff'])):
                if result['significant_mask'][c]:
                    if c < depth.shape[1]:
                        if result['diff'][c] > 0:
                            composite_overlay[r, c, 0] = max(composite_overlay[r, c, 0], 
                                                              min(1.0, result['diff'][c] / (np.max(np.abs(result['diff'])) + 1e-6)))
                        else:
                            composite_overlay[r, c, 2] = max(composite_overlay[r, c, 2], 
                                                              min(1.0, np.abs(result['diff'][c]) / (np.max(np.abs(result['diff'])) + 1e-6)))
    
    ax.imshow(composite_overlay, alpha=0.6, aspect='auto')
    
    for result in results:
        ax.axhline(y=result['row'], color='#c4a35a', linewidth=1.5, linestyle='--', alpha=0.8)
    
    ax.set_title('Bilateral Wound Analysis Overlay\nRed = Swelling | Blue = Depression', 
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (px)', color='white')
    ax.set_ylabel('Row (px)', color='white')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    legend_text = 'Red = Left side swelling / Right side depression\nBlue = Left side depression / Right side swelling'
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, color='white', 
            fontsize=9, verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8, edgecolor='#c4a35a'))
    
    plt.tight_layout()
    return fig

def print_catalog(results):
    print("\n" + "="*90)
    print("BILATERAL WOUND CATALOG")
    print("="*90)
    print(f"{'Slice':<10} {'Max Asymmetry':<15} {'Location (px)':<15} {'Direction':<12} {'Interpretation':<40}")
    print("-"*90)
    
    for result in results:
        print(f"{result['name']:<10} {result['max_asymmetry']:<15.4f} {result['location_px']:<15} {result['direction']:<12} {result['interpretation']:<40}")
    
    print("="*90)
    print("\nSUMMARY:")
    print("-"*90)
    
    avg_asymmetry = np.mean([r['max_asymmetry'] for r in results])
    print(f"Average Maximum Asymmetry: {avg_asymmetry:.4f}")
    
    swelling_count = sum(1 for r in results if r['direction'] == 'swelling')
    depression_count = sum(1 for r in results if r['direction'] == 'depression')
    
    print(f"Swelling regions: {swelling_count}")
    print(f"Depression regions: {depression_count}")
    
    print("\nBILATERAL WOUND ANALYSIS COMPLETE")
    print("="*90)

def main():
    print("\n" + "="*90)
    print("BILATERAL WOUND CATALOG ANALYSIS")
    print("="*90 + "\n")
    
    os.makedirs('output/analysis', exist_ok=True)
    os.makedirs('docs/images', exist_ok=True)
    
    depth = load_and_prepare_depth()
    
    slices = define_slices()
    print(f"\nDefined {len(slices)} horizontal slices:")
    for name, row in slices.items():
        print(f"  {name}: row {row}")
    
    print("\nAnalyzing slices...")
    results = []
    for name, row in slices.items():
        print(f"  Processing {name} slice at row {row}...")
        result = analyze_slice(depth, row, name)
        results.append(result)
        print(f"    Max asymmetry: {result['max_asymmetry']:.4f}, Location: {result['location_px']} px, Direction: {result['direction']}")
    
    print_catalog(results)
    
    print("\nGenerating figures...")
    
    print("  Creating bilateral_wound_slices.png...")
    fig_slices = create_slices_figure(depth, results)
    fig_slices.savefig('output/analysis/bilateral_wound_slices.png', 
                       facecolor='#1a1a1a', edgecolor='none', dpi=150, bbox_inches='tight')
    fig_slices.savefig('docs/images/bilateral_wound_slices.png', 
                       facecolor='#1a1a1a', edgecolor='none', dpi=150, bbox_inches='tight')
    plt.close(fig_slices)
    print("    Saved to output/analysis/ and docs/images/")
    
    print("  Creating bilateral_wound_catalog.png...")
    fig_catalog = create_catalog_heatmap(results)
    fig_catalog.savefig('output/analysis/bilateral_wound_catalog.png', 
                        facecolor='#1a1a1a', edgecolor='none', dpi=150, bbox_inches='tight')
    fig_catalog.savefig('docs/images/bilateral_wound_catalog.png', 
                        facecolor='#1a1a1a', edgecolor='none', dpi=150, bbox_inches='tight')
    plt.close(fig_catalog)
    print("    Saved to output/analysis/ and docs/images/")
    
    print("  Creating bilateral_wound_overlay.png...")
    fig_overlay = create_overlay_figure(depth, results)
    fig_overlay.savefig('output/analysis/bilateral_wound_overlay.png', 
                        facecolor='#1a1a1a', edgecolor='none', dpi=150, bbox_inches='tight')
    fig_overlay.savefig('docs/images/bilateral_wound_overlay.png', 
                        facecolor='#1a1a1a', edgecolor='none', dpi=150, bbox_inches='tight')
    plt.close(fig_overlay)
    print("    Saved to output/analysis/ and docs/images/")
    
    print("\n" + "="*90)
    print("BILATERAL WOUND CATALOG GENERATION COMPLETE")
    print("="*90)
    print(f"\nOutput files:")
    print("  - output/analysis/bilateral_wound_slices.png")
    print("  - output/analysis/bilateral_wound_catalog.png")
    print("  - output/analysis/bilateral_wound_overlay.png")
    print("  - docs/images/bilateral_wound_slices.png")
    print("  - docs/images/bilateral_wound_catalog.png")
    print("  - docs/images/bilateral_wound_overlay.png")

if __name__ == '__main__':
    main()