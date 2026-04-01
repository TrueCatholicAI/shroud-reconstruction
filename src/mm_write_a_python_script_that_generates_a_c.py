import matplotlib
matplotlib.use('Agg')
import os
import json
import subprocess
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction'

results = {}

try:
    src_path = os.path.join(PROJECT_ROOT, 'src')
    docs_path = os.path.join(PROJECT_ROOT, 'docs')
    docs_images_path = os.path.join(PROJECT_ROOT, 'docs', 'images')
    output_path = os.path.join(PROJECT_ROOT, 'output')
    task_results_path = os.path.join(PROJECT_ROOT, 'output', 'task_results')

    os.makedirs(task_results_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'analysis'), exist_ok=True)
    os.makedirs(docs_images_path, exist_ok=True)

    scripts = glob.glob(os.path.join(src_path, '*.py'))
    results['total_scripts'] = len(scripts)

    html_pages = glob.glob(os.path.join(docs_path, '*.html'))
    results['total_html_pages'] = len(html_pages)

    images = glob.glob(os.path.join(docs_images_path, '*'))
    images = [f for f in images if os.path.isfile(f)]
    results['total_images'] = len(images)

    output_files = []
    for root, dirs, files in os.walk(output_path):
        output_files.extend([os.path.join(root, f) for f in files])
    results['total_output_files'] = len(output_files)

    json_files = [f for f in output_files if f.endswith('.json')]
    results['total_json_files'] = len(json_files)

    total_lines = 0
    for script in scripts:
        try:
            with open(script, 'r', encoding='utf-8') as f:
                total_lines += len(f.readlines())
        except:
            pass
    results['total_lines_python'] = total_lines

    try:
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--count'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            results['total_commits'] = int(result.stdout.strip())
        else:
            results['total_commits'] = 0
    except:
        results['total_commits'] = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    categories = ['Scripts', 'HTML Pages', 'Images', 'Output Files', 'JSON Files', 'Commits']
    values = [
        results.get('total_scripts', 0),
        results.get('total_html_pages', 0),
        results.get('total_images', 0),
        results.get('total_output_files', 0),
        results.get('total_json_files', 0),
        results.get('total_commits', 0)
    ]

    colors = ['#c4a35a' if v > 0 else '#555555' for v in values]
    bars = ax.bar(categories, values, color=colors, edgecolor='#c4a35a', linewidth=1.5)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    color='white')

    ax.set_ylabel('Count', color='white', fontsize=12)
    ax.set_title('Shroud Reconstruction Project Statistics', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')

    for spine in ax.spines.values():
        spine.set_color('#c4a35a')

    ax.xaxis.label.set_color('white')
    plt.tight_layout()

    chart_path_1 = os.path.join(output_path, 'analysis', 'project_stats.png')
    chart_path_2 = os.path.join(docs_images_path, 'project_stats.png')
    plt.savefig(chart_path_1, facecolor='#1a1a1a', edgecolor='none', dpi=150)
    plt.savefig(chart_path_2, facecolor='#1a1a1a', edgecolor='none', dpi=150)
    plt.close()

    results['image_files'] = [
        os.path.relpath(chart_path_1, PROJECT_ROOT),
        os.path.relpath(chart_path_2, PROJECT_ROOT)
    ]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor('#1a1a1a')
    ax2.set_facecolor('#1a1a1a')

    stats_text = f"Total Python Lines: {results.get('total_lines_python', 0)}\n"
    stats_text += f"Total Scripts: {results.get('total_scripts', 0)}\n"
    stats_text += f"Total Commits: {results.get('total_commits', 0)}"

    ax2.text(0.5, 0.5, stats_text,
             transform=ax2.transAxes,
             fontsize=16,
             verticalalignment='center',
             horizontalalignment='center',
             color='#c4a35a',
             fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', edgecolor='#c4a35a', linewidth=2))

    ax2.axis('off')
    ax2.set_title('Code Metrics Summary', color='white', fontsize=14, fontweight='bold')

    summary_path_1 = os.path.join(output_path, 'analysis', 'code_metrics.png')
    summary_path_2 = os.path.join(docs_images_path, 'code_metrics.png')
    plt.savefig(summary_path_1, facecolor='#1a1a1a', edgecolor='none', dpi=150)
    plt.savefig(summary_path_2, facecolor='#1a1a1a', edgecolor='none', dpi=150)
    plt.close()

    results['image_files'].extend([
        os.path.relpath(summary_path_1, PROJECT_ROOT),
        os.path.relpath(summary_path_2, PROJECT_ROOT)
    ])

    print("=" * 50)
    print("PROJECT STATISTICS SUMMARY")
    print("=" * 50)
    for key, value in results.items():
        if key != 'image_files':
            print(f"{key}: {value}")
    print("-" * 50)
    print(f"Charts saved to: {chart_path_1}")
    print(f"Charts saved to: {chart_path_2}")
    print("=" * 50)

    json_output_path = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\output\task_results\write_a_python_script_that_generates_a_c_results.json'
    print("\nJSON OUTPUT:")
    print(json.dumps(results, indent=2))
    json.dump(results, open(json_output_path, 'w'), indent=2)
    print(f"\nJSON saved to: {json_output_path}")

except Exception as e:
    print(f"ERROR: {str(e)}")
    error_results = {
        'error': str(e),
        'total_scripts': 0,
        'total_html_pages': 0,
        'total_images': 0,
        'total_output_files': 0,
        'total_json_files': 0,
        'total_lines_python': 0,
        'total_commits': 0,
        'image_files': []
    }
    print(json.dumps(error_results, indent=2))
    json.dump(error_results, open(r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction\output\task_results\write_a_python_script_that_generates_a_c_results.json', 'w'), indent=2)