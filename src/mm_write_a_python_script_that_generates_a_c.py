import matplotlib
matplotlib.use('Agg')
import os
import json
import glob
import subprocess
from pathlib import Path

PROJECT_ROOT = r'C:\Users\nickh\Documents\claude-sandbox\shroud-reconstruction'

def count_files(directory, pattern):
    try:
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        return len(files)
    except Exception as e:
        print(f"Error counting files in {directory} with pattern {pattern}: {e}")
        return 0

def count_lines_of_code(directory):
    total_lines = 0
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += sum(1 for line in f)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    except Exception as e:
        print(f"Error walking directory {directory}: {e}")
    return total_lines

def count_git_commits():
    try:
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
        else:
            print(f"Git command failed: {result.stderr}")
            return 0
    except FileNotFoundError:
        print("Git not found in PATH")
        return 0
    except Exception as e:
        print(f"Error counting git commits: {e}")
        return 0

def create_stats_visualization(stats):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    categories = ['Python Files', 'HTML Files', 'Images', 'JSON Files', 'Lines of Code', 'Git Commits']
    values = [
        stats.get('py_files', 0),
        stats.get('html_files', 0),
        stats.get('image_files', 0),
        stats.get('json_files', 0),
        stats.get('lines_of_code', 0),
        stats.get('git_commits', 0)
    ]

    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, values, color='#c4a35a', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, color='white', rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Count', color='white', fontsize=12)
    ax.set_title('Project Statistics Summary', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', fontsize=9)

    plt.tight_layout()
    return fig

def main():
    print("=" * 60)
    print("GENERATING PROJECT STATISTICS SUMMARY")
    print("=" * 60)

    src_dir = os.path.join(PROJECT_ROOT, 'src')
    docs_dir = os.path.join(PROJECT_ROOT, 'docs')
    docs_images_dir = os.path.join(PROJECT_ROOT, 'docs', 'images')
    output_analysis_dir = os.path.join(PROJECT_ROOT, 'output', 'analysis')
    task_results_dir = os.path.join(PROJECT_ROOT, 'output', 'task_results')

    os.makedirs(output_analysis_dir, exist_ok=True)
    os.makedirs(task_results_dir, exist_ok=True)
    os.makedirs(docs_images_dir, exist_ok=True)

    print("\nCounting files...")

    py_files_count = count_files(src_dir, '*.py')
    print(f"  Python files in src/: {py_files_count}")

    html_files_count = count_files(docs_dir, '**/*.html')
    print(f"  HTML files in docs/: {html_files_count}")

    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg', '*.bmp', '*.webp']
    image_files_count = 0
    for ext in image_extensions:
        image_files_count += count_files(docs_images_dir, ext)
    print(f"  Image files in docs/images/: {image_files_count}")

    json_files_count = count_files(task_results_dir, '*.json')
    print(f"  JSON files in output/task_results/: {json_files_count}")

    print("\nCounting lines of code...")
    lines_of_code = count_lines_of_code(src_dir)
    print(f"  Total lines of Python code: {lines_of_code:,}")

    print("\nCounting git commits...")
    git_commits = count_git_commits()
    print(f"  Total git commits: {git_commits}")

    print("\nCreating visualization...")
    stats = {
        'py_files': py_files_count,
        'html_files': html_files_count,
        'image_files': image_files_count,
        'json_files': json_files_count,
        'lines_of_code': lines_of_code,
        'git_commits': git_commits
    }

    fig = create_stats_visualization(stats)

    stats_chart_path = os.path.join(output_analysis_dir, 'project_stats_chart.png')
    docs_chart_path = os.path.join(docs_images_dir, 'project_stats_chart.png')

    fig.savefig(stats_chart_path, facecolor='#1a1a1a', edgecolor='none', dpi=150)
    print(f"  Saved chart to: {stats_chart_path}")

    fig.savefig(docs_chart_path, facecolor='#1a1a1a', edgecolor='none', dpi=150)
    print(f"  Saved chart to: {docs_chart_path}")

    import matplotlib.pyplot as plt
    plt.close(fig)

    image_files = [
        os.path.relpath(stats_chart_path, PROJECT_ROOT),
        os.path.relpath(docs_chart_path, PROJECT_ROOT)
    ]

    results = {
        'py_files': py_files_count,
        'html_files': html_files_count,
        'image_files': image_files_count,
        'json_files': json_files_count,
        'lines_of_code': lines_of_code,
        'git_commits': git_commits,
        'image_files': image_files
    }

    output_json_path = os.path.join(task_results_dir, 'project_stats.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_json_path}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS (JSON):")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)
    print("PROJECT STATISTICS SUMMARY COMPLETE")
    print("=" * 60)

    return results

if __name__ == '__main__':
    main()