"""Generate rotating 3D surface animation GIF of the Enrie depth map."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEPTH_PATH = PROJECT / "data" / "final" / "depth_150x150_g15.npy"
OUT_PATH = PROJECT / "docs" / "images" / "3d-rotation.gif"
RESULTS_JSON = PROJECT / "output" / "task_results" / "rotation_gif_results.json"

RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

depth = np.load(DEPTH_PATH).astype(np.float64)
print(f"Loaded depth: {depth.shape}")

x = np.arange(depth.shape[1])
y = np.arange(depth.shape[0])
X, Y = np.meshgrid(x, y)

N_FRAMES = 36
DEGREES_PER_FRAME = 10

BG = '#1a1a1a'
GOLD = '#c4a35a'

frames = []

for i in range(N_FRAMES):
    azim = i * DEGREES_PER_FRAME
    fig = plt.figure(figsize=(5, 5), facecolor=BG)
    ax = fig.add_subplot(111, projection='3d', facecolor=BG)

    ax.plot_surface(X, Y, depth, cmap='inferno', rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=True)

    ax.set_xlabel('X', color='white', fontsize=8)
    ax.set_ylabel('Y', color='white', fontsize=8)
    ax.set_zlabel('Depth', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=25, azim=azim)

    # Render to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    frames.append(img)

    if i % 6 == 0:
        print(f"  Frame {i+1}/{N_FRAMES} (azim={azim}°)")

# Save as GIF
frames[0].save(
    OUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=100,  # ms per frame
    loop=0,
)

print(f"Saved {N_FRAMES}-frame GIF to {OUT_PATH.relative_to(PROJECT)}")
print(f"File size: {OUT_PATH.stat().st_size} bytes")

results = {
    "frames": N_FRAMES,
    "degrees_per_frame": DEGREES_PER_FRAME,
    "duration_ms_per_frame": 100,
    "total_duration_ms": N_FRAMES * 100,
    "output_path": str(OUT_PATH.relative_to(PROJECT)),
    "image_files": [str(OUT_PATH.relative_to(PROJECT))],
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
