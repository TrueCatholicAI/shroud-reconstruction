"""Generate morph animation GIF between original and healed depth maps."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEPTH_ORIG = PROJECT / "data" / "final" / "depth_150x150_g15.npy"
DEPTH_HEALED = PROJECT / "data" / "final" / "depth_healed_150.npy"
OUT_PATH = PROJECT / "docs" / "images" / "morph-animation.gif"
RESULTS_JSON = PROJECT / "output" / "task_results" / "morph_gif_results.json"

RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

original = np.load(DEPTH_ORIG).astype(np.float64)
healed = np.load(DEPTH_HEALED).astype(np.float64)
print(f"Original: {original.shape}, Healed: {healed.shape}")

N_FRAMES = 20
frames = []

for i in range(N_FRAMES):
    # Forward and back: 0->1->0 over the cycle
    t = i / N_FRAMES
    # Ping-pong: go forward first half, backward second half
    if t < 0.5:
        alpha = t * 2  # 0 -> 1
    else:
        alpha = 2 - t * 2  # 1 -> 0

    blended = (1 - alpha) * original + alpha * healed
    # Normalize to 0-255
    blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)
    frames.append(Image.fromarray(blended_u8, mode='L'))

# Save as GIF
frames[0].save(
    OUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=150,  # ms per frame
    loop=0,
)

print(f"Saved {N_FRAMES}-frame GIF to {OUT_PATH.relative_to(PROJECT)}")
print(f"File size: {OUT_PATH.stat().st_size} bytes")

results = {
    "frames": N_FRAMES,
    "duration_ms_per_frame": 150,
    "total_duration_ms": N_FRAMES * 150,
    "output_path": str(OUT_PATH.relative_to(PROJECT)),
    "image_files": [str(OUT_PATH.relative_to(PROJECT))],
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
