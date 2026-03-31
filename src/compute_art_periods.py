"""Generate ControlNet reconstructions across 5 art period styles."""
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEPTH_PATH = PROJECT / "data" / "final" / "controlnet_depth_512.png"
OUT_DIR = PROJECT / "output" / "art_periods"
RESULTS_JSON = PROJECT / "output" / "task_results" / "art_periods_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)

# Art period prompts
ART_PERIODS = [
    {
        "name": "Byzantine Mosaic",
        "slug": "byzantine",
        "prompt": "a face depicted as a Byzantine mosaic, gold tesserae background, solemn expression, frontal pose, Hagia Sophia style, religious icon",
        "negative": "modern, photograph, realistic skin, 3d render",
    },
    {
        "name": "Romanesque Stone",
        "slug": "romanesque",
        "prompt": "a face carved in rough limestone, Romanesque cathedral sculpture, 12th century style, weathered stone texture, architectural relief",
        "negative": "modern, smooth, polished, photograph",
    },
    {
        "name": "Gothic Cathedral",
        "slug": "gothic",
        "prompt": "a face as a Gothic cathedral stone sculpture, 13th century French Gothic style, naturalistic features, limestone, jamb figure",
        "negative": "modern, photograph, painted, colorful",
    },
    {
        "name": "Renaissance Marble",
        "slug": "renaissance",
        "prompt": "a face sculpted in white Carrara marble, Renaissance style, Michelangelo influence, smooth polished marble surface, classical proportions",
        "negative": "modern, photograph, painted, rough",
    },
    {
        "name": "Modern Forensic Clay",
        "slug": "forensic_clay",
        "prompt": "a face sculpted in gray forensic reconstruction clay, neutral lighting, museum display, forensic facial reconstruction, matte clay surface",
        "negative": "painting, artistic, stylized, colorful",
    },
]

SEED = 44
CONTROLNET_SCALE = 0.95
NUM_STEPS = 30

print("Loading ControlNet pipeline...")
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
pipe.enable_attention_slicing()
print("Pipeline loaded.")

# Load depth conditioning image
depth_img = Image.open(DEPTH_PATH).convert("RGB")
print(f"Depth conditioning: {depth_img.size}")

image_paths = []
period_results = []

for period in ART_PERIODS:
    print(f"\nGenerating: {period['name']}...")
    generator = torch.Generator("cuda").manual_seed(SEED)

    result = pipe(
        prompt=period["prompt"],
        negative_prompt=period["negative"],
        image=depth_img,
        num_inference_steps=NUM_STEPS,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        generator=generator,
    )

    img = result.images[0]
    img_path = OUT_DIR / f"{period['slug']}_seed{SEED}.png"
    img.save(img_path)
    rel_path = str(img_path.relative_to(PROJECT))
    image_paths.append(rel_path)

    period_results.append({
        "name": period["name"],
        "slug": period["slug"],
        "prompt": period["prompt"],
        "image_path": rel_path,
    })
    print(f"  Saved: {img_path.name}")

# Generate comparison grid
import matplotlib.pyplot as plt

BG = '#1a1a1a'
GOLD = '#c4a35a'

fig, axes = plt.subplots(1, 5, figsize=(20, 5), facecolor=BG)
for ax, period in zip(axes, period_results):
    img = Image.open(PROJECT / period["image_path"])
    ax.imshow(img)
    ax.set_title(period["name"], color=GOLD, fontsize=11, fontweight='bold')
    ax.axis('off')

fig.suptitle(f'Same Depth Map, Five Art Periods (Seed {SEED}, ControlNet {CONTROLNET_SCALE})',
             color=GOLD, fontsize=14, fontweight='bold')
plt.tight_layout()
grid_path = OUT_DIR / 'art_periods_grid.png'
fig.savefig(grid_path, dpi=150, facecolor=BG)
plt.close(fig)
image_paths.append(str(grid_path.relative_to(PROJECT)))

# Also include depth conditioning image for reference
depth_ref_path = OUT_DIR / 'depth_conditioning.png'
depth_img.save(depth_ref_path)
image_paths.append(str(depth_ref_path.relative_to(PROJECT)))

results = {
    "depth_conditioning": str(DEPTH_PATH.relative_to(PROJECT)),
    "seed": SEED,
    "controlnet_scale": CONTROLNET_SCALE,
    "num_steps": NUM_STEPS,
    "model": "stable-diffusion-v1-5 + control_v11f1p_sd15_depth",
    "periods": period_results,
    "image_files": image_paths,
}

with open(RESULTS_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_JSON.relative_to(PROJECT)}")
print(json.dumps({k: v for k, v in results.items() if k != 'periods'}, indent=2))
