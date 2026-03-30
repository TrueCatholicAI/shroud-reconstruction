"""
Step 4c: Healed face reconstruction.

Generates reconstructions from the symmetrized depth map — the face
geometry with trauma effects (asymmetry, swelling) algorithmically removed.

This is NOT a claim about pre-Passion appearance. It is an analytical step
showing what the geometry looks like when visible trauma is removed from
the data.

Uses seeds 44 and 45 (best results from prior round).
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

PROJECT_ROOT = Path(__file__).parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "output" / "reconstructions" / "healed"

MATERIALS = [
    (
        "gray_clay",
        "neutral gray clay forensic reconstruction model, matte clay surface, "
        "forensic sculpting clay, even studio lighting, medical reconstruction",
    ),
    (
        "sandstone",
        "carved sandstone relief, limestone sculpture, first-century carved stone, "
        "warm natural stone texture, archaeological artifact",
    ),
    (
        "white_marble",
        "white Carrara marble sculpture, Roman marble bust, smooth polished stone, "
        "soft diffuse lighting, museum display",
    ),
]

BASE_PROMPT = (
    "sculptural bust of a man, age 30-35, "
    "short curly hair cropped close to the head, short beard, "
    "prominent nose, deep-set eyes, strong brow ridge, lean face, "
    "symmetrical face, balanced features, "
    "based on forensic reconstruction from burial cloth, "
    "{material_desc}, "
    "highly detailed surface texture, professional photography"
)

NEGATIVE_PROMPT = (
    "long hair, flowing hair, Renaissance style, shoulder length hair, "
    "painted, colorful, photorealistic skin, living person, "
    "white, black, asian, caucasian, african, european, "
    "skin tone, flesh color, pink, tan, brown skin, "
    "cartoon, anime, illustration, blurry, low quality, "
    "watermark, text, deformed, disfigured, extra limbs, "
    "religious imagery, halo, crown of thorns, blood, wounds, "
    "swollen, bruised, injured, trauma, asymmetric, "
    "smile, happy, laughing"
)


def setup_pipeline():
    """Initialize SD 1.5 + ControlNet Depth."""
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
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def generate(pipe, depth_image, material_name, material_desc, seeds):
    """Generate images for specific seeds."""
    prompt = BASE_PROMPT.format(material_desc=material_desc)
    images = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=depth_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.95,
            generator=generator,
        )
        img = result.images[0]
        images.append(img)
        out_path = OUTPUT_DIR / f"healed_{material_name}_seed{seed}.png"
        img.save(out_path)
        print(f"  Saved: {out_path.name}")
    return images


def create_comparison_grid(original_images, healed_images, orig_depth, healed_depth):
    """Side-by-side: original vs healed for each material."""
    cell = 512
    n_materials = len(original_images)
    margin_top = 70
    label_width = 180
    section_gap = 20

    # Layout: label | orig_depth | orig_render | gap | healed_depth | healed_render
    cols_per_section = 2  # depth + render
    grid_w = label_width + cols_per_section * cell + section_gap + cols_per_section * cell
    grid_h = margin_top + n_materials * cell
    grid = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(grid)

    # Title
    draw.text((label_width + 10, 5),
              "Shroud of Turin - Trauma Analysis",
              fill=(220, 220, 220))

    # Section headers
    orig_x = label_width
    healed_x = label_width + cols_per_section * cell + section_gap

    draw.text((orig_x + 10, 25),
              "As Recorded on the Shroud (at death)",
              fill=(180, 140, 140))
    draw.text((orig_x + 10, 42), "Depth Map", fill=(120, 120, 120))
    draw.text((orig_x + cell + 10, 42), "Reconstruction", fill=(120, 120, 120))

    draw.text((healed_x + 10, 25),
              "Estimated Appearance Before the Passion (healed)",
              fill=(140, 180, 140))
    draw.text((healed_x + 10, 42), "Depth Map", fill=(120, 120, 120))
    draw.text((healed_x + cell + 10, 42), "Reconstruction", fill=(120, 120, 120))

    orig_depth_resized = orig_depth.resize((cell, cell))
    healed_depth_resized = healed_depth.resize((cell, cell))

    materials = list(original_images.keys())
    for row, mat in enumerate(materials):
        y = margin_top + row * cell
        label = mat.replace("_", " ").title()
        draw.text((10, y + cell // 2 - 10), label, fill=(200, 200, 200))

        # Original side
        grid.paste(orig_depth_resized, (orig_x, y))
        grid.paste(original_images[mat].resize((cell, cell)), (orig_x + cell, y))

        # Healed side
        grid.paste(healed_depth_resized, (healed_x, y))
        grid.paste(healed_images[mat].resize((cell, cell)), (healed_x + cell, y))

    out_path = OUTPUT_DIR / "original_vs_healed_comparison.png"
    grid.save(out_path)
    print(f"\nSaved comparison: {out_path}")
    return out_path


def run():
    """Generate healed reconstructions and comparison grid."""
    print("=" * 60)
    print("  HEALED FACE RECONSTRUCTION")
    print("  Symmetrized depth map + trauma removal")
    print("=" * 60 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [44, 45]

    # Load depth images
    orig_depth = Image.open(FINAL_DIR / "controlnet_depth_512_rgb.png").convert("RGB")
    healed_depth = Image.open(FINAL_DIR / "controlnet_depth_healed_512_rgb.png").convert("RGB")
    print(f"Original depth: {orig_depth.size}")
    print(f"Healed depth:   {healed_depth.size}\n")

    # Setup pipeline
    print("Loading pipeline...")
    pipe = setup_pipeline()
    print("Pipeline ready.\n")

    # Generate healed versions
    print("--- Generating HEALED reconstructions ---\n")
    healed_images = {}
    for mat_name, mat_desc in MATERIALS:
        print(f"  {mat_name}:")
        imgs = generate(pipe, healed_depth, mat_name, mat_desc, seeds)
        healed_images[mat_name] = imgs[0]  # use first seed for comparison

    # Generate matching originals with same seeds for fair comparison
    print("\n--- Generating ORIGINAL reconstructions (matching seeds) ---\n")
    original_images = {}
    for mat_name, mat_desc in MATERIALS:
        print(f"  {mat_name}:")
        imgs = generate(pipe, orig_depth, mat_name, mat_desc, seeds)
        original_images[mat_name] = imgs[0]  # use first seed for comparison

    # Create comparison grid
    print("\nCreating comparison grid...")
    create_comparison_grid(original_images, healed_images, orig_depth, healed_depth)

    # Also create a healed-only grid
    print("Creating healed-only grid...")
    from src.reconstruct_sculptural import create_grid
    all_healed = {}
    for mat_name, mat_desc in MATERIALS:
        # Reload both seeds
        imgs = []
        for s in seeds:
            p = OUTPUT_DIR / f"healed_{mat_name}_seed{s}.png"
            imgs.append(Image.open(p))
        all_healed[mat_name] = imgs

    cell = 512
    rows = len(all_healed)
    n_seeds = len(seeds)
    cols = 1 + n_seeds
    label_width = 180
    margin_top = 50
    grid_w = label_width + cols * cell
    grid_h = margin_top + rows * cell
    grid = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(grid)
    draw.text((label_width + 10, 10),
              "Healed Face - Sculptural Reconstruction (ControlNet 0.95)",
              fill=(220, 220, 220))
    headers = ["Healed Depth"] + [f"Seed {s}" for s in seeds]
    for c, header in enumerate(headers):
        draw.text((label_width + c * cell + 10, 30), header, fill=(160, 160, 160))
    depth_resized = healed_depth.resize((cell, cell))
    for row, (mat_name, images) in enumerate(all_healed.items()):
        y = margin_top + row * cell
        label = mat_name.replace("_", " ").title()
        draw.text((10, y + cell // 2 - 10), label, fill=(200, 200, 200))
        grid.paste(depth_resized, (label_width, y))
        for col, img in enumerate(images):
            grid.paste(img.resize((cell, cell)), (label_width + (col + 1) * cell, y))
    grid.save(OUTPUT_DIR / "healed_sculptural_grid.png")
    print(f"Saved healed grid: {OUTPUT_DIR / 'healed_sculptural_grid.png'}")

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
