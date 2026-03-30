"""
Study 2: Vernon Miller 1978 STURP — Sculptural Reconstructions

Generates clay (seed 44) and sandstone (seed 45) sculptural busts from
the Miller depth maps — both original and healed — then creates a
4-panel comparison grid.

Seeds 44 and 45 are used (locked from Study 1 best results).
ControlNet conditioning scale: 0.95 (same as Study 1).
"""

import torch
from PIL import Image, ImageDraw
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

PROJECT_ROOT = Path(__file__).parent.parent
STUDY2_DIR = PROJECT_ROOT / "output" / "study2_miller"
OUTPUT_DIR = STUDY2_DIR / "reconstructions"

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

SEEDS = [44, 45]


def setup_pipeline():
    """Initialize SD 1.5 + ControlNet Depth pipeline."""
    print("Loading ControlNet Depth...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    )
    print("Loading Stable Diffusion 1.5...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    print("Pipeline ready.\n")
    return pipe


def generate(pipe, depth_image, material_name, material_desc, variant, seeds):
    """Generate images for specific seeds. variant = 'original' or 'healed'."""
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
        fname = f"{variant}_{material_name}_seed{seed}.png"
        img.save(str(OUTPUT_DIR / fname))
        print(f"  Saved: {fname}")
    return images


def create_comparison_grid(orig_images, healed_images, orig_depth, healed_depth):
    """4-panel grid: original vs healed for clay and sandstone."""
    cell = 512
    n_materials = len(orig_images)
    margin_top = 75
    label_w = 160
    section_gap = 20
    cols_per_section = 2  # depth + render

    grid_w = label_w + cols_per_section * cell + section_gap + cols_per_section * cell
    grid_h = margin_top + n_materials * cell
    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)

    # Title
    draw.text((label_w + 10, 6),
              "Study 2: Vernon Miller 1978 STURP — Sculptural Reconstruction",
              fill=(220, 220, 220))

    # Section headers
    orig_x = label_w
    healed_x = label_w + cols_per_section * cell + section_gap

    draw.text((orig_x + 10, 26),
              "As Recorded on the Shroud (at death)",
              fill=(180, 140, 140))
    draw.text((orig_x + 10, 44), "Depth Map", fill=(110, 110, 110))
    draw.text((orig_x + cell + 10, 44), "Reconstruction", fill=(110, 110, 110))

    draw.text((healed_x + 10, 26),
              "Estimated Appearance Before the Passion (healed)",
              fill=(140, 180, 140))
    draw.text((healed_x + 10, 44), "Depth Map", fill=(110, 110, 110))
    draw.text((healed_x + cell + 10, 44), "Reconstruction", fill=(110, 110, 110))

    orig_d = orig_depth.resize((cell, cell))
    healed_d = healed_depth.resize((cell, cell))

    for row, mat in enumerate(orig_images.keys()):
        y = margin_top + row * cell
        label = mat.replace("_", " ").title()
        draw.text((10, y + cell // 2 - 10), label, fill=(200, 200, 200))

        grid.paste(orig_d, (orig_x, y))
        grid.paste(orig_images[mat].resize((cell, cell)), (orig_x + cell, y))

        grid.paste(healed_d, (healed_x, y))
        grid.paste(healed_images[mat].resize((cell, cell)), (healed_x + cell, y))

    path = OUTPUT_DIR / "study2_original_vs_healed.png"
    grid.save(str(path))
    print(f"\nSaved comparison grid: {path.name}")


def create_seed_grid(all_orig, all_healed, orig_depth, healed_depth):
    """Seed-level grid showing both seeds for each material/variant."""
    cell = 512
    rows = len(MATERIALS)  # clay, sandstone
    # cols: orig_depth | orig_s44 | orig_s45 | gap_col | healed_depth | healed_s44 | healed_s45
    label_w = 140
    dep_col = 1
    seed_cols = len(SEEDS)
    section_gap = 10
    col_count = dep_col + seed_cols + dep_col + seed_cols  # 6 content cols + gap

    grid_w = label_w + (dep_col + seed_cols) * cell + section_gap + (dep_col + seed_cols) * cell
    margin_top = 60
    grid_h = margin_top + rows * cell
    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)

    draw.text((label_w + 10, 5),
              "Study 2 — Miller 1978 STURP  |  Seeds 44 & 45",
              fill=(210, 210, 210))

    orig_x0 = label_w
    healed_x0 = label_w + (dep_col + seed_cols) * cell + section_gap
    headers_orig = ["Depth"] + [f"Seed {s}" for s in SEEDS]
    headers_healed = ["Depth (healed)"] + [f"Seed {s}" for s in SEEDS]
    for c, h in enumerate(headers_orig):
        draw.text((orig_x0 + c * cell + 8, 30), h, fill=(150, 150, 150))
    for c, h in enumerate(headers_healed):
        draw.text((healed_x0 + c * cell + 8, 30), h, fill=(150, 150, 150))

    orig_d = orig_depth.resize((cell, cell))
    healed_d = healed_depth.resize((cell, cell))

    for row, (mat_name, _) in enumerate(MATERIALS):
        y = margin_top + row * cell
        label = mat_name.replace("_", " ").title()
        draw.text((8, y + cell // 2 - 8), label, fill=(200, 200, 200))

        # Original side
        grid.paste(orig_d, (orig_x0, y))
        for c, img in enumerate(all_orig[mat_name]):
            grid.paste(img.resize((cell, cell)), (orig_x0 + (c + 1) * cell, y))

        # Healed side
        grid.paste(healed_d, (healed_x0, y))
        for c, img in enumerate(all_healed[mat_name]):
            grid.paste(img.resize((cell, cell)), (healed_x0 + (c + 1) * cell, y))

    path = OUTPUT_DIR / "study2_seed_grid.png"
    grid.save(str(path))
    print(f"Saved seed grid: {path.name}")


def run():
    print("=" * 65)
    print("  STUDY 2: SCULPTURAL RECONSTRUCTIONS")
    print("  Vernon Miller 1978 STURP — Clay + Sandstone, Seeds 44 & 45")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ControlNet depth images
    orig_depth = Image.open(STUDY2_DIR / "controlnet_depth_original_512.png").convert("RGB")
    healed_depth = Image.open(STUDY2_DIR / "controlnet_depth_healed_512.png").convert("RGB")
    print(f"Original depth: {orig_depth.size}")
    print(f"Healed depth:   {healed_depth.size}\n")

    # Setup pipeline
    print("Loading pipeline...")
    pipe = setup_pipeline()

    all_orig = {}
    all_healed = {}

    for mat_name, mat_desc in MATERIALS:
        print(f"--- {mat_name} ---")
        print(f"  Original:")
        imgs_orig = generate(pipe, orig_depth, mat_name, mat_desc, "original", SEEDS)
        all_orig[mat_name] = imgs_orig

        print(f"  Healed:")
        imgs_healed = generate(pipe, healed_depth, mat_name, mat_desc, "healed", SEEDS)
        all_healed[mat_name] = imgs_healed

    print("\nCreating comparison grids...")
    # Use seed 44 for the original vs healed comparison
    orig_best = {k: v[0] for k, v in all_orig.items()}
    healed_best = {k: v[0] for k, v in all_healed.items()}
    create_comparison_grid(orig_best, healed_best, orig_depth, healed_depth)
    create_seed_grid(all_orig, all_healed, orig_depth, healed_depth)

    print("\n" + "=" * 65)
    print("  DONE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    run()
