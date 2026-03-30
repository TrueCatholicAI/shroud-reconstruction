"""
Step 4: Constrained facial reconstruction using Stable Diffusion + ControlNet.

Uses the approved 150x150 depth map as ControlNet depth conditioning to
generate a face that matches the Shroud's 3D geometry. The AI is constrained
by the measured depth topology — it cannot freely invent facial structure.

Biological parameters from forensic anthropology for a 1st-century
Levantine Jewish male, ~30-35 years old, ~180cm tall.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

PROJECT_ROOT = Path(__file__).parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "output" / "reconstructions"

# Anthropological constraints for a 1st-century Levantine Jewish male
# Based on forensic anthropology, population genetics, and archaeological evidence
BIOLOGICAL_PARAMS = {
    "ethnicity": "Levantine Semitic (1st century Judean)",
    "age": "30-35 years",
    "height": "175-185 cm (estimated from Shroud)",
    "build": "lean, muscular (manual laborer)",
    "skin": {
        "base": "olive-brown (Fitzpatrick type IV-V)",
        "variants": [
            ("light_olive", "light olive Mediterranean skin tone"),
            ("medium_olive", "medium olive-brown Middle Eastern skin tone"),
            ("darker_brown", "warm brown skin tone, darker complexion"),
        ],
    },
    "hair": "dark brown to black, wavy, shoulder-length",
    "beard": "short to medium beard, dark brown to black",
    "eyes": "brown eyes, deep-set",
    "facial_hair": "short beard following jawline, no mustache gap",
    "nose": "prominent, slightly aquiline (common in Levantine populations)",
    "brow": "prominent brow ridge",
}

# Prompt components
BASE_PROMPT = (
    "photorealistic portrait of a man, age 30-35, "
    "{skin_desc}, "
    "dark brown almost black wavy hair to shoulders, "
    "brown eyes, deep-set eyes under prominent brow ridge, "
    "short dark beard, "
    "prominent slightly aquiline nose, "
    "lean face with visible cheekbones, "
    "serious dignified expression, "
    "neutral studio lighting, "
    "forensic reconstruction quality, anatomically accurate, "
    "8k, detailed skin texture, photographic"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, artistic, stylized, "
    "western european features, pale white skin, blue eyes, blond hair, "
    "medieval art, icon, halo, religious imagery, crown of thorns, "
    "blood, wounds, scars, gore, "
    "deformed, disfigured, bad anatomy, extra fingers, "
    "blurry, low quality, watermark, text, "
    "smile, happy expression, laughing"
)


def load_controlnet_depth():
    """Load the approved ControlNet depth image."""
    depth_path = FINAL_DIR / "controlnet_depth_512_rgb.png"
    if not depth_path.exists():
        raise FileNotFoundError(f"ControlNet depth input not found: {depth_path}")
    img = Image.open(depth_path).convert("RGB")
    return img


def setup_pipeline():
    """Initialize SD 1.5 + ControlNet Depth pipeline."""
    print("Loading ControlNet Depth model...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    )

    print("Loading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Use efficient scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # RTX 2080 Ti (11GB) can fit SD1.5 + ControlNet in fp16 (~8GB)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    print("Pipeline ready.")
    return pipe


def generate_reconstruction(
    pipe,
    depth_image: Image.Image,
    skin_desc: str,
    variant_name: str,
    num_images: int = 2,
    seed: int = 42,
):
    """Generate reconstruction images for a given skin tone variant."""
    prompt = BASE_PROMPT.format(skin_desc=skin_desc)

    print(f"\n--- Generating: {variant_name} ---")
    print(f"  Prompt: {prompt[:100]}...")
    print(f"  Seed: {seed}, Images: {num_images}")

    images = []
    for i in range(num_images):
        generator = torch.Generator(device="cpu").manual_seed(seed + i)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=depth_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.8,
            generator=generator,
        )

        img = result.images[0]
        images.append(img)

        # Save individual image
        out_path = OUTPUT_DIR / f"{variant_name}_seed{seed + i}.png"
        img.save(out_path)
        print(f"  Saved: {out_path}")

    return images


def create_comparison_grid(all_images: dict, depth_image: Image.Image):
    """Create a comparison grid of all variants."""
    from PIL import ImageDraw, ImageFont

    variants = list(all_images.keys())
    n_variants = len(variants)
    n_per_variant = len(next(iter(all_images.values())))

    # Grid: depth map + variants
    cell_size = 512
    cols = n_per_variant + 1  # +1 for depth reference
    rows = n_variants

    grid = Image.new("RGB", (cols * cell_size, rows * cell_size + 40), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for row, variant_name in enumerate(variants):
        y_offset = row * cell_size + 40

        # First column: depth map
        if row == 0:
            depth_resized = depth_image.resize((cell_size, cell_size))
            grid.paste(depth_resized, (0, y_offset))
        else:
            grid.paste(depth_resized, (0, y_offset))

        # Variant images
        for col, img in enumerate(all_images[variant_name]):
            img_resized = img.resize((cell_size, cell_size))
            grid.paste(img_resized, ((col + 1) * cell_size, y_offset))

        # Label
        draw.text((10, y_offset + 5), variant_name.replace("_", " ").title(),
                   fill=(255, 255, 255))

    # Header
    draw.text((10, 5), "Shroud of Turin - Constrained Facial Reconstruction",
               fill=(255, 255, 255))

    grid_path = OUTPUT_DIR / "reconstruction_comparison.png"
    grid.save(grid_path)
    print(f"\nSaved comparison grid: {grid_path}")
    return grid_path


def run_reconstruction():
    """Full reconstruction pipeline."""
    print("=" * 60)
    print("  STEP 4: CONSTRAINED FACIAL RECONSTRUCTION")
    print("  Stable Diffusion 1.5 + ControlNet Depth")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load depth conditioning
    print("\n1. Loading ControlNet depth input...")
    depth_image = load_controlnet_depth()
    print(f"   Depth image: {depth_image.size}")

    # Save depth reference
    depth_image.save(OUTPUT_DIR / "depth_reference.png")

    # Setup pipeline
    print("\n2. Setting up SD + ControlNet pipeline...")
    pipe = setup_pipeline()

    # Generate variants
    print("\n3. Generating reconstructions...")
    skin_variants = BIOLOGICAL_PARAMS["skin"]["variants"]

    all_images = {}
    for variant_name, skin_desc in skin_variants:
        images = generate_reconstruction(
            pipe, depth_image, skin_desc, variant_name,
            num_images=2, seed=42,
        )
        all_images[variant_name] = images

    # Create comparison grid
    print("\n4. Creating comparison grid...")
    grid_path = create_comparison_grid(all_images, depth_image)

    print("\n" + "=" * 60)
    print("  RECONSTRUCTION COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    return all_images


if __name__ == "__main__":
    run_reconstruction()
