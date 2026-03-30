"""
Step 4b: Sculptural reconstruction — material-based rendering.

Renders the Shroud's depth topology as neutral sculptural busts in
classical materials. This approach lets the geometry speak without
the bias of pigmentation assumptions.

ControlNet conditioning scale = 0.95 for maximum depth map adherence.
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
FINAL_DIR = PROJECT_ROOT / "data" / "final"
OUTPUT_DIR = PROJECT_ROOT / "output" / "reconstructions" / "sculptural"

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
    "smile, happy, laughing"
)


def load_depth():
    """Load the approved ControlNet depth image."""
    path = FINAL_DIR / "controlnet_depth_512_rgb.png"
    return Image.open(path).convert("RGB")


def setup_pipeline():
    """Initialize SD 1.5 + ControlNet Depth pipeline."""
    print("Loading ControlNet Depth model...")
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


def generate(pipe, depth_image, material_name, material_desc, num_seeds=4, seed=42):
    """Generate images for a given material."""
    prompt = BASE_PROMPT.format(material_desc=material_desc)

    print(f"--- {material_name} ({num_seeds} seeds) ---")
    images = []
    for i in range(num_seeds):
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
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

        out_path = OUTPUT_DIR / f"{material_name}_seed{seed + i}.png"
        img.save(out_path)
        print(f"  Saved: {out_path.name}")

    return images


def create_grid(all_images, depth_image):
    """Create grid: depth | seeds for each material."""
    cell = 512
    rows = len(all_images)
    n_seeds = len(next(iter(all_images.values())))
    cols = 1 + n_seeds  # depth + seeds
    margin_top = 50
    label_width = 180

    grid_w = label_width + cols * cell
    grid_h = margin_top + rows * cell
    grid = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(grid)

    draw.text((label_width + 10, 10),
              "Shroud of Turin — Sculptural Reconstruction (ControlNet 0.95)",
              fill=(220, 220, 220))

    headers = ["Depth"] + [f"Seed {42 + i}" for i in range(n_seeds)]
    for c, header in enumerate(headers):
        draw.text((label_width + c * cell + 10, 30), header, fill=(160, 160, 160))

    depth_resized = depth_image.resize((cell, cell))

    for row, (material_name, images) in enumerate(all_images.items()):
        y = margin_top + row * cell
        label = material_name.replace("_", " ").title()
        draw.text((10, y + cell // 2 - 10), label, fill=(200, 200, 200))
        grid.paste(depth_resized, (label_width, y))
        for col, img in enumerate(images):
            grid.paste(img.resize((cell, cell)), (label_width + (col + 1) * cell, y))

    out_path = OUTPUT_DIR / "sculptural_comparison.png"
    grid.save(out_path)
    print(f"\nSaved grid: {out_path}")
    return out_path


def run():
    """Full sculptural reconstruction pipeline."""
    print("=" * 60)
    print("  SCULPTURAL RECONSTRUCTION")
    print("  ControlNet conditioning scale: 0.95")
    print("=" * 60 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    depth = load_depth()
    print(f"Depth image: {depth.size}\n")

    pipe = setup_pipeline()

    all_images = {}
    for material_name, material_desc in MATERIALS:
        images = generate(pipe, depth, material_name, material_desc, num_seeds=4)
        all_images[material_name] = images

    print("\nCreating comparison grid...")
    create_grid(all_images, depth)

    print("\n" + "=" * 60)
    print("  DONE — 12 images generated")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
