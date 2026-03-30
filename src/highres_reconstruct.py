"""Phase 2.2 cont: ControlNet reconstructions from higher-res Miller depth maps."""
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

device = "cuda"
print("Loading ControlNet Depth + SD 1.5...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
pipe.enable_attention_slicing()
print("Pipeline loaded.")

prompt = "sculptural bust of a man, neutral gray clay material, museum lighting, photorealistic sculpture, no color, monochromatic gray, professional studio photography"
negative = "painting, cartoon, anime, illustration, colorful, bright colors, skin texture, hair detail, eyes open, modern clothing"

configs = [
    ("output/highres_miller/controlnet_depth_300x300_g21_512.png", "highres_300"),
    ("output/highres_miller/controlnet_depth_500x500_g31_512.png", "highres_500"),
]

for depth_path, label in configs:
    print(f"\n--- Generating: {label} (seed 44) ---")
    depth_img = Image.open(depth_path).convert("RGB")

    generator = torch.Generator(device=device).manual_seed(44)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=depth_img,
        num_inference_steps=30,
        controlnet_conditioning_scale=0.95,
        generator=generator,
    ).images[0]

    out_path = f"output/highres_miller/reconstruction_{label}_clay_s44.png"
    result.save(out_path)
    print(f"  Saved: {out_path}")
    torch.cuda.empty_cache()

# Also reconstruct original 150x150 for direct comparison
print("\n--- Generating: original 150x150 (seed 44) for comparison ---")
depth_150 = Image.open("output/study2_miller/controlnet_depth_original_512.png").convert("RGB")
generator = torch.Generator(device=device).manual_seed(44)
result_150 = pipe(
    prompt=prompt,
    negative_prompt=negative,
    image=depth_150,
    num_inference_steps=30,
    controlnet_conditioning_scale=0.95,
    generator=generator,
).images[0]
result_150.save("output/highres_miller/reconstruction_150_clay_s44.png")
print("  Saved: reconstruction_150_clay_s44.png")

# Comparison grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a1a')

imgs = [
    ("output/highres_miller/reconstruction_150_clay_s44.png", "150x150 + G15"),
    ("output/highres_miller/reconstruction_highres_300_clay_s44.png", "300x300 + G21 (FFT-filtered)"),
    ("output/highres_miller/reconstruction_highres_500_clay_s44.png", "500x500 + G31 (FFT-filtered)"),
]

for ax, (path, title) in zip(axes, imgs):
    img = Image.open(path)
    ax.imshow(np.array(img))
    ax.set_title(title, color='white', fontsize=12)
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')

fig.suptitle('Miller Resolution Comparison - Gray Clay Seed 44', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/highres_miller/reconstruction_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: reconstruction_comparison.png")
print("=== High-Res Reconstructions Complete ===")
