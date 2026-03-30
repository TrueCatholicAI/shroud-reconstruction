"""Task B: Full-body ControlNet reconstructions from depth map."""
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

device = "cuda"
print("=== Full-Body Sculptural Reconstruction ===")

# Load full-body depth
depth = np.load('output/full_body/depth_body_smooth.npy')  # 618x300
print(f"Body depth: {depth.shape}, range [{depth.min()}, {depth.max()}]")

# Create healed (symmetrized) version
rows, cols = depth.shape
mid_x = cols // 2
healed = depth.astype(np.float64).copy()
for y in range(rows):
    for x in range(cols):
        mirror_x = 2 * mid_x - x
        if 0 <= mirror_x < cols:
            healed[y, x] = (depth[y, x].astype(np.float64) + depth[y, mirror_x].astype(np.float64)) / 2.0
healed = gaussian_filter(healed, sigma=2.5)
healed = np.clip(healed, 0, 255).astype(np.uint8)
np.save('output/full_body/depth_body_healed.npy', healed)
cv2.imwrite('output/full_body/depth_body_healed.png', healed)
print(f"Healed body depth: {healed.shape}, range [{healed.min()}, {healed.max()}]")

# Resize to 512x768 portrait aspect for SD 1.5
# Original is 618x300 (height x width) ~2.06:1 aspect
# 512x768 would be width x height = 512 wide, 768 tall... actually SD wants W x H
# Let's use 384x768 (1:2 aspect, closer to body proportions)
target_w, target_h = 384, 768

depth_cn = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
healed_cn = cv2.resize(healed, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

cv2.imwrite('output/full_body/controlnet_depth_original.png', depth_cn)
cv2.imwrite('output/full_body/controlnet_depth_healed.png', healed_cn)
print(f"ControlNet inputs: {target_w}x{target_h}")

# Load pipeline
print("Loading ControlNet + SD 1.5...")
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

negative = "painting, cartoon, anime, illustration, colorful, bright colors, skin texture, modern clothing, background, text"

configs = [
    ("gray clay", "sculptural full body statue of a man, neutral gray clay material, museum lighting, photorealistic sculpture, no color, monochromatic gray, standing figure, arms crossed over pelvis, classical sculpture", "clay"),
    ("sandstone", "sculptural full body statue of a man, warm sandstone material, museum lighting, photorealistic ancient sculpture, no color, monochromatic, standing figure, arms crossed over pelvis, classical sculpture", "sandstone"),
]

for depth_type, depth_arr, cn_path, suffix in [
    ("original", depth_cn, 'output/full_body/controlnet_depth_original.png', 'original'),
    ("healed", healed_cn, 'output/full_body/controlnet_depth_healed.png', 'healed'),
]:
    depth_img = Image.open(cn_path).convert("RGB")

    for mat_name, prompt, mat_label in configs:
        print(f"\n--- {mat_name} / {depth_type} / seed 44 ---")
        generator = torch.Generator(device=device).manual_seed(44)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=depth_img,
            num_inference_steps=30,
            controlnet_conditioning_scale=0.95,
            generator=generator,
            width=target_w,
            height=target_h,
        ).images[0]

        out_path = f"output/full_body/{suffix}_{mat_label}_s44.png"
        result.save(out_path)
        print(f"  Saved: {out_path}")
        torch.cuda.empty_cache()

# Comparison grid
print("\nGenerating comparison grid...")
fig, axes = plt.subplots(2, 3, figsize=(18, 24))
fig.patch.set_facecolor('#1a1a1a')

panels = [
    (0, 0, 'output/full_body/controlnet_depth_original.png', 'Original Depth'),
    (0, 1, 'output/full_body/original_clay_s44.png', 'Clay (Original)'),
    (0, 2, 'output/full_body/original_sandstone_s44.png', 'Sandstone (Original)'),
    (1, 0, 'output/full_body/controlnet_depth_healed.png', 'Healed Depth'),
    (1, 1, 'output/full_body/healed_clay_s44.png', 'Clay (Healed)'),
    (1, 2, 'output/full_body/healed_sandstone_s44.png', 'Sandstone (Healed)'),
]

for r, c, path, title in panels:
    img = Image.open(path)
    axes[r, c].imshow(np.array(img), cmap='gray' if 'Depth' in title else None)
    axes[r, c].set_title(title, color='white', fontsize=13)
    axes[r, c].axis('off')
    axes[r, c].set_facecolor('#1a1a1a')

fig.suptitle('Full-Body Sculptural Reconstruction', color='#c4a35a', fontsize=18, y=0.98)
plt.tight_layout()
plt.savefig('output/full_body/fullbody_comparison_grid.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("Saved: fullbody_comparison_grid.png")
print("=== Full-Body Reconstruction Complete ===")
