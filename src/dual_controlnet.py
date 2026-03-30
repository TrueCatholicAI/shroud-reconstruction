"""Phase 3.1: Dual ControlNet reconstruction — Depth + Canny edge simultaneously."""
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

device = "cuda"
print("=== Dual ControlNet: Depth + Canny ===")

# Load both ControlNet models
print("Loading ControlNet Depth...")
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16,
)

print("Loading ControlNet Canny...")
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16,
)

print("Loading SD 1.5 pipeline with dual ControlNet...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=[controlnet_depth, controlnet_canny],
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
pipe.enable_attention_slicing()
print("Pipeline loaded.")

# Prepare Enrie depth input (512x512)
# Use the existing ControlNet depth input from Study 1
# Need to find or create it
depth_150 = np.load('data/final/depth_healed_150.npy')  # Use original for better comparison
# Actually use the smoothed depth from study 1
from scipy.ndimage import zoom
full_depth = np.load('data/processed/depth_map_smooth_15.npy')
h, w = full_depth.shape
depth_150 = zoom(full_depth.astype(np.float32), (150/h, 150/w), order=1).astype(np.uint8)
depth_512 = cv2.resize(depth_150, (512, 512), interpolation=cv2.INTER_LINEAR)
depth_img = Image.fromarray(depth_512).convert("RGB")
print(f"Depth input: 512x512")

# Prepare Canny edge input from original Enrie photograph
enrie_src = cv2.imread('data/source/enrie_1931_face_hires.jpg', cv2.IMREAD_GRAYSCALE)
enrie_512 = cv2.resize(enrie_src, (512, 512), interpolation=cv2.INTER_LINEAR)
# Canny edge detection
canny = cv2.Canny(enrie_512, 50, 150)
canny_img = Image.fromarray(canny).convert("RGB")
print(f"Canny input: 512x512, edges detected: {np.sum(canny > 0)} pixels")

# Save inputs for reference
cv2.imwrite('output/dual_controlnet/input_depth_512.png', depth_512)
cv2.imwrite('output/dual_controlnet/input_canny_512.png', canny)

prompt = "sculptural bust of a man, neutral gray clay material, museum lighting, photorealistic sculpture, no color, monochromatic gray, professional studio photography"
negative = "painting, cartoon, anime, illustration, colorful, bright colors, skin texture, hair detail, eyes open, modern clothing"

# Dual ControlNet generation
print("\n--- Dual ControlNet: depth=0.7, canny=0.4, seed 44 ---")
generator = torch.Generator(device=device).manual_seed(44)
result_dual = pipe(
    prompt=prompt,
    negative_prompt=negative,
    image=[depth_img, canny_img],
    num_inference_steps=30,
    controlnet_conditioning_scale=[0.7, 0.4],
    generator=generator,
).images[0]
result_dual.save("output/dual_controlnet/dual_depth_canny_s44.png")
print("  Saved: dual_depth_canny_s44.png")

torch.cuda.empty_cache()

# Depth-only for comparison (same weight 0.7 for fair comparison)
print("\n--- Depth-only reference: depth=0.95, seed 44 ---")
# Need to reload single-controlnet pipeline
del pipe
torch.cuda.empty_cache()

pipe_single = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet_depth,
    torch_dtype=torch.float16,
)
pipe_single.scheduler = UniPCMultistepScheduler.from_config(pipe_single.scheduler.config)
pipe_single.to(device)
pipe_single.enable_attention_slicing()

generator = torch.Generator(device=device).manual_seed(44)
result_depth_only = pipe_single(
    prompt=prompt,
    negative_prompt=negative,
    image=depth_img,
    num_inference_steps=30,
    controlnet_conditioning_scale=0.95,
    generator=generator,
).images[0]
result_depth_only.save("output/dual_controlnet/depth_only_s44.png")
print("  Saved: depth_only_s44.png")

# Comparison figure
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.patch.set_facecolor('#1a1a1a')

inputs_and_results = [
    (np.array(depth_img), "Depth Input"),
    (np.array(canny_img), "Canny Edge Input"),
    (np.array(result_depth_only), "Depth Only (0.95)"),
    (np.array(result_dual), "Depth (0.7) + Canny (0.4)"),
]

for ax, (img, title) in zip(axes, inputs_and_results):
    ax.imshow(img)
    ax.set_title(title, color='white', fontsize=12)
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')

fig.suptitle('Enrie Study 1 - Dual ControlNet Comparison', color='#c4a35a', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('output/dual_controlnet/comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
plt.close()
print("\nSaved: comparison.png")

del pipe_single
torch.cuda.empty_cache()
print("=== Dual ControlNet Complete ===")
