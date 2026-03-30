"""Phase 3.2: SDXL + ControlNet depth reconstruction comparison."""
import torch
import gc
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = "cuda"
print("=== SDXL ControlNet Depth Reconstruction ===")

# Check available VRAM
free_mem = torch.cuda.mem_get_info()[0] / 1024**3
total_mem = torch.cuda.mem_get_info()[1] / 1024**3
print(f"VRAM: {free_mem:.1f} / {total_mem:.1f} GB free")

# Clear everything first
gc.collect()
torch.cuda.empty_cache()

try:
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
    from diffusers import UniPCMultistepScheduler

    print("Loading SDXL ControlNet Depth...")
    # Use diffusers/controlnet-depth-sdxl-1.0 which is the official SDXL depth controlnet
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    print("Loading SDXL base pipeline...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    # Try xformers if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled")
    except Exception:
        print("xformers not available, using attention slicing only")

    print("SDXL pipeline loaded!")

    # Prepare depth input at 1024x1024
    from scipy.ndimage import zoom
    full_depth = np.load('data/processed/depth_map_smooth_15.npy')
    h, w = full_depth.shape
    depth_150 = zoom(full_depth.astype(np.float32), (150/h, 150/w), order=1).astype(np.uint8)
    depth_1024 = cv2.resize(depth_150, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    depth_img = Image.fromarray(depth_1024).convert("RGB")

    prompt = "sculptural bust of a man, neutral gray clay material, museum lighting, photorealistic sculpture, no color, monochromatic gray, professional studio photography, highly detailed, 8k"
    negative = "painting, cartoon, anime, illustration, colorful, bright colors, skin texture, hair detail, eyes open, modern clothing, low quality"

    print("\n--- Generating SDXL 1024x1024, seed 44 ---")
    generator = torch.Generator(device=device).manual_seed(44)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=depth_img,
        num_inference_steps=25,  # Fewer steps for VRAM
        controlnet_conditioning_scale=0.8,
        generator=generator,
    ).images[0]

    result.save("output/sdxl/sdxl_clay_s44_1024.png")
    print("  Saved: sdxl_clay_s44_1024.png")

    # Load SD 1.5 for comparison
    sd15_img = Image.open("output/reconstructions/sculptural/gray_clay_seed44.png")

    # Comparison
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#1a1a1a')

    axes[0].imshow(np.array(depth_img))
    axes[0].set_title('Depth Input', color='white', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(np.array(sd15_img))
    axes[1].set_title('SD 1.5 (512x512)', color='white', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(np.array(result))
    axes[2].set_title('SDXL (1024x1024)', color='white', fontsize=13)
    axes[2].axis('off')

    for ax in axes:
        ax.set_facecolor('#1a1a1a')
    fig.suptitle('SD 1.5 vs SDXL ControlNet Depth Comparison', color='#c4a35a', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('output/sdxl/sd15_vs_sdxl_comparison.png', dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print("  Saved: sd15_vs_sdxl_comparison.png")

    sdxl_success = True

except torch.cuda.OutOfMemoryError as e:
    print(f"\nSDXL VRAM exceeded: {e}")
    print("11GB RTX 2080 Ti insufficient for SDXL ControlNet at 1024x1024.")
    print("Attempting 768x768...")
    gc.collect()
    torch.cuda.empty_cache()

    try:
        depth_768 = cv2.resize(depth_150, (768, 768), interpolation=cv2.INTER_LINEAR)
        depth_img_768 = Image.fromarray(depth_768).convert("RGB")

        generator = torch.Generator(device=device).manual_seed(44)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=depth_img_768,
            num_inference_steps=20,
            controlnet_conditioning_scale=0.8,
            generator=generator,
            width=768,
            height=768,
        ).images[0]

        result.save("output/sdxl/sdxl_clay_s44_768.png")
        print("  Saved: sdxl_clay_s44_768.png (768x768)")
        sdxl_success = True
    except torch.cuda.OutOfMemoryError:
        print("768x768 also failed. SDXL skipped — insufficient VRAM.")
        sdxl_success = False
        # Write a note
        with open('output/sdxl/SKIPPED.txt', 'w') as f:
            f.write("SDXL ControlNet reconstruction skipped.\n")
            f.write("RTX 2080 Ti (11GB VRAM) insufficient for SDXL + ControlNet.\n")
            f.write("Tested 1024x1024 and 768x768, both OOM.\n")

except Exception as e:
    print(f"\nSDXL failed: {type(e).__name__}: {e}")
    sdxl_success = False
    with open('output/sdxl/SKIPPED.txt', 'w') as f:
        f.write(f"SDXL ControlNet reconstruction failed.\n")
        f.write(f"Error: {type(e).__name__}: {e}\n")

gc.collect()
torch.cuda.empty_cache()
print(f"\n=== SDXL Phase Complete (success={sdxl_success}) ===")
