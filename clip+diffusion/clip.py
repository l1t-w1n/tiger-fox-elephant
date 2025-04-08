import torch
from tqdm import tqdm
import diffusers
from PIL import Image
import os
from diffusers import DDPMPipeline
from accelerate import Accelerator
import clip
import torch.nn.functional as F
from torchvision import transforms

from pathlib import Path
import sys
project_root = Path.cwd()
sys.path.append(str(project_root))
from diffusion import Config


def clip_guided_inference(config, prompt, guidance_scale=3.0, num_iters=5):
    """
    Generate images from an unconditional diffusion model, guided by CLIP to match `prompt`.
    
    :param config: Configuration object (paths, device, etc.).
    :param prompt: Text prompt (e.g. "a photo of a cat").
    :param guidance_scale: Strength of CLIP guidance. High values can destabilize images.
    :param num_iters: How many separate inference runs (each produces a grid).
    """
    device = config.device
    
    # 1) Load CLIP model & get text embedding
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    text_input = clip.tokenize([prompt]).to(device)
    with torch.inference_mode():
        text_features = clip_model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 2) Load your diffusion pipeline
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    pipeline = DDPMPipeline.from_pretrained(config.WEIGHTS_DIR).to(device)
    unet = accelerator.prepare(pipeline.unet)
    scheduler = pipeline.scheduler

    # 3) Run multiple iterations (each produces a grid)
    for iter_idx in range(num_iters):
        # 3a) Random initialization (x_T), in [-1,1]
        sample = torch.randn(
            (config.eval_batch_size, 3, config.image_size, config.image_size),
            device=device
        )

        # 3b) Denoising loop
        for t in tqdm(scheduler.timesteps, desc=f"Inference run {iter_idx+1}"):
            # --(i) UNet forward (no grad needed)
            with torch.inference_mode():
                noise_pred = unet(sample, t).sample

            # --(ii) Scheduler step => partial denoise
            sample = scheduler.step(noise_pred, t, sample).prev_sample

            # --(iii) CLIP guidance
            # We want to push 'sample' to better match the prompt
            sample.requires_grad_(True)

            # Convert to [0,1] for CLIP
            sample_0_1 = (sample + 1) / 2  
            # Resize to 224 (ViT-B/32 input)
            sample_0_1 = F.interpolate(sample_0_1, size=(224, 224), mode="bilinear", align_corners=False)
            
            # CLIP normalizing
            clip_input = transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )(sample_0_1)
            
            # Encode image, measure similarity
            image_features = clip_model.encode_image(clip_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features * text_features).sum()
            loss = -similarity  # we want to maximize similarity => minimize negative

            grad = torch.autograd.grad(loss, sample)[0]  # Grad wrt the original sample

            # Guidance step
            with torch.inference_mode():
                sample = sample - guidance_scale * grad
                sample = sample.clamp(-1, 1)  # Keep in [-1,1]

        # 3c) Convert final sample -> PIL and save
        sample_0_1 = (sample + 1) / 2
        sample_0_1 = sample_0_1.clamp(0, 1)
        images_np = sample_0_1.permute(0,2,3,1).cpu().numpy()  # [B,H,W,C]

        pil_images = []
        for i in range(images_np.shape[0]):
            arr = (images_np[i]*255).astype("uint8")
            pil_images.append(Image.fromarray(arr))

        # Make and save grid
        grid = diffusers.utils.make_image_grid(pil_images, rows=4, cols=4)
        out_path = config.output_dir / f"clip_guided_{prompt.replace(' ', '_')}_{iter_idx}.png"
        grid.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    config = Config()
    clip_guided_inference(config, prompt="a photo of a cat", guidance_scale=3.0)
    clip_guided_inference(config, prompt="a photo of a dog", guidance_scale=3.0)
