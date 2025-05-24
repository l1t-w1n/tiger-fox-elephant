#!/usr/bin/env python3
# clip_guided_generation_fixed.py  (no CLI login needed)
# ------------------------------------------------------
import os, pathlib, torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPModel, AutoProcessor
from torchvision import transforms as T

# ───────────────────────── settings ─────────────────────────
CKPT_DIR = "clip_2.0/out/v3/final"         # fine-tuned CLIP
SD_NAME  = "runwayml/stable-diffusion-v1-5"
OUT_DIR  = pathlib.Path("gen_out"); OUT_DIR.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
STEPS    = 25
GUIDE    = 50.0
SEED     = 42
SIZE     = 512  # height = width and divisible by 8

# ─────────────────── load Stable Diffusion ──────────────────
print("▸ loading Stable Diffusion …")
pipe = StableDiffusionPipeline.from_pretrained(
    SD_NAME,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(DEVICE)

# ───────────────────── load CLIP judge ──────────────────────
print("▸ loading fine-tuned CLIP …")
clip = CLIPModel.from_pretrained(CKPT_DIR).to(DEVICE).eval()
proc = AutoProcessor.from_pretrained(CKPT_DIR)

@torch.no_grad()
def clip_txt_emb(prompt: str):
    tok = proc.tokenizer(prompt, return_tensors="pt").to(DEVICE)
    emb = clip.get_text_features(**tok)
    return torch.nn.functional.normalize(emb, dim=-1)

@torch.no_grad()
def sd_cond_emb(prompt: str):
    tok = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    ).to(DEVICE)
    return pipe.text_encoder(tok.input_ids)[0]   # [1,77,768]

# ───────────────────── guided sampler ───────────────────────
def clip_guided(prompt: str):
    g      = torch.Generator(DEVICE).manual_seed(SEED)
    clip_e = clip_txt_emb(prompt)
    sd_e   = sd_cond_emb(prompt)

    lat = torch.randn(
        (1, pipe.unet.config.in_channels, SIZE//8, SIZE//8),
        generator=g, device=DEVICE, dtype=torch.float16
    )

    pipe.scheduler.set_timesteps(STEPS, DEVICE)
    for i, t in enumerate(pipe.scheduler.timesteps):
        lat_in = pipe.scheduler.scale_model_input(lat, t)

        # --- UNet denoise prediction (no grad) ---
        with torch.no_grad():
            noise_pred = pipe.unet(lat_in, t, encoder_hidden_states=sd_e).sample
            lat = pipe.scheduler.step(noise_pred, t, lat).prev_sample

        # --- VAE decode + CLIP loss (with grad) ---
        lat.requires_grad_(True)                       # attach graph
        img = pipe.vae.decode(1/0.18215 * lat).sample  # [-1,1]
        img_clip = (img / 2 + 0.5).clamp(0, 1)
        img_224  = T.Resize((224,224))(img_clip)

        img_f = clip.get_image_features(pixel_values=img_224)
        img_f = torch.nn.functional.normalize(img_f, dim=-1)

        loss = -torch.cosine_similarity(img_f, clip_e).mean()
        grad = torch.autograd.grad(loss, lat)[0]

        lat = (lat - GUIDE * grad).detach()            # gradient step
        print(f"step {i+1:02d}/{STEPS}  clip-loss {loss.item():.4f}")

    # final decode (no grad needed)
    with torch.no_grad():
        final = pipe.vae.decode(1/0.18215 * lat).sample[0]
    final = (final / 2 + 0.5).clamp(0,1).cpu()
    return T.ToPILImage()(final)

# ─────────────────── generate two images ───────────────────
prompts = {
    "cat_face": "a close-up photo of a cat face",
    "dog_face": "a close-up photo of a dog face",
}
for name, prm in prompts.items():
    img = clip_guided(prm)
    img.save(OUT_DIR / f"{name}.png")
    print("✓ saved →", OUT_DIR / f"{name}.png")
