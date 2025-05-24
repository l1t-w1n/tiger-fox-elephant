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

    N_STEPS = 50
    pipe.scheduler.set_timesteps(N_STEPS, DEVICE)
    timesteps = pipe.scheduler.timesteps

    for i, t in enumerate(timesteps):
        lat_in = pipe.scheduler.scale_model_input(lat, t)

        # ---- UNet prediction ----
        with torch.no_grad():
            noise_pred = pipe.unet(lat_in, t, encoder_hidden_states=sd_e).sample
            lat = pipe.scheduler.step(noise_pred, t, lat).prev_sample

        # ---- CLIP guidance only on the last 60 % steps ----
        pct = i / (N_STEPS - 1)
        if pct < 0.4:
            continue                        # skip early, let UNet do the heavy lift

        lat.requires_grad_(True)
        img = pipe.vae.decode(1/0.18215 * lat).sample
        img_clamp = (img / 2 + 0.5).clamp(0, 1)
        img_224   = T.Resize((224,224))(img_clamp)

        img_f = clip.get_image_features(pixel_values=img_224)
        img_f = torch.nn.functional.normalize(img_f, dim=-1)

        loss = -torch.cosine_similarity(img_f, clip_e).mean()
        grad = torch.autograd.grad(loss, lat)[0]

        # σ-scaled step (smaller as noise level drops)
        sigma = pipe.scheduler.sigmas[i]
        lat = (lat - 10.0 * (sigma**2) * grad).detach()   # 10 = guidance strength

        print(f"{i+1:02d}/{N_STEPS}  σ={sigma:.3f}  loss={loss.item():.4f}")

    with torch.no_grad():
        final = pipe.vae.decode(1/0.18215 * lat).sample[0]
    return T.ToPILImage()((final / 2 + 0.5).clamp(0,1).cpu())


# ─────────────────── generate two images ───────────────────
prompts = {
    "cat_face": "a close-up photo of a cat face",
    "dog_face": "a close-up photo of a dog face",
}
for name, prm in prompts.items():
    img = clip_guided(prm)
    img.save(OUT_DIR / f"{name}.png")
    print("✓ saved →", OUT_DIR / f"{name}.png")
