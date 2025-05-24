#!/usr/bin/env python3
# clip_guided_generation.py
# ---------------------------------------------------------
# CLIP-guided diffusion with a fine-tuned CLIP encoder
# (cats vs dogs) + Stable Diffusion v1.5.
# ---------------------------------------------------------

import os, math, pathlib, torch
from transformers import CLIPModel, AutoProcessor
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from torchvision import transforms as T

# ────────────────────────────── paths / settings ─────────────────────────────
ckpt_dir  = "clip_2.0/out/v3/final"        # ← your fine-tuned CLIP folder
sd_id     = "runwayml/stable-diffusion-v1-5"
out_dir   = pathlib.Path("gen_out"); out_dir.mkdir(exist_ok=True)

HF_TOKEN  = os.getenv("HUGGINGFACE_TOKEN", "hf_your_token_here")  # paste once
device    = "cuda" if torch.cuda.is_available() else "cpu"

steps     = 25          # diffusion steps (30-40 = higher fidelity, slower)
guidance  = 50.0        # CLIP guidance scale (20-80 typical)
seed      = 42          # RNG seed
size      = 512         # final image resolution (divisible by 8)

# ──────────────────────────── load Stable Diffusion ──────────────────────────
print("▸ loading Stable Diffusion …")
pipe = StableDiffusionPipeline.from_pretrained(
    sd_id,
    torch_dtype=torch.float16,
    safety_checker=None,          # optional
    use_auth_token=HF_TOKEN,      # auto-downloads weights once
)
pipe.to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# ───────────────────────────── load CLIP judge ───────────────────────────────
print("▸ loading fine-tuned CLIP …")
clip = CLIPModel.from_pretrained(ckpt_dir).to(device).eval()
proc = AutoProcessor.from_pretrained(ckpt_dir)

@torch.no_grad()
def clip_text_embed(prompt: str):
    tok = proc.tokenizer(prompt, return_tensors="pt").to(device)
    e   = clip.get_text_features(**tok)
    return torch.nn.functional.normalize(e, dim=-1)

@torch.no_grad()
def sd_text_embed(prompt: str):
    tok = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    ).to(device)
    return pipe.text_encoder(tok.input_ids)[0]      # shape [1,77,768]

def clip_guided(prompt: str):
    g = torch.Generator(device).manual_seed(seed)

    clip_e = clip_text_embed(prompt)     # CLIP space (for guidance)
    sd_e   = sd_text_embed(prompt)       # SD encoder_hidden_states

    # initial latent
    lat = torch.randn(
        (1, pipe.unet.config.in_channels, size//8, size//8),
        generator=g, device=device, dtype=torch.float16
    )

    pipe.scheduler.set_timesteps(steps)
    timesteps = pipe.scheduler.timesteps.to(device)

    for i, t in enumerate(timesteps):
        lat.requires_grad_(True)

        # 1) denoise prediction
        with torch.no_grad():
            noise_pred = pipe.unet(lat, t, encoder_hidden_states=sd_e).sample
            lat = pipe.scheduler.step(noise_pred, t, lat).prev_sample

        # 2) decode for CLIP
        with torch.no_grad():
            img = pipe.vae.decode(1/0.18215 * lat).sample       # [-1,1]
        img_clip = (img / 2 + 0.5).clamp(0, 1)
        img_224  = T.Resize((224, 224))(img_clip)

        img_f = clip.get_image_features(pixel_values=img_224)
        img_f = torch.nn.functional.normalize(img_f, dim=-1)

        # 3) CLIP guidance gradient
        loss = -torch.cosine_similarity(img_f, clip_e).mean()
        grad = torch.autograd.grad(loss, lat)[0]

        lat  = (lat - guidance * grad).detach()   # gradient step

        print(f"step {i+1:02d}/{steps}  clip-loss {loss.item():.4f}")

    with torch.no_grad():
        final = pipe.vae.decode(1/0.18215 * lat).sample[0]
    final = (final / 2 + 0.5).clamp(0, 1).cpu()
    return T.ToPILImage()(final)

# ─────────────────────────── generate two samples ────────────────────────────
tests = {
    "cat_face": "a close-up photo of a cat face",
    "dog_face": "a close-up photo of a dog face",
}

for name, prm in tests.items():
    pil_img = clip_guided(prm)
    pil_img.save(out_dir / f"{name}.png")
    print("✓ saved →", out_dir / f"{name}.png")
