# clip_guided_generation.py  –  zero-CLI version
# -------------------------------------------------------
# • Uses your fine-tuned CLIP judge (ckpt_dir)
# • Pulls Stable Diffusion 1.5 from Hugging Face on first run
#   via the `use_auth_token` kwarg.  Put your HF token in the
#   HF_TOKEN variable or export HUGGINGFACE_TOKEN=... first.

import os, math, pathlib, torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPModel, AutoProcessor
from torchvision import transforms as T
from PIL import Image

# ─────── settings ─────────────────────────────────────────
ROOT = pathlib.Path.cwd()
ckpt_dir   = ROOT / "clip_2.0/out/v3/best"             # your fine-tuned CLIP
sd_model   = "runwayml/stable-diffusion-v1-5"   # SD checkpoint name
out_dir    = ROOT / "clip_2.0/out/stable_diff"; out_dir.mkdir(exist_ok=True)

HF_TOKEN   = os.getenv("HUGGINGFACE_TOKEN", "hf_your_token_here")  # <─ paste once
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ─────── load Stable Diffusion (auto-download) ────────────
print("▸ loading Stable Diffusion… (first run downloads ~4 GB)")
pipe = StableDiffusionPipeline.from_pretrained(
    sd_model,
    torch_dtype=torch.float16,
    safety_checker=None,               # disable NSFW filter; optional
    use_auth_token=HF_TOKEN,           # ← this triggers automatic download
)
pipe.to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# ─────── load the fine-tuned CLIP judge ───────────────────
print("▸ loading fine-tuned CLIP…")
clip = CLIPModel.from_pretrained(ckpt_dir).to(device).eval()
proc = AutoProcessor.from_pretrained(ckpt_dir)

@torch.no_grad()
def text_embed(prompt: str):
    tok = proc.tokenizer(prompt, return_tensors="pt").to(device)
    e   = clip.get_text_features(**tok)
    return torch.nn.functional.normalize(e, dim=-1)

def clip_guided(prompt: str,
                steps: int = 25,
                guidance: float = 50.0,
                seed: int = 0,
                size: int = 512):
    g = torch.Generator(device).manual_seed(seed)
    text_e = text_embed(prompt)

    lat = torch.randn((1, pipe.unet.in_channels, size//8, size//8),
                      device=device, dtype=torch.float16, generator=g)

    sigmas = pipe.scheduler.sigmas.to(device)[:steps]

    for i, sigma in enumerate(sigmas):
        lat.requires_grad_(True)

        # --- standard diffusion step ---
        with torch.no_grad():
            v = pipe.unet(lat, sigma).sample
            lat = lat - sigma * v

        # --- decode & CLIP guidance ---
        img = pipe.vae.decode(1/0.18215 * lat).sample      # [-1,1]
        img_clamp = (img / 2 + 0.5).clamp(0, 1)
        img_224   = T.Resize((224, 224))(img_clamp)

        img_f = clip.get_image_features(pixel_values=img_224)
        img_f = torch.nn.functional.normalize(img_f, dim=-1)

        loss = -torch.cosine_similarity(img_f, text_e).mean()
        grad = torch.autograd.grad(loss, lat)[0]

        lat  = lat - guidance * (sigma**2) * grad
        lat  = lat.detach()

        print(f"step {i+1:02d}/{steps} | clip-loss {loss.item():.4f}")

    with torch.no_grad():
        final = pipe.vae.decode(1/0.18215 * lat).sample[0]
    final = (final / 2 + 0.5).clamp(0, 1).cpu()
    return T.ToPILImage()(final)

# ─────── quick demo ───────────────────────────────────────
for name, prm in {
    "cat_face": "a close-up photo of a cat face",
    "dog_face": "a close-up photo of a dog face",
}.items():
    img = clip_guided(prm, steps=25, guidance=50.0, seed=42)
    img.save(out_dir / f"{name}.png")
    print("✓ saved", out_dir / f"{name}.png")
