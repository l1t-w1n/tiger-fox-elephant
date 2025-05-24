#!/usr/bin/env python3
# compare_guidance_no_cli.py
# ----------------------------------------------------------
import pathlib, os, torch, random, math, time
import torchvision.transforms as T
from tqdm.auto import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import CLIPModel, AutoProcessor
from torchvision.transforms.functional import resize, to_pil_image

# ────────────────────────── CONFIG ─────────────────────────
PROMPT      = "a close-up photo of a dog face"
N_IMAGES    = 32                          # per strategy
OUT_ROOT    = pathlib.Path("clip_2.0/cmp_out_nc")  # output folder
DATASET_DIR = pathlib.Path("data/diffusion/cat_and_dog_face")
CLIP_TUNED  = "clip_2.0/out/v3/final"     # your fine-tuned checkpoint
SD_NAME     = "runwayml/stable-diffusion-v1-5"
HF_TOKEN    = "hf_your_token_here"        # leave "" if already cached
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

STEPS       = 50        # Euler steps
GUIDE_MAX   = 10.0      # top guidance strength after ramp
IMG_SIZE    = 512

# ─────────────────────── LOAD MODELS ──────────────────────
print("• Stable Diffusion …")
pipe = StableDiffusionPipeline.from_pretrained(
    SD_NAME, torch_dtype=torch.float16,
    safety_checker=None, use_auth_token=HF_TOKEN or None
).to(DEVICE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

print("• CLIPs …")
clip_base  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
clip_tuned = CLIPModel.from_pretrained(CLIP_TUNED).to(DEVICE).eval()
proc_base  = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
proc_tuned = AutoProcessor.from_pretrained(CLIP_TUNED)

metric_clip, metric_proc = clip_base, proc_base   # for distance metric

# ────────────────── DATASET EMBEDDINGS (ONCE) ─────────────
cache = OUT_ROOT / "dataset_feats.pt"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

if cache.exists():
    feats_ds = torch.load(cache)
else:
    print("• encoding dataset images once (this may take a minute)")
    tf224 = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),                # gives 0-1 float tensor C×H×W
    ])
    feats = []
    img_paths = sorted(DATASET_DIR.glob("*.[jp][pn]*g"))
    for pth in tqdm(img_paths, desc="embed-dataset"):
        ten = tf224(Image.open(pth).convert("RGB")).unsqueeze(0).to(DEVICE)
        feat = metric_clip.get_image_features(pixel_values=ten)
        feats.append(torch.nn.functional.normalize(feat, dim=-1).cpu())
    feats_ds = torch.cat(feats)
    torch.save(feats_ds, cache)

def nearest_cosine(feat):        # feat: [B,512] cpu tensor, l2-normed
    return 1 - (feat @ feats_ds.T).max(dim=1).values

# ────────────────── GENERATION HELPER ─────────────────────
def clip_guided_sample(seed, clip_model=None, clip_proc=None):
    g = torch.Generator(DEVICE).manual_seed(seed)

    # SD text-encoder embedding
    tok_sd = pipe.tokenizer(
        PROMPT, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    ).to(DEVICE)
    sd_emb = pipe.text_encoder(tok_sd.input_ids)[0]

    # optional CLIP embedding for guidance
    if clip_model:
        tok = clip_proc.tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
        clip_emb = torch.nn.functional.normalize(
            clip_model.get_text_features(**tok), dim=-1)

    lat = torch.randn(
        (1, pipe.unet.config.in_channels, IMG_SIZE//8, IMG_SIZE//8),
        generator=g, device=DEVICE, dtype=torch.float16)

    pipe.scheduler.set_timesteps(STEPS, DEVICE)
    sigmas = pipe.scheduler.sigmas

    for i, (t, sigma) in enumerate(zip(pipe.scheduler.timesteps, sigmas)):
        lat_in = pipe.scheduler.scale_model_input(lat, t)
        with torch.no_grad():
            eps = pipe.unet(lat_in, t, encoder_hidden_states=sd_emb).sample
            lat = pipe.scheduler.step(eps, t, lat).prev_sample

        if not clip_model:    # no guidance
            continue

        pct = (i+1)/STEPS
        if pct < 0.5:         # start guidance half-way
            continue

        lat.requires_grad_(True)
        img = pipe.vae.decode(1/0.18215*lat).sample
        img01 = (img/2+0.5).clamp(0,1)
        img_224 = resize(img01, (224,224))
        f = clip_model.get_image_features(pixel_values=img_224)
        f = torch.nn.functional.normalize(f, dim=-1)
        loss = -torch.cosine_similarity(f, clip_emb).mean()
        grad = torch.autograd.grad(loss, lat)[0]
        strength = GUIDE_MAX * (pct-0.5)/0.5 * (sigma**2)
        lat = (lat - strength*grad).detach()

    img = pipe.vae.decode(1/0.18215*lat).sample[0]
    return to_pil_image((img/2+0.5).clamp(0,1).cpu())

# ───────────────────── EVALUATION LOOP ────────────────────
strategies = {
    "plain": (None, None),
    "base" : (clip_base , proc_base ),
    "tuned": (clip_tuned, proc_tuned),
}

scores = {}
for tag, (cm, cp) in strategies.items():
    out_dir = OUT_ROOT / tag; out_dir.mkdir(exist_ok=True)
    dists = []
    for k in tqdm(range(N_IMAGES), desc=f"gen-{tag}"):
        img = clip_guided_sample(seed=10+k, clip_model=cm, clip_proc=cp)
        img.save(out_dir/f"{k:03d}.png")
        ten = resize((torch.tensor(img).float()/255).permute(2,0,1), (224,224)).unsqueeze(0)
        feat = metric_clip.get_image_features(pixel_values=ten.to(DEVICE))
        feat = torch.nn.functional.normalize(feat, dim=-1).cpu()
        dists.append( nearest_cosine(feat)[0].item() )
    scores[tag] = dists

# ───────────────────────── RESULTS ────────────────────────
print(f"\nPrompt:  {PROMPT}")
print("Cosine distance to nearest dataset image  (↓ better)")
for tag, ds in scores.items():
    print(f"{tag:<6}  mean {sum(ds)/len(ds):.3f}   median {sorted(ds)[len(ds)//2]:.3f}")

print("\nImages saved under:", OUT_ROOT.resolve())
