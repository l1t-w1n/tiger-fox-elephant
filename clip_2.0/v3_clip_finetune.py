from __future__ import annotations

"""
Fine‑tune OpenAI CLIP (ViT‑B/32) on a balanced **cat‑face vs dog‑face** corpus *using a supervised/multi‑positive contrastive loss*.

❶ **Multi‑positive InfoNCE** (a.k.a. *Supervised Contrastive*): every image is
   encouraged to be close to *all* text prompts of *its own class* and far from
   all prompts of the opposite class. This removes the false‑negative trap that
   plagued vanilla CLIP loss when multiple correct captions existed per class.
❷ **Larger effective batch via gradient accumulation** (512) ⇒ many
   negatives → sharper alignment.
❸ **Slightly wider unfreezing**: the last *two* ViT blocks + projections +
   logit_scale. Still lightweight (<4 M trainable params) but offers more
   capacity than the single‑block variant.
❹ Class‑aware validation metrics:
      • *Class Top‑1*: does the highest‑sim text belong to the correct class?
      • *Diag retrieval* (optional) kept for reference, but no longer the only
        success metric.
"""

import math, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel, CLIPTokenizer

# ───────────────────────── cfg ──────────────────────────
ROOT = Path.cwd()

@dataclass
class CFG:
    img_root: Path = ROOT / "data/diffusion/cat_and_dog_face"
    out_dir:  Path = ROOT / "clip_2.0/out/v3"

    img_size: int = 224
    batch: int = 256            # micro‑batch
    accum: int = 2              # => 512 effective
    epochs: int = 6
    lr: float = 4e-5
    wd: float = 1e-4
    warmup_steps: int = 500
    seed: int = 42

    bf16: bool = True           # disable if older GPU/driver

    cat_templates: Tuple[str, ...] = (
        "a photo of a cat",
        "a close‑up photo of a cat face",
        "a portrait of a cat",
        "a headshot of a cat",
        "a cat face",
        "a cute feline closeup",
    )
    dog_templates: Tuple[str, ...] = (
        "a photo of a dog",
        "a close‑up photo of a dog face",
        "a portrait of a dog",
        "a headshot of a dog",
        "a dog face",
        "a cute canine closeup",
    )

    def save(self, p: Path):
        p.write_text("\n".join(f"{k}={v}" for k, v in asdict(self).items()))

# ─────────────────── dataset & collate ──────────────────
class CatDogDataset(data.Dataset):
    def __init__(self, cfg: CFG, train: bool, split: float = 0.8):
        files = [p for p in cfg.img_root.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        cats  = sorted(p for p in files if p.stem.startswith("cat_"))
        dogs  = sorted(p for p in files if p.stem.startswith("dog_"))
        if not cats or not dogs:
            raise RuntimeError("Images must be named cat_* / dog_*")
        sc, sd = int(len(cats)*split), int(len(dogs)*split)
        self.paths  = cats[:sc]+dogs[:sd] if train else cats[sc:]+dogs[sd:]
        self.labels = [0]*(sc if train else len(cats)-sc) + [1]*(sd if train else len(dogs)-sd)
        self.tfs = T.Compose([
            T.RandomResizedCrop(cfg.img_size, scale=(0.8,1.0)),
            T.ColorJitter(0.4,0.4,0.4,0.1),
            T.RandomHorizontalFlip(),
        ]) if train else T.Compose([
            T.Resize(256), T.CenterCrop(cfg.img_size)
        ])
        self.cat_t, self.dog_t = cfg.cat_templates, cfg.dog_templates
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.tfs(img)
        lbl = self.labels[idx]
        cap = random.choice(self.cat_t if lbl==0 else self.dog_t)
        return img, cap, lbl

def make_collate(processor):
    def _coll(batch):
        imgs, caps, labels = zip(*batch)
        enc = processor(images=list(imgs), text=list(caps), return_tensors="pt", padding=True)
        enc["labels"] = torch.tensor(labels)
        return enc
    return _coll

# ───────────── supervised‑contrastive loss ──────────────

def supcon_clip_loss(img_f, txt_f, labels, logit_scale):
    sim = logit_scale * img_f @ txt_f.T
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B,B]
    log_i = F.log_softmax(sim, dim=1)
    log_t = F.log_softmax(sim.T, dim=1)
    loss_i = -(log_i * mask.float()).sum(1) / mask.sum(1)
    loss_t = -(log_t * mask.float()).sum(1) / mask.sum(1)
    return (loss_i.mean() + loss_t.mean()) / 2

# ──────────────────── validation utils ──────────────────
@torch.inference_mode()
def build_proto(tok:CLIPTokenizer, model:CLIPModel, tmpls:dict, device):
    out = {}
    for cls, ts in tmpls.items():
        feats = [F.normalize(model.get_text_features(**tok(t, return_tensors="pt").to(device)), dim=-1) for t in ts]
        out[cls] = F.normalize(torch.stack(feats).mean(0), dim=-1)
    return out

@torch.inference_mode()
def class_top1(model:CLIPModel, loader, proto, device):
    model.eval(); proto_mat = torch.cat(list(proto.values()))
    ok = tot = 0
    for enc in loader:
        imgs = enc["pixel_values"].to(device)
        lbls = enc["labels"].to(device)
        feats = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
        preds = (feats @ proto_mat.T).argmax(1)
        ok += preds.eq(lbls).sum().item(); tot += lbls.size(0)
    return ok / tot

# ────────────────────────── main ────────────────────────

def main():
    cfg = CFG(); cfg.out_dir.mkdir(parents=True, exist_ok=True); cfg.save(cfg.out_dir/"hparams.txt")
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # precision setup
    bf16_ready = cfg.bf16 and torch.cuda.is_bf16_supported()
    amp_dtype  = torch.bfloat16 if bf16_ready else torch.float16
    scaler = None if bf16_ready else torch.amp.GradScaler("cuda")

    print(f"Loading CLIP ViT‑B/32 …  (autocast={amp_dtype})")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", do_resize=False, do_center_crop=False)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = processor.tokenizer

    # freeze / unfreeze  — last 4 ViT & last 2 text layers
    for p in model.parameters():
        p.requires_grad_(False)
    unf = [
        "visual_projection", "text_projection", "logit_scale",
        "vision_model.encoder.layers.8.", "vision_model.encoder.layers.9.",
        "vision_model.encoder.layers.10.", "vision_model.encoder.layers.11.",
        "text_model.encoder.layers.10.", "text_model.encoder.layers.11."
    ]
    for n, p in model.named_parameters():
        if any(n.startswith(t) for t in unf):
            p.requires_grad_(True)
    print(f"Trainable params ≈ {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")

    # compile (PyTorch 2) for speed
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # data
    tr_ds, va_ds = CatDogDataset(cfg, True), CatDogDataset(cfg, False)
    tr_ld = data.DataLoader(tr_ds, cfg.batch, True, num_workers=os.cpu_count(), pin_memory=True, collate_fn=make_collate(processor), drop_last=True)
    va_ld = data.DataLoader(va_ds, cfg.batch, False, num_workers=os.cpu_count(), pin_memory=True, collate_fn=make_collate(processor))

    # optimiser with WD mask
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if p.ndim < 2 or "bias" in n or "LayerNorm" in n else decay).append(p)
    opt = AdamW([
        {"params": decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, weight_decay=cfg.wd)

    total_steps = cfg.epochs * len(tr_ld) // cfg.accum
    sched = SequentialLR(
        opt,
        [LinearLR(opt, 0.1, 1.0, cfg.warmup_steps), CosineAnnealingLR(opt, total_steps - cfg.warmup_steps)],
        milestones=[cfg.warmup_steps],
    )

    writer = SummaryWriter(cfg.out_dir / "tb")

    best = 0.0
    global_step = 0

    for ep in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        tl = 0.0
        opt.zero_grad(set_to_none=True)

        progress = tqdm(tr_ld, desc=f"Epoch {ep}/{cfg.epochs-1}", unit="batch", leave=False)
        for step, enc in enumerate(progress):
            imgs = enc["pixel_values"].to(device, non_blocking=True)
            lbls = enc["labels"].to(device, non_blocking=True)
            txt  = {k: v.to(device, non_blocking=True) for k, v in enc.items() if k in {"input_ids", "attention_mask"}}

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out    = model(pixel_values=imgs, **txt)
                img_f  = F.normalize(out.image_embeds, dim=-1)
                txt_f  = F.normalize(out.text_embeds,  dim=-1)
                loss   = supcon_clip_loss(img_f, txt_f, lbls, model.logit_scale.exp())

            if scaler:  # FP16 path
                scaler.scale(loss / cfg.accum).backward()
            else:       # BF16 path
                (loss / cfg.accum).backward()

            if (step + 1) % cfg.accum == 0:
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                # keep temperature in sane range
                model.logit_scale.data.clamp_(0, math.log(100))

            tl += loss.item() * imgs.size(0)
            global_step += 1

        # ─ validation ─
        proto = build_proto(tokenizer, model, {"cat": cfg.cat_templates, "dog": cfg.dog_templates}, device)
        top1  = class_top1(model, va_ld, proto, device)
        tr_loss = tl / len(tr_ld.dataset)

        writer.add_scalar("epoch/loss", tr_loss, ep)
        writer.add_scalar("epoch/class_top1", top1, ep)
        print(f"Ep {ep:02d} | loss {tr_loss:.3f} | top‑1 {top1:.3f} | {(time.time() - epoch_start):.1f}s")

        if top1 > best:
            best = top1
            save_dir = cfg.out_dir / "best"
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  new best {best:.3f} saved → {save_dir}")

    writer.close()
    final_dir = cfg.out_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Final checkpoint saved to {final_dir}")
    print("Done. Best class‑top‑1:", best)

if __name__ == "__main__":
    main()
