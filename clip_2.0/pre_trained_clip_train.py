from __future__ import annotations

"""Fine‑tune an *already pre‑trained* CLIP (ViT‑B/32) on the cat‑vs‑dog set.

Why this version?
-----------------
Our tiny scratch model could not climb out of the random‑guess regime. The most
reliable fix on limited data is to **start from OpenAI’s CLIP checkpoint** and
fine‑tune *only* a small number of parameters. Here we:

1. Load **openai/clip‑vit‑base‑patch32** weights using the *transformers* CLIP
   implementation (same code that Hugging Face hosts).
2. **Freeze the vision & text encoders completely** and optimise just:
      – the 2 projection layers (`visual_projection`, `text_projection`)
      – a learnable **logit_scale**
   That is ~200 k parameters instead of 150 M.
3. Keep the contrastive InfoNCE loss; we are simply nudging the projections so
   the pre‑trained image/text spaces better separate *our* two classes.
4. Batch size 256 fits into 12 GB GPU with fp16; adjust if needed.
"""

import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.optim import AdamW
from torch.utils.data.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoProcessor  # transformers >=4.38

# -----------------------------------------------------------------------------
# 1. Hyper‑parameters ----------------------------------------------------------
# -----------------------------------------------------------------------------

project_root = Path.cwd()


@dataclass
class CFG:
    img_root: Path = project_root / "data/diffusion/cat_and_dog_face"
    img_size: int = 224  # CLIP ViT‑B expects 224

    batch: int = 256     # adjust to GPU
    epochs: int = 10     # fewer epochs needed with pre‑training
    lr: float = 5e-5
    wd: float = 1e-4

    out_dir: Path = project_root / "clip_finetune/out"
    seed: int = 42

    cat_templates: Tuple[str, ...] = (
        "a photo of a cat",
        "an image of a feline",
    )
    dog_templates: Tuple[str, ...] = (
        "a photo of a dog",
        "an image of a canine",
    )

    def save(self, path: Path):
        path.write_text("\n".join(f"{k}={v}" for k, v in asdict(self).items()))


# -----------------------------------------------------------------------------
# 2. Dataset -------------------------------------------------------------------
# -----------------------------------------------------------------------------


class CatDogDataset(data.Dataset):
    def __init__(self, cfg: CFG, train: bool, split: float = 0.8):
        files = [p for p in cfg.img_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        cats = sorted([p for p in files if p.stem.lower().startswith("cat_")])
        dogs = sorted([p for p in files if p.stem.lower().startswith("dog_")])
        if not cats or not dogs:
            raise RuntimeError("No cat_* or dog_* images found in img_root")

        split_c = int(len(cats) * split)
        split_d = int(len(dogs) * split)
        self.samples = (
            cats[:split_c] + dogs[:split_d] if train else cats[split_c:] + dogs[split_d:]
        )
        self.labels = [0]* (split_c if train else len(cats)-split_c) + [1]* (split_d if train else len(dogs)-split_d)

        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.processor.image_processor.size = {"height": cfg.img_size, "width": cfg.img_size}
        self.cat_tmpl = cfg.cat_templates
        self.dog_tmpl = cfg.dog_templates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        caption = random.choice(self.cat_tmpl if label == 0 else self.dog_tmpl)
        return img, caption, label


# ----------------------------------------------------------------------------
# 3. Helper: collate ----------------------------------------------------------
# ----------------------------------------------------------------------------


def collate(batch, processor):
    imgs, caps, labels = zip(*batch)
    enc = processor(text=list(caps), images=list(imgs), return_tensors="pt", padding=True)
    labels = torch.tensor(labels)
    return enc, labels


# ----------------------------------------------------------------------------
# 4. Training & evaluation ----------------------------------------------------
# ----------------------------------------------------------------------------


def clip_loss(img_logits, txt_logits):
    gt = torch.arange(img_logits.size(0), device=img_logits.device)
    return (F.cross_entropy(img_logits, gt) + F.cross_entropy(txt_logits, gt)) / 2


def accuracy(img_logits, txt_logits):
    gt = torch.arange(img_logits.size(0), device=img_logits.device)
    return ((img_logits.argmax(1) == gt).float().mean() + (txt_logits.argmax(1) == gt).float().mean()).item() / 2


@torch.inference_mode()
def eval_epoch(model, loader, device):
    model.eval()
    tot_acc = 0.0
    for enc, _ in tqdm(loader, leave=False):
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        tot_acc += accuracy(out.logits_per_image, out.logits_per_text) * enc["pixel_values"].size(0)
    return tot_acc / len(loader.dataset)


# ----------------------------------------------------------------------------
# 5. Main ---------------------------------------------------------------------
# ----------------------------------------------------------------------------


def main():
    cfg = CFG()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("▶ loading pre‑trained CLIP …")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # freeze everything except projection & logit_scale ---------------------
    for name, param in model.named_parameters():
        if name not in {"visual_projection", "text_projection", "logit_scale"}:
            param.requires_grad_(False)

    train_ds = CatDogDataset(cfg, True)
    val_ds   = CatDogDataset(cfg, False)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=lambda b: collate(b, processor),
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=os.cpu_count(),
        collate_fn=lambda b: collate(b, processor),
    )

    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler("cuda")
    writer = SummaryWriter(cfg.out_dir / "tb")

    best = 0.0
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        tl = ta = 0.0
        for enc, _ in tqdm(train_loader, leave=False):
            enc = {k: v.to(device) for k, v in enc.items()}
            optim.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(**enc)
                loss = clip_loss(out.logits_per_image, out.logits_per_text)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            # log_temperature range safeguard
            model.logit_scale.data.clamp_(0, math.log(100))

            acc = accuracy(out.logits_per_image.detach(), out.logits_per_text.detach())
            tl += loss.item() * enc["pixel_values"].size(0)
            ta += acc * enc["pixel_values"].size(0)
            writer.add_scalar("Batch/Loss", loss.item(), global_step)
            writer.add_scalar("Batch/Acc", acc, global_step)
            global_step += 1

        tr_loss = tl / len(train_loader.dataset)
        tr_acc  = ta / len(train_loader.dataset)
        val_acc = eval_epoch(model, val_loader, device)

        writer.add_scalar("Epoch/Loss", tr_loss, epoch)
        writer.add_scalar("Epoch/TrainAcc", tr_acc, epoch)
        writer.add_scalar("Epoch/ValAcc", val_acc, epoch)

        print(f"Ep {epoch:02d} | train_loss {tr_loss:.4f} | train_acc {tr_acc:.4f} | val_acc {val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            ck = cfg.out_dir / "best.pt"
            model.save_pretrained(cfg.out_dir / "best")
            print(f"✔ saved new best ({val_acc:.4f}) → {ck}")

    writer.close()
    print("done | best val_acc", best)


if __name__ == "__main__":
    main()
