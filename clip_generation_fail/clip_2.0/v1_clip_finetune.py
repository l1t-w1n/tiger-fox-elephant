from __future__ import annotations

"""Train a tiny CLIP‑style model that distinguishes cat vs. dog images.

Latest fixes (v final)
----------------------
* **Unique captions** → append the image filename stem to every caption so there
  are no duplicate texts inside a batch. This boosts the InfoNCE signal on small
  data sets.
* **Clamp `logit_scale`** after every optimiser step to `[0, ln 100]` to avoid
  early saturation or infinities.
* **Complete script** – previous snippet was truncated; this is the full,
  runnable file.

Retains earlier tweaks: dropout = 0, batch = 64, freeze ResNet‑50 for 5 epochs,
then unfreeze with LR × 0.1.
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
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.optim import AdamW
from torch.utils.data.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Hyper‑parameter container -------------------------------------------------
# -----------------------------------------------------------------------------

project_root = Path.cwd()


@dataclass
class CFG:
    # data
    img_root: Path = project_root / "data/diffusion/cat_and_dog_face"
    img_size: int = 224

    # model
    proj_dim: int = 256
    txt_width: int = 256
    txt_layers: int = 4
    txt_heads: int = 8
    dropout: float = 0.0

    # optimisation
    batch: int = 64
    epochs: int = 30
    lr: float = 1e-4
    wd: float = 1e-2
    freeze_epochs: int = 5

    # misc
    out_dir: Path = project_root / "clip_2.0/out"
    seed: int = 42

    # caption templates
    cat_templates: Tuple[str, ...] = (
        "a photo of a cat",
        "a close-up photo of a cat face",
        "a studio portrait of a cat",
        "a cute domestic cat",
        "an image of a feline",
    )
    dog_templates: Tuple[str, ...] = (
        "a photo of a dog",
        "a close-up photo of a dog face",
        "a studio portrait of a dog",
        "a cute domestic dog",
        "an image of a canine",
    )
    adjectives: Tuple[str, ...] = (
        "happy",
        "sleepy",
        "fluffy",
        "outdoor",
        "indoor",
    )

    def save(self, path: Path):
        path.write_text("\n".join(f"{k}={v}" for k, v in asdict(self).items()))


# -----------------------------------------------------------------------------
# 2. Dataset + balanced sampler ------------------------------------------------
# -----------------------------------------------------------------------------


class CatDogDataset(data.Dataset):
    """Loads cat_####/dog_#### images and synthesises *unique* captions."""

    def __init__(self, cfg: CFG, train: bool, split: float = 0.8):
        self.cfg = cfg
        self.train = train

        files = [p for p in cfg.img_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        cats = sorted([p for p in files if p.stem.lower().startswith("cat_")])
        dogs = sorted([p for p in files if p.stem.lower().startswith("dog_")])
        if not cats or not dogs:
            raise RuntimeError("No cat_* or dog_* images found in img_root")

        split_c = int(len(cats) * split)
        split_d = int(len(dogs) * split)
        self.samples = (
            [(p, 0) for p in cats[:split_c]] + [(p, 1) for p in dogs[:split_d]] if train else
            [(p, 0) for p in cats[split_c:]] + [(p, 1) for p in dogs[split_d:]]
        )

        tf = [transforms.Resize(cfg.img_size + 16)]
        if train:
            tf += [transforms.RandomCrop(cfg.img_size),
                   transforms.RandomHorizontalFlip(0.5),
                   transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)]
        else:
            tf += [transforms.CenterCrop(cfg.img_size)]
        tf += [transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(tf)

    def __len__(self):
        return len(self.samples)

    def _make_caption(self, label: int, stem: str) -> str:
        tmpl = random.choice(self.cfg.cat_templates if label == 0 else self.cfg.dog_templates)
        adj = random.choice(self.cfg.adjectives)
        return tmpl.replace("a ", f"a {adj} ", 1) + f" ({stem})"

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self._make_caption(label, path.stem), label


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], batch: int):
        if batch % 2:
            raise ValueError("batch size must be even")
        self.batch = batch
        self.cat_idx = [i for i, y in enumerate(labels) if y == 0]
        self.dog_idx = [i for i, y in enumerate(labels) if y == 1]
        self.num_batches = min(len(self.cat_idx), len(self.dog_idx)) // (batch // 2)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        cats = np.random.permutation(self.cat_idx)
        dogs = np.random.permutation(self.dog_idx)
        half = self.batch // 2
        for i in range(self.num_batches):
            yield list(cats[i*half:(i+1)*half]) + list(dogs[i*half:(i+1)*half])


# -----------------------------------------------------------------------------
# 3. Model ---------------------------------------------------------------------
# -----------------------------------------------------------------------------


class TextEncoder(nn.Module):
    def __init__(self, vocab, width, layers, heads, dropout, eos):
        super().__init__()
        self.eos = eos
        self.token = nn.Embedding(vocab, width)
        self.pos = nn.Parameter(torch.randn(77, width) * 0.01)
        block = nn.TransformerEncoderLayer(width, heads, dropout=dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(block, layers)
        self.ln = nn.LayerNorm(width)

    def forward(self, ids, mask):
        x = self.token(ids) + self.pos[: ids.size(1)]
        x = self.tr(x, src_key_padding_mask=~mask.bool())
        x = self.ln(x)
        eos_pos = (ids == self.eos).float().argmax(1)
        return x[torch.arange(x.size(0), device=x.device), eos_pos]


class CLIP(nn.Module):
    def __init__(self, cfg: CFG, tok):
        super().__init__()
        # vision branch
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        vis_out = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.img_proj = nn.Linear(vis_out, cfg.proj_dim)
        # text branch
        self.text = TextEncoder(tok.vocab_size, cfg.txt_width, cfg.txt_layers, cfg.txt_heads, cfg.dropout, tok.eos_token_id)
        self.txt_proj = nn.Linear(cfg.txt_width, cfg.proj_dim)
        # temperature
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))
        self.tok = tok

    def encode_image(self, img):
        return F.normalize(self.img_proj(self.backbone(img)), dim=-1)

    def encode_text(self, caps):
        tokens = self.tok(caps, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(next(self.parameters()).device)
        return F.normalize(self.txt_proj(self.text(tokens["input_ids"], tokens["attention_mask"])), dim=-1)

    def forward(self, img, caps):
        im, tx = self.encode_image(img), self.encode_text(caps)
        scale = self.logit_scale.exp()
        logits = scale * im @ tx.t()
        return logits, logits.t()


# -----------------------------------------------------------------------------
# 4. Tokeniser helper ----------------------------------------------------------
# -----------------------------------------------------------------------------

from transformers import AutoTokenizer

def get_tokenizer():
    tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.padding_side = "right"
    return tok


# -----------------------------------------------------------------------------
# 5. Loss & metrics ------------------------------------------------------------
# -----------------------------------------------------------------------------

def clip_loss(li, lt):
    gt = torch.arange(li.size(0), device=li.device)
    return (F.cross_entropy(li, gt) + F.cross_entropy(lt, gt)) / 2


def accuracy(li, lt):
    gt = torch.arange(li.size(0), device=li.device)
    return ((li.argmax(1) == gt).float().mean() + (lt.argmax(1) == gt).float().mean()).item() / 2


# -----------------------------------------------------------------------------
# 6. Training helpers ----------------------------------------------------------
# -----------------------------------------------------------------------------

def train_epoch(model, loader, opt, scaler, device, writer, epoch, gstep):
    model.train()
    tl = ta = 0.0
    for imgs, caps, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
            li, lt = model(imgs, caps)
            loss = clip_loss(li, lt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # clamp temperature ----------------------------------------------
        model.logit_scale.data.clamp_(0, math.log(100))

        ba = accuracy(li.detach(), lt.detach())
        tl += loss.item() * imgs.size(0)
        ta += ba * imgs.size(0)
        writer.add_scalar("Batch/Loss", loss.item(), gstep)
        writer.add_scalar("Batch/Acc", ba, gstep)
        gstep += 1
    n = len(loader.dataset)
    return tl/n, ta/n, gstep


@torch.inference_mode()
def eval_epoch(model, loader, device):
    model.eval()
    acc = 0.0
    for imgs, caps, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        li, lt = model(imgs, caps)
        acc += accuracy(li, lt) * imgs.size(0)
    return acc / len(loader.dataset)


# -----------------------------------------------------------------------------
# 7. Entry-point ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    cfg = CFG()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # reproducibility ---------------------------------------------------------
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(cfg.out_dir / "tb")

    # data --------------------------------------------------------------------
    train_ds = CatDogDataset(cfg, True)
    val_ds   = CatDogDataset(cfg, False)

    train_loader = data.DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler([lbl for _, lbl in train_ds.samples], cfg.batch),
        num_workers=os.cpu_count(), pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=cfg.batch, shuffle=False,
        num_workers=os.cpu_count(), pin_memory=True,
    )

    # model -------------------------------------------------------------------
    tok = get_tokenizer()
    model = CLIP(cfg, tok).to(device)

    # freeze backbone for first `freeze_epochs` ------------------------------
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    # optimiser --------------------------------------------------------------
    rest_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.lr * 0.1},  # will be unfrozen later
        {"params": rest_params,                 "lr": cfg.lr},
    ], weight_decay=cfg.wd)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    gstep = 0
    best  = 0.0
    for epoch in range(cfg.epochs):
        if epoch == cfg.freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            print("🔓  Unfroze ResNet backbone")

        t0 = time.time()
        tr_loss, tr_acc, gstep = train_epoch(model, train_loader, opt, scaler, device, writer, epoch, gstep)
        val_acc = eval_epoch(model, val_loader, device)
        dt = (time.time() - t0) / 60

        writer.add_scalar("Epoch/Loss", tr_loss, epoch)
        writer.add_scalar("Epoch/TrainAcc", tr_acc, epoch)
        writer.add_scalar("Epoch/ValAcc", val_acc, epoch)
        writer.add_scalar("Epoch/LR", opt.param_groups[0]["lr"], epoch)

        tqdm.write(f"Ep {epoch:02d} | train_loss {tr_loss:.4f} | train_acc {tr_acc:.4f} | val_acc {val_acc:.4f} | {dt:.1f} min")
        if val_acc > best:
            best = val_acc
            ckpt = cfg.out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, ckpt)
            tqdm.write(f"✔ Saved best ckpt to {ckpt} (val_acc {val_acc:.4f})")

    writer.close()
    print("Training complete! Best val_acc:", best)


if __name__ == "__main__":
    main()
