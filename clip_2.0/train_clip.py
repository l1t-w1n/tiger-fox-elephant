from __future__ import annotations

import argparse
import math
import os
import random
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.optim import AdamW
from torch.utils.data.sampler import Sampler
from torchvision import models, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# 1. Hyper‑parameter container -------------------------------------------------
# -----------------------------------------------------------------------------

project_root = Path.cwd()
sys.path.append(str(project_root))

@dataclass
class CFG:
    # data
    img_root: int = project_root / "data/diffusion/cat_and_dog_face"
    img_size: int = 224

    # model
    device = torch.device("cuda")
    proj_dim: int = 256
    txt_width: int = 256
    txt_layers: int = 4
    txt_heads: int = 8
    dropout: float = 0.1

    # optimisation
    batch: int = 128  # must be even (half cat / half dog)
    epochs: int = 30
    lr: float = 1e-4
    wd: float = 1e-2

    # misc
    out_dir: Path = project_root / "clip_2.0/out"
    seed: int = 42

    # templates
    cat_templates: Tuple[str, ...] = (
        "a photo of a cat",
        "a close‑up photo of a cat face",
        "a studio portrait of a cat",
        "a cute domestic cat",
        "an image of a feline",
    )
    dog_templates: Tuple[str, ...] = (
        "a photo of a dog",
        "a close‑up photo of a dog face",
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
    """Loads images from a single folder and makes synthetic captions."""

    def __init__(self, cfg: CFG, train: bool, split: float = 0.8):
        self.cfg = cfg
        self.train = train

        # discover files – expect cat_#### and dog_#### patterns
        all_files = [p for p in cfg.img_root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        cats = sorted([p for p in all_files if p.stem.lower().startswith("cat_")])
        dogs = sorted([p for p in all_files if p.stem.lower().startswith("dog_")])
        assert cats and dogs, "No cat_* or dog_* images found in img_root"

        # 80/20 stratified split
        split_c = int(len(cats) * split)
        split_d = int(len(dogs) * split)
        if train:
            self.samples = [(p, 0) for p in cats[:split_c]] + [(p, 1) for p in dogs[:split_d]]
        else:
            self.samples = [(p, 0) for p in cats[split_c:]] + [(p, 1) for p in dogs[split_d:]]

        tfs = [
            transforms.Resize(cfg.img_size + 16)
        ]
        
        if train:
            tfs += [
                transforms.RandomCrop(cfg.img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ]
        else:
            tfs += [
                transforms.CenterCrop(cfg.img_size)
            ]
            
        tfs += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.samples)

    def _make_caption(self, label: int) -> str:
        template = random.choice(self.cfg.cat_templates if label == 0 else self.cfg.dog_templates)
        adj = random.choice(self.cfg.adjectives)
        return template.replace("a ", f"a {adj} ", 1)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        caption = self._make_caption(label)
        return img, caption, label


class BalancedBatchSampler(Sampler[List[int]]):
    """Ensures each batch has equal cat & dog images."""

    def __init__(self, labels: List[int], batch_size: int):
        self.cat_idx = [i for i, y in enumerate(labels) if y == 0]
        self.dog_idx = [i for i, y in enumerate(labels) if y == 1]
        self.batch = batch_size
        assert self.batch % 2 == 0, "batch size must be even"
        self.num_batches = min(len(self.cat_idx), len(self.dog_idx)) // (self.batch // 2)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # reshuffle every epoch
        cat_perm = np.random.permutation(self.cat_idx)
        dog_perm = np.random.permutation(self.dog_idx)
        half = self.batch // 2
        for i in range(self.num_batches):
            yield list(cat_perm[i * half:(i + 1) * half]) + list(dog_perm[i * half:(i + 1) * half])


# -----------------------------------------------------------------------------
# 3. Model ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Tiny Transformer encoder for caption templates."""

    def __init__(self, vocab_size: int, width: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Parameter(torch.randn(77, width) * 0.01)  # 77 tokens max as in CLIP
        enc_layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.ln_final = nn.LayerNorm(width)

    def forward(self, token_ids: torch.Tensor):
        x = self.token_emb(token_ids) + self.pos_emb[: token_ids.size(1)]
        x = self.transformer(x)
        x = self.ln_final(x)

        # first (left-most) occurrence of eos_token_id in each row
        eos_id = self.token_emb.weight.size(0) - 1          # = tokenizer.eos_token_id
        eos_pos = (token_ids == eos_id).float().argmax(dim=1)
        return x[torch.arange(x.size(0)), eos_pos]


class CLIP(nn.Module):
    def __init__(self, cfg: CFG, tokenizer):
        super().__init__()
        # vision
        self.vision_backbone = models.resnet50(weights='IMAGENET1K_V1')
        vis_out = self.vision_backbone.fc.in_features
        self.vision_backbone.fc = nn.Identity()
        self.img_proj = nn.Linear(vis_out, cfg.proj_dim)
        # text
        self.text_encoder = TextEncoder(tokenizer.vocab_size, cfg.txt_width, cfg.txt_layers, cfg.txt_heads, cfg.dropout)
        self.txt_proj = nn.Linear(cfg.txt_width, cfg.proj_dim)
        # temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.tokenizer = tokenizer

    def encode_image(self, img):
        x = self.vision_backbone(img)
        x = self.img_proj(x)
        return F.normalize(x, dim=-1)

    def encode_text(self, captions: List[str]):
        tokens = self.tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        token_ids = tokens["input_ids"].to(next(self.parameters()).device)
        x = self.text_encoder(token_ids)
        x = self.txt_proj(x)
        return F.normalize(x, dim=-1)

    def forward(self, img, captions):
        img_emb = self.encode_image(img)
        txt_emb = self.encode_text(captions)
        scale = self.logit_scale.exp()
        logits_img = scale * img_emb @ txt_emb.t()
        return logits_img, logits_img.t()


# -----------------------------------------------------------------------------
# 4. Tokeniser helper ----------------------------------------------------------
# -----------------------------------------------------------------------------

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError:
    AutoTokenizer = None


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    if tok.pad_token is None:                       # add new [PAD] = id 0
        tok.add_special_tokens({'pad_token': '[PAD]'})
    tok.padding_side = "right"
    return tok



# -----------------------------------------------------------------------------
# 5. Loss, accuracy, train & eval ---------------------------------------------
# -----------------------------------------------------------------------------

def clip_loss(img_logits, txt_logits):
    gt = torch.arange(img_logits.size(0), device=img_logits.device)
    return (F.cross_entropy(img_logits, gt) + F.cross_entropy(txt_logits, gt)) / 2


def accuracy(img_logits, txt_logits):
    gt = torch.arange(img_logits.size(0), device=img_logits.device)
    acc = (img_logits.argmax(1) == gt).float().mean()
    acc += (txt_logits.argmax(1) == gt).float().mean()
    return (acc / 2).item()


# batch‑level training loop (logs every `log_interval` batches)

def train_epoch(model, loader, opt, scaler, device, writer, epoch: int, log_interval: int = 50, global_step: int = 0):
    model.train()
    tot_loss = tot_acc = 0.0
    for batch_idx, (imgs, caps, _) in enumerate(tqdm(loader, leave=False)):
        imgs = imgs.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
            l_img, l_txt = model(imgs, caps)
            loss = clip_loss(l_img, l_txt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        batch_acc = accuracy(l_img.detach(), l_txt.detach())
        tot_loss += loss.item() * imgs.size(0)
        tot_acc += batch_acc * imgs.size(0)

        # --- TensorBoard batch logging -------------------------------------
        writer.add_scalar("Batch/Loss", loss.item(), global_step)
        writer.add_scalar("Batch/Accuracy", batch_acc, global_step)
        global_step += 1

        if batch_idx % log_interval == 0:
            tqdm.write(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | loss {loss.item():.4f} | acc {batch_acc:.4f}")

    n = len(loader.dataset)
    return (tot_loss / n, tot_acc / n, global_step)


@torch.inference_mode()
def eval_epoch(model, loader, device):
    model.eval()
    tot_acc = 0.0
    for imgs, caps, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        l_img, l_txt = model(imgs, caps)
        tot_acc += accuracy(l_img, l_txt) * imgs.size(0)
    return tot_acc / len(loader.dataset)


# -----------------------------------------------------------------------------
# 6. Entry‑point ---------------------------------------------------------------
# -----------------------------------------------------------------------------


def main():
    cfg = CFG()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = CFG.device
    writer = SummaryWriter(cfg.out_dir / "tb")

    # datasets & loaders
    train_ds = CatDogDataset(cfg, train=True)
    val_ds   = CatDogDataset(cfg, train=False)
    train_sampler = BalancedBatchSampler([lbl for _, lbl in train_ds.samples], cfg.batch)
    train_loader  = data.DataLoader(train_ds, batch_sampler=train_sampler, num_workers=os.cpu_count(), pin_memory=True)
    val_loader    = data.DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    tokenizer = get_tokenizer()
    model = CLIP(cfg, tokenizer).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler("cuda", enabled=False)

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        start = time.time()
        train_loss, train_acc, global_step = train_epoch(model, train_loader, opt, scaler, device, writer, epoch, global_step=global_step)
        val_acc = eval_epoch(model, val_loader, device)
        duration = time.time() - start

        # epoch‑level logging --------------------------------------------------
        writer.add_scalar("Epoch/Loss", train_loss, epoch)
        writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch)
        writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)
        writer.add_scalar("Epoch/LR", opt.param_groups[0]["lr"], epoch)

        tqdm.write(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | {duration/60:.1f} min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = cfg.out_dir / "best.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, ckpt_path)
            tqdm.write(f"✔ Saved best checkpoint to {ckpt_path} (val_acc {val_acc:.4f})")

    writer.close()
    print("Training complete! Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
