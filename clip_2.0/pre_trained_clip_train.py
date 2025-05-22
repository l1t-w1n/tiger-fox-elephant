from __future__ import annotations
"""
Fine-tune OpenAI’s CLIP (ViT-B/32) for a balanced cat-vs-dog-face dataset.

Changes vs. the original
------------------------
✔  Correct freeze logic – *only* projection layers + logit_scale stay trainable.  
✔  Unfreeze the **last ViT block** for a pinch of visual adaptation.  
✔  More diverse text templates (6× per class).  
✔  Lightweight image augmentations during training.  
✔  Smaller batch (64) – stronger InfoNCE signal for two classes.  
✔  Extra metric: **binary class accuracy** using prototypical text embeddings.  
✔  Clearer logging; TensorBoard still supported.

Expect > 95 % val accuracy in ≤ 4 epochs on 5 k + 5 k face crops.
"""

import math, os, random, time
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
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer, AutoProcessor

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

project_root = Path.cwd()


@dataclass
class CFG:
    img_root: Path = project_root / "data/diffusion/cat_and_dog_face"
    img_size: int = 224                     # CLIP ViT-B expects 224×224
    batch: int = 64                         # smaller batch ⇒ better negatives
    epochs: int = 10
    lr: float = 1e-4
    wd: float = 1e-4
    out_dir: Path = project_root / "clip_finetune/out"
    seed: int = 42

    # ✎ Diverse prompt templates
    cat_templates: Tuple[str, ...] = (
        "a photo of a cat",
        "a photo of a cat face",
        "a close-up photo of a cat face",
        "a portrait of a cat",
        "a headshot of a cat",
        "a closeup of a cat face",
    )
    dog_templates: Tuple[str, ...] = (
        "a photo of a dog",
        "a photo of a dog face",
        "a close-up photo of a dog face",
        "a portrait of a dog",
        "a headshot of a dog",
        "a closeup of a dog face",
    )

    def save(self, path: Path):
        path.write_text("\n".join(f"{k}={v}" for k, v in asdict(self).items()))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────────────────────


class CatDogDataset(data.Dataset):
    def __init__(self, cfg: CFG, train: bool, split: float = 0.8):
        files = [p for p in cfg.img_root.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        cats = sorted([p for p in files if p.stem.lower().startswith("cat_")])
        dogs = sorted([p for p in files if p.stem.lower().startswith("dog_")])
        if not cats or not dogs:
            raise RuntimeError("No cat_* or dog_* images found in img_root")

        # simple stratified split
        split_c = int(len(cats) * split)
        split_d = int(len(dogs) * split)
        self.samples = (
            cats[:split_c] + dogs[:split_d]
            if train else cats[split_c:] + dogs[split_d:]
        )
        self.labels = (
            [0] * (split_c if train else len(cats) - split_c)
            + [1] * (split_d if train else len(dogs) - split_d)
        )

        # Augmentations (only spatial / colour – CLIP normalisation later)
        self.transform = (
            T.Compose([
                T.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.RandomHorizontalFlip(),
            ])
            if train else
            T.Compose([
                T.Resize(256),
                T.CenterCrop(cfg.img_size),
            ])
        )

        self.cat_tmpl = cfg.cat_templates
        self.dog_tmpl = cfg.dog_templates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)            # PIL → augmented PIL
        label = self.labels[idx]
        caption = random.choice(self.cat_tmpl if label == 0 else self.dog_tmpl)
        return img, caption, label


# ──────────────────────────────────────────────────────────────────────────────
# 3. Collate
# ──────────────────────────────────────────────────────────────────────────────


def make_collate(processor):
    def collate(batch):
        imgs, caps, labels = zip(*batch)
        # processor applies CLIP’s resize-centre-crop-norm – we already cropped,
        # so just use its normalisation / tensor conversion
        enc = processor(
            text=list(caps),
            images=list(imgs),
            return_tensors="pt",
            padding=True,
        )
        enc["labels"] = torch.tensor(labels)
        return enc
    return collate


# ──────────────────────────────────────────────────────────────────────────────
# 4. Metrics
# ──────────────────────────────────────────────────────────────────────────────


def clip_loss(img_logits, txt_logits):
    gt = torch.arange(img_logits.size(0), device=img_logits.device)
    return (F.cross_entropy(img_logits, gt)
            + F.cross_entropy(txt_logits, gt)) / 2


@torch.no_grad()
def build_class_text_embeds(
    tokenizer: CLIPTokenizer,
    model: CLIPModel,
    templates: dict[str, Tuple[str, ...]],
    device,
):
    """Return ℓ2-normalised prototypes for each class name."""
    embeds = {}
    for cls, tmpls in templates.items():
        feats = []
        for t in tmpls:
            tok = tokenizer(t, return_tensors="pt").to(device)
            f = model.get_text_features(**tok)
            f = F.normalize(f, dim=-1)
            feats.append(f)
        proto = F.normalize(torch.stack(feats).mean(0), dim=-1)  # [1,dim]
        embeds[cls] = proto
    return embeds


@torch.no_grad()
def class_accuracy(model, loader, class_embeds, device):
    model.eval()
    correct = tot = 0
    classes = list(class_embeds.keys())
    proto = torch.cat([class_embeds[c] for c in classes])   # [K,dim]
    for enc in loader:
        imgs = enc["pixel_values"].to(device)
        labels = enc["labels"].to(device)
        feats = F.normalize(model.get_image_features(imgs), dim=-1)  # [B,dim]
        sim = feats @ proto.T                        # [B,K]
        preds = sim.argmax(1)
        correct += (preds == labels).sum().item()
        tot += labels.size(0)
    return correct / tot


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    cfg = CFG()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.out_dir / "hparams.txt")

    # Reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("▶ loading CLIP ViT-B/32 …")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = processor.tokenizer

    # ── Freeze / unfreeze ────────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)

    for name, p in model.named_parameters():
        if (
            name.startswith("visual_projection")
            or name.startswith("text_projection")
            or name == "logit_scale"
            or name.startswith("vision_model.encoder.layers.11.")  # last ViT blk
        ):
            p.requires_grad_(True)

    # Sanity-check
    n_train_elems   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_train_tensors = sum(p.requires_grad for p in model.parameters())
    print(f"✔ {n_train_elems:,} parameters in {n_train_tensors} tensors are trainable")


    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = CatDogDataset(cfg, train=True)
    val_ds   = CatDogDataset(cfg, train=False)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=make_collate(processor),
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=os.cpu_count(),
        collate_fn=make_collate(processor),
    )

    # Pre-compute text prototypes
    class_embeds = build_class_text_embeds(
        tokenizer,
        model,
        {"cat": cfg.cat_templates, "dog": cfg.dog_templates},
        device,
    )

    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(cfg.out_dir / "tb")

    best_cls = 0.0
    global_step = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        tl = ta = 0.0
        for enc in tqdm(train_loader, leave=False):
            imgs = enc["pixel_values"].to(device)
            txt  = {k: v.to(device) for k, v in enc.items()
                    if k in {"input_ids", "attention_mask"}}

            optim.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(pixel_values=imgs, **txt)
                loss = clip_loss(out.logits_per_image, out.logits_per_text)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            model.logit_scale.data.clamp_(0, math.log(100))

            # Diagonal-match accuracy (optional)
            bsz = imgs.size(0)
            gt = torch.arange(bsz, device=device)
            acc = ((out.logits_per_image.argmax(1) == gt).float().mean()
                   + (out.logits_per_text.argmax(1) == gt).float().mean()) / 2

            tl += loss.item() * bsz
            ta += acc.item()   * bsz

            writer.add_scalar("Batch/Loss", loss.item(), global_step)
            writer.add_scalar("Batch/DiagAcc", acc.item(), global_step)
            global_step += 1

        tr_loss = tl / len(train_loader.dataset)
        tr_acc  = ta / len(train_loader.dataset)

        # ─ Evaluation ────────────────────────────────────────────────────
        class_embeds = build_class_text_embeds(
            tokenizer, model,
            {"cat": cfg.cat_templates, "dog": cfg.dog_templates},
            device,
        )
        val_cls_acc = class_accuracy(model, val_loader, class_embeds, device)

        writer.add_scalar("Epoch/Loss",      tr_loss,      epoch)
        writer.add_scalar("Epoch/DiagAcc",   tr_acc,       epoch)
        writer.add_scalar("Epoch/ClassAcc",  val_cls_acc,  epoch)

        print(f"Ep {epoch:02d} | loss {tr_loss:.3f} | diag_acc {tr_acc:.3f} "
              f"| cls_acc {val_cls_acc:.3f} | {(time.time()-t0):.1f}s")

        if val_cls_acc > best_cls:
            best_cls = val_cls_acc
            save_dir = cfg.out_dir / "best"
            model.save_pretrained(save_dir)
            print(f"  ✔ new best cls_acc {best_cls:.3f} → {save_dir}")

    writer.close()
    save_dir = cfg.out_dir / "final"
    model.save_pretrained(save_dir)
    print("✔ model saved to", save_dir)
    print("done | best class accuracy", best_cls)


if __name__ == "__main__":
    main()
