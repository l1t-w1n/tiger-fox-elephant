from __future__ import annotations
"""
Fixed CLIP fine-tuning with:
- Correct image preprocessing pipeline
- Dynamic text prototype evaluation
- Better layer unfreezing
- Meaningful training dynamics
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
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameters (Revised)
# ──────────────────────────────────────────────────────────────────────────────

project_root = Path.cwd()

@dataclass
class CFG:
    img_root: Path = project_root / "data/diffusion/cat_and_dog_face"
    img_size: int = 224
    batch: int = 128                        # Increased batch size
    epochs: int = 15
    lr: float = 3e-5                       # Adjusted learning rate
    wd: float = 0.0                         # Start without weight decay
    out_dir: Path = project_root / "clip_finetune/out"
    seed: int = 42

    # Enhanced templates
    cat_templates: Tuple[str, ...] = (
        "a photo of a cat", "a close-up of a cat face",
        "a portrait of a domestic cat", "image of a tabby cat",
        "a cute cat photo", "a cat looking at the camera"
    )
    dog_templates: Tuple[str, ...] = (
        "a photo of a dog", "a close-up of a dog face",
        "a portrait of a canine", "image of a golden retriever",
        "a cute dog photo", "a dog looking at the camera"
    )

    def save(self, path: Path):
        path.write_text("\n".join(f"{k}={v}" for k, v in asdict(self).items()))

# ──────────────────────────────────────────────────────────────────────────────
# 2. Fixed Dataset & Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

class CatDogDataset(data.Dataset):
    def __init__(self, cfg: CFG, train: bool, split: float = 0.8):
        files = [p for p in cfg.img_root.iterdir() 
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        cats = sorted([p for p in files if p.stem.lower().startswith("cat_")])
        dogs = sorted([p for p in files if p.stem.lower().startswith("dog_")])
        
        # Stratified split
        split_c = int(len(cats) * split)
        split_d = int(len(dogs) * split)
        self.samples = (
            cats[:split_c] + dogs[:split_d] if train 
            else cats[split_c:] + dogs[split_d:]
        )
        self.labels = (
            [0] * split_c + [1] * split_d if train
            else [0] * (len(cats)-split_c) + [1] * (len(dogs)-split_d)
        )

        # Augmentations (PIL-only)
        self.transform = T.Compose([
            T.RandomResizedCrop(cfg.img_size, scale=(0.67, 1.0), 
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]) if train else T.Compose([T.Resize(cfg.img_size), T.CenterCrop(cfg.img_size)])

        self.class_templates = {
            0: cfg.cat_templates,
            1: cfg.dog_templates
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)  # Returns PIL Image
        label = self.labels[idx]
        return img, label

# ──────────────────────────────────────────────────────────────────────────────
# 3. Correct Collate Function
# ──────────────────────────────────────────────────────────────────────────────

def make_collate(processor: CLIPImageProcessor, tokenizer: CLIPTokenizer):
    clip_norm = T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    
    def collate(batch):
        imgs, labels = zip(*batch)
        
        # Process images: PIL -> Tensor -> Normalize
        pixel_values = torch.stack([
            clip_norm(T.functional.to_tensor(img)) 
            for img in imgs
        ])
        
        # Generate fresh captions
        captions = [
            random.choice(processor.class_templates[label])
            for label in labels
        ]
        
        # Tokenize text
        text = tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text.input_ids,
            "attention_mask": text.attention_mask,
            "labels": torch.tensor(labels)
        }
    return collate

# ──────────────────────────────────────────────────────────────────────────────
# 4. Dynamic Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calculate_class_accuracy(model, loader, cfg, tokenizer, device):
    """Real-time prototype calculation with current model weights"""
    model.eval()
    
    # Build fresh prototypes
    prototypes = []
    for class_id in [0, 1]:
        templates = cfg.cat_templates if class_id == 0 else cfg.dog_templates
        text_inputs = tokenizer(
            [random.choice(templates) for _ in range(32)],  # Sample 32 templates
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        
        with torch.cuda.amp.autocast():
            text_feats = model.get_text_features(**text_inputs)
            prototypes.append(text_feats.mean(dim=0))
    
    prototypes = torch.stack(prototypes)
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    
    # Calculate accuracy
    correct = 0
    total = 0
    for batch in loader:
        imgs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.cuda.amp.autocast():
            img_feats = model.get_image_features(imgs)
            img_feats = F.normalize(img_feats, p=2, dim=-1)
            
            logits = img_feats @ prototypes.T
            preds = logits.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# ──────────────────────────────────────────────────────────────────────────────
# 5. Main Training Loop (Fixed)
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

    print("▶ Initializing CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # ── Model Unfreezing ──────────────────────────────────────────────────
    # Freeze all except:
    # - Projection layers
    # - Last 2 ViT blocks
    # - Logit scale
    for name, param in model.named_parameters():
        param.requires_grad = False
        if any([
            "visual_projection" in name,
            "text_projection" in name,
            "vision_model.encoder.layers.10" in name,  # Unfreeze last 2 blocks
            "vision_model.encoder.layers.11" in name,
            "logit_scale" in name
        ]):
            param.requires_grad = True
    
    model = model.to(device)
    print(f"✔ Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Data Loading ──────────────────────────────────────────────────────
    train_ds = CatDogDataset(cfg, train=True)
    val_ds = CatDogDataset(cfg, train=False)

    collate_fn = make_collate(processor, tokenizer)
    
    train_loader = data.DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # ── Optimization ──────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.wd
    )
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(cfg.out_dir / "logs")

    best_acc = 0.0
    global_step = 0

    print("▶ Starting training...")
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for batch in progress:
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = clip_loss(
                    outputs.logits_per_image,
                    outputs.logits_per_text,
                    model.logit_scale.exp()
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            epoch_loss += loss.item() * inputs["pixel_values"].size(0)
            progress.set_postfix({"loss": f"{loss.item():.3f}"})
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # ── Validation ────────────────────────────────────────────────────
        val_acc = calculate_class_accuracy(
            model, val_loader, cfg, tokenizer, device
        )
        train_loss = epoch_loss / len(train_ds)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(cfg.out_dir / "best")
            print(f"💾 New best model: {val_acc:.2%} accuracy")

        print(f"Ep {epoch+1:02d} | "
              f"Loss: {train_loss:.3f} | "
              f"Val Acc: {val_acc:.2%}")
        
    model.save_pretrained(cfg.out_dir / "final")
    writer.close()
    print(f"✅ Training complete. Best validation accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    main()