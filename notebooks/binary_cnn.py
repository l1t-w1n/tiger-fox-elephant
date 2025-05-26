# binary_cnn.py -------------------------------------------------------
# Train ImprovedBinaryCNN for each target class (fox / tiger / elephant)
# --------------------------------------------------------------------
import sys, random, logging, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---------- 1. configuration ----------------------------------------
@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path.cwd())

    img_size:    int   = 224
    batch_size:  int   = 64
    num_workers: int   = 12
    lr:          float = 1e-3
    epochs:      int   = 15
    random_seed: int   = 42

    data_dir:    Path = field(init=False)
    weights_dir: Path = field(init=False)
    logs_dir:    Path = field(init=False)
    device:      torch.device = field(init=False)

    def __post_init__(self):
        self.data_dir    = self.project_root / "data" / "resized_and_split"
        self.weights_dir = self.project_root / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        self.logs_dir    = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

# ---------- 2. dataset ----------------------------------------------
class BinaryImageDataset(Dataset):
    """positive/ (1) and negative/ (0) sub-dirs under root."""
    EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root: Path, img_size: int, transform=None):
        self.root, self.img_size, self.transform = Path(root), img_size, transform
        self.samples: List[Tuple[Path, int]] = []
        for sub, lab in [("positive", 1), ("negative", 0)]:
            for p in (self.root / sub).glob("*"):
                if p.suffix.lower() in self.EXT:
                    self.samples.append((p, lab))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, lab = self.samples[idx]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(lab, dtype=torch.float32)

# ---------- 3. model ----------------------------------
class ImprovedBinaryCNN(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.act1  = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.act2  = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(2)           # 224→112

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.act3  = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(2)           # 112→56

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)  # logits

# ---------- 4. helpers ----------------------------------------------
def _run_epoch(model, loader, crit, opt, device, train=True, desc=""):
    if train:
        model.train()
    else:
        model.eval()

    loss_sum, correct, n_seen = 0.0, 0, 0
    bar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):
        for x, y in bar:
            x, y = x.to(device), y.to(device)

            if train:
                opt.zero_grad()

            logits = model(x).squeeze()
            loss   = crit(logits, y)

            if train:
                loss.backward()
                opt.step()

            preds = (torch.sigmoid(logits) > 0.5).float()

            loss_sum += loss.item() * y.size(0)
            correct  += (preds == y).sum().item()
            n_seen   += y.size(0)

            bar.set_postfix(
                loss=f"{loss_sum / n_seen:.4f}",
                acc =f"{correct  / n_seen:.4f}"
            )

    return loss_sum / n_seen, correct / n_seen

def train_one(cfg: Config, target: str, log):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tr = BinaryImageDataset(cfg.data_dir/target/"train",       cfg.img_size, tfm)
    va = BinaryImageDataset(cfg.data_dir/target/"validation",  cfg.img_size, tfm)
    log.info("Class %-8s | train %5d | val %5d", target, len(tr), len(va))

    tr_ld = DataLoader(tr, cfg.batch_size, True,  num_workers=cfg.num_workers, pin_memory=True)
    va_ld = DataLoader(va, cfg.batch_size, False, num_workers=cfg.num_workers, pin_memory=True)

    model = ImprovedBinaryCNN().to(cfg.device)
    log.info("Architecture for %s:%s", target, model)

    crit  = nn.BCEWithLogitsLoss()
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    sch   = optim.lr_scheduler.StepLR(opt, 3, 0.1)

    hist = {"tl":[], "ta":[], "vl":[], "va":[]}

    for ep in range(1, cfg.epochs+1):
        t0 = time.time()
        tl, ta = _run_epoch(model, tr_ld, crit, opt, cfg.device, True,
                            desc=f"[{target}] train ep {ep}/{cfg.epochs}")
        vl, va = _run_epoch(model, va_ld, crit, opt, cfg.device, False,
                            desc=f"[{target}] val   ep {ep}/{cfg.epochs}")
        sch.step()
        epoch_dur = time.time() - t0

        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)

        log.info(
            "[%s] Ep %02d/%d | "
            "Train loss=%.4f acc=%.4f | "
            "Val loss=%.4f acc=%.4f | "
            "time=%.1fs | lr=%.3e",
            target, ep, cfg.epochs,
            tl, ta, vl, va,
            epoch_dur, sch.get_last_lr()[0]
        )

    torch.save(model.state_dict(), cfg.weights_dir/f"{target}.pth")
    np.savez(cfg.weights_dir/f"{target}_hist.npz", **hist)
    return hist

def plot(hist, ttl=""):
    ep = range(1, len(hist["tl"])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(ep,hist["tl"],"o-",label="train")
    plt.plot(ep,hist["vl"],"o-",label="val");   plt.title(ttl+" loss"); plt.grid()
    plt.subplot(1,2,2); plt.plot(ep,hist["ta"],"o-",label="train")
    plt.plot(ep,hist["va"],"o-",label="val");   plt.title(ttl+" acc");  plt.grid()
    plt.tight_layout(); plt.show()

# ---------- 5. main --------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(str(Config().logs_dir/"binary_cnn.log"),"a"),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger("binary_cnn")

    cfg = Config()
    log.info(
        "CFG img=%d batch=%d epochs=%d lr=%.3g device=%s",
        cfg.img_size, cfg.batch_size, cfg.epochs, cfg.lr, cfg.device
    )

    for cls in ["fox", "tiger", "elephant"]:
        hist = train_one(cfg, cls, log)
        plot(hist, cls)
