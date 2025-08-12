"""
Minimal training script for a tiny edge-band refiner (optional micro-training).
Goal: Learn residual on unknown band to fix color-ambiguous boundaries without replacing RMBG-2.0.

Data expectation:
- A dataset folder providing (image, alpha_gt) pairs (e.g., DIM/P3M-style). For quick POC, you can also use your internal data.

Usage (example):
  python train_refiner.py --data_root /path/to/data --epochs 5 --batch_size 8 --lr 2e-4

This is intentionally compact; replace dataset loader / loss combos as needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class Args:
    data_root: str
    epochs: int = 5
    batch_size: int = 8
    lr: float = 2e-4
    input_size: Tuple[int, int] = (512, 512)
    out_dir: str = "refiner_ckpts"


class SimpleMattingSet(Dataset):
    def __init__(self, root: str, size: Tuple[int, int] = (512, 512)) -> None:
        self.root = Path(root)
        self.images = sorted([p for p in (self.root / "images").glob("*.png")]) + \
                       sorted([p for p in (self.root / "images").glob("*.jpg")])
        self.size = size
        self.t_img = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        self.t_alpha = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        ip = self.images[idx]
        ap = (self.root / "alphas" / ip.name).with_suffix(".png")
        img = Image.open(ip).convert("RGB")
        alp = Image.open(ap).convert("L")
        return self.t_img(img), self.t_alpha(alp)


class TinyRefiner(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        # U-Net like tiny
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


def unknown_band(alpha: torch.Tensor, low: float = 0.05, high: float = 0.95) -> torch.Tensor:
    return ((alpha > low) & (alpha < high)).float()


def train(args: Args):
    ds = SimpleMattingSet(args.data_root, args.input_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = TinyRefiner(in_ch=4).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss(reduction="none")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        for img, alpha_gt in dl:
            img = img.cuda()  # [B,3,H,W]
            alpha_gt = alpha_gt.cuda()  # [B,1,H,W] in 0..1

            # For POC, synthesize a noisy alpha as if from RMBG (here: blur + noise)
            alpha_pred = torch.clamp(alpha_gt + 0.05*torch.randn_like(alpha_gt), 0, 1)

            band = unknown_band(alpha_pred)
            x = torch.cat([img, alpha_pred], dim=1)  # [B,4,H,W]
            delta = model(x)
            alpha_ref = torch.clamp(alpha_pred + delta, 0, 1)

            loss = (l1(alpha_ref, alpha_gt) * (0.5 + band)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        torch.save(model.state_dict(), out_dir / f"tiny_refiner_ep{ep+1}.pth")
        print(f"[ep {ep+1}] saved ckpt, last loss ~ {float(loss):.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--input_size", type=int, nargs=2, default=[512, 512])
    p.add_argument("--out_dir", type=str, default="refiner_ckpts")
    a = p.parse_args()
    args = Args(
        data_root=a.data_root,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        input_size=(a.input_size[0], a.input_size[1]),
        out_dir=a.out_dir,
    )
    train(args)


