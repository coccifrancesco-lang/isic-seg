"""Training baseline U-Net su ISIC 2018 Task 1.

Esempi:
  # 1) SANITY CHECK: overfitting su 32 immagini (la loss deve crollare, il Dice salire verso ~1)
  python train.py --img_dir ISIC2018_Task1-2_Training_Input --mask_dir ISIC2018_Task1_Training_GroundTruth \
                  --overfit_n 32 --epochs 60 --batch_size 8 --out_dir runs/overfit

  # 2) Training vero
  python train.py --img_dir ISIC2018_Task1-2_Training_Input --mask_dir ISIC2018_Task1_Training_GroundTruth \
                  --epochs 50 --batch_size 16 --loss bce_dice --out_dir runs/baseline_bcedice
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ISICDataset, list_ids, split_ids
from losses import get_loss
from metrics import dice_iou
from model import UNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            loss = loss_fn(logits.float(), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp):
    model.eval()
    total_loss, n = 0.0, 0
    dices, ious = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
        logits = logits.float()
        total_loss += loss_fn(logits, masks).item() * imgs.size(0)
        n += imgs.size(0)
        d, i = dice_iou(logits, masks)
        dices.append(d.cpu())
        ious.append(i.cpu())
    return total_loss / n, torch.cat(dices).mean().item(), torch.cat(ious).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--out_dir", default="runs/baseline")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="bce_dice", choices=["bce", "dice", "bce_dice"])
    p.add_argument("--base", type=int, default=64, help="canali del primo livello della U-Net")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit_n", type=int, default=0, help="se >0: usa solo N immagini sia per train sia per val")
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Device: {device} | AMP: {use_amp}")

    # --- Dati -------------------------------------------------------------
    ids = list_ids(args.img_dir)
    assert len(ids) > 0, f"Nessuna immagine ISIC_*.jpg trovata in {args.img_dir}"
    if args.overfit_n > 0:
        train_ids = val_ids = ids[: args.overfit_n]
        augment_train = False  # nel test di overfitting niente augmentation
    else:
        train_ids, val_ids = split_ids(ids, args.val_frac, args.seed)
        augment_train = True
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_ds = ISICDataset(args.img_dir, args.mask_dir, train_ids, args.img_size, augment=augment_train)
    val_ds = train_ds if args.overfit_n > 0 else ISICDataset(args.img_dir, args.mask_dir, val_ids, args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- Modello, loss, ottimizzatore ---------------------------------------
    model = UNet(base=args.base).to(device)
    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Loop di training ---------------------------------------------------
    best_dice = -1.0
    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, use_amp)
            val_loss, val_dice, val_iou = evaluate(model, val_loader, loss_fn, device, use_amp)
            dt = time.time() - t0
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_dice:.4f}", f"{val_iou:.4f}", f"{dt:.1f}"])
            f.flush()
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
                  f"| Dice {val_dice:.4f} | IoU {val_iou:.4f} | {dt:.0f}s")
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), out_dir / "best.pt")
    print(f"Fatto. Miglior Dice di validazione: {best_dice:.4f} (checkpoint: {out_dir / 'best.pt'})")


if __name__ == "__main__":
    main()
