"""Dataset ISIC 2018 Task 1 (segmentazione di lesioni).

Struttura attesa (dopo aver estratto gli zip):
  <img_dir>  = .../ISIC2018_Task1-2_Training_Input/        ISIC_0000000.jpg, ...
  <mask_dir> = .../ISIC2018_Task1_Training_GroundTruth/    ISIC_0000000_segmentation.png, ...
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def list_ids(img_dir):
    """Ritorna gli id (es. 'ISIC_0000000') di tutte le immagini nella cartella, in ordine."""
    return sorted(p.stem for p in Path(img_dir).glob("ISIC_*.jpg"))


def split_ids(ids, val_frac=0.2, seed=42):
    """Split riproducibile train/val: stesso seed => stesso split, sempre."""
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return ids[n_val:], ids[:n_val]


class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ids, img_size=256, augment=False, cache=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.ids = list(ids)
        self.img_size = img_size
        self.augment = augment
        self.cache = None
        if cache:  # pre-carica tutto in RAM (ridotto a img_size): molto piu' veloce su Colab/Drive
            print(f"Carico {len(self.ids)} immagini in RAM (una tantum)...")
            self.cache = [self._load(i) for i in self.ids]

    def _load(self, image_id):
        s = self.img_size
        img = Image.open(self.img_dir / f"{image_id}.jpg")
        img.draft("RGB", (s, s))  # decodifica JPEG a risoluzione ridotta: molto piu' veloce
        img = img.convert("RGB").resize((s, s), Image.BILINEAR)
        mask = Image.open(self.mask_dir / f"{image_id}_segmentation.png").convert("L")
        mask = mask.resize((s, s), Image.NEAREST)  # NEAREST: non inventa valori intermedi
        return np.asarray(img, dtype=np.uint8), (np.asarray(mask) > 127).astype(np.uint8)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, mask = self.cache[idx] if self.cache is not None else self._load(self.ids[idx])
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W) in [0,1]
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1,H,W) in {0,1}

        if self.augment:  # le STESSE trasformazioni geometriche su immagine e maschera
            if random.random() < 0.5:
                img, mask = img.flip(2), mask.flip(2)  # flip orizzontale
            if random.random() < 0.5:
                img, mask = img.flip(1), mask.flip(1)  # flip verticale
            k = random.randint(0, 3)
            img, mask = torch.rot90(img, k, (1, 2)), torch.rot90(mask, k, (1, 2))

        img = (img - 0.5) / 0.5  # normalizza in [-1, 1]
        return img, mask
