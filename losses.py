"""Loss per segmentazione binaria. Tutte ricevono LOGITS (B,1,H,W) e target in {0,1} (B,1,H,W)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """1 - Dice, calcolato per immagine e poi mediato sul batch (versione 'soft', differenziabile)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(dim=1)
        dice = (2 * inter + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combinazione pesata di BCE (stabile, per-pixel) e Dice (guarda la sovrapposizione globale)."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * self.dice(logits, targets)


class BCELoss(nn.Module):
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)


def get_loss(name):
    losses = {"bce": BCELoss, "dice": DiceLoss, "bce_dice": BCEDiceLoss}
    if name not in losses:
        raise ValueError(f"Loss sconosciuta: {name}. Scegli tra {list(losses)}")
    return losses[name]()
