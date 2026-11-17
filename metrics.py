"""Metriche di segmentazione, calcolate PER IMMAGINE (come nella challenge ISIC, che media gli score per-immagine)."""
import torch


@torch.no_grad()
def dice_iou(logits, targets, threshold=0.5, eps=1e-7):
    """Ritorna due tensori di shape (B,): Dice e IoU (Jaccard) per ogni immagine del batch."""
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = targets.flatten(1)
    inter = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    targ_sum = targets.sum(dim=1)
    union = pred_sum + targ_sum - inter
    dice = (2 * inter + eps) / (pred_sum + targ_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou
