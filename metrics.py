import torch

def dice_coeff(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2. * inter + smooth) / (pred.sum() + target.sum() + smooth)

def iou(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)

def accuracy(pred, target):
    pred = (pred > 0.5).float()
    return (pred == target).float().mean()

def sensitivity(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return tp / (tp + fn + eps)

def specificity(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return tn / (tn + fp + eps)
