#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from models.unet import UNet
from models.simple_encoder import SimpleEncoderDecoder
from dataloader import *
from metrics import dice_coeff, iou, accuracy, sensitivity, specificity
import matplotlib.pyplot as plt


def train_model(args):
    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets ---
    train_loader, test_loader = make_dataloaders(
        batch_size=args.batch_size, img_size=(args.img_size, args.img_size)
    )

    # --- Model ---
    if args.model == "simple":
        model = SimpleEncoderDecoder(in_channels=3, out_channels=1).to(device)
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Loss function ---
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "wbce":
        pos_weight = torch.tensor([args.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == "dice":
        def dice_loss(pred, target, smooth=1.0):
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum(dim=(2, 3))
            union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = (2. * intersection + smooth) / (union + smooth)
            return 1 - dice.mean()
        criterion = dice_loss
    elif args.loss == "bce_dice":
        bce = nn.BCEWithLogitsLoss()
        def bce_dice_loss(pred, target):
            pred_sigmoid = torch.sigmoid(pred)
            intersection = (pred_sigmoid * target).sum(dim=(2, 3))
            union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = (2. * intersection + 1.0) / (union + 1.0)
            return 0.5 * bce(pred, target) + 0.5 * (1 - dice.mean())
        criterion = bce_dice_loss
    elif args.loss == "focal":
        def focal_loss(pred, target, alpha=0.8, gamma=2.0):
            bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
            pt = torch.exp(-bce)
            focal = alpha * (1 - pt) ** gamma * bce
            return focal.mean()
        criterion = focal_loss
    else:
        raise ValueError("Unsupported loss type")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Track metrics per epoch ---
    epoch_losses = []
    epoch_metrics = {"dice": [], "iou": [], "acc": [], "sens": [], "spec": []}

    for epoch in range(args.epochs):
        # --- Training phase ---
        model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # --- Evaluation phase per epoch ---
        model.eval()
        metrics_sum = {k: 0 for k in epoch_metrics.keys()}
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = torch.sigmoid(model(imgs))
                metrics_sum["dice"] += dice_coeff(preds, masks)
                metrics_sum["iou"] += iou(preds, masks)
                metrics_sum["acc"] += accuracy(preds, masks)
                metrics_sum["sens"] += sensitivity(preds, masks)
                metrics_sum["spec"] += specificity(preds, masks)

        # Average metrics per epoch
        for k in metrics_sum:
            value = metrics_sum[k] / len(test_loader)
            epoch_metrics[k].append(value)
            print(f"  {k}: {value:.4f}")

    # --- Save checkpoint each epoch ---
    torch.save(model.state_dict(),
                os.path.join(args.output_dir, f"{args.model}_model.pth"))
    # --- Save metric curves ---
    plt.figure(figsize=(8, 6))
    for k, values in epoch_metrics.items():
        plt.plot(range(1, args.epochs+1), values, marker='o', label=k)
    plt.title(f"Metrics per Epoch ({args.model}, {args.loss})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"metrics_curve_{args.model}_{args.loss}.png")
    plt.savefig(plot_path)
    plt.close()

    # --- Save final averaged metrics ---
    results_path = os.path.join(args.output_dir, f"results_{args.model}_{args.loss}.txt")
    with open(results_path, "w") as f:
        for k, v in {k: epoch_metrics[k][-1] for k in epoch_metrics}.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"[INFO] Metrics saved → {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "simple"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bce", "wbce", "dice", "bce_dice", "focal"])
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--output-dir", type=str, default="./results_unet")
    args = parser.parse_args()

    train_model(args)
