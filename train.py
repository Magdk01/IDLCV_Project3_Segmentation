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


def train_model(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_loader, test_loader = make_dataloaders(
        batch_size=args.batch_size, img_size=(args.img_size, args.img_size)
    )

    # Model
    if args.model == "simple":
        model = SimpleEncoderDecoder(in_channels=3, out_channels=1).to(device)
    else:
        model = UNet(in_channels=3, out_channels=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss function
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "wbce":
        pos_weight = torch.tensor([args.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError("Loss must be 'bce' or 'wbce'")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(args.epochs):
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
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"unet_epoch{epoch+1}.pth"))

    # ---------------------------
    # Evaluation
    # ---------------------------
    model.eval()
    metrics = {"dice": 0, "iou": 0, "acc": 0, "sens": 0, "spec": 0}

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Evaluating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs))
            metrics["dice"] += dice_coeff(preds, masks)
            metrics["iou"] += iou(preds, masks)
            metrics["acc"] += accuracy(preds, masks)
            metrics["sens"] += sensitivity(preds, masks)
            metrics["spec"] += specificity(preds, masks)

    for k in metrics:
        metrics[k] /= len(test_loader)
        print(f"{k}: {metrics[k]:.4f}")

    # --- Save metrics to a model-specific file ---
    results_filename = f"results_{args.model}.txt"
    results_path = os.path.join(args.output_dir, results_filename)

    with open(results_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on HPC for segmentation")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "simple"],
                    help="Choose segmentation model: unet or simple")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=256, help="Resize image to this size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "wbce"], help="Loss function type")
    parser.add_argument("--pos-weight", type=float, default=3.0, help="Positive class weight for wbce")
    parser.add_argument("--output-dir", type=str, default="./results_unet", help="Output directory for results")
    args = parser.parse_args()

    train_model(args)
