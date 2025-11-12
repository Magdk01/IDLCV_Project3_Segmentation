#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from models.unet import UNet
from models.simple_encoder import SimpleEncoderDecoder
from dataloader_points import make_dataloaders
from point_loss import create_point_loss_function
from metrics import dice_coeff, iou, accuracy, sensitivity, specificity
import matplotlib.pyplot as plt


def train_model(args):
    # --- Device setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # --- Datasets ---
    # Allow dataset root override via command line or environment variable
    dataset_root = args.dataset_root if args.dataset_root else None
    if dataset_root:
        print(f"[INFO] Using dataset root from command line: {dataset_root}")

    train_loader, test_loader = make_dataloaders(
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        correct_points=args.correct_points,
        incorrect_points=args.incorrect_points,
        dataset_root=dataset_root,
    )

    # --- Model ---
    if args.model == "simple":
        model = SimpleEncoderDecoder(in_channels=3, out_channels=1).to(device)
    else:
        model = UNet(n_channels=3, n_classes=1, dropout_rate=0.5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Loss function ---
    # Point-level loss with image-level term:
    # (1) Image-level: Encourages both fg/bg classes to appear somewhere confidently
    # (2) Point-level: BCE loss at annotated point locations (sparse supervision)
    # This helps prevent degenerate solutions with very sparse annotations
    criterion = create_point_loss_function()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Track metrics per epoch ---
    # Note: Metrics and validation loss are computed against GROUND TRUTH masks during evaluation
    # to verify that training on point-based sparse supervision actually learns
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_metrics = {"dice": [], "iou": [], "acc": [], "sens": [], "spec": []}

    # Create validation loss function (uses full masks, not points)
    if args.loss == "bce":
        val_criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "mse":
        val_criterion = nn.MSELoss()
    elif args.loss == "focal":

        def focal_loss(pred, target, alpha=0.8, gamma=2.0):
            bce = nn.BCEWithLogitsLoss(reduction="none")(pred, target)
            pt = torch.exp(-bce)
            focal = alpha * (1 - pt) ** gamma * bce
            return focal.mean()

        val_criterion = focal_loss
    else:
        val_criterion = nn.BCEWithLogitsLoss()  # fallback

    for epoch in range(args.epochs):
        # --- Training phase ---
        model.train()
        total_loss = 0.0

        for imgs, masks, points_list, labels_list in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False
        ):
            imgs = imgs.to(device)
            preds = model(imgs)

            # Move points and labels to device
            points_list_device = [p.to(device) for p in points_list]
            labels_list_device = [l.to(device) for l in labels_list]

            # Training loss: Point-level loss with image-level term
            # - Point term: BCE at annotated point locations (sparse supervision)
            # - Image term: Encourages both classes to appear somewhere in prediction
            loss = criterion(preds, points_list_device, labels_list_device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(
            f"Epoch {epoch+1}/{args.epochs} - Train Loss (point + image-level): {avg_train_loss:.4f}"
        )

        # --- Evaluation phase per epoch ---
        # Evaluate against GROUND TRUTH masks to verify learning
        model.eval()
        val_loss_sum = 0.0
        metrics_sum = {k: 0 for k in epoch_metrics.keys()}

        with torch.no_grad():
            for imgs, masks, points_list, labels_list in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds_logits = model(imgs)  # Model outputs logits

                # Validation loss: Use logits (not thresholded) for proper loss computation
                # This uses full mask supervision (not sparse points)
                val_loss = val_criterion(preds_logits, masks)

                # Convert to binary predictions (0 or 1) for metrics comparison
                # Metrics compare binary predictions vs binary ground truth masks
                preds_probs = torch.sigmoid(preds_logits)  # Probabilities
                preds_binary = (preds_probs > 0.5).float()  # Binary (0 or 1)
                val_loss_sum += val_loss.item()

                # Metrics against GROUND TRUTH masks (using binary predictions)
                # Note: metrics functions also threshold internally, but we pass binary for consistency
                metrics_sum["dice"] += dice_coeff(preds_binary, masks)
                metrics_sum["iou"] += iou(preds_binary, masks)
                metrics_sum["acc"] += accuracy(preds_binary, masks)
                metrics_sum["sens"] += sensitivity(preds_binary, masks)
                metrics_sum["spec"] += specificity(preds_binary, masks)

        # Average validation loss and metrics per epoch (against GROUND TRUTH)
        avg_val_loss = val_loss_sum / len(test_loader)
        epoch_val_losses.append(avg_val_loss)
        print(f"  Val Loss (vs GT): {avg_val_loss:.4f}")
        print(f"  Metrics vs GROUND TRUTH:")
        for k in metrics_sum:
            value = metrics_sum[k] / len(test_loader)
            epoch_metrics[k].append(value)
            print(f"    {k}: {value:.4f}")

        # --- Save checkpoint each epoch ---
        torch.save(
            model.state_dict(),
            os.path.join(args.output_dir, f"{args.model}_epoch{epoch+1}.pth"),
        )

    # --- Save loss curves ---
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, args.epochs + 1),
        epoch_train_losses,
        marker="o",
        label="Train Loss (point + image-level)",
    )
    plt.plot(
        range(1, args.epochs + 1),
        epoch_val_losses,
        marker="s",
        label="Val Loss (vs GT)",
    )
    plt.title(f"Loss Curves ({args.model})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # --- Save metric curves ---
    plt.subplot(1, 2, 2)
    for k, values in epoch_metrics.items():
        # Convert torch tensors to floats if necessary
        values_cpu = [v.item() if torch.is_tensor(v) else v for v in values]
        plt.plot(range(1, args.epochs + 1), values_cpu, marker="o", label=k)
    plt.title(f"Metrics vs Ground Truth ({args.model})")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"curves_{args.model}_points.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved loss and metrics plot → {plot_path}")

    # --- Save final averaged metrics ---
    # Note: These metrics are computed against GROUND TRUTH masks
    results_path = os.path.join(args.output_dir, f"results_{args.model}_points.txt")
    with open(results_path, "w") as f:
        f.write("# Metrics and validation loss computed against GROUND TRUTH masks\n")
        f.write(
            "# (Training uses point-based sparse supervision, evaluation uses full GT)\n\n"
        )
        f.write(f"train_loss: {epoch_train_losses[-1]:.4f}\n")
        f.write(f"val_loss: {epoch_val_losses[-1]:.4f}\n")
        f.write("\n")
        for k, v in {k: epoch_metrics[k][-1] for k in epoch_metrics}.items():
            value = v.item() if torch.is_tensor(v) else v
            f.write(f"{k}: {value:.4f}\n")
    print(f"[INFO] Metrics (vs Ground Truth) saved → {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "simple"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--loss", type=str, default="bce", choices=["bce", "mse", "focal"]
    )
    parser.add_argument("--output-dir", type=str, default="./results_points")

    # Point sampling parameters
    parser.add_argument(
        "--correct-points",
        type=int,
        default=10,
        help="Number of positive points to sample per mask",
    )
    parser.add_argument(
        "--incorrect-points",
        type=int,
        default=5,
        help="Number of negative points to sample per mask",
    )

    # Dataset path override
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/PH2_Dataset_images",
        help="Override PH2 dataset root path (overrides env var if provided)",
    )

    args = parser.parse_args()

    train_model(args)