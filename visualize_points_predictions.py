#!/usr/bin/env python3
"""
Visualization script for point-based segmentation predictions.

Loads the last checkpoint, samples 10 images from the test set, and displays:
- Original image with annotated points overlaid
- Predicted mask
- Ground truth mask
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet
from models.simple_encoder import SimpleEncoderDecoder
from dataloader_points import make_dataloaders
from torchvision import transforms


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
        return True
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}")
        return False


def find_last_checkpoint(output_dir, model_name):
    """Find the last checkpoint based on epoch number."""
    checkpoints = []
    for filename in os.listdir(output_dir):
        if filename.startswith(f"{model_name}_epoch") and filename.endswith(".pth"):
            # Extract epoch number
            epoch_num = int(filename.split("epoch")[1].split(".")[0])
            checkpoints.append((epoch_num, os.path.join(output_dir, filename)))

    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]  # Return path of latest checkpoint
    return None


def visualize_predictions(
    model,
    test_loader,
    device,
    num_images=10,
    output_path="visualization_points.png",
):
    """
    Visualize predictions for sample images.

    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to run inference on
        num_images: Number of images to visualize
        output_path: Path to save visualization
    """
    model.eval()

    # Collect samples
    samples = []
    with torch.no_grad():
        for imgs, masks, points_list, labels_list in test_loader:
            imgs = imgs.to(device)

            # Get predictions
            preds_logits = model(imgs)  # Model outputs logits
            preds_probs = torch.sigmoid(preds_logits)  # Convert to probabilities

            # Handle different output shapes: (B, 1, H, W) or (B, H, W)
            if len(preds_probs.shape) == 4 and preds_probs.shape[1] == 1:
                preds_probs = preds_probs.squeeze(1)  # (B, H, W)

            # Debug: Check value ranges (only print once)
            if len(samples) == 0:
                print(
                    f"[DEBUG] Logits range: [{preds_logits.min().item():.4f}, {preds_logits.max().item():.4f}]"
                )
                print(
                    f"[DEBUG] Probabilities range: [{preds_probs.min().item():.4f}, {preds_probs.max().item():.4f}]"
                )
                print(f"[DEBUG] Probabilities mean: {preds_probs.mean().item():.4f}")
                print(
                    f"[DEBUG] Probabilities > 0.5: {(preds_probs > 0.5).float().mean().item() * 100:.2f}%"
                )

            # Convert probabilities to binary predictions (0 or 1) by thresholding at 0.5
            # This matches how metrics are computed - binary predictions vs binary ground truth
            preds_binary = (preds_probs > 0.5).float()

            # Move to CPU for visualization
            imgs_cpu = imgs.cpu()
            masks_cpu = (
                masks.cpu().squeeze(1) if masks.dim() == 4 else masks.cpu()
            )  # Handle (B, 1, H, W)
            preds_cpu = (
                preds_binary.cpu()
            )  # Use binary predictions (0 or 1) for visualization

            # Collect samples
            for b in range(len(imgs_cpu)):
                if len(samples) >= num_images:
                    break

                img = imgs_cpu[b]
                mask_gt = masks_cpu[b]
                pred = preds_cpu[b]
                points = points_list[b].cpu().numpy()
                labels = labels_list[b].cpu().numpy()

                # Store both probabilities and binary for visualization
                pred_probs = preds_probs[b].cpu()  # Store probabilities for debugging
                samples.append(
                    {
                        "image": img,
                        "gt_mask": mask_gt,
                        "pred_mask": pred,  # Binary prediction
                        "pred_probs": pred_probs,  # Probabilities for debugging
                        "points": points,
                        "labels": labels,
                    }
                )

            if len(samples) >= num_images:
                break

    # Limit to requested number
    samples = samples[:num_images]

    # Create visualization
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, sample in enumerate(samples):
        img = sample["image"]
        gt_mask = sample["gt_mask"]
        pred_mask = sample["pred_mask"]  # Already binary (0 or 1)
        pred_probs = sample.get("pred_probs", pred_mask)  # Probabilities for debugging
        points = sample["points"]
        labels = sample["labels"]

        # Ensure masks are 2D
        if gt_mask.dim() > 2:
            gt_mask = gt_mask.squeeze()
        if pred_mask.dim() > 2:
            pred_mask = pred_mask.squeeze()

        # Convert image tensor to numpy (C, H, W) -> (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        # Denormalize if needed (assuming ToTensor normalized to [0,1])
        img_np = np.clip(img_np, 0, 1)

        # Convert masks to numpy
        # pred_mask is already binary (0 or 1) from thresholding above
        gt_mask_np = gt_mask.numpy()
        pred_mask_binary = pred_mask.numpy()  # Already binary (0 or 1)

        # Get probabilities for debugging (if available)
        if isinstance(pred_probs, torch.Tensor):
            pred_mask_probs_np = pred_probs.numpy()
        else:
            pred_mask_probs_np = pred_mask_binary  # Fallback to binary if no probs

        # Plot 1: Original image with points overlaid
        axes[idx, 0].imshow(img_np)
        if len(points) > 0:
            # Separate positive and negative points
            pos_points = points[labels == 1]
            neg_points = points[labels == 0]

            if len(pos_points) > 0:
                axes[idx, 0].scatter(
                    pos_points[:, 0],
                    pos_points[:, 1],
                    c="green",
                    s=50,
                    marker="+",
                    linewidths=2,
                    label="Positive",
                )
            if len(neg_points) > 0:
                axes[idx, 0].scatter(
                    neg_points[:, 0],
                    neg_points[:, 1],
                    c="red",
                    s=50,
                    marker="x",
                    linewidths=2,
                    label="Negative",
                )
        axes[idx, 0].set_title("Image with Points\n(Green=+, Red=-)")
        axes[idx, 0].axis("off")
        if idx == 0:
            axes[idx, 0].legend(loc="upper right", fontsize=8)

        # Plot 2: Predicted mask (binary, thresholded at 0.5)
        # Also show probability values for debugging
        axes[idx, 1].imshow(pred_mask_binary, cmap="gray", vmin=0, vmax=1)
        prob_min = pred_mask_probs_np.min()
        prob_max = pred_mask_probs_np.max()
        prob_mean = pred_mask_probs_np.mean()
        binary_pct = (pred_mask_binary > 0.5).sum() / pred_mask_binary.size * 100
        title = f"Predicted Mask (Binary)\nProb: [{prob_min:.3f}, {prob_max:.3f}], mean={prob_mean:.3f}\nBinary pixels: {binary_pct:.1f}%"
        axes[idx, 1].set_title(title, fontsize=8)
        axes[idx, 1].axis("off")

        # Plot 3: Ground truth mask (ensure binary)
        gt_mask_binary = (gt_mask_np > 0.5).astype(np.float32)
        axes[idx, 2].imshow(gt_mask_binary, cmap="gray", vmin=0, vmax=1)
        axes[idx, 2].set_title("Ground Truth Mask")
        axes[idx, 2].axis("off")

        # Plot 4: Overlay comparison (prediction in red, GT in green)
        axes[idx, 3].imshow(img_np)
        # Overlay masks with transparency (using binary predictions)
        pred_overlay = np.zeros_like(img_np)
        pred_overlay[:, :, 0] = pred_mask_binary  # Red channel for prediction
        gt_overlay = np.zeros_like(img_np)
        gt_overlay[:, :, 1] = gt_mask_np  # Green channel for GT

        axes[idx, 3].imshow(pred_overlay, alpha=0.5 * pred_mask_binary)
        axes[idx, 3].imshow(gt_overlay, alpha=0.5 * gt_mask_np)
        axes[idx, 3].set_title("Overlay\n(Red=Pred, Green=GT)")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize point-based segmentation predictions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "simple"],
        help="Model architecture",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (if not provided, uses last checkpoint from output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results_points",
        help="Output directory where checkpoints are stored",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to visualize",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Image size (should match training)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override PH2 dataset root path",
    )
    parser.add_argument(
        "--correct-points",
        type=int,
        default=10,
        help="Number of positive points (should match training)",
    )
    parser.add_argument(
        "--incorrect-points",
        type=int,
        default=5,
        help="Number of negative points (should match training)",
    )
    parser.add_argument(
        "--output-vis",
        type=str,
        default="visualization_points.png",
        help="Output path for visualization",
    )

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    if args.model == "simple":
        model = SimpleEncoderDecoder(in_channels=3, out_channels=1).to(device)
    else:
        model = UNet(n_channels=3, n_classes=1).to(device)

    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_last_checkpoint(args.output_dir, args.model)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    else:
        print("[WARNING] No checkpoint found, using untrained model")

    # Create dataloader
    _, test_loader = make_dataloaders(
        batch_size=4,
        img_size=(args.img_size, args.img_size),
        correct_points=args.correct_points,
        incorrect_points=args.incorrect_points,
        dataset_root=args.dataset_root,
    )

    # Visualize
    visualize_predictions(
        model,
        test_loader,
        device,
        num_images=args.num_images,
        output_path=args.output_vis,
    )


if __name__ == "__main__":
    main()
