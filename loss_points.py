"""
Point-based loss for simulating expert annotation scenarios.

This module implements a loss function that computes loss ONLY at annotated point locations.
This provides pure sparse supervision without introducing false information through interpolation.

Usage:
- Expert clicks points on foreground/background regions
- Model learns to predict correct labels at those specific points
- No interpolation - only uses what is actually annotated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def point_loss(pred_logits, points, labels, base_loss="bce"):
    """
    Calculate loss ONLY at annotated point locations (pure sparse supervision).

    This approach uses only the actual point annotations without any interpolation,
    avoiding the introduction of false information.

    Args:
        pred_logits: Model predictions, shape (batch_size, 1, H, W) or (batch_size, H, W)
        points: Point coordinates, list of tensors of shape (num_points_i, 2) per batch item
        labels: Point labels (1 for positive, 0 for negative), list of tensors of shape (num_points_i,)
        base_loss: Base loss type ('bce', 'mse', 'focal')

    Returns:
        loss: Scalar loss value
    """
    batch_size = pred_logits.shape[0]

    # Ensure pred_logits is (batch_size, H, W)
    if len(pred_logits.shape) == 4:
        if pred_logits.shape[1] == 1:
            pred_logits = pred_logits.squeeze(1)  # (batch_size, H, W)

    H, W = pred_logits.shape[1], pred_logits.shape[2]
    total_loss = 0.0
    total_weight = 0.0

    for b in range(batch_size):
        batch_points = points[b]  # (num_points, 2)
        batch_labels = labels[b]  # (num_points,)

        if len(batch_points) == 0:
            continue

        pred_b = pred_logits[b]  # (H, W)

        # Points are (x, y) coordinates, but PyTorch tensors are (H, W) = (y, x)
        y_coords = batch_points[:, 1].long()  # row indices
        x_coords = batch_points[:, 0].long()  # col indices

        # Clamp to valid range
        y_coords = torch.clamp(y_coords, 0, H - 1)
        x_coords = torch.clamp(x_coords, 0, W - 1)

        # Get predictions at point locations
        point_preds = pred_b[y_coords, x_coords]  # (num_points,)

        # Compute loss at point locations
        if base_loss == "bce":
            point_losses = F.binary_cross_entropy_with_logits(
                point_preds, batch_labels, reduction="none"
            )  # (num_points,)
        elif base_loss == "mse":
            point_probs = torch.sigmoid(point_preds)
            point_losses = F.mse_loss(point_probs, batch_labels, reduction="none")
        elif base_loss == "focal":
            bce = F.binary_cross_entropy_with_logits(
                point_preds, batch_labels, reduction="none"
            )
            pt = torch.exp(-bce)
            point_losses = 0.8 * (1 - pt) ** 2.0 * bce
        else:
            raise ValueError(
                f"Unknown base_loss: {base_loss}. Choose from: bce, mse, focal"
            )

        # Average loss across all points for this sample
        loss = point_losses.mean()
        total_loss += loss
        total_weight += 1.0

    # Average across batch
    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)


def create_point_loss_function(base_loss="bce"):
    """
    Factory function to create a point-based loss function.

    Args:
        base_loss: Base loss type ('bce', 'mse', 'focal')

    Returns:
        loss_fn: Function that takes (pred_logits, points, labels) and returns loss
    """

    def loss_fn(pred_logits, points, labels):
        return point_loss(pred_logits, points, labels, base_loss=base_loss)

    return loss_fn
