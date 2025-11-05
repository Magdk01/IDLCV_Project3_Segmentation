"""
Custom Loss Function: Point-Level Supervision for Binary Segmentation
---------------------------------------------------------------------

This script implements a custom loss suitable for training a CNN when:

  - You have **two classes**: foreground (fg) and background (bg)
  - Both classes are **always present** in every image
  - You have **sparse supervision**: only a few labeled points per image
    (e.g., 5 foreground and 5 background points)

The loss combines:
  (1) An image-level cross-entropy term to ensure both classes appear somewhere,
  (2) A point-level cross-entropy term for the annotated pixels.

Mathematically:

    L_point(S, G) =
        - 1/2 [ log(S_{t_fg, fg}) + log(S_{t_bg, bg}) ]
        - Σ_{i∈I_s} α_i log(S_{i, G_i})

where:
  S_{ic} = softmax score for class c at pixel i
  t_fg, t_bg = argmax pixels where each class is most confident
  I_s = set of annotated pixel indices
  α_i = weighting factor for pixel i (typically 1 / |I_s|)

This encourages:
  - At least one pixel to be strongly foreground and one strongly background
  - The few annotated pixels to match their ground truth labels

---------------------------------------------------------------------
"""

import torch
import torch.nn.functional as F


def point_level_loss(pred_logits, points, labels):
    """
    Compute the point-level supervision loss for binary segmentation.
    
    This loss combines:
    1. Image-level term: Encourages both fg/bg classes to appear somewhere
    2. Point-level term: Standard BCE on annotated points
    
    Args:
        pred_logits (torch.Tensor): Model output logits of shape (B, 1, H, W).
        points (list of torch.Tensor): List of length B, each tensor of shape (num_points, 2)
                                       with (x, y) coordinates.
        labels (list of torch.Tensor): List of length B, each tensor of shape (num_points,)
                                       with binary labels (1=foreground, 0=background).
    
    Returns:
        torch.Tensor: Scalar loss (mean over batch).
    """
    batch_size = pred_logits.shape[0]
    
    # Ensure pred_logits is (batch_size, H, W)
    if len(pred_logits.shape) == 4:
        if pred_logits.shape[1] == 1:
            pred_logits = pred_logits.squeeze(1)  # (batch_size, H, W)
    
    H, W = pred_logits.shape[1], pred_logits.shape[2]
    
    # Convert single-channel logits to 2-channel probabilities
    # Channel 0: foreground (label=1), Channel 1: background (label=0)
    probs_fg = torch.sigmoid(pred_logits)  # (B, H, W)
    probs_bg = 1 - probs_fg  # (B, H, W)
    
    total_loss = 0.0
    total_weight = 0.0
    
    for b in range(batch_size):
        batch_points = points[b]  # (num_points, 2)
        batch_labels = labels[b]  # (num_points,)
        
        if len(batch_points) == 0:
            continue
        
        # --- (1) IMAGE-LEVEL TERM ---
        # Encourage at least one pixel to be strongly foreground and one strongly background
        S_fg = probs_fg[b]  # (H, W)
        S_bg = probs_bg[b]  # (H, W)
        
        # Find the pixel where each class is most confident
        t_fg = torch.argmax(S_fg)
        t_bg = torch.argmax(S_bg)
        
        # Compute image-level presence loss
        L_img = -0.5 * (
            torch.log(S_fg.flatten()[t_fg] + 1e-8) +
            torch.log(S_bg.flatten()[t_bg] + 1e-8)
        )
        
        # --- (2) POINT-LEVEL TERM ---
        # Standard BCE loss at annotated point locations
        # Points are (x, y) coordinates, but PyTorch tensors are (H, W) = (y, x)
        y_coords = batch_points[:, 1].long()  # row indices
        x_coords = batch_points[:, 0].long()  # col indices
        
        # Clamp to valid range
        y_coords = torch.clamp(y_coords, 0, H - 1)
        x_coords = torch.clamp(x_coords, 0, W - 1)
        
        # Get logits at point locations
        pred_logits_b = pred_logits[b]  # (H, W)
        point_logits = pred_logits_b[y_coords, x_coords]  # (num_points,)
        
        # Compute BCE loss at point locations
        L_points = F.binary_cross_entropy_with_logits(
            point_logits, batch_labels, reduction='mean'
        )
        
        # Combine both terms
        total_loss += L_img + L_points
        total_weight += 1.0
    
    # Average across batch
    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)


def create_point_loss_function():
    """
    Factory function to create a point-level loss function.
    
    This is compatible with the loss_points.py interface but adds
    an image-level term to encourage both classes to appear.
    
    Returns:
        loss_fn: Function that takes (pred_logits, points, labels) and returns loss
    """
    def loss_fn(pred_logits, points, labels):
        return point_level_loss(pred_logits, points, labels)
    
    return loss_fn


# ---------------------------------------------------------------------
# Example usage:
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example dummy data
    B, H, W = 2, 64, 64
    torch.manual_seed(0)
    
    # Model outputs (B, 1, H, W)
    pred_logits = torch.randn(B, 1, H, W)
    
    # Define annotated points as list of tensors (x, y) format
    # label=1: foreground, label=0: background
    points = [
        torch.tensor([[10, 20], [15, 25], [30, 40]], dtype=torch.float32),  # Image 0: 3 points
        torch.tensor([[5, 10], [50, 50]], dtype=torch.float32),  # Image 1: 2 points
    ]
    
    labels = [
        torch.tensor([1.0, 1.0, 0.0]),  # Image 0: 2 fg, 1 bg
        torch.tensor([0.0, 1.0]),  # Image 1: 1 bg, 1 fg
    ]
    
    # Create loss function
    criterion = create_point_loss_function()
    loss = criterion(pred_logits, points, labels)
    print(f"Point-level supervision loss: {loss.item():.6f}")
    
    # Compare with simple BCE-only loss
    print("\nComparison:")
    print(f"  - Image-level term: Forces both classes to appear somewhere")
    print(f"  - Point-level term: BCE at annotated points only")
    print(f"  - Total loss encourages fuller segmentation with sparse supervision")

