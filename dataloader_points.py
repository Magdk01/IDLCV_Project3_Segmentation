import os
import glob
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def collate_points_fn(batch):
    """
    Custom collate function to handle variable-length point arrays.
    Returns points and labels as lists of tensors (one per batch item).
    """
    images = []
    masks = []
    points_list = []
    labels_list = []
    
    for image, mask, points, labels in batch:
        images.append(image)
        masks.append(mask)
        points_list.append(points)
        labels_list.append(labels)
    
    # Stack images and masks
    images = torch.stack(images)
    masks = torch.stack(masks)
    
    # Points and labels stay as lists (variable length per batch item)
    return images, masks, points_list, labels_list


def mask_to_points(mask_path, correct_points=5, incorrect_points=5):
    """
    Take a mask path and return two numpy arrays of shape (num_points, 2) 
    each containing the (x, y) coordinates of points sampled from black and white regions of the mask respectively.
    
    Args:
        mask_path: Path to the mask image
        correct_points: Number of points to sample from white regions (positive class)
        incorrect_points: Number of points to sample from black regions (negative class)
    
    Returns:
        correct_points: Array of (x, y) coordinates from white regions
        incorrect_points: Array of (x, y) coordinates from black regions
    """
    mask = np.array(Image.open(mask_path).convert('L'))
    ys, xs = np.where(mask > 128)  # Get coordinates of white pixels
    # get coordinates of black pixels
    yn, xn = np.where(mask <= 128)  # Get coordinates of black pixels
    
    if len(xs) == 0 or len(ys) == 0:
        return np.array([]), np.array([])  # No white pixels found
    
    # Randomly select points from white pixels
    num_white_points = correct_points
    white_indices = np.random.choice(len(xs), min(num_white_points, len(xs)), replace=False)
    
    # Randomly select points from black pixels
    num_black_points = incorrect_points 
    num_black_points = min(num_black_points, len(xn))
    black_indices = np.random.choice(len(xn), num_black_points, replace=False)
    
    # Combine the selected points
    incorrect_points_arr = np.array(list(zip(xn[black_indices], yn[black_indices])))
    correct_points_arr = np.array(list(zip(xs[white_indices], ys[white_indices])))
    
    return correct_points_arr, incorrect_points_arr


class PointSegmentationDataset(Dataset):
    """
    Dataset that loads images, masks, and sampled points for point-based segmentation.
    """
    def __init__(self, image_paths, mask_paths, transform=None, correct_points=10, incorrect_points=5):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.correct_points = correct_points
        self.incorrect_points = incorrect_points
        
        # Pre-generate all points for reproducibility (or generate on-the-fly)
        # For now, we'll generate on-the-fly to allow different random samples each epoch
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Get the size after transform
        H, W = image.shape[1], image.shape[2]  # (height, width) for tensor
        
        # Convert mask to numpy for point generation
        # mask is a tensor, convert to numpy array
        mask_np = (mask.squeeze(0).numpy() * 255).astype(np.uint8)  # (H, W)
        
        # Generate points from transformed mask
        # mask_to_points expects a mask path, but we have a numpy array
        # So we'll do the point sampling directly here
        ys, xs = np.where(mask_np > 128)  # Get coordinates of white pixels
        yn, xn = np.where(mask_np <= 128)  # Get coordinates of black pixels
        
        correct_points = np.array([])
        incorrect_points = np.array([])
        
        if len(xs) > 0 and len(ys) > 0:
            # Randomly select points from white pixels
            num_white_points = min(self.correct_points, len(xs))
            white_indices = np.random.choice(len(xs), num_white_points, replace=False)
            correct_points = np.array(list(zip(xs[white_indices], ys[white_indices])))
        
        if len(xn) > 0 and len(yn) > 0:
            # Randomly select points from black pixels
            num_black_points = min(self.incorrect_points, len(xn))
            black_indices = np.random.choice(len(xn), num_black_points, replace=False)
            incorrect_points = np.array(list(zip(xn[black_indices], yn[black_indices])))
        
        # Points are now in (x, y) format where x is column (0 to W-1) and y is row (0 to H-1)
        # Clip to valid range (already should be, but just to be safe)
        if len(correct_points) > 0:
            correct_points = correct_points.astype(np.int32)
            correct_points[:, 0] = np.clip(correct_points[:, 0], 0, W - 1)
            correct_points[:, 1] = np.clip(correct_points[:, 1], 0, H - 1)
        
        if len(incorrect_points) > 0:
            incorrect_points = incorrect_points.astype(np.int32)
            incorrect_points[:, 0] = np.clip(incorrect_points[:, 0], 0, W - 1)
            incorrect_points[:, 1] = np.clip(incorrect_points[:, 1], 0, H - 1)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        # Convert points to tensors
        # Store as: correct_points (N, 2), incorrect_points (M, 2)
        # Also store labels: 1 for correct, 0 for incorrect
        all_points = []
        all_labels = []
        
        if len(correct_points) > 0:
            all_points.append(correct_points)
            all_labels.append(np.ones(len(correct_points)))
        
        if len(incorrect_points) > 0:
            all_points.append(incorrect_points)
            all_labels.append(np.zeros(len(incorrect_points)))
        
        if len(all_points) > 0:
            all_points = np.vstack(all_points)
            all_labels = np.concatenate(all_labels)
        else:
            all_points = np.array([]).reshape(0, 2)
            all_labels = np.array([])
        
        return image, mask, torch.tensor(all_points, dtype=torch.long), torch.tensor(all_labels, dtype=torch.float32)


def load_ph2_dataset(root="/dtu/datasets1/02516/PH2_Dataset_images"):
    """
    Load PH2 dataset image and mask paths.
    
    Args:
        root: Root directory of PH2 dataset. Can be overridden via PH2_DATASET_ROOT environment variable.
    
    Returns:
        x_train, x_test, y_train_mask, y_test_mask: Train/test split of image and mask paths
    """
    # Allow override via environment variable
    if "PH2_DATASET_ROOT" in os.environ:
        root = os.environ["PH2_DATASET_ROOT"]
        print(f"[INFO] Using PH2 dataset root from environment: {root}")
    else:
        print(f"[INFO] Using default PH2 dataset root: {root}")
    
    imgs, masks = [], []
    for case_dir in sorted(glob.glob(os.path.join(root, "IMD*"))):
        cid = os.path.basename(case_dir)
        img = os.path.join(case_dir, f"{cid}_Dermoscopic_Image/{cid}.bmp")
        mask = os.path.join(case_dir, f"{cid}_lesion/{cid}_lesion.bmp")
        if os.path.exists(img) and os.path.exists(mask):
            imgs.append(img)
            masks.append(mask)
    
    print(f"[INFO] PH2 found {len(imgs)} images and {len(masks)} masks")
    
    return train_test_split(imgs, masks, test_size=0.2, random_state=42)


def make_dataloaders(batch_size=4, img_size=(256, 256), correct_points=10, incorrect_points=5, 
                     dataset_root=None):
    """
    Create dataloaders for PH2 dataset with point-based sampling.
    
    Args:
        batch_size: Batch size for dataloaders
        img_size: Target image size (height, width)
        correct_points: Number of positive points to sample per mask
        incorrect_points: Number of negative points to sample per mask
        dataset_root: Optional override for dataset root path (overrides env var if provided)
    
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    # Override root if provided
    if dataset_root is not None:
        os.environ["PH2_DATASET_ROOT"] = dataset_root
    
    ph2_train_imgs, ph2_test_imgs, ph2_train_masks, ph2_test_masks = load_ph2_dataset()
    
    # Ensure same length
    n_train = min(len(ph2_train_imgs), len(ph2_train_masks))
    n_test = min(len(ph2_test_imgs), len(ph2_test_masks))
    ph2_train_imgs, ph2_train_masks = ph2_train_imgs[:n_train], ph2_train_masks[:n_train]
    ph2_test_imgs, ph2_test_masks = ph2_test_imgs[:n_test], ph2_test_masks[:n_test]
    
    # Create transforms
    t = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_ds = PointSegmentationDataset(
        ph2_train_imgs, ph2_train_masks, t, 
        correct_points=correct_points, 
        incorrect_points=incorrect_points
    )
    test_ds = PointSegmentationDataset(
        ph2_test_imgs, ph2_test_masks, t,
        correct_points=correct_points,
        incorrect_points=incorrect_points
    )
    
    print(f"[INFO] Created train dataset: {len(train_ds)} samples")
    print(f"[INFO] Created test dataset: {len(test_ds)} samples")
    
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_points_fn),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_points_fn)
    )

