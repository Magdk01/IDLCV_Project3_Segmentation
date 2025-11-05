import os
import glob
from PIL import Image
import torch
import re
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()  # binarize
        return image, mask

def normalize_name(path):
    """Return a comparable basename without suffixes like _mask or _manual1."""
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r'(_mask|_manual1)$', '', base)
    return base

def align_pairs(imgs, masks):
    """Match images and masks by normalized filename."""
    img_basenames = {normalize_name(p) for p in imgs}
    mask_basenames = {normalize_name(p) for p in masks}
    valid = img_basenames & mask_basenames
    imgs = [p for p in imgs if normalize_name(p) in valid]
    masks = [p for p in masks if normalize_name(p) in valid]
    return sorted(imgs), sorted(masks)

def load_drive_dataset(root="/dtu/datasets1/02516/DRIVE"):
    """Load DRIVE dataset image and mask paths from training/ and test/ folders."""
    img_train = sorted(glob.glob(os.path.join(root, "training/images/*_training.tif")))
    mask_train = sorted(glob.glob(os.path.join(root, "training/mask/*_training_mask.gif")))

    img_test = sorted(glob.glob(os.path.join(root, "test/images/*_test.tif")))
    mask_test = sorted(glob.glob(os.path.join(root, "test/mask/*_test_mask.gif")))

    print(f"[INFO] DRIVE found {len(img_train)} training images and {len(mask_train)} training masks")
    print(f"[INFO] DRIVE found {len(img_test)} test images and {len(mask_test)} test masks")

    img_train, mask_train = align_pairs(img_train, mask_train)
    img_test, mask_test = align_pairs(img_test, mask_test)

    print(f"[INFO] After align_pairs: train={len(img_train)}, test={len(img_test)}")

    return (img_train, mask_train), (img_test, mask_test)


def load_ph2_dataset(root="/dtu/datasets1/02516/PH2_Dataset_images"):
    imgs, masks = [], []
    for case_dir in sorted(glob.glob(os.path.join(root, "IMD*"))):
        cid = os.path.basename(case_dir)
        img = os.path.join(case_dir, f"{cid}_Dermoscopic_Image/{cid}.bmp")
        mask = os.path.join(case_dir, f"{cid}_lesion/{cid}_lesion.bmp")
        if os.path.exists(img) and os.path.exists(mask):
            imgs.append(img)
            masks.append(mask)
    return train_test_split(imgs, masks, test_size=0.2, random_state=42)

def make_dataloaders(batch_size=4, img_size=(256, 256)):
    """Create combined dataloaders for DRIVE + PH2 datasets."""
    t = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    (drive_train_imgs, drive_train_masks), (drive_test_imgs, drive_test_masks) = load_drive_dataset()
    ph2_train_imgs, ph2_test_imgs, ph2_train_masks, ph2_test_masks = load_ph2_dataset()

    drive_train_imgs = drive_train_imgs + drive_test_imgs
    drive_train_masks = drive_train_masks + drive_test_masks
    drive_test_imgs, drive_test_masks = [], []

    train_imgs = list(drive_train_imgs) + list(ph2_train_imgs)
    train_masks = list(drive_train_masks) + list(ph2_train_masks)
    test_imgs = list(drive_test_imgs) + list(ph2_test_imgs)
    test_masks = list(drive_test_masks) + list(ph2_test_masks)

    n = min(len(train_imgs), len(train_masks))
    train_imgs, train_masks = train_imgs[:n], train_masks[:n]
    n = min(len(test_imgs), len(test_masks))
    test_imgs, test_masks = test_imgs[:n], test_masks[:n]

    train_ds = SegmentationDataset(train_imgs, train_masks, t)
    test_ds = SegmentationDataset(test_imgs, test_masks, t)

    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )