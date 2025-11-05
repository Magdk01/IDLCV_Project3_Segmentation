import os
import glob
from PIL import Image
import torch
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


def load_drive_dataset(root="/dtu/datasets1/02516/DRIVE"):
    train_imgs = sorted(glob.glob(os.path.join(root, "training/images/*.tif")))
    train_masks = sorted(glob.glob(os.path.join(root, "training/mask/*.tif")))
    test_imgs = sorted(glob.glob(os.path.join(root, "test/images/*.tif")))
    test_masks = sorted(glob.glob(os.path.join(root, "test/mask/*.tif")))
    return (train_imgs, train_masks), (test_imgs, test_masks)


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
    t = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    # --- Load DRIVE ---
    (drive_train_imgs, drive_train_masks), (drive_test_imgs, drive_test_masks) = load_drive_dataset()

    # --- Load PH2 ---
    train_imgs_ph2, test_imgs_ph2, train_masks_ph2, test_masks_ph2 = load_ph2_dataset()

    # --- Combine both datasets ---
    train_imgs = drive_train_imgs + train_imgs_ph2
    train_masks = drive_train_masks + train_masks_ph2
    test_imgs = drive_test_imgs + test_imgs_ph2
    test_masks = drive_test_masks + test_masks_ph2

    # --- Debug info (shows up in .out file) ---
    print(f"[INFO] DRIVE train: {len(drive_train_imgs)} | PH2 train: {len(train_imgs_ph2)}")
    print(f"[INFO] Combined train: {len(train_imgs)} | Combined test: {len(test_imgs)}")
    print(f"[DEBUG] train_imgs: {len(train_imgs)}, train_masks: {len(train_masks)}")
    print(f"[DEBUG] test_imgs:  {len(test_imgs)}, test_masks:  {len(test_masks)}")

    # Safety: drop extra samples if mismatch happens
    n = min(len(train_imgs), len(train_masks))
    train_imgs, train_masks = train_imgs[:n], train_masks[:n]
    n = min(len(test_imgs), len(test_masks))
    test_imgs, test_masks = test_imgs[:n], test_masks[:n]


    # --- Create datasets ---
    train_ds = SegmentationDataset(train_imgs, train_masks, t)
    test_ds = SegmentationDataset(test_imgs, test_masks, t)

    # --- Return dataloaders ---
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )

