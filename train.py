import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import make_dataloaders
from models.unet import UNet
from metrics import dice_coeff, iou, accuracy, sensitivity, specificity

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = make_dataloaders(batch_size=2)
model = UNet(n_channels=3, n_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} | Loss {loss.item():.4f}")

# Validation / Testing
model.eval()
metrics = {'dice': 0, 'iou': 0, 'acc': 0, 'sens': 0, 'spec': 0}
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = torch.sigmoid(model(imgs))
        metrics['dice'] += dice_coeff(preds, masks)
        metrics['iou'] += iou(preds, masks)
        metrics['acc'] += accuracy(preds, masks)
        metrics['sens'] += sensitivity(preds, masks)
        metrics['spec'] += specificity(preds, masks)

for k in metrics:
    metrics[k] /= len(test_loader)
print(metrics)
