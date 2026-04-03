import os, random, argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

# ================= CONFIG =================
NUM_CLASSES = 10

CLASS_MAP = {
    100:(0,"Trees"), 200:(1,"Lush Bushes"), 300:(2,"Dry Grass"),
    500:(3,"Dry Bushes"), 550:(4,"Ground Clutter"), 600:(5,"Flowers"),
    700:(6,"Logs"), 800:(7,"Rocks"), 7100:(8,"Landscape"), 10000:(9,"Sky")
}

# 🔥 STRONG weights
CLASS_WEIGHTS = torch.tensor([2,4,2,4,4,6,6,5,1,0.5], dtype=torch.float32)

# ================= DATASET =================
class DesertDataset(Dataset):
    def __init__(self, root, split, size=384, augment=False):
        self.root = Path(root)
        self.size = (size, size)
        self.augment = augment

        self.rgb_dir = self.root / split / "Color_Images"
        self.mask_dir = self.root / split / "Segmentation"

        self.images = sorted(list(self.rgb_dir.glob("*")))

        self.lut = np.zeros(65536, dtype=np.uint8)
        for raw,(idx,_) in CLASS_MAP.items():
            self.lut[raw] = idx

        self.tf = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __getitem__(self, i):
        img_path = self.images[i]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 🔥 Resize BOTH
        img = img.resize(self.size)
        mask = mask.resize(self.size, Image.NEAREST)

        mask = np.array(mask)
        mask = self.lut[mask]

        if self.augment and random.random()>0.5:
            img = TF.hflip(img)
            mask = np.fliplr(mask).copy()

        img = self.tf(img)
        mask = torch.tensor(mask).long()

        return img, mask

    def __len__(self): return len(self.images)

# ================= MODEL =================
def build_model():
    model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, 1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, 1)
    return model

# ================= LOSS =================
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = F.one_hot(target, NUM_CLASSES).permute(0,3,1,2)

        inter = (pred * target).sum((2,3))
        union = pred.sum((2,3)) + target.sum((2,3))

        dice = (2*inter + 1)/(union + 1)
        return 1 - dice.mean()

# ================= IOU =================
def compute_iou(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    ious=[]
    for c in range(NUM_CLASSES):
        inter = ((pred==c)&(target==c)).sum().item()
        union = ((pred==c)|(target==c)).sum().item()
        ious.append(inter/union if union else 0)
    return np.mean(ious)

# ================= TRAIN =================
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = DesertDataset(args.data,"train",augment=True)
    val_ds   = DesertDataset(args.data,"val")

    train_dl = DataLoader(train_ds,batch_size=args.batch,shuffle=True)
    val_dl   = DataLoader(val_ds,batch_size=args.batch)

    model = build_model().to(device)

    ce = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    dice = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total=0

        for img,mask in train_dl:
            img,mask = img.to(device),mask.to(device)

            out = model(img)["out"]
            loss = ce(out,mask)+dice(out,mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total+=loss.item()

        # VALIDATION
        model.eval()
        preds=[]; labels=[]
        with torch.no_grad():
            for img,mask in val_dl:
                img = img.to(device)
                out = model(img)["out"]
                preds.append(out.argmax(1).cpu())
                labels.append(mask)

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        iou = compute_iou(preds,labels)

        print(f"Epoch {epoch+1} | Loss {total/len(train_dl):.3f} | IoU {iou:.3f}")

# ================= MAIN =================
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)

    args, unknown = p.parse_known_args()  # 🔥 FIXED

    train(args)