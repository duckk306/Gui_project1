'''

Dataset: *.npy
- images: (3430, 180, 320, 3) uint8
- labels: (3430, 180, 320) uint8, classes {0,1,2}

Ghi chú:
- H=180 không chia hết cho 16/32 -> thay vì resize sẽ méo ảnh -> ta PAD chiều cao lên 192 để UNet/FCN chạy đẹp.
- Khi tính metric + lưu ảnh, ta cắt về lại 180.

'''

import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchmetrics.segmentation import MeanIoU
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassJaccardIndex
from modelUnet import UNet

# Giảm thời gian train
torch.backends.cudnn.benchmark = True

CLASS_NAMES = {
    0: "direct",
    1: "alternative",
    2: "background",
}

COLOR_MAP = np.array([
    [255, 0, 0],        # class 0: direct
    [0, 255, 0],        # class 1: alternative
    [0, 0, 0],          # class 2: background (không overlay)
], dtype=np.uint8)


# 1. Data: Load data từ file *.npy
IMAGE_NPY = "image_180_320.npy"
LABEL_NPY = "label_180_320.npy"

images = np.load(IMAGE_NPY)      # (N,180,320,3) uint8
labels = np.load(LABEL_NPY)      # (N,180,320)   uint8


##############################################################################
# Câu 1. Hãy trực quan hóa dataset: hiển thị k ảnh đầu tiên trong dataset
##############################################################################
import cv2
import matplotlib.pyplot as plt

def visualize(images, labels, k=6):
    for i in range(k):
        image = images[i]                          # RGB uint8
        mask = labels[i].astype("uint8")           # (H,W)
        mask_rgb = COLOR_MAP[mask]                   # RGB uint8

        overlay_rgb = cv2.addWeighted(image, 1.0, mask_rgb, .5, 0.0)

        plt.figure(figsize=(12, 4))
        for j, (title, data) in enumerate(
            [("Image", image), ("Mask", mask_rgb), ("Overlay", overlay_rgb)]
        ):
            plt.subplot(1, 3, j + 1)
            plt.imshow(data)
            plt.title(title)
            plt.axis("off")
        plt.show()



##############################################################################
# Câu 2. Chia tập dữ liệu với tỷ lệ: Train:Val là 80% : 20%
##############################################################################

# PAD: 180 -> 192 (bội của 32). 320 đã là bội của 32.
ORIG_H, ORIG_W = 180, 320
PAD_H = 192
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def pad_to_192(img_chw: torch.Tensor, mask_hw: torch.Tensor):
    # img_chw: (3,180,320), mask_hw: (180,320)
    pad_bottom = PAD_H - ORIG_H     # 12
    img = F.pad(img_chw, (0, 0, 0, pad_bottom), mode="constant", value=0.0)      # (3,192,320)
    mask = F.pad(mask_hw, (0, 0, 0, pad_bottom), mode="constant", value=0)       # (192,320)
    return img, mask


class BDDDataset(Dataset):
    def __init__(self, images_raw, labels_raw):
        self.images = images_raw
        self.labels = labels_raw

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # image: uint8 HWC -> float32 CHW [0,1]
        img = torch.from_numpy(self.images[idx]).float() / 255.0   # (180,320,3)
        img = img.permute(2, 0, 1)                                  # (3,180,320)

        # label: uint8 HW -> long
        mask = torch.from_numpy(self.labels[idx]).long()            # (180,320)

        # pad + normalize
        img, mask = pad_to_192(img, mask)
        img = (img - MEAN) / STD

        return img, mask



def split_data(full_ds):
    N = images.shape[0]  # Số lượng ảnh trong dataset
    TRAIN_RATIO = 0.8
    n_train = int(TRAIN_RATIO * N)
    n_val = N - n_train

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    return train_ds, val_ds



##############################################################################
# Câu 3. Hãy định nghĩa metric
##############################################################################
NUM_CLASSES = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
miou_metric = MeanIoU(num_classes=NUM_CLASSES, include_background=True,
                      per_class=True, input_format="index").to(device)


##############################################################################
# Câu 4. Chọn 1 mô hình trong torchvision và fine tune model hình để giải quyết bài toán
##############################################################################
def create_model_Unet():
    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES
    )
    return model

def create_mode():
    weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.fcn_resnet50(weights=weights)

    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1)

    # FREEZE BACKBONE
    for p in model.backbone.parameters():
        p.requires_grad = False

    return model



##############################################################################
# Câu 5. Huấn luyện mô hình
##############################################################################
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, NUM_CLASSES).permute(0,3,1,2).float()

        intersection = (probs * targets_oh).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_oh.sum(dim=(2,3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def forward_logits(model, x):
    """
    Chuẩn hoá output của model:
    - DeepLabV3 → model(x)["out"]
    - UNet      → model(x)
    """
    out = model(x)
    if isinstance(out, dict):
        return out["out"]
    return out


def train_model(model, train_loader, val_loader, miou_metric, device, EPOCHS, FILE_NAME):

    LR = 3e-4
    WEIGHT_DECAY = 1e-4

    # 6. TRAIN + VAL (mIoU)
    class_weights = torch.tensor([2.0, 2.0, 0.5]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss()

    best_miou = -1.0

    for epoch in range(1, EPOCHS + 1):
        # 1. Train
        model.train()
        train_loss = 0.0
        steps = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):


                logits = forward_logits(model, x)
                loss = ce_loss(logits, y) + dice_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()
            steps += 1
            bar.set_postfix(train_loss = train_loss / steps)

        # 2. Validation
        miou_metric.reset()
        model.eval()
        val_loss = 0.0
        steps = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = forward_logits(model, x)
                loss = ce_loss(logits, y) + dice_loss(logits, y)
                val_loss += loss.item()
                steps += 1

                pred = torch.argmax(logits, dim=1)
                pred = pred[:, :ORIG_H, :]
                y = y[:, :ORIG_H, :]

                miou_metric.update(pred, y)

        # per-class IoU (Tensor [C])
        iou_t = miou_metric.compute()
        # TorchMetrics có thể trả -1 nếu class hoàn toàn vắng mặt ở cả pred và target -> đổi về 0 để giống cách cũ
        iou_t = torch.where(iou_t < 0, torch.zeros_like(iou_t), iou_t)

        iou = [float(v) for v in iou_t.detach().cpu().tolist()]
        miou = sum(iou) / NUM_CLASSES

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | mIoU={miou:.4f} | IoU={['%.3f' % v for v in iou]}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), FILE_NAME)

        scheduler.step()

    print("Best mIoU:", best_miou, "| Saved:", FILE_NAME)



def Inference(model, val_ds, k):
    with torch.inference_mode():
        for i in range(k):
            img, _ = val_ds[i]                      # (3,192,320) normalized
            x = img.unsqueeze(0).to(device)

            logits = forward_logits(model, x)
            pred = torch.argmax(logits, dim=1)[0]

            pred = pred[:ORIG_H, :ORIG_W].cpu().numpy()
            mask_rgb = COLOR_MAP[pred]

            img_np = (img[:, :ORIG_H, :ORIG_W] * STD + MEAN)
            img_np = (img_np * 255).byte().permute(1, 2, 0).numpy()

            overlay = cv2.addWeighted(img_np, 1.0, mask_rgb, 0.5, 0.0)

            cv2.imshow("Overlay", overlay)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Câu 1.
    visualize(images, labels, k=2)

    # Câu 2.
    full_ds = BDDDataset(images, labels)
    train_ds, val_ds = split_data(full_ds)
    print(len(full_ds), len(train_ds), len(val_ds))

    # Câu 3. Định nghĩa metric
    print(miou_metric)

    # Câu 4. Tạo Model
    # model = create_mode()
    model = create_model_Unet()
    model = model.to(device)


    # print(model)
    print("Model device:", next(model.parameters()).device)
    print("Using device:", device)

    # Câu 5. Train
    BATCH_SIZE = 16
    num_workers = 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    #
    FILE_NAME = f"best_{model.__class__.__name__}.pt"

    EPOCHS = 5
    train_model(model, train_loader, val_loader, miou_metric, device, EPOCHS, FILE_NAME)
    #
    # Load model
    model.load_state_dict(torch.load(FILE_NAME, map_location="cpu"))
    model = model.to(device).eval()

    # Câu 6. Chạy kiểm tra
    k=1
    Inference(model, val_ds, k)