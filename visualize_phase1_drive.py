import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.MiMo import MiMo
from datasets.BDD100kDriveDataset import BDD100kDriveDataset


# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "bdd100k_seg/bdd100k/seg"
SPLIT = "val"   # train / val
CKPT_PATH = "checkpoints_bdd100k/mimo_bdd100k_phase1_epoch29.pt"

IMG_SIZE = 640
NUM_SAMPLES = 6
THRESH = 0.5
# =========================================


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm(img):
    """Tensor (3,H,W) -> uint8 image"""
    img = img.permute(1, 2, 0).cpu().numpy()
    img = img * STD + MEAN
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def main():
    # -------- Dataset --------
    dataset = BDD100kDriveDataset(
        root=DATA_ROOT,
        split=SPLIT,
        img_size=IMG_SIZE
    )

    # -------- Model --------
    model = MiMo(num_classes=3)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.to(DEVICE).eval()

    idxs = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)

    plt.figure(figsize=(15, 5 * NUM_SAMPLES))

    with torch.no_grad():
        for i, idx in enumerate(idxs):
            sample = dataset[idx]

            img = sample["image"].unsqueeze(0).to(DEVICE)
            gt = sample["drive_area"][0].cpu().numpy()   # (H,W)

            # -------- Inference --------
            out = model(img)
            prob = torch.sigmoid(out["drive_area"])[0, 0].cpu().numpy()
            pred = (prob > THRESH).astype(np.uint8)

            # -------- Visualization --------
            img_vis = denorm(sample["image"])

            overlay = img_vis.copy()

            # True Positive - Green
            overlay[(pred == 1) & (gt == 1)] = [0, 255, 0]

            # False Positive - Red
            overlay[(pred == 1) & (gt == 0)] = [255, 0, 0]

            # False Negative - Blue
            overlay[(pred == 0) & (gt == 1)] = [0, 0, 255]

            # -------- Plot --------
            plt.subplot(NUM_SAMPLES, 3, i * 3 + 1)
            plt.imshow(img_vis)
            plt.title("Input")
            plt.axis("off")

            plt.subplot(NUM_SAMPLES, 3, i * 3 + 2)
            plt.imshow(gt, cmap="gray")
            plt.title("GT Drivable Area")
            plt.axis("off")

            plt.subplot(NUM_SAMPLES, 3, i * 3 + 3)
            plt.imshow(overlay)
            plt.title("Prediction Overlay (TP/FP/FN)")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
