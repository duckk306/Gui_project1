import cv2
import torch
import numpy as np

from models.MiMo import MiMo
from utils.BboxUtils import decode_bbox


# -------------------------
# Config
# -------------------------
IMG_SIZE = 640
STRIDES = [8, 16, 32]
NUM_CLASSES = 3
CONF_THRESH = 0.4


# -------------------------
# Utils
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def collect_detections(outputs, strides):
    """
    Convert raw detection outputs to bbox list.
    Return: list of (x1, y1, x2, y2, score, class_id)
    """
    dets = []

    for out, stride in zip(outputs, strides):
        cls_logits = out["cls"][0]   # (C,H,W)
        reg = out["reg"][0]          # (4,H,W)

        cls_prob = torch.sigmoid(cls_logits)

        boxes = decode_bbox(reg.unsqueeze(0), stride)[0]  # (4,H,W)

        C, H, W = cls_prob.shape
        for c in range(C):
            mask = cls_prob[c] > CONF_THRESH
            ys, xs = mask.nonzero(as_tuple=True)

            for y, x in zip(ys, xs):
                score = cls_prob[c, y, x].item()
                cx, cy, w, h = boxes[:, y, x]

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                dets.append((x1, y1, x2, y2, score, c))

    return dets


def draw_detections(img, dets):
    for x1, y1, x2, y2, score, cls_id in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"C{cls_id}:{score:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return img


# -------------------------
# Main
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MiMo(num_classes=NUM_CLASSES).to(device)
    model.eval()

    # Load image
    img = cv2.imread("traffic.jpg")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_vis = img.copy()

    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)

    # -------- Drive area --------
    drive_logits = outputs["drive_area"][0, 0]
    drive_mask = (torch.sigmoid(drive_logits) > 0.5).cpu().numpy()

    overlay = img_vis.copy()
    overlay[drive_mask] = (overlay[drive_mask] * 0.5 + np.array([0, 0, 255]) * 0.5)

    # -------- Detection --------
    dets = collect_detections(outputs["detection"], STRIDES)
    overlay = draw_detections(overlay, dets)

    # Show
    cv2.imshow("MiMo Inference", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
