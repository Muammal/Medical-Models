# brain_model.py

import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchvision import models, transforms

# ==========================================
# CONFIG
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEG_IMG_SIZE = 224
CLS_IMG_SIZE = 224

CLASSES = {
    0: 'No Tumor',
    1: 'Glioma Tumor',
    2: 'Meningioma Tumor',
    3: 'Pituitary Tumor'
}

# ==========================================
# LOAD MODELS
# ==========================================

def load_seg_model():
    model = smp.Unet(
        encoder_name="tu-swin_tiny_patch4_window7_224",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

    checkpoint = torch.load("models\\swin_unet_best.pth", map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


def load_cls_model():
    model = models.efficientnet_b3(weights=None)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 4)

    checkpoint = torch.load("models\\efficientnet_b3_cls.pth", map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


# Load once (IMPORTANT for API speed)
seg_model = load_seg_model()
cls_model = load_cls_model()

# ==========================================
# PREPROCESSING
# ==========================================

seg_transform = A.Compose([
    A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

cls_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# MAIN FUNCTION (USED BY API)
# ==========================================

def predict_brain_tumor(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -------- Classification --------
    cls_input = cls_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_out = cls_model(cls_input)
        probs = torch.softmax(cls_out, dim=1)[0]

    confidences = {
        CLASSES[i]: float(probs[i]) for i in range(4)
    }

    top_class_id = torch.argmax(probs).item()

    # -------- Segmentation --------
    h, w = image.shape[:2]

    aug = seg_transform(image=image)
    seg_input = aug['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_out = seg_model(seg_input)
        pred_mask = (torch.sigmoid(seg_out) > 0.5).float().cpu().numpy().squeeze()

    pred_mask = cv2.resize(pred_mask, (w, h))

    # -------- Visualization --------
    output_image = image.copy()

    tumor_detected = bool(np.any(pred_mask))

    if tumor_detected:
        overlay = output_image.copy()

        colors = {
            0: (255, 0, 0),
            1: (0, 0, 255),
            2: (212, 28, 15),
            3: (0, 255, 0)
        }

        color = colors.get(top_class_id, (255, 0, 0))

        overlay[pred_mask == 1] = color

        output_image = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

        contours, _ = cv2.findContours(
            pred_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(output_image, contours, -1, color, 2)

    return {
        "tumor_detected": tumor_detected,
        "class": CLASSES[top_class_id],
        "confidences": confidences,
        "image": output_image  # numpy image
    }