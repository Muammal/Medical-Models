import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import uuid
import logging
from transformers import AutoModelForImageClassification, AutoImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATIC_DIR = os.path.join(os.getcwd(), "static", "cams")

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

logger.info(f"Vision model running on: {DEVICE}")

# Load model and processor
try:
    _raw_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    _raw_model.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    logger.info("Model and processor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

class ModelWrapper(torch.nn.Module):
    """
    Wrapper to ensure the model returns a Tensor (logits) instead of a HF Output object.
    This prevents Grad-CAM from iterating over the keys of the output object.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits

# Use the wrapper for CAM
wrapped_model = ModelWrapper(_raw_model)

def reshape_transform(tensor):
    """
    Reshape transform for DINOv2 (ViT) architecture.
    Converts (batch, 257, 768) -> (batch, 768, 16, 16)
    """
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    
    # Remove the CLS token (first token)
    # DINOv2 usually has 257 tokens for 224x224 input
    if tensor.size(1) == 257:
        result = tensor[:, 1:, :]
    else:
        result = tensor

    # Calculate grid size (sqrt of patch tokens)
    # 256 -> 16
    grid_size = int(result.size(1)**0.5)
    
    # Reshape: (batch, height, width, hidden_dim)
    result = result.reshape(result.size(0), grid_size, grid_size, result.size(2))
    
    # Permute to (batch, hidden_dim, height, width) for CAM library
    result = result.permute(0, 3, 1, 2)
    return result

def predict_skin_disease(image_path):
    """
    Predicts skin disease and generates Grad-CAM visualization.
    Only returns predicted_disease and gradcam_image_path.
    Confidence score is completely removed.
    """
    try:
        # Load and process image
        orig_image = Image.open(image_path).convert("RGB")
        # Resize to 224x224 as standard for ViT
        image_224 = orig_image.resize((224, 224))
        inputs = processor(images=image_224, return_tensors="pt").to(DEVICE)
        
        # 1. Classification Prediction
        with torch.no_grad():
            outputs = _raw_model(**inputs)
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        
        # Get predicted labels
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        predicted_label = _raw_model.config.id2label[predicted_idx]
        
        logger.info(f"Model prediction: {predicted_label}")

        # 2. Grad-CAM Generation
        gradcam_path = None
        try:
            # Target the last layer of the DINOv2 encoder
            # Using the raw model's sub-modules for targeting
            target_layers = [_raw_model.dinov2.encoder.layer[-1]]
            
            # Initialize Grad-CAM with the wrapped model and reshape transform
            cam = GradCAM(model=wrapped_model, 
                          target_layers=target_layers, 
                          reshape_transform=reshape_transform)
            
            # Target the predicted class
            targets = [ClassifierOutputTarget(predicted_idx)]
            
            # Generate mask (uses wrapped_model internally)
            # Pass the raw pixel values tensor
            grayscale_cam = cam(input_tensor=inputs['pixel_values'], targets=targets)
            
            # Process result
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay heatmap on original image (resized to 224)
            img_array = np.array(image_224) / 255.0
            visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
            
            # 3. Save Grad-CAM image
            filename = f"cam_{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(STATIC_DIR, filename)
            
            # Save using OpenCV (requires BGR for imwrite)
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
            gradcam_path = f"/static/cams/{filename}"
            
            if os.path.exists(save_path):
                logger.info(f"Grad-CAM saved successfully: {gradcam_path} ({os.path.getsize(save_path)} bytes)")
            else:
                logger.error("Grad-CAM file was not created.")
                gradcam_path = None

        except Exception as cam_err:
            logger.error(f"Grad-CAM failed: {cam_err}")
            # Ensure we still return the prediction even if CAM fails
            gradcam_path = None

        return {
            "predicted_disease": predicted_label,
            "gradcam_image_path": gradcam_path
        }

    except Exception as e:
        logger.error(f"Critical error in vision flow: {e}")
        # Final fallback classification
        try:
            img = Image.open(image_path).convert("RGB").resize((224, 224))
            inp = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = _raw_model(**inp)
            label = _raw_model.config.id2label[torch.argmax(out.logits, -1).item()]
            return {
                "predicted_disease": label,
                "gradcam_image_path": None
            }
        except Exception as f_err:
            logger.error(f"Deep failure: {f_err}")
            raise e

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_input.jpg"
    if os.path.exists(path):
        print("\n=== ANALYSIS RESULT ===")
        print(predict_skin_disease(path))
    else:
        print(f"Usage: python skin.py <image_path>")