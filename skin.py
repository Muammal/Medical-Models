import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

MODEL_NAME = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Vision model running on: {DEVICE}")

# Load model
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

def predict_skin_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    return predicted_label

# Test run if executed directly
if __name__ == "__main__":

    image_path = input("Enter image path: ")
    disease = predict_skin_disease(image_path)
    print("\nDetected disease:", disease)