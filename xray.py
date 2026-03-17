# cxr_model.py

import torch
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from PIL import Image

# ==========================================
# CONFIG
# ==========================================

MODEL_NAME = "nvidia/NV-Reason-CXR-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading NV-Reason-CXR-3B...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

model.eval()

print("✅ CXR model loaded")

# ==========================================
# MAIN FUNCTION
# ==========================================

def analyze_cxr(image_path, prompt=None):
    """
    Analyze chest X-ray image with reasoning
    """

    image = Image.open(image_path).convert("RGB")

    if prompt is None:
        prompt = "Analyze this chest X-ray. Describe findings and possible diagnosis."

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    response = processor.decode(output[0], skip_special_tokens=True)

    return {
        "model": "cxr",
        "analysis": response
    }