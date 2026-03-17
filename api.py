# api_server.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import tempfile
import cv2
import base64

from main import run_model

app = FastAPI(title="MedicHub AI API")

# Allow website connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------
# HEALTH CHECK
# ----------------------------------

@app.get("/")
def home():
    return {"status": "MedicHub API running"}


# ----------------------------------
# IMAGE ENCODER (for returning images)
# ----------------------------------

def encode_image(image_np):

    _, buffer = cv2.imencode(".jpg", image_np)

    return base64.b64encode(buffer).decode("utf-8")


# ----------------------------------
# MAIN ANALYSIS ENDPOINT
# ----------------------------------

@app.post("/analyze")

async def analyze(
    model: str = Form(...),
    file: UploadFile = File(...)
):

    # Save uploaded image
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:

        tmp.write(contents)

        image_path = tmp.name

    # Run selected model
    result = run_model(model, image_path)

    # Encode segmentation image if brain model
    if model == "brain":

        if "image" in result.get("prediction", {}):
            img = result["prediction"].pop("image")

            result["prediction"]["segmentation_image"] = encode_image(img)

    return result