from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tempfile
from skin import predict_skin_disease
from llm import get_disease_info

app = FastAPI(title="Medic Hub AI API")

# Enable website access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later to your website domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Medic Hub API running locally"}

@app.post("/analyze")
async def analyze_skin_image(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Convert to PIL image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run vision model
    predicted_disease = predict_skin_disease(temp_path)

    # Run local LLM
    disease_info = get_disease_info(predicted_disease)

    return {
        "predicted_disease": predicted_disease,
        "llm_explanation": disease_info
    }