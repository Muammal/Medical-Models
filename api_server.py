from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import tempfile
import logging
from skin import predict_skin_disease
from llm import get_disease_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medic Hub API")

# Setup Static Directory
# We use the absolute path to avoid confusion with working directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CAMS_DIR = os.path.join(STATIC_DIR, "cams")
os.makedirs(CAMS_DIR, exist_ok=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for Grad-CAM images
# Important: Ensure the mount point matches what skin.py returns (/static/cams/...)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return {
        "message": "Medic Hub API running locally",
        "static_dir": STATIC_DIR
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Endpoint to analyze skin medical images.
    Accepts an image, returns predicted disease, LLM explanation, and Grad-CAM image path.
    """
    logger.info(f"Received analysis request for file: {file.filename}")
    
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    temp_path = None
    try:
        # Save upload to a temporary file
        ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, 'wb') as tmp:
            shutil.copyfileobj(file.file, tmp)

        # 1. Run Skin Disease Prediction and Grad-CAM
        logger.info("Running vision model analysis...")
        vision_result = predict_skin_disease(temp_path)
        
        predicted_disease = vision_result.get("predicted_disease", "Unknown")
        gradcam_image = vision_result.get("gradcam_image_path")
        
        # 2. Get LLM explanation
        logger.info(f"Fetching LLM explanation for: {predicted_disease}")
        llm_explanation = get_disease_info(predicted_disease)
        
        # 3. Comprehensive Logging
        logger.info("Analysis complete.")
        logger.info(f"Result: {predicted_disease}")
        logger.info(f"Grad-CAM Path: {gradcam_image}")
        
        # Return result WITHOUT confidence score as requested
        return {
            "predicted_disease": predicted_disease,
            "llm_explanation": llm_explanation,
            "gradcam_image": gradcam_image
        }

    except Exception as e:
        logger.error(f"API Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Server error during analysis: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    # Host 0.0.0.0 for external access, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
