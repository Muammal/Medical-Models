# main.py

from skin import predict_skin_disease
from brain import predict_brain_tumor
from xray import analyze_cxr
from llm import get_disease_info


def run_model(model_name: str, image_path: str):

    model_name = model_name.lower()

    # -----------------------------
    # SKIN MODEL
    # -----------------------------
    if model_name == "skin":

        disease = predict_skin_disease(image_path)

        explanation = get_disease_info(disease)

        return {
            "model": "skin",
            "prediction": disease,
            "explanation": explanation
        }

    # -----------------------------
    # BRAIN MRI MODEL
    # -----------------------------
    elif model_name == "brain":

        result = predict_brain_tumor(image_path)

        explanation = get_disease_info(result["class"])

        return {
            "model": "brain",
            "prediction": result,
            "explanation": explanation
        }

    # -----------------------------
    # CHEST X-RAY MODEL
    # -----------------------------
    elif model_name == "cxr":

        result = analyze_cxr(image_path)

        return {
            "model": "cxr",
            "prediction": result["analysis"]
        }

    # -----------------------------
    # INVALID MODEL
    # -----------------------------
    else:

        return {
            "error": "Invalid model name. Use: skin | brain | cxr"
        }