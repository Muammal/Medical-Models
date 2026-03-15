MedicHub – AI Skin Disease Detector

Welcome to MedicHub, your friendly AI dermatologist!
Upload a picture of a skin condition, get a disease prediction, and even ask questions about it – all running locally (no cloud servers required 😎).

Features 🚀

Vision Model: Detects skin diseases from images using dinov2-base-finetuned-SkinDisease.

Local LLM: Provides detailed explanations and answers questions about the detected disease.

FastAPI Server: Turns your model into a local API for websites or apps.

100% Offline Option: No Groq or external API required.

MedicHub/
│
├── main.py              # CLI terminal app
├── vision_model.py      # Skin disease classifier
├── llm_module.py        # Local LLM handler
├── api_server.py        # FastAPI server for web
└── requirements.txt     # Python dependencies

🛠 Installation

Clone the repo:

git clone https://github.com/yourusername/medichub.git
cd medichub

Create a virtual environment:

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

Install dependencies:

pip install -r requirements.txt

Download your local LLM (Ollama example):

ollama pull llama3
⚡ Running the Apps
1️⃣ Terminal Mode
python main.py

Follow the prompts:

Enter path to skin image: images/acne.jpg
Detected Disease: Acne
Explanation: Acne is caused by...
Ask questions: "What causes it?" → MedicHub answers!

2️⃣ API Mode (For Web)

Start the FastAPI server:

uvicorn api_server:app --reload

API response example:

{
  "predicted_disease": "Eczema",
  "llm_explanation": "Eczema is a chronic inflammatory skin condition..."
}