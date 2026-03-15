# 🩺 MedicHub – AI Skin Disease Detector

![MedicHub Sticker](https://media.giphy.com/media/l0Exk8EUzSLsrErEQ/giphy.gif)

Welcome to **MedicHub**, your **friendly AI dermatologist**!  
Upload a picture of a skin condition, get a **disease prediction**, and even **ask questions about it** – all running **locally** (no cloud servers required 😎).

---

## Features 🚀

- **Vision Model**: Detects skin diseases from images using `dinov2-base-finetuned-SkinDisease`.  
- **Local LLM**: Provides **detailed explanations** and answers questions about the detected disease.  
- **FastAPI Server**: Turns your model into a **local API** for websites or apps.  
- **100% Offline Option**: No Groq or external API required.  

![Funny doctor](https://media.giphy.com/media/l2JHRhAtnJSDNJ2py/giphy.gif)

---

## 📂 Project Structure
MedicHub/
│
├── main.py # CLI terminal app
├── vision_model.py # Skin disease classifier
├── llm_module.py # Local LLM handler
├── api_server.py # FastAPI server for web
└── requirements.txt # Python dependencies

````markdown
## 🛠 Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/yourusername/medichub.git
cd medichub
````

---

### 2️⃣ Create a virtual environment

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download your local LLM (Ollama example)

```bash
ollama pull llama3
```

```

✅ Key points:

- Every code block **opens and closes with triple backticks**.  
- Add `bash` (or `powershell` for Windows) after the opening backticks to get proper syntax highlighting.  
- Don’t put extra spaces or empty lines before the closing backticks.  
- Separate sections with `---` to keep things clean and readable.  

---

If you want, I can **rewrite the entire README.md** for MedicHub using this same formatting style so **all code blocks (terminal, API commands, JSON responses) are perfectly highlighted** for GitHub.  

Do you want me to do that next?
```
