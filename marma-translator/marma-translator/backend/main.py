from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os

from model_loader import load_llama_model
from translate import generate_translation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Marma LLaMA-3 Translator API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Request/Response Schemas
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str

# Load Model on Startup
@app.on_event("startup")
async def startup_event():
    # UPDATE THIS PATH to where your saved Unsloth model is in Drive
    model_path = os.getenv("MODEL_PATH", "/content/drive/MyDrive/Final_Llama_Model")
    logger.info(f"Initializing API... Loading model from {model_path}")
    try:
        load_llama_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Startup failed: {e}")

@app.get("/")
async def root():
    return {"status": "online", "model": "LLaMA-3-8B (Marma Fine-tuned)"}

@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    try:
        logger.info(f"Translating: {request.source_lang} -> {request.target_lang}")
        
        # Call the inference logic
        translated_text = generate_translation(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        
        return TranslationResponse(
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))