"""
Research Assistant Translator - FastAPI Backend
A scalable translation API with dynamic model routing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Assistant Translator API",
    description="Multi-language translation API with dynamic model routing",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL ROUTING SYSTEM - Add your trained models here
# ============================================================================

class ModelRouter:
    """
    Scalable model routing system using a dictionary-based approach.
    
    To add your trained models:
    1. Load each model in the load_models() method
    2. Add the model reference to MODEL_MAP using the key format: "SourceLang_TargetLang"
    3. Update SUPPORTED_LANGUAGES list if adding new languages
    """
    
    def __init__(self):
        self.MODEL_MAP: Dict[str, any] = {}
        self.SUPPORTED_LANGUAGES = ["English", "Bangla", "Marma English Transliterated", "Marma Bangla Transliterated"]
        self.load_models()
    
    def load_models(self):
        """
        Load all translation models here.
        Replace the dummy models with your actual PyTorch/HuggingFace models.
        
        Example for loading real models:
        --------------------------------
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        # Load English -> Marma model
        tokenizer_eng_marma = AutoTokenizer.from_pretrained("./models/eng_marma")
        model_eng_marma = AutoModelForSeq2SeqLM.from_pretrained("./models/eng_marma")
        
        self.MODEL_MAP["English_Marma"] = {
            "model": model_eng_marma,
            "tokenizer": tokenizer_eng_marma
        }
        """
        
        logger.info("Loading translation models...")
        
        # ====================================================================
        # REPLACE THESE DUMMY MODELS WITH YOUR ACTUAL TRAINED MODELS
        # ====================================================================
        
        # All translation directions between 4 languages
        self.MODEL_MAP = {
            # English to other languages
            "English_Bangla": self._create_dummy_model("English", "Bangla"),
            "English_Marma English Transliterated": self._create_dummy_model("English", "Marma English Transliterated"),
            "English_Marma Bangla Transliterated": self._create_dummy_model("English", "Marma Bangla Transliterated"),
            
            # Bangla to other languages
            "Bangla_English": self._create_dummy_model("Bangla", "English"),
            "Bangla_Marma English Transliterated": self._create_dummy_model("Bangla", "Marma English Transliterated"),
            "Bangla_Marma Bangla Transliterated": self._create_dummy_model("Bangla", "Marma Bangla Transliterated"),
            
            # Marma English Transliterated to other languages
            "Marma English Transliterated_English": self._create_dummy_model("Marma English Transliterated", "English"),
            "Marma English Transliterated_Bangla": self._create_dummy_model("Marma English Transliterated", "Bangla"),
            "Marma English Transliterated_Marma Bangla Transliterated": self._create_dummy_model("Marma English Transliterated", "Marma Bangla Transliterated"),
            
            # Marma Bangla Transliterated to other languages
            "Marma Bangla Transliterated_English": self._create_dummy_model("Marma Bangla Transliterated", "English"),
            "Marma Bangla Transliterated_Bangla": self._create_dummy_model("Marma Bangla Transliterated", "Bangla"),
            "Marma Bangla Transliterated_Marma English Transliterated": self._create_dummy_model("Marma Bangla Transliterated", "Marma English Transliterated"),
        }
        
        logger.info(f"Loaded {len(self.MODEL_MAP)} translation models")
        logger.info(f"Available directions: {list(self.MODEL_MAP.keys())}")
    
    def _create_dummy_model(self, source: str, target: str):
        """
        Dummy model for demonstration purposes.
        Replace this with actual model loading logic.
        """
        return {
            "model": f"dummy_model_{source}_{target}",
            "tokenizer": f"dummy_tokenizer_{source}_{target}",
            "source": source,
            "target": target
        }
    
    def get_model(self, source_lang: str, target_lang: str):
        """
        Retrieve the appropriate model based on source and target languages.
        
        Args:
            source_lang: Source language name
            target_lang: Target language name
            
        Returns:
            Model object for the specified translation direction
            
        Raises:
            KeyError: If translation direction is not available
        """
        model_key = f"{source_lang}_{target_lang}"
        
        if model_key not in self.MODEL_MAP:
            available = ", ".join(self.MODEL_MAP.keys())
            raise KeyError(
                f"Translation from '{source_lang}' to '{target_lang}' is not available. "
                f"Available directions: {available}"
            )
        
        return self.MODEL_MAP[model_key]
    
    def get_available_directions(self):
        """Get all available translation directions."""
        return list(self.MODEL_MAP.keys())
    
    def get_supported_languages(self):
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES


# Initialize the model router
model_router = ModelRouter()

# ============================================================================
# API MODELS (Request/Response Schemas)
# ============================================================================

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "source_lang": "English",
                "target_lang": "Marma"
            }
        }


class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    model_used: str


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Research Assistant Translator API",
        "version": "1.0.0"
    }


@app.get("/languages")
async def get_languages():
    """Get list of supported languages."""
    return {
        "languages": model_router.get_supported_languages()
    }


@app.get("/directions")
async def get_directions():
    """Get all available translation directions."""
    return {
        "available_directions": model_router.get_available_directions(),
        "total_count": len(model_router.get_available_directions())
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text from source language to target language.
    
    This endpoint dynamically selects the appropriate model based on
    the source and target language pair.
    """
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Translation text cannot be empty"
            )
        
        # Get the appropriate model for this translation direction
        model_info = model_router.get_model(request.source_lang, request.target_lang)
        model_key = f"{request.source_lang}_{request.target_lang}"
        
        logger.info(f"Translating: {request.source_lang} -> {request.target_lang}")
        
        # ================================================================
        # PERFORM ACTUAL TRANSLATION HERE
        # Replace this dummy logic with your model inference code
        # ================================================================
        
        # Example for real translation:
        # tokenizer = model_info["tokenizer"]
        # model = model_info["model"]
        # inputs = tokenizer(request.text, return_tensors="pt", padding=True)
        # outputs = model.generate(**inputs, max_length=512)
        # translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Dummy translation (remove this in production)
        translated_text = f"[{request.target_lang} translation of: '{request.text}']"
        
        return TranslationResponse(
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            model_used=model_key
        )
        
    except KeyError as e:
        logger.error(f"Translation direction not available: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during translation: {str(e)}"
        )


# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting Research Assistant Translator API")
    print("=" * 60)
    print(f"📊 Loaded {len(model_router.get_available_directions())} translation models")
    print(f"🌐 Supported languages: {', '.join(model_router.get_supported_languages())}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
