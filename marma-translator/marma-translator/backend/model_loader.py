from unsloth import FastLanguageModel
import torch
import logging
import os

# Configure logging to see status in the Colab console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to ensure the model stays in GPU memory across requests
model = None
tokenizer = None

def load_llama_model(model_path: str):
    """
    Loads the specialized Unsloth Llama-3 model.
    This bypasses standard HuggingFace 'weight' errors by using 
    Unsloth's native optimized loader.
    """
    global model, tokenizer
    
    # Singleton pattern: don't reload if already in memory
    if model is not None and tokenizer is not None:
        return model, tokenizer

    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        raise FileNotFoundError(f"Could not find model at {model_path}")

    logger.info(f"🚀 Loading specialized Unsloth model from: {model_path}")
    
    try:
        # FastLanguageModel handles the 4-bit/LoRA structure correctly
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,   # Matches your training configuration
            dtype = None,            # Automatically detects (Float16/Bfloat16)
            load_in_4bit = True,     # Essential for Colab memory management
        )
        
        # Enable 2x faster inference mode (Unsloth optimization)
        FastLanguageModel.for_inference(model)
        
        logger.info("✅ Model and Tokenizer loaded successfully using Unsloth!")
        return model, tokenizer

    except Exception as e:
        logger.error(f"❌ Critical error during model loading: {str(e)}")
        raise e

def get_model_and_tokenizer():
    """
    Retrieves the loaded model and tokenizer. 
    Raises an error if the startup_event hasn't finished yet.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("Model is not loaded. Ensure the FastAPI startup_event finished successfully.")
    return model, tokenizer