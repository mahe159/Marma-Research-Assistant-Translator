import torch
import re
from model_loader import get_model_and_tokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This MUST match the exact prompt structure you used during unsloth SFT training
# Llama-3-8B is highly sensitive to the exact formatting of these tags.
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Translate this {source_lang} sentence into {target_lang}.

### Input:
{text}

### Response:
"""

def generate_translation(text: str, source_lang: str, target_lang: str) -> str:
    """
    Formats the Alpaca prompt, runs GPU-accelerated inference, 
    and cleans the output to return only the translated string.
    """
    try:
        model, tokenizer = get_model_and_tokenizer()
        
        # 1. Format the prompt based on user selection
        prompt = ALPACA_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text=text
        )
        
        # 2. Tokenize input and move to GPU
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 3. Generate response with fixed parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,          # Standard length for sentences
                temperature=0.1,             # Low temp for high accuracy
                do_sample=False,             # Greedy decoding for stable translation
                pad_token_id=tokenizer.eos_token_id,
                
                # --- CRITICAL FIX FOR BROADCAST SHAPE ERROR ---
                # Setting use_cache to False forces the model to recompute 
                # rather than using stored KV states that cause shape mismatches.
                use_cache=False              
            )
        
        # 4. Decode the raw output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 5. Extract ONLY the text after the Response tag
        if "### Response:" in decoded_output:
            translation = decoded_output.split("### Response:")[-1].strip()
        else:
            # Fallback if the model format shifted
            translation = decoded_output.strip()

        # Final cleanup: remove potential leftover special tokens
        translation = translation.replace(tokenizer.eos_token, "").strip()
        
        logger.info(f"Successfully translated: {text[:20]}...")
        return translation
    
    except Exception as e:
        logger.error(f"❌ Translation Generation Error: {str(e)}")
        return f"Internal Error: {str(e)}"

# Example of a manual test call (internal use)
# print(generate_translation("Hello", "English", "Marma English Transliterated"))