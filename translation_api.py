import os
import torch
from fastapi import FastAPI, HTTPException, Body
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from pydantic import BaseModel
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Initialize FastAPI App
app = FastAPI(
    title="NLLB Translation API",
    description="An API to translate English text to 12 other languages using a quantized NLLB-200 model.",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

# --- Language Mapping ---
# Maps simple language codes to NLLB-200's specific format
LANGUAGE_CODE_MAP = {
    "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn",
    "hi": "hin_Deva", "ar": "arb_Arab", "ru": "rus_Cyrl",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "it": "ita_Latn",
    "pt": "por_Latn", "ko": "kor_Hang", "nl": "nld_Latn"
}

# 2. Load Model and Tokenizer on Startup
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    model_id = "facebook/nllb-200-distilled-600M"

    # Forcing CPU to avoid bitsandbytes issues on Windows
    device = "cpu"
    logger.info(f"Using device: {device}")

    try:
        logger.info(f"Loading tokenizer for model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")

        logger.info(f"Loading model: {model_id} (no quantization)")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(device)
        logger.info("✅ Model and tokenizer loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load model or tokenizer: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

# 3. Define Request and Response Models
class TranslationRequest(BaseModel):
    text: str
    target_language: str

class TranslationResponse(BaseModel):
    translated_text: str
    target_language: str
    generation_time_seconds: float

# 4. Create the /translate Endpoint
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate English text to a specified target language.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. The service might be starting up.")

    # Validate target language
    if request.target_language not in LANGUAGE_CODE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target language. Supported codes: {list(LANGUAGE_CODE_MAP.keys())}"
        )

    try:
        start_time = time.time()
        # Map simple code to NLLB format
        nllb_lang_code = LANGUAGE_CODE_MAP[request.target_language]
        logger.info(f"Translating to: {request.target_language} ({nllb_lang_code})")

        # Tokenize the input text (src_lang is already set on the tokenizer)
        inputs = tokenizer(request.text, return_tensors="pt")

        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate translated tokens
        generation_start = time.time()
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_lang_code),
            max_length=512
        )
        generation_time = time.time() - generation_start
        logger.info(f"🧠 Model generation took: {generation_time:.4f} seconds")

        # Decode the tokens to a string
        decoded_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        total_time = time.time() - start_time
        logger.info(f"Translation successful. Output: {decoded_text}")
        logger.info(f"⏱️ Total request time: {total_time:.4f} seconds")

        return TranslationResponse(
            translated_text=decoded_text,
            target_language=request.target_language,
            generation_time_seconds=round(generation_time, 4)
        )

    except Exception as e:
        logger.error(f"An error occurred during translation: {e}")
        raise HTTPException(status_code=500, detail="Translation failed due to a server error.")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None and tokenizer is not None}
