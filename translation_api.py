import os
import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel
import logging
import time
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Language Mapping ---
# Maps simple language codes to NLLB-200's specific format
LANGUAGE_CODE_MAP = {
    "en": "eng_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn",
    "hi": "hin_Deva", "ar": "arb_Arab", "ru": "rus_Cyrl",
    "zh": "zho_Hans", "mandarin": "zho_Hans", "cn": "zho_Hans",
    "fa": "pes_Arab",
    "ja": "jpn_Jpan", "it": "ita_Latn",
    "pt": "por_Latn", "ko": "kor_Hang", "nl": "nld_Latn"
}

# --- Request and Response Models ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "en"  # Default to English

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    generation_time_seconds: float

# --- Thread-Safe Translator Class ---
class Translator:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.lock = Lock()

    def load(self):
        logger.info(f"Loading model and tokenizer for {self.model_id} on device {self.device}")
        try:
            # Load tokenizer without a default source language
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            logger.info("✅ Model and tokenizer loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load model or tokenizer: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> tuple[str, float]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")

        # Ensure thread-safe tokenization
        with self.lock:
            self.tokenizer.src_lang = src_lang
            inputs = self.tokenizer(text, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_start = time.time()
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=512
        )
        generation_time = time.time() - generation_start

        decoded_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return decoded_text, generation_time

# --- FastAPI App ---
app = FastAPI(
    title="NLLB Translation API",
    description="An API to translate text between supported languages using the NLLB-200 model.",
    version="1.1.0"
)

translator: Translator = None

@app.on_event("startup")
async def startup_event():
    global translator
    model_id = "facebook/nllb-200-distilled-600M"
    translator = Translator(model_id)
    translator.load()

# --- API Endpoints ---
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text from a source language to a target language.
    """
    if not translator:
        raise HTTPException(status_code=503, detail="Model not loaded. The service might be starting up.")

    # Validate source and target languages
    if request.source_language not in LANGUAGE_CODE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source language. Supported codes: {list(LANGUAGE_CODE_MAP.keys())}"
        )
    if request.target_language not in LANGUAGE_CODE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target language. Supported codes: {list(LANGUAGE_CODE_MAP.keys())}"
        )

    try:
        start_time = time.time()
        nllb_src_code = LANGUAGE_CODE_MAP[request.source_language]
        nllb_tgt_code = LANGUAGE_CODE_MAP[request.target_language]

        logger.info(f"Translating from {request.source_language} ({nllb_src_code}) to {request.target_language} ({nllb_tgt_code})")

        decoded_text, generation_time = translator.translate(request.text, nllb_src_code, nllb_tgt_code)

        total_time = time.time() - start_time
        logger.info(f"🧠 Model generation took: {generation_time:.4f} seconds")
        logger.info(f"Translation successful. Output: {decoded_text}")
        logger.info(f"⏱️ Total request time: {total_time:.4f} seconds")

        return TranslationResponse(
            translated_text=decoded_text,
            source_language=request.source_language,
            target_language=request.target_language,
            generation_time_seconds=round(generation_time, 4)
        )

    except Exception as e:
        logger.error(f"An error occurred during translation: {e}")
        raise HTTPException(status_code=500, detail="Translation failed due to a server error.")

@app.get("/health")
async def health_check():
    model_loaded = translator is not None and translator.model is not None and translator.tokenizer is not None
    return {"status": "ok", "model_loaded": model_loaded}
