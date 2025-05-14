from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from deepseek_vl2.utils.io import load_pil_images
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
import torch
import logging
import os

app = FastAPI(
    title="DeepSeek-VL2 Inference API",
    description="Upload an image and prompt; returns VL2-generated text."
)

# Load model & processor at startup
try:
    dtype = torch.float16
    processor = DeepseekVLV2Processor.from_pretrained("deepseek-ai/deepseek-vl2-tiny")
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-vl2-tiny",
        trust_remote_code=True,
        torch_dtype=dtype
    ).cuda().eval()
    logging.info("Model and processor loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or processor: {e}")
    raise RuntimeError(f"Model or processor loading failed: {e}")

@app.post("/infer/")
async def infer(file: UploadFile = File(...), prompt: str = Form(...)):
    logging.info(f"Received request: file={file.filename if file else None}, content_type={file.content_type if file else None}, prompt={prompt}")
    try:
        if not file:
            logging.error("No file uploaded.")
            raise HTTPException(status_code=400, detail="No file uploaded.")
        if not prompt:
            logging.error("Prompt is required.")
            raise HTTPException(status_code=400, detail="Prompt is required.")
        content = await file.read()
        logging.info(f"File content length: {len(content)} bytes")
        if not content:
            logging.error("Uploaded file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            logging.info("Attempting to load image with PIL...")
            images = load_pil_images([{"images": [content]}])
            logging.info("Image loaded successfully.")
        except Exception as e:
            logging.error(f"Invalid image file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        # Prepare inputs
        try:
            logging.info("Preparing model inputs...")
            inputs = processor(
                conversations=[{"role": "<|User|>", "content": prompt}],
                images=images,
                force_batchify=True
            ).to(model.device)
            logging.info("Model inputs prepared.")
        except Exception as e:
            logging.error(f"Input processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Input processing failed: {e}")
        # Generate embeddings & output
        try:
            logging.info("Generating embeddings and running inference...")
            embeds = model.prepare_inputs_embeds(**inputs)
            outputs = model.language_model.generate(
                inputs_embeds=embeds,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256
            )
            text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info("Inference complete.")
        except Exception as e:
            logging.error(f"Model inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
        logging.info(f"Returning response: {text}")
        return {"response": text}
    except HTTPException as he:
        logging.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {e}"})
