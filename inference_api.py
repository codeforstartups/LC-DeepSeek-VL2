from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepseek_vl2.utils.io import load_pil_images
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
import torch
import logging

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
except Exception as e:
    logging.error(f"Failed to load model or processor: {e}")
    raise RuntimeError(f"Model or processor loading failed: {e}")

@app.post("/infer/")
async def infer(file: UploadFile = File(...), prompt: str = ""):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            images = load_pil_images([{"images": [content]}])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        # Prepare inputs
        try:
            inputs = processor(
                conversations=[{"role": "<|User|>", "content": prompt}],
                images=images,
                force_batchify=True
            ).to(model.device)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Input processing failed: {e}")
        # Generate embeddings & output
        try:
            embeds = model.prepare_inputs_embeds(**inputs)
            outputs = model.language_model.generate(
                inputs_embeds=embeds,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256
            )
            text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
        return {"response": text}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {e}"})
