from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
import torch
import logging
import os
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DeepSeek-VL2 Medical Imaging API",
    description="Upload MRI/CT scan images for AI-assisted cancer detection analysis using DeepSeek VL2."
)

# Add CORS middleware to allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    logging.info("Medical imaging model and processor loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or processor: {e}")
    raise RuntimeError(f"Model or processor loading failed: {e}")

def resize_to_multiple(img, multiple=14, fixed_size=512):
    # Always resize to fixed_size x fixed_size, both multiples of 14
    if img.size != (fixed_size, fixed_size):
        logging.info(f"Resizing medical image from {img.size} to ({fixed_size}, {fixed_size}) to match patch size multiple {multiple} and reduce memory usage.")
        img = img.resize((fixed_size, fixed_size), Image.BICUBIC)
    return img

@app.get("/health")
async def health_check():
    """Health check endpoint for medical imaging API"""
    return {
        "status": "healthy",
        "service": "Medical Imaging Cancer Detection API",
        "model": "deepseek-ai/deepseek-vl2-tiny"
    }

@app.post("/analyze_scan/")
async def analyze_medical_scan(file: UploadFile = File(...), prompt: str = Form(...)):
    """
    Analyze MRI/CT scan images for potential cancer detection

    Args:
        file: Medical scan image file (DICOM, PNG, JPEG)
        prompt: Analysis request or specific focus area
    """
    logging.info(f"Received medical scan analysis request: file={file.filename if file else None}, prompt={prompt}")

    try:
        if not file:
            logging.error("No medical scan file uploaded.")
            raise HTTPException(status_code=400, detail="No medical scan file uploaded.")
        if not prompt:
            logging.error("Prompt is required.")
            raise HTTPException(status_code=400, detail="Prompt is required.")

        content = await file.read()
        logging.info(f"Medical scan file content length: {len(content)} bytes")

        if not content:
            logging.error("Uploaded medical scan file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded medical scan file is empty.")

        try:
            logging.info("Attempting to load medical scan image with PIL...")
            pil_image = Image.open(BytesIO(content))
            pil_image = pil_image.convert("RGB")
            pil_image = resize_to_multiple(pil_image, multiple=14, fixed_size=512)
            images = [pil_image]
            logging.info("Medical scan image loaded successfully.")
        except Exception as e:
            logging.error(f"Invalid medical scan image file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid medical scan image file: {e}")

        # Prepare medical imaging analysis inputs
        try:
            logging.info("Preparing medical imaging analysis inputs...")

            system_prompt = (
                "You are an expert medical imaging AI assistant specializing in radiological analysis. "
                "Your task is to analyze medical scans (MRI, CT, X-Ray) and provide detailed observations about potential abnormalities. "
                "Focus on identifying suspicious areas, masses, density changes, and other radiological findings that may indicate cancer or other pathologies. "
                "Always emphasize that your analysis is for educational purposes and cannot replace professional medical diagnosis. "
                "Provide clear, structured observations with appropriate medical terminology while remaining accessible. "
                "Include confidence levels and recommendations for professional medical consultation."
            )

            inputs = processor(
                conversations=[
                    {
                        "role": "<|User|>",
                        "content": "<image>\n" + prompt,
                        "images": ["medical_scan.png"]
                    },
                    {"role": "<|Assistant|>", "content": ""}
                ],
                images=images,
                force_batchify=True,
                system_prompt=system_prompt
            ).to(model.device, dtype=torch.float16)
            logging.info("Medical imaging analysis inputs prepared.")
        except Exception as e:
            logging.error(f"Medical imaging input processing failed: {repr(e)}")
            raise HTTPException(status_code=500, detail=f"Input processing failed: {e}")

        # Generate medical analysis
        try:
            logging.info("Generating medical imaging analysis...")
            embeds = model.prepare_inputs_embeds(**inputs)
            outputs = model.generate(
                inputs_embeds=embeds,
                input_ids=inputs.input_ids,
                images=inputs.images,
                images_seq_mask=inputs.images_seq_mask,
                images_spatial_crop=inputs.images_spatial_crop,
                attention_mask=inputs.attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=512,  # Longer output for detailed medical analysis
                do_sample=True,
                temperature=0.3,  # Lower temperature for more consistent medical analysis
                top_p=0.8,
                repetition_penalty=1.1,
                use_cache=True
            )
            analysis_text = processor.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):].cpu().tolist(),
                skip_special_tokens=True
            )
            logging.info("Medical imaging analysis complete.")
        except Exception as e:
            logging.error(f"Medical imaging analysis failed: {repr(e)}")
            raise HTTPException(status_code=500, detail=f"Medical analysis failed: {e}")

        logging.info(f"Returning medical analysis response for {file.filename}")
        return {"response": analysis_text}

    except HTTPException as he:
        logging.error(f"HTTPException in medical analysis: {he.detail}")
        raise he
    except Exception as e:
        logging.error(f"Unexpected error in medical analysis: {repr(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected server error in medical analysis: {e}"}
        )
