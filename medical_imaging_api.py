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
    title="DeepSeek-VL2 Unified Analysis API",
    description="Upload images for AI-assisted analysis using DeepSeek VL2. Supports both medical imaging analysis and OCR text extraction."
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
    """Health check endpoint for unified analysis API"""
    return {
        "status": "healthy",
        "service": "Unified Analysis API (Medical + OCR)",
        "model": "deepseek-ai/deepseek-vl2-tiny",
        "supported_types": ["medical", "ocr"]
    }

@app.post("/analyze/")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    analysis_type: str = Form(default="medical")
):
    """
    Unified image analysis endpoint supporting multiple analysis types

    Args:
        file: Image file (medical scans, documents, photos, etc.)
        prompt: Analysis request or specific focus area
        analysis_type: Type of analysis ("medical" or "ocr")
    """
    logging.info(f"Received analysis request: file={file.filename if file else None}, type={analysis_type}, prompt={prompt}")

    # Validate analysis type
    if analysis_type not in ["medical", "ocr"]:
        raise HTTPException(status_code=400, detail="analysis_type must be 'medical' or 'ocr'")

    try:
        if not file:
            logging.error("No image file uploaded.")
            raise HTTPException(status_code=400, detail="No image file uploaded.")
        if not prompt:
            logging.error("Prompt is required.")
            raise HTTPException(status_code=400, detail="Prompt is required.")

        content = await file.read()
        logging.info(f"Image file content length: {len(content)} bytes")

        if not content:
            logging.error("Uploaded image file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

        try:
            logging.info(f"Attempting to load image for {analysis_type} analysis...")
            pil_image = Image.open(BytesIO(content))
            pil_image = pil_image.convert("RGB")
            pil_image = resize_to_multiple(pil_image, multiple=14, fixed_size=512)
            images = [pil_image]
            logging.info("Image loaded successfully.")
        except Exception as e:
            logging.error(f"Invalid image file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Prepare analysis inputs with type-specific system prompts
        try:
            logging.info(f"Preparing {analysis_type} analysis inputs...")

            # Define system prompts for different analysis types
            if analysis_type == "medical":
                system_prompt = (
                    "You are an expert medical imaging AI assistant specializing in radiological analysis. "
                    "Your task is to analyze medical scans (MRI, CT, X-Ray) and provide detailed observations about potential abnormalities. "
                    "Focus on identifying suspicious areas, masses, density changes, and other radiological findings that may indicate cancer or other pathologies. "
                    "Always emphasize that your analysis is for educational purposes and cannot replace professional medical diagnosis. "
                    "Provide clear, structured observations with appropriate medical terminology while remaining accessible. "
                    "Include confidence levels and recommendations for professional medical consultation."
                )
                max_tokens = 256
                temperature = 0.3
                top_p = 0.8
            elif analysis_type == "ocr":
                system_prompt = (
                    "You are an expert OCR assistant. When the user provides an image, your task is to extract all visible text from the image as accurately as possible. "
                    "Return only the recognized text, formatted clearly and preserving the original structure (such as lines or paragraphs) if possible. "
                    "Do not add any extra commentary or explanation—just output the extracted text."
                )
                max_tokens = 256
                temperature = 0.4
                top_p = 0.9

            inputs = processor(
                conversations=[
                    {
                        "role": "<|User|>",
                        "content": "<image>\n" + prompt,
                        "images": [f"{analysis_type}_image.png"]
                    },
                    {"role": "<|Assistant|>", "content": ""}
                ],
                images=images,
                force_batchify=True,
                system_prompt=system_prompt
            ).to(model.device, dtype=torch.float16)
            logging.info(f"{analysis_type.capitalize()} analysis inputs prepared.")
        except Exception as e:
            logging.error(f"{analysis_type.capitalize()} input processing failed: {repr(e)}")
            raise HTTPException(status_code=500, detail=f"Input processing failed: {e}")

        # Generate analysis with type-specific parameters
        try:
            logging.info(f"Generating {analysis_type} analysis...")
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
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                use_cache=True
            )
            analysis_text = processor.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):].cpu().tolist(),
                skip_special_tokens=True
            )
            logging.info(f"{analysis_type.capitalize()} analysis complete.")
        except Exception as e:
            logging.error(f"{analysis_type.capitalize()} analysis failed: {repr(e)}")
            raise HTTPException(status_code=500, detail=f"{analysis_type} analysis failed: {e}")

        # Return response with analysis type info
        response = {
            "response": analysis_text,
            "analysis_type": analysis_type,
            "filename": file.filename
        }

        # Add disclaimer for medical analysis
        if analysis_type == "medical":
            response["disclaimer"] = "This AI analysis is for educational and research purposes only. Always consult with qualified medical professionals for proper diagnosis and treatment."

        logging.info(f"Returning {analysis_type} analysis response for {file.filename}")
        return response

    except HTTPException as he:
        logging.error(f"HTTPException in {analysis_type} analysis: {he.detail}")
        raise he
    except Exception as e:
        logging.error(f"Unexpected error in {analysis_type} analysis: {repr(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected server error in {analysis_type} analysis: {e}"}
        )
