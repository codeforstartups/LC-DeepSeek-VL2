from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
import torch
import torch.cuda
import gc
import logging
import os
from io import BytesIO
from PIL import Image
from contextlib import contextmanager

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

# 🔥 GPU Memory Optimization Configuration (GPU-only)
MEMORY_CONFIG = {
    "max_image_size": 320,          # Smaller default size to save memory
    "use_mixed_precision": True,    # FP16 to save memory
    "clear_cache_before_inference": True,
    "clear_cache_after_inference": True,
    "max_retries": 3,
    "retry_image_sizes": [320, 256, 224],  # Progressive size reduction
    "use_gradient_checkpointing": True,
    "max_memory_fraction": 0.85     # Use max 85% of GPU memory
}

# 🔥 GPU Memory Management Functions
def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**2
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved,
            "free_mb": memory_free,
            "total_mb": total_memory,
            "free_percent": (memory_free / total_memory) * 100
        }
    return None

def clear_gpu_memory():
    """Comprehensive GPU memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logging.info("GPU memory cache cleared")

def optimize_gpu_memory():
    """Set GPU memory optimization settings"""
    if torch.cuda.is_available():
        # Set memory fraction to avoid fragmentation
        torch.cuda.set_per_process_memory_fraction(MEMORY_CONFIG["max_memory_fraction"])

        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,roundup_power2_divisions:16"

        logging.info(f"GPU memory optimization enabled - using {MEMORY_CONFIG['max_memory_fraction']*100}% of GPU memory")

@contextmanager
def gpu_inference_context():
    """Context manager for GPU inference with memory optimization"""
    try:
        # Clear memory before inference
        if MEMORY_CONFIG["clear_cache_before_inference"]:
            clear_gpu_memory()

        # Log memory status
        memory_info = get_gpu_memory_info()
        if memory_info:
            logging.info(f"Pre-inference GPU memory: {memory_info['free_mb']:.1f}MB free ({memory_info['free_percent']:.1f}%)")

        yield

    finally:
        # Always clean up after inference
        if MEMORY_CONFIG["clear_cache_after_inference"]:
            clear_gpu_memory()

def optimize_model_for_inference(model):
    """Apply GPU inference optimizations to the model"""
    try:
        model.eval()

        # Disable gradient computation for all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Enable gradient checkpointing if supported
        if MEMORY_CONFIG["use_gradient_checkpointing"] and hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable()
                logging.info("Gradient checkpointing enabled for memory efficiency")
            except Exception as e:
                logging.warning(f"Could not enable gradient checkpointing: {e}")

        logging.info("Model optimized for GPU inference")
        return model
    except Exception as e:
        logging.warning(f"Model optimization failed: {e}")
        return model

# Initialize GPU optimization
optimize_gpu_memory()

# Load model & processor at startup with GPU optimization
try:
    dtype = torch.float16 if MEMORY_CONFIG["use_mixed_precision"] else torch.float32

    logging.info(f"Loading model with dtype: {dtype}")

    # Clear memory before loading model
    clear_gpu_memory()

    processor = DeepseekVLV2Processor.from_pretrained("deepseek-ai/deepseek-vl2-tiny")
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-vl2-tiny",
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
    ).cuda().eval()

    # Apply inference optimizations
    model = optimize_model_for_inference(model)

    logging.info("Medical imaging model and processor loaded successfully.")

    # Log initial memory status
    memory_info = get_gpu_memory_info()
    if memory_info:
        logging.info(f"Post-loading GPU memory: {memory_info['allocated_mb']:.1f}MB allocated, "
                    f"{memory_info['free_mb']:.1f}MB free ({memory_info['free_percent']:.1f}%)")

except Exception as e:
    logging.error(f"Failed to load model or processor: {e}")
    raise RuntimeError(f"Model or processor loading failed: {e}")

def resize_to_multiple(img, multiple=14, fixed_size=None):
    """Resize image with memory-conscious approach"""
    if fixed_size is None:
        fixed_size = MEMORY_CONFIG["max_image_size"]

    # Always resize to fixed_size x fixed_size, both multiples of 14
    target_size = (fixed_size // multiple) * multiple  # Ensure multiple of 14

    if img.size != (target_size, target_size):
        logging.info(f"Resizing image from {img.size} to ({target_size}, {target_size}) for memory optimization")
        img = img.resize((target_size, target_size), Image.BICUBIC)
    return img

def safe_gpu_inference(model, inputs, max_tokens, temperature, top_p, attempt_size=None):
    """Perform GPU inference with memory management and retry logic"""

    # Use provided size or start with default
    if attempt_size is None:
        attempt_size = MEMORY_CONFIG["max_image_size"]

    for attempt, size in enumerate(MEMORY_CONFIG["retry_image_sizes"]):
        if attempt_size and size > attempt_size:
            continue

        try:
            logging.info(f"GPU inference attempt {attempt + 1} with image size {size}")

            with gpu_inference_context():
                with torch.no_grad():
                    if MEMORY_CONFIG["use_mixed_precision"]:
                        # Use automatic mixed precision for memory efficiency
                        with torch.cuda.amp.autocast():
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
                                use_cache=False  # Disable cache to save memory
                            )
                    else:
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
                            use_cache=False
                        )

            logging.info(f"GPU inference successful on attempt {attempt + 1}")
            return outputs

        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"CUDA OOM on attempt {attempt + 1} with size {size}: {e}")
            clear_gpu_memory()

            if attempt < len(MEMORY_CONFIG["retry_image_sizes"]) - 1:
                logging.info(f"Retrying with smaller image size: {MEMORY_CONFIG['retry_image_sizes'][attempt + 1]}")
                continue
            else:
                logging.error("All GPU inference attempts failed due to memory constraints")
                raise HTTPException(
                    status_code=500,
                    detail=f"GPU memory insufficient for inference. Try with a smaller image or reduce max_tokens. Error: {str(e)}"
                )

    raise HTTPException(500, "GPU inference failed after all attempts")

@app.get("/health")
async def health_check():
    """Health check endpoint with GPU memory status"""
    health_info = {
        "status": "healthy",
        "service": "Unified Analysis API (Medical + OCR)",
        "model": "deepseek-ai/deepseek-vl2-tiny",
        "supported_types": ["medical", "ocr", "object_detection"],
        "device": str(model.device) if hasattr(model, 'device') else "unknown",
        "memory_config": MEMORY_CONFIG
    }

    # Add GPU memory info
    memory_info = get_gpu_memory_info()
    if memory_info:
        health_info["gpu_memory"] = {
            "allocated_mb": round(memory_info["allocated_mb"], 1),
            "free_mb": round(memory_info["free_mb"], 1),
            "total_mb": round(memory_info["total_mb"], 1),
            "free_percent": round(memory_info["free_percent"], 1)
        }

    return health_info

@app.post("/analyze/")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    analysis_type: str = Form(default="medical")
):
    """
    Unified image analysis endpoint with GPU memory optimization

    Args:
        file: Image file (medical scans, documents, photos, etc.)
        prompt: Analysis request or specific focus area
        analysis_type: Type of analysis ("medical", "ocr", or "object_detection")
    """
    logging.info(f"Received analysis request: file={file.filename if file else None}, type={analysis_type}")

    # Log initial memory status
    memory_info = get_gpu_memory_info()
    if memory_info:
        logging.info(f"Pre-analysis GPU memory: {memory_info['free_mb']:.1f}MB free ({memory_info['free_percent']:.1f}%)")

    # Validate analysis type
    if analysis_type not in ["medical", "ocr", "object_detection"]:
        raise HTTPException(status_code=400, detail="analysis_type must be 'medical', 'ocr', or 'object_detection'")

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
            pil_image = resize_to_multiple(pil_image, multiple=14, fixed_size=MEMORY_CONFIG["max_image_size"])
            images = [pil_image]
            logging.info("Image loaded and resized successfully.")
        except Exception as e:
            logging.error(f"Invalid image file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Prepare analysis inputs with type-specific system prompts
        try:
            logging.info(f"Preparing {analysis_type} analysis inputs...")

            # Define system prompts with reduced token limits for memory efficiency
            if analysis_type == "medical":
                system_prompt = (
                    "You are an expert medical imaging AI assistant specializing in radiological analysis. "
                    "Your task is to analyze medical scans (MRI, CT, X-Ray) and provide detailed observations about potential abnormalities. "
                    "Focus on identifying suspicious areas, masses, density changes, and other radiological findings. "
                    "Always emphasize that your analysis is for educational purposes and cannot replace professional medical diagnosis. "
                    "Provide clear, structured observations with appropriate medical terminology while remaining accessible."
                )
                max_tokens = 200  # Reduced for memory efficiency
                temperature = 0.3
                top_p = 0.8
            elif analysis_type == "ocr":
                system_prompt = (
                    "You are an expert OCR assistant. Extract all visible text from the image as accurately as possible. "
                    "Return only the recognized text, formatted clearly and preserving the original structure. "
                    "Do not add any extra commentary or explanation—just output the extracted text."
                )
                max_tokens = 200  # Reduced for memory efficiency
                temperature = 0.2
                top_p = 0.9
            elif analysis_type == "object_detection":
                system_prompt = (
                    "You are an expert object detection AI assistant. "
                    "Analyze images and detect specific objects, people, animals, or items as requested. "
                    "Provide clear, direct responses about what you observe in the image. "
                    "Be precise and factual. If the requested object is not present, clearly state this."
                )
                max_tokens = 150  # Reduced for memory efficiency
                temperature = 0.2
                top_p = 0.7

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
            ).to(model.device, dtype=torch.float16 if MEMORY_CONFIG["use_mixed_precision"] else torch.float32)
            logging.info(f"{analysis_type.capitalize()} analysis inputs prepared.")
        except Exception as e:
            logging.error(f"{analysis_type.capitalize()} input processing failed: {repr(e)}")
            raise HTTPException(status_code=500, detail=f"Input processing failed: {e}")

        # Generate analysis with GPU memory-safe inference
        try:
            logging.info(f"Generating {analysis_type} analysis...")

            outputs = safe_gpu_inference(
                model=model,
                inputs=inputs,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
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

        # Log final memory status
        memory_info = get_gpu_memory_info()
        if memory_info:
            logging.info(f"Post-analysis GPU memory: {memory_info['free_mb']:.1f}MB free ({memory_info['free_percent']:.1f}%)")

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
    finally:
        # Always clean up memory after request
        clear_gpu_memory()

@app.get("/gpu_memory")
async def gpu_memory_status():
    """Get current GPU memory status"""
    memory_info = get_gpu_memory_info()
    if memory_info:
        return {
            "gpu_memory": memory_info,
            "config": MEMORY_CONFIG,
            "status": "healthy" if memory_info["free_percent"] > 10 else "low_memory"
        }
    else:
        return {"error": "CUDA not available"}

@app.post("/clear_gpu_cache")
async def clear_gpu_cache():
    """Manually clear GPU memory cache"""
    try:
        clear_gpu_memory()
        memory_info = get_gpu_memory_info()
        return {
            "message": "GPU cache cleared successfully",
            "memory_after_clear": memory_info
        }
    except Exception as e:
        return {"error": f"Failed to clear GPU cache: {e}"}
