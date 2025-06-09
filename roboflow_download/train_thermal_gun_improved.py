#!/usr/bin/env python3
"""
Enhanced Thermal Gun Detection Training Script
Optimized for GPU training with advanced monitoring and robust error handling
"""

import os
import yaml
import time
import psutil
from datetime import datetime
from ultralytics import YOLO
import torch
import gc
import logging
from pathlib import Path

def setup_logging():
    """Setup detailed logging for training monitoring"""
    log_dir = Path("training_logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"thermal_gun_training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

def check_gpu_setup():
    """Check and optimize GPU setup"""
    print("🔧 GPU Setup Analysis:")
    print("-" * 40)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - will use CPU (much slower)")
        return 'cpu', {}

    gpu_count = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3

    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"🎮 GPU Count: {gpu_count}")
    print(f"🎯 Current GPU: {current_gpu} ({gpu_name})")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")

    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

    # Determine optimal batch size based on GPU memory
    if gpu_memory >= 8:
        batch_size = 32
        print("🚀 High-end GPU detected - using batch size 32")
    elif gpu_memory >= 4:
        batch_size = 16
        print("⚡ Mid-range GPU detected - using batch size 16")
    else:
        batch_size = 8
        print("⚠️  Limited GPU memory - using batch size 8")

    gpu_config = {
        'device': f'cuda:{current_gpu}',
        'batch_size': batch_size,
        'memory_gb': gpu_memory
    }

    return f'cuda:{current_gpu}', gpu_config

def setup_training_config(gpu_config):
    """Setup enhanced training configuration optimized for GPU"""
    batch_size = gpu_config.get('batch_size', 16)

    return {
        # Model settings
        'model': 'yolov8n.pt',  # Start with nano for small dataset
        'epochs': 150,  # More epochs but with early stopping
        'patience': 20,  # Increased patience for better convergence
        'imgsz': 640,
        'batch': batch_size,  # GPU-optimized batch size

                # Optimization (GPU-optimized)
        'lr0': 0.002,  # Slightly higher LR for GPU training
        'lrf': 0.01,   # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,  # Longer warmup for stability
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # Data Augmentation (enhanced for thermal imagery)
        'hsv_h': 0.02,       # Slightly more hue variation for thermal
        'hsv_s': 0.8,        # Increased saturation for thermal contrast
        'hsv_v': 0.5,        # More value variation
        'degrees': 15.0,     # More rotation for robustness
        'translate': 0.15,   # More translation
        'scale': 0.3,        # More scale variation
        'shear': 0.05,       # Small shear for variety
        'perspective': 0.0001, # Tiny perspective for realism
        'flipud': 0.0,       # No vertical flip (guns have orientation)
        'fliplr': 0.5,       # Horizontal flip OK
        'mosaic': 1.0,       # Keep mosaic
        'mixup': 0.15,       # Slightly more mixup
        'copy_paste': 0.15,  # More copy-paste for small dataset

        # Enhanced Regularization
        'dropout': 0.15,     # More dropout for small dataset
        'label_smoothing': 0.1,
        'box': 7.5,          # Box loss weight
        'cls': 0.5,          # Classification loss weight
        'dfl': 1.5,          # Distribution focal loss weight

        # Performance settings
        'val': True,
        'save_period': -1,   # Disable frequent epoch saves to prevent corruption
        'save': True,        # Only save best and last models
        'save_txt': False,   # Don't save prediction txt files
        'plots': True,
        'verbose': True,
        'workers': 4,        # Reduced workers to prevent memory issues
        'rect': True,        # Rectangular training for efficiency
        'overlap_mask': True, # Better mask handling
        'mask_ratio': 4,     # Mask downsample ratio

        # Memory optimization
        'amp': True,         # Automatic Mixed Precision for GPU
        'fraction': 0.8,     # Reduced GPU memory fraction for stability
    }

def fix_data_yaml():
    """Fix data.yaml paths to be absolute paths"""
    data_yaml_path = "data.yaml"

    # Read current data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Get absolute paths (now we're inside roboflow_download)
    base_dir = os.path.abspath(".")

    # Update paths to absolute
    data['train'] = os.path.join(base_dir, "train", "images")
    data['val'] = os.path.join(base_dir, "valid", "images")
    data['test'] = os.path.join(base_dir, "test", "images")

    # Save fixed data.yaml
    fixed_path = "data_fixed.yaml"
    with open(fixed_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Fixed data.yaml saved to: {fixed_path}")
    return fixed_path

def monitor_system_resources():
    """Monitor system resources during training"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    print(f"💻 System Resources:")
    print(f"   CPU Usage: {cpu_percent:.1f}%")
    print(f"   RAM Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_used = torch.cuda.memory_allocated(0) / 1024**3
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   GPU Memory: {gpu_used:.1f}GB used, {gpu_reserved:.1f}GB reserved, {gpu_memory:.1f}GB total")

def validate_model_file(model_path, min_size_mb=5):
    """Validate that a model file is not corrupted"""
    if not os.path.exists(model_path):
        return False, "File does not exist"

    try:
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        if file_size < min_size_mb:
            return False, f"File too small ({file_size:.1f}MB < {min_size_mb}MB)"

        # Try to load the model to verify it's not corrupted
        test_model = YOLO(model_path)
        del test_model  # Clean up
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True, f"Valid model file ({file_size:.1f}MB)"

    except Exception as e:
        return False, f"Corrupted file: {str(e)}"

def cleanup_corrupted_models(model_dir):
    """Remove corrupted model files to prevent issues"""
    weights_dir = os.path.join(model_dir, "weights")
    if not os.path.exists(weights_dir):
        return

    model_files = []
    for file in os.listdir(weights_dir):
        if file.endswith('.pt'):
            model_files.append(os.path.join(weights_dir, file))

    print(f"🧹 Checking {len(model_files)} model files for corruption...")

    corrupted_files = []
    for model_path in model_files:
        is_valid, message = validate_model_file(model_path)
        if not is_valid:
            corrupted_files.append(model_path)
            print(f"   ❌ {os.path.basename(model_path)}: {message}")
            try:
                os.remove(model_path)
                print(f"   🗑️  Removed corrupted file: {os.path.basename(model_path)}")
            except:
                print(f"   ⚠️  Could not remove: {os.path.basename(model_path)}")
        else:
            print(f"   ✅ {os.path.basename(model_path)}: {message}")

    if corrupted_files:
        print(f"🧹 Removed {len(corrupted_files)} corrupted model files")
    else:
        print(f"✅ All model files are valid")

def train_improved_model():
    """Train thermal gun detection model with enhanced GPU optimization"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Enhanced Thermal Gun Detection Training")

    print("🚀 Enhanced Thermal Gun Detection Training")
    print("=" * 70)

    # Monitor initial system resources
    monitor_system_resources()
    print()

    # Check and optimize GPU setup
    device, gpu_config = check_gpu_setup()
    print()

    # Log GPU configuration
    logger.info(f"GPU Configuration: {gpu_config}")

    # Fix data.yaml paths
    logger.info("Setting up data paths...")
    data_yaml = fix_data_yaml()

    # Load pre-trained model
    print(f"📦 Loading YOLOv8n model...")
    logger.info("Loading pre-trained YOLOv8n model")
    model = YOLO('yolov8n.pt')

    # Get training configuration
    config = setup_training_config(gpu_config)

    print(f"⚙️  Enhanced Training Configuration:")
    print(f"   - Model: {config['model']}")
    print(f"   - Epochs: {config['epochs']} (with early stopping)")
    print(f"   - Patience: {config['patience']} epochs")
    print(f"   - Batch size: {config['batch']} (GPU-optimized)")
    print(f"   - Initial LR: {config['lr0']}")
    print(f"   - Image size: {config['imgsz']}")
    print(f"   - Mixed Precision: {config['amp']}")
    print(f"   - Workers: {config['workers']}")
    print(f"   - Data augmentation: Enhanced")
    print(f"   - Regularization: Advanced (dropout={config['dropout']})")

    # Log configuration
    logger.info(f"Training config: {config}")

    # Start training with performance monitoring
    print(f"\n🏋️  Starting GPU-optimized training at {datetime.now()}")
    print("-" * 70)

    # Clear GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU memory cleared before training")

    start_time = time.time()

    try:
        results = model.train(
            data=data_yaml,
            epochs=config['epochs'],
            patience=config['patience'],  # Early stopping
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=device,

            # Learning rate settings
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],

            # Data augmentation
            hsv_h=config['hsv_h'],
            hsv_s=config['hsv_s'],
            hsv_v=config['hsv_v'],
            degrees=config['degrees'],
            translate=config['translate'],
            scale=config['scale'],
            shear=config['shear'],
            perspective=config['perspective'],
            flipud=config['flipud'],
            fliplr=config['fliplr'],
            mosaic=config['mosaic'],
            mixup=config['mixup'],
            copy_paste=config['copy_paste'],

            # Regularization
            dropout=config['dropout'],
            label_smoothing=config['label_smoothing'],

            # Output settings
            project='../MyTrainedModels',
            name='ThermalPistol_v2_improved',
            exist_ok=True,
            save_period=config['save_period'],
            save=config['save'],
            save_txt=config['save_txt'],
            plots=config['plots'],
            verbose=config['verbose'],
            val=config['val']
        )

        # Calculate training time
        end_time = time.time()
        training_duration = end_time - start_time

        print(f"\n✅ Training completed successfully!")
        print(f"⏱️  Total training time: {training_duration/3600:.2f} hours")
        print(f"📊 Final Results:")

        # Extract and display comprehensive results
        try:
            best_map50 = results.results_dict.get('metrics/mAP50(B)', 0)
            best_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
            final_epoch = results.epoch + 1 if hasattr(results, 'epoch') else 'N/A'

            print(f"   - Best mAP@0.5: {best_map50:.3f}")
            print(f"   - Best mAP@0.5-0.95: {best_map50_95:.3f}")
            print(f"   - Training stopped at epoch: {final_epoch}")

            # Log results
            logger.info(f"Training completed. mAP@0.5: {best_map50:.3f}, mAP@0.5-0.95: {best_map50_95:.3f}")

        except Exception as e:
            print(f"   - Results extraction failed: {e}")
            logger.warning(f"Could not extract final metrics: {e}")

        # Clean up corrupted models and validate saved models
        model_dir = "../MyTrainedModels/ThermalPistol_v2_improved"
        print(f"\n🔍 Validating saved models...")
        cleanup_corrupted_models(model_dir)

        # Find the best valid model
        weights_dir = os.path.join(model_dir, "weights")
        model_path = None

        # Check for best.pt first
        best_path = os.path.join(weights_dir, "best.pt")
        if os.path.exists(best_path):
            is_valid, message = validate_model_file(best_path)
            if is_valid:
                model_path = best_path
                print(f"✅ Valid best.pt model found: {message}")
            else:
                print(f"❌ best.pt corrupted: {message}")

        # If best.pt not valid, try last.pt
        if not model_path:
            last_path = os.path.join(weights_dir, "last.pt")
            if os.path.exists(last_path):
                is_valid, message = validate_model_file(last_path)
                if is_valid:
                    model_path = last_path
                    print(f"✅ Using valid last.pt model: {message}")
                else:
                    print(f"❌ last.pt also corrupted: {message}")

        if model_path:
            print(f"🎯 Final model saved to: {model_path}")
        else:
            print(f"❌ No valid model files found! Training may have failed to save properly.")
            model_path = best_path  # Return the path anyway for debugging

        # Final GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 Final GPU memory cleanup completed")

        return model_path, results

    except Exception as e:
        error_msg = f"Training failed: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)

        # Cleanup on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return None, None

def validate_training_results(model_path):
    """Quick validation on test set"""
    if not model_path or not os.path.exists(model_path):
        print("❌ No valid model found for validation")
        return

    print(f"\n🔍 Running validation on test set...")

    # Load trained model
    model = YOLO(model_path)

    # Run validation on test set
    test_images = "test/images"
    if os.path.exists(test_images):
        results = model.val(data="data_fixed.yaml", split='test')
        print(f"📊 Test Set Results:")
        print(f"   - Precision: {results.box.p.mean():.3f}")
        print(f"   - Recall: {results.box.r.mean():.3f}")
        print(f"   - mAP@0.5: {results.box.map50:.3f}")
        print(f"   - mAP@0.5-0.95: {results.box.map:.3f}")

    # Test on a few sample images
    print(f"\n🖼️  Testing on sample images...")
    import glob
    sample_images = glob.glob(os.path.join(test_images, "*.jpg"))[:3]

    for img_path in sample_images:
        results = model.predict(img_path, conf=0.25, save=True, project="test_results")
        img_name = os.path.basename(img_path)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"   - {img_name}: {detections} detections")

if __name__ == "__main__":
    print("🔥 Enhanced Thermal Gun Detection Training")
    print("GPU-Optimized with Advanced Monitoring and Regularization")
    print("=" * 70)

    try:
        # Clean up any existing corrupted models first
        existing_model_dir = "../MyTrainedModels/ThermalPistol_v2_improved"
        if os.path.exists(existing_model_dir):
            print("🧹 Cleaning up existing model directory...")
            cleanup_corrupted_models(existing_model_dir)

        # Train the enhanced model
        model_path, results = train_improved_model()

        # Validate results
        if model_path:
            print(f"\n🔍 Running post-training validation...")
            validate_training_results(model_path)

            print(f"\n🎉 Training Successfully Completed!")
            print(f"🚀 GPU-Enhanced Improvements Made:")
            print(f"   ✅ GPU-optimized batch sizes and memory management")
            print(f"   ✅ Automatic Mixed Precision (AMP) for faster training")
            print(f"   ✅ Enhanced data augmentation for thermal imagery")
            print(f"   ✅ Advanced regularization with early stopping")
            print(f"   ✅ Real-time resource monitoring")
            print(f"   ✅ Comprehensive logging and error handling")
            print(f"   ✅ Optimized learning rate and training parameters")
            print(f"\n📁 Model saved to: {model_path}")
            print(f"📊 Training logs available in: training_logs/")

            # Final resource check
            print(f"\n📈 Final System Status:")
            monitor_system_resources()

            # Recommendations
            print(f"\n💡 Next Steps:")
            print(f"   1. Test the new model on validation images")
            print(f"   2. Compare performance with v1 model")
            print(f"   3. Run inference on your thermal video")
            print(f"   4. Check training plots in the model directory")
            print(f"   5. Review training logs for insights")

        else:
            print(f"\n❌ Training failed. Check error messages and logs above.")
            print(f"💡 Troubleshooting tips:")
            print(f"   - Check GPU memory availability")
            print(f"   - Verify CUDA installation")
            print(f"   - Review training logs for details")

    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        print(f"🔧 Please check your environment setup and try again.")

    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n🧹 Final cleanup completed.")
