#!/usr/bin/env python3
"""
Test the Enhanced V2 Thermal Gun Detection Model
Clean and focused testing script
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import glob

def test_v2_model():
    """Test the V2 improved model with clean, focused approach"""
    print("🔍 Testing Enhanced V2 Thermal Gun Detection Model")
    print("=" * 60)

    # Model path - use best.pt from successful training
    v2_model_path = "../MyTrainedModels/ThermalPistol_v2_improved/weights/best.pt"

    # Check if model exists
    if not os.path.exists(v2_model_path):
        print(f"❌ V2 model not found at: {v2_model_path}")
        return

    print(f"✅ V2 model found: {v2_model_path}")

    # Load V2 model
    print("📦 Loading V2 model...")
    model_v2 = YOLO(v2_model_path)

    # Get validation images
    val_images_dir = "valid/images"
    image_files = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    print(f"📊 Found {len(image_files)} validation images")

    # Test with optimal confidence threshold
    print(f"\n🎯 Testing V2 Model Performance (conf=0.25):")
    print("-" * 50)

    total_detections = 0
    images_with_detections = 0

    for img_path in image_files:
        results = model_v2(img_path, conf=0.25, verbose=False)
        detections = len(results[0].boxes) if len(results) > 0 and results[0].boxes is not None else 0
        total_detections += detections
        if detections > 0:
            images_with_detections += 1

    print(f"Images with detections: {images_with_detections}/{len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(image_files):.2f}")

    # Test on sample images with visualization
    print(f"\n🖼️  Sample Image Results:")
    print("-" * 30)

    sample_images = image_files[:3]  # Test first 3 images
    results_dir = "v2_test_results"
    os.makedirs(results_dir, exist_ok=True)

    for i, img_path in enumerate(sample_images):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        # Run inference
        results = model_v2(img_path, conf=0.25, verbose=False)

        # Process results
        detections = 0
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            detections = len(boxes)

            # Draw detections
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                confidence = boxes.conf[j].cpu().numpy()

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'Gun {confidence:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"{img_name}: {detections} detections")

        # Save visualization
        output_path = os.path.join(results_dir, f"v2_{i+1:02d}_{img_name}")
        cv2.imwrite(output_path, img)

    print(f"\n📁 Visualizations saved to: {results_dir}/")

    # Run validation metrics
    print(f"\n🔍 Running Validation Metrics:")
    print("-" * 35)

    try:
        val_results = model_v2.val(data="data_fixed.yaml", split='val', verbose=False)
        precision = val_results.box.p.mean()
        recall = val_results.box.r.mean()
        map50 = val_results.box.map50
        map50_95 = val_results.box.map

        print(f"Precision:     {precision:.3f}")
        print(f"Recall:        {recall:.3f}")
        print(f"mAP@0.5:       {map50:.3f}")
        print(f"mAP@0.5-0.95:  {map50_95:.3f}")

        # Performance assessment
        if map50 > 0.8:
            print(f"\n🎉 EXCELLENT performance! (mAP@0.5 > 0.8)")
        elif map50 > 0.5:
            print(f"\n✅ GOOD performance! (mAP@0.5 > 0.5)")
        else:
            print(f"\n⚠️  Model needs improvement (mAP@0.5 < 0.5)")

    except Exception as e:
        print(f"❌ Validation failed: {e}")

    print(f"\n🎯 V2 Model Testing Complete!")
    print(f"Ready for real-world thermal video testing.")

if __name__ == "__main__":
    test_v2_model()
